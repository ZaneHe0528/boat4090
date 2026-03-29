#!/usr/bin/env python3
"""
警戒线导航脚本 —— 识别两岸警戒线，计算中心线，连接船头生成平滑航行轨迹，
输出小船偏航角。支持模型推理模式和实时相机模式。


source /home/yanmin/ws_boat/src/boat/boat/.venv/bin/activate
python scripts/realtime_pilot_v4.py --camera /dev/video2 --model models/segformer_river/best_model.pth --mavros

运行前请在终端执行（按实际路径修改 venv 与 cusparselt）：

  source .venv/bin/activate
  export LD_LIBRARY_PATH="/path/to/.venv/lib/python3.10/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH}"

【模型推理模式】对图片/目录进行推理，输出可视化图片 + 偏航角 CSV：

  python scripts/realtime_pilot_v4.py \
      --images dataset_final/images/test \
      --model  models/segformer_river/best_model.pth \
      --model-name nvidia/segformer-b0-finetuned-ade-512-512 \
      --output vis_boundary/test

  # 同时弹窗查看
  python scripts/Boundary_identification.py \
      --images dataset_final/images/test \
      --model  models/segformer_river/best_model.pth \
      --output vis_boundary --show

【实时相机模式】从摄像头读帧，实时输出偏航角：
  python scripts/realtime_pilot_v4.py \
      --camera 2 \
      --model  models/segformer_river/best_model.pth


  # 向 MAVROS 发布速度 setpoint（前进 0.5 m/s，角速度由 yaw_deg 换算），并以不低于 --mavros-min-hz
  # 的频率重复上一帧指令；启动约 --mavros-prepare-delay 秒后自动请求 OFFBOARD 并 arm。
  # 默认视觉暂时失败时仍保持前进（仅不转向）；若要停船请加 --mavros-stop-when-not-ok。
  python scripts/realtime_pilot_v4.py \
      --camera 2 \
      --model  models/segformer_river/best_model.pth \
      --mavros

  # 指定相机设备 + 显示画面
  python scripts/Boundary_identification.py \
      --camera /dev/video0 \
      --model  models/segformer_river/best_model.pth \
      --show
"""

import argparse
import math
import sys
import threading
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from visualize_centerline import (
    BOUNDARY_CLASS, WATER_CLASS,
    load_segformer, infer_mask,
    collect_images,
    scan_comp_rows, smooth_row_dict,
    _extrapolate_dict,
)

# ─── 样式常量 ──────────────────────────────────────────────────────────────────
C_YELLOW   = (0, 255, 255)   # BGR 黄色
C_CENTER   = (0, 210, 0)     # BGR 亮绿（中线）
C_SAIL     = (255, 180, 0)   # BGR 蓝色（航行路径）
C_WARN     = (0, 140, 255)   # BGR 橙色（警戒线）
DASH_LEN   = 20
GAP_LEN    = 10
THICKNESS  = 3
MIN_AREA   = 30              # 连通域最少像素数，过滤噪声

ROI_TOP_RATIO    = 0.1       # 去除图像最顶部比例（天空噪声）
WIDTH_OUTLIER_TH = 0.4       # 航道宽度偏差超过此比例的行视为异常

# ─── PIL 中文字体工具 ─────────────────────────────────────────────────────────
_CN_FONT_CACHE: Dict[int, Any] = {}


def _get_pil_font(size: int) -> Any:
    if size in _CN_FONT_CACHE:
        return _CN_FONT_CACHE[size]
    from PIL import ImageFont
    candidates = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc',
        '/usr/share/fonts/truetype/arphic/uming.ttc',
        '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
    ]
    font = None
    for p in candidates:
        if Path(p).exists():
            try:
                font = ImageFont.truetype(p, size)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()
    _CN_FONT_CACHE[size] = font
    return font


def _cn_text_size(text: str, font_size: int) -> Tuple[int, int]:
    font = _get_pil_font(font_size)
    try:
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        from PIL import Image, ImageDraw
        draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
        return draw.textsize(text, font=font)


def _put_cn_texts(canvas: np.ndarray, items: List[Tuple]) -> None:
    try:
        from PIL import Image, ImageDraw
        img_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        for item in items:
            text, xy, fs, cbgr = item[:4]
            stroke_w   = item[4] if len(item) > 4 else 0
            stroke_bgr = item[5] if len(item) > 5 else (0, 0, 0)
            crgb = (int(cbgr[2]), int(cbgr[1]), int(cbgr[0]))
            srgb = (int(stroke_bgr[2]), int(stroke_bgr[1]), int(stroke_bgr[0]))
            font = _get_pil_font(fs)
            if stroke_w > 0:
                draw.text(xy, text, font=font, fill=crgb,
                          stroke_width=stroke_w, stroke_fill=srgb)
            else:
                draw.text(xy, text, font=font, fill=crgb)
        canvas[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception:
        pass


# ─── EMA 时序滤波器 ───────────────────────────────────────────────────────────

class YawFilter:
    """指数滑动平均滤波器，平滑偏航角帧间抖动。"""

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self._prev: Optional[float] = None

    def update(self, value: float) -> float:
        if self._prev is None:
            self._prev = value
            return value
        filtered = self.alpha * value + (1.0 - self.alpha) * self._prev
        self._prev = filtered
        return filtered

    def reset(self):
        self._prev = None


# ─── 黄色虚线绘制 ──────────────────────────────────────────────────────────────

def _draw_dashed_line(
    canvas: np.ndarray,
    pts: List[Tuple[int, int]],
    color: Tuple[int, int, int] = C_YELLOW,
    thickness: int = THICKNESS,
    dash: int = DASH_LEN,
    gap: int = GAP_LEN,
) -> None:
    """沿折线点序列绘制虚线。"""
    if len(pts) < 2:
        return
    drawing = True
    seg_remain = dash

    for i in range(1, len(pts)):
        p1 = np.array(pts[i - 1], dtype=np.float64)
        p2 = np.array(pts[i],     dtype=np.float64)
        seg_len = float(np.linalg.norm(p2 - p1))
        if seg_len < 1e-3:
            continue
        d   = (p2 - p1) / seg_len
        pos = 0.0

        while pos < seg_len:
            step = min(seg_remain, seg_len - pos)
            q1   = p1 + d * pos
            q2   = p1 + d * (pos + step)
            if drawing:
                cv2.line(
                    canvas,
                    (int(round(q1[0])), int(round(q1[1]))),
                    (int(round(q2[0])), int(round(q2[1]))),
                    color, thickness, cv2.LINE_AA,
                )
            pos        += step
            seg_remain -= step
            if seg_remain <= 0:
                drawing    = not drawing
                seg_remain = dash if drawing else gap


# ─── 边界折线提取 ──────────────────────────────────────────────────────────────

def _polyline_side(pts: List[Tuple[int, int]], img_w: int) -> str:
    mean_x = sum(p[0] for p in pts) / len(pts)
    return 'left' if mean_x < img_w / 2 else 'right'


def _min_y(pts: List[Tuple[int, int]]) -> int:
    return min(p[1] for p in pts)


def extract_boundary_polylines(
    mask: np.ndarray,
) -> Tuple[List[List[Tuple[int, int]]], int]:
    """
    从分割 mask 中提取边界折线，每侧最多保留一条（倒影过滤）。
    同侧多条折线会合并为一条。
    返回: (polylines, n_filtered)
    """
    h, w = mask.shape[:2]

    bnd_bin = (mask == BOUNDARY_CLASS).astype(np.uint8)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bnd_bin = cv2.morphologyEx(bnd_bin, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bnd_bin)

    candidates: List[List[Tuple[int, int]]] = []
    for lid in range(1, num_labels):
        if int(stats[lid, cv2.CC_STAT_AREA]) < MIN_AREA:
            continue
        comp_mask = (labels == lid)
        row_dict  = scan_comp_rows(comp_mask)
        row_dict  = smooth_row_dict(row_dict, window=11)
        pts = [(int(round(x)), y) for y, x in sorted(row_dict.items())]
        if len(pts) >= 2:
            candidates.append(pts)

    groups: dict = {'left': [], 'right': []}
    for pts in candidates:
        side = _polyline_side(pts, w)
        groups[side].append(pts)

    polylines: List[List[Tuple[int, int]]] = []
    n_filtered = 0
    for side, side_pts in groups.items():
        if not side_pts:
            continue
        if len(side_pts) == 1:
            polylines.append(side_pts[0])
        else:
            # 合并同侧所有碎片
            merged: dict = {}
            for pts in side_pts:
                for x, y in pts:
                    merged.setdefault(y, []).append(float(x))
            row_dict = {y: float(np.median(xs)) for y, xs in merged.items()}
            row_dict = smooth_row_dict(row_dict, window=11)
            merged_pts = [(int(round(x)), y) for y, x in sorted(row_dict.items())]
            if len(merged_pts) >= 2:
                polylines.append(merged_pts)
            n_filtered += len(side_pts) - 1

    return polylines, n_filtered


# ─── 构建航行路径：船头 → 平滑过渡 → 中线 ──────────────────────────────────────

def _build_sailing_path(
    center_pts: List[Tuple[int, int]],
    img_w: int,
    img_h: int,
    n_interp: int = 30,
) -> List[Tuple[int, int]]:
    """
    从船头位置（图像底部中心）到中线起点做样条平滑过渡，
    拼接成完整航行路径。
    center_pts: 中线点列表，按 y 降序（[0]=最近端, y 最大）。
    """
    boat_x, boat_y = img_w // 2, img_h - 1
    cl_x, cl_y = center_pts[0]

    if abs(cl_y - boat_y) < 5 and abs(cl_x - boat_x) < 5:
        return [(boat_x, boat_y)] + center_pts

    try:
        from scipy.interpolate import CubicSpline
        n_cl = min(5, len(center_pts))
        ctrl_ys = [float(boat_y)] + [float(center_pts[i][1]) for i in range(n_cl)]
        ctrl_xs = [float(boat_x)] + [float(center_pts[i][0]) for i in range(n_cl)]

        clean_ys, clean_xs = [ctrl_ys[0]], [ctrl_xs[0]]
        for yy, xx in zip(ctrl_ys[1:], ctrl_xs[1:]):
            if yy < clean_ys[-1]:
                clean_ys.append(yy)
                clean_xs.append(xx)

        if len(clean_ys) < 3:
            return [(boat_x, boat_y)] + center_pts

        rev_ys = list(reversed(clean_ys))
        rev_xs = list(reversed(clean_xs))
        cs = CubicSpline(rev_ys, rev_xs, bc_type='natural')

        trans_ys = np.linspace(cl_y, boat_y, n_interp + 2)
        trans_xs = cs(trans_ys)

        transition = [
            (int(round(float(x))), int(round(float(y))))
            for x, y in zip(reversed(trans_xs), reversed(trans_ys))
        ]
        return transition + center_pts[1:]

    except Exception:
        n = n_interp
        path = []
        for i in range(n + 1):
            t = i / n
            x = int(round(boat_x + t * (cl_x - boat_x)))
            y = int(round(boat_y + t * (cl_y - boat_y)))
            path.append((x, y))
        path += center_pts[1:]
        return path


# ─── 单帧核心处理：警戒线 → 中线 → 航行轨迹 → 偏航角 ─────────────────────────

def process_frame(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    yaw_filter: Optional[YawFilter] = None,
    roi_top_ratio: float = ROI_TOP_RATIO,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    处理单帧：
      1. 提取左右警戒线
      2. 计算中心线
      3. 构建船头→中线的平滑航行轨迹
      4. 计算偏航角
    返回: (可视化图像, 控制信号字典)
    """
    h, w = img_bgr.shape[:2]
    canvas = img_bgr.copy()

    empty_result = {
        'yaw_deg': 0.0,
        'status': 'no centerline',
        'n_filtered': 0,
        'center_pts': [],
        'path_pts': [],
    }

    # ── 1. 提取左右警戒线 ──
    polylines, n_filtered = extract_boundary_polylines(mask)

    # 边界区域半透明红色底色
    bnd_mask = (mask == BOUNDARY_CLASS)
    if bnd_mask.any():
        tint = canvas.copy()
        tint[bnd_mask] = np.clip(
            tint[bnd_mask].astype(np.int32) + np.array([0, 0, 100], np.int32),
            0, 255,
        ).astype(np.uint8)
        cv2.addWeighted(tint, 0.55, canvas, 0.45, 0, canvas)

    if len(polylines) != 2:
        for pts in polylines:
            _draw_dashed_line(canvas, pts, C_YELLOW)
        empty_result['n_filtered'] = n_filtered
        empty_result['status'] = f'boundary count={len(polylines)}, need 2'
        return canvas, empty_result

    # ── 2. 区分左右 ──
    avg_x = [sum(p[0] for p in pts) / len(pts) for pts in polylines]
    if avg_x[0] <= avg_x[1]:
        left_pts, right_pts = polylines[0], polylines[1]
    else:
        left_pts, right_pts = polylines[1], polylines[0]

    # ── 3. 构建行字典 ──
    sky_cut = int(h * roi_top_ratio)
    left_dict  = {y: float(x) for x, y in left_pts  if y >= sky_cut}
    right_dict = {y: float(x) for x, y in right_pts if y >= sky_cut}

    # ── 4. 插值 + 外推 → 稠密行覆盖 ──
    all_ys = sorted(set(left_dict.keys()) | set(right_dict.keys()))
    if len(all_ys) < 2:
        empty_result['n_filtered'] = n_filtered
        empty_result['status'] = 'too few boundary rows'
        return canvas, empty_result

    y_min_u, y_max_u = all_ys[0], all_ys[-1]
    left_dense  = _extrapolate_dict(left_dict,  y_min_u, y_max_u)
    right_dense = _extrapolate_dict(right_dict, y_min_u, y_max_u)
    overlap_ys  = sorted(set(left_dense.keys()) & set(right_dense.keys()))

    if len(overlap_ys) < 3:
        empty_result['n_filtered'] = n_filtered
        empty_result['status'] = 'overlap rows < 3'
        return canvas, empty_result

    # ── 5. 异常宽度过滤 ──
    width_dict = {y: right_dense[y] - left_dense[y] for y in overlap_ys}
    near_rows = [y for y in overlap_ys
                 if y >= max(overlap_ys) - (max(overlap_ys) - min(overlap_ys)) * 0.3]
    if not near_rows:
        near_rows = overlap_ys
    mean_width = float(np.mean([width_dict[y] for y in near_rows]))

    if mean_width < 10:
        empty_result['n_filtered'] = n_filtered
        empty_result['status'] = 'channel too narrow'
        return canvas, empty_result

    valid_ys = [
        y for y in overlap_ys
        if abs(width_dict[y] - mean_width) / mean_width < WIDTH_OUTLIER_TH
    ]

    if len(valid_ys) < 3:
        empty_result['n_filtered'] = n_filtered
        empty_result['status'] = 'too few valid rows after width filter'
        return canvas, empty_result

    # ── 6. 逐行取中点 → 中线（按 y 降序 = 从近到远）──
    center_pts = [
        (int(round((left_dense[y] + right_dense[y]) / 2.0)), y)
        for y in sorted(valid_ys, reverse=True)
    ]

    # ── 7. 样条平滑中线 ──
    if len(center_pts) >= 5:
        try:
            from scipy.interpolate import splprep, splev
            pts_arr = np.array(center_pts, dtype=np.float64)
            tck, _ = splprep([pts_arr[:, 0], pts_arr[:, 1]],
                             s=len(center_pts) * 2, k=3)
            u_new = np.linspace(0, 1, len(center_pts))
            xs, ys = splev(u_new, tck)
            center_pts = [
                (int(round(float(x))), int(round(float(y))))
                for x, y in zip(xs, ys)
            ]
        except Exception:
            pass

    # ── 8. 构建航行路径：船头 → 平滑过渡 → 中线 ──
    sailing_path = _build_sailing_path(center_pts, w, h)

    # ── 9. 计算偏航角 ──
    if len(sailing_path) >= 5:
        look_idx = min(5, len(sailing_path) // 4)
        dx = float(sailing_path[look_idx][0] - sailing_path[0][0])
        dy = float(sailing_path[0][1] - sailing_path[look_idx][1])
    elif len(sailing_path) >= 2:
        dx = float(sailing_path[-1][0] - sailing_path[0][0])
        dy = float(sailing_path[0][1] - sailing_path[-1][1])
    else:
        dx, dy = 0.0, 1.0

    if abs(dx) > 0.1 or abs(dy) > 0.1:
        yaw_deg = math.degrees(math.atan2(dx, max(dy, 0.1)))
    else:
        yaw_deg = 0.0

    if yaw_filter is not None:
        yaw_deg = yaw_filter.update(yaw_deg)

    # ── 10. 可视化 ──
    _draw_navigation_overlay(
        canvas, left_dense, right_dense, valid_ys,
        center_pts, sailing_path, yaw_deg,
    )

    result = {
        'yaw_deg': round(yaw_deg, 2),
        'status': 'ok',
        'n_filtered': n_filtered,
        'center_pts': center_pts,
        'path_pts': sailing_path,
    }
    return canvas, result


# ─── 可视化叠加层 ──────────────────────────────────────────────────────────────

def _draw_navigation_overlay(
    canvas: np.ndarray,
    left_dense: dict,
    right_dense: dict,
    valid_ys: list,
    center_pts: List[Tuple[int, int]],
    sailing_path: List[Tuple[int, int]],
    yaw_deg: float,
) -> None:
    """
    叠加：走廊填充、警戒线、中线、航行路径、船位标记、转向条、HUD。
    """
    h, w = canvas.shape[:2]

    # ① 走廊半透明绿色填充
    if len(valid_ys) >= 2:
        fill_layer = canvas.copy()
        for y in valid_ys:
            xl = max(0, int(left_dense[y]))
            xr = min(w - 1, int(right_dense[y]))
            if xr > xl:
                cv2.line(fill_layer, (xl, y), (xr, y), (0, 180, 60), 1)
        cv2.addWeighted(fill_layer, 0.25, canvas, 0.75, 0, canvas)

    # ② 左右警戒线（橙色粗实线）
    if valid_ys:
        left_line  = [(int(left_dense[y]),  y) for y in sorted(valid_ys)]
        right_line = [(int(right_dense[y]), y) for y in sorted(valid_ys)]
        for pts in [left_line, right_line]:
            for i in range(1, len(pts)):
                cv2.line(canvas, pts[i-1], pts[i], C_WARN, 3, cv2.LINE_AA)

    # ③ 中线白色虚线
    if len(center_pts) >= 2:
        _draw_dashed_line(canvas, center_pts, (200, 200, 200), 1)

    # ④ 蓝色航行路径（粗线 + 方向箭头）
    if len(sailing_path) >= 2:
        for i in range(1, len(sailing_path)):
            cv2.line(canvas, sailing_path[i-1], sailing_path[i],
                     (255, 255, 255), 5, cv2.LINE_AA)
        for i in range(1, len(sailing_path)):
            cv2.line(canvas, sailing_path[i-1], sailing_path[i],
                     C_SAIL, 3, cv2.LINE_AA)
        arrow_step = max(3, len(sailing_path) // 6)
        for i in range(0, len(sailing_path) - arrow_step, arrow_step):
            cv2.arrowedLine(canvas, sailing_path[i], sailing_path[i + arrow_step],
                            C_SAIL, 2, cv2.LINE_AA, tipLength=0.25)

    # ⑤ 船位标记（底部中心三角形）
    boat_cx, boat_cy = w // 2, h - 12
    boat_pts = np.array([
        [boat_cx, boat_cy - 18],
        [boat_cx - 10, boat_cy + 4],
        [boat_cx + 10, boat_cy + 4],
    ], np.int32)
    cv2.fillPoly(canvas, [boat_pts], (0, 220, 255))
    cv2.polylines(canvas, [boat_pts], True, (255, 255, 255), 1)

    # ⑥ 转向指示条
    _draw_steering_bar(canvas, yaw_deg, w, h)

    # ⑦ HUD
    _draw_hud(canvas, yaw_deg, w, h)


def _draw_steering_bar(
    canvas: np.ndarray, yaw_deg: float, w: int, h: int,
    max_steer: float = 90.0,
) -> None:
    bar_w, bar_h = 200, 32
    x0 = w // 2 - bar_w // 2
    y0 = h - bar_h - 4

    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + bar_w, y0 + bar_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)
    cv2.rectangle(canvas, (x0, y0), (x0 + bar_w, y0 + bar_h), (120, 120, 120), 1)

    mid_x = x0 + bar_w // 2
    cv2.line(canvas, (mid_x, y0 + 4), (mid_x, y0 + bar_h - 4), (180, 180, 180), 1)

    ratio = max(-1.0, min(1.0, yaw_deg / max_steer))
    fill_len = int(abs(ratio) * (bar_w // 2 - 4))
    if ratio > 0:
        cv2.rectangle(canvas, (mid_x, y0 + 5),
                      (mid_x + fill_len, y0 + bar_h - 5), (0, 100, 255), -1)
    elif ratio < 0:
        cv2.rectangle(canvas, (mid_x - fill_len, y0 + 5),
                      (mid_x, y0 + bar_h - 5), (255, 180, 0), -1)

    if abs(yaw_deg) < 3:
        label, col_t = "直行", (0, 255, 120)
    elif yaw_deg > 0:
        label, col_t = f"右转 {yaw_deg:.0f}°", (0, 100, 255)
    else:
        label, col_t = f"左转 {-yaw_deg:.0f}°", (255, 200, 0)

    fs_bar = 15
    tw, th = _cn_text_size(label, fs_bar)
    _put_cn_texts(canvas, [(label, (mid_x - tw // 2, y0 + (bar_h - th) // 2),
                            fs_bar, col_t, 2, (0, 0, 0))])


def _draw_hud(
    canvas: np.ndarray, yaw_deg: float, w: int, h: int,
) -> None:
    yaw_col = ((0, 255, 120) if abs(yaw_deg) < 5
               else (0, 180, 255) if abs(yaw_deg) < 15
               else (0, 80, 255))
    yaw_dir = "右偏" if yaw_deg > 0 else "左偏" if yaw_deg < 0 else "直行"
    yaw_txt = f"偏航 {yaw_dir} {abs(yaw_deg):.1f}°"
    s_txt = "导航中"
    s_col = (0, 230, 80)

    fs1 = 18
    ts_w, _ = _cn_text_size(s_txt, fs1)

    _put_cn_texts(canvas, [
        (yaw_txt, (8, 6),            fs1, yaw_col, 2, (0, 0, 0)),
        (s_txt,   (w - ts_w - 8, 6), fs1, s_col,   2, (0, 0, 0)),
    ])


# ─── MAVROS setpoint_velocity（定频重复上一帧）────────────────────────────────

def _twist_cmd_vel(linear_x: float, yaw_deg: float, max_wz: float):
    """linear.x 前进速度；angular.z 由 yaw_deg（度）换算为 rad/s 并限幅。"""
    from geometry_msgs.msg import Twist

    msg = Twist()
    msg.linear.x = float(linear_x)
    wz = math.radians(float(yaw_deg))
    wz = max(-max_wz, min(max_wz, wz))
    msg.angular.z = wz
    return msg


def _twist_stop():
    from geometry_msgs.msg import Twist

    return Twist()


def _mavros_wait_service(client, timeout_sec: float) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if client.wait_for_service(timeout_sec=0.5):
            return True
    return False


def _mavros_call_blocking(node, client, request, timeout_sec: float = 10.0):
    """在独立 Executor 线程已 spin 时，用异步调用阻塞等待结果。"""
    if not _mavros_wait_service(client, min(timeout_sec, 15.0)):
        return None
    fut = client.call_async(request)
    t0 = time.time()
    while not fut.done() and time.time() - t0 < timeout_sec:
        time.sleep(0.02)
    if not fut.done():
        return None
    return fut.result()


def _mav_result_text(code: Optional[int]) -> str:
    table = {
        0: 'ACCEPTED',
        1: 'TEMPORARILY_REJECTED',
        2: 'DENIED',
        3: 'UNSUPPORTED',
        4: 'FAILED',
        5: 'IN_PROGRESS',
        6: 'CANCELLED',
    }
    if code is None:
        return 'NONE'
    return table.get(int(code), f'UNKNOWN({int(code)})')


class _MavrosStateMonitor:
    """订阅 /mavros/state，提供线程安全状态快照。"""

    def __init__(self, node, topic: str):
        from mavros_msgs.msg import State

        self._lock = threading.Lock()
        self.connected = False
        self.armed = False
        self.mode = ''
        self._sub = node.create_subscription(State, topic, self._on_state, 10)

    def _on_state(self, msg) -> None:
        with self._lock:
            self.connected = bool(msg.connected)
            self.armed = bool(msg.armed)
            self.mode = str(msg.mode)

    def snapshot(self) -> Tuple[bool, bool, str]:
        with self._lock:
            return self.connected, self.armed, self.mode


def _mavros_wait_connected(state_mon: _MavrosStateMonitor, timeout_sec: float) -> bool:
    deadline = time.time() + max(0.5, float(timeout_sec))
    while time.time() < deadline:
        connected, _, _ = state_mon.snapshot()
        if connected:
            return True
        time.sleep(0.05)
    return False


def _mavros_set_offboard_and_arm(
    node,
    state_mon: _MavrosStateMonitor,
    prepare_delay: float,
    set_mode_srv: str,
    arm_srv: str,
    max_retry: int = 8,
) -> None:
    """
    PX4 进入 OFFBOARD 前需已持续收到 setpoint（由定频 repeater 提供）。
    顺序：等待 prepare_delay → SetMode(OFFBOARD) → 解锁 arm。
    """
    from mavros_msgs.srv import CommandBool, SetMode

    if not _mavros_wait_connected(state_mon, timeout_sec=10.0):
        print('[MAVROS] ⚠️ 未检测到 FCU 连接（state.connected=False）')
        print('[MAVROS] 请检查: PX4 是否运行、MAVROS bridge 是否已连接、命名空间是否正确')
        return

    connected, armed, mode = state_mon.snapshot()
    print(f'[MAVROS] 初始状态 connected={connected} mode={mode} armed={armed}')

    time.sleep(max(0.0, float(prepare_delay)))

    mode_cli = node.create_client(SetMode, set_mode_srv)
    arm_cli = node.create_client(CommandBool, arm_srv)

    mode_req = SetMode.Request()
    mode_req.base_mode = 0
    mode_req.custom_mode = 'OFFBOARD'

    # 重试到状态真正进入 OFFBOARD，而不仅仅是 service 返回。
    for attempt in range(1, max_retry + 1):
        _, _, mode = state_mon.snapshot()
        if mode == 'OFFBOARD':
            print('[MAVROS] 当前已是 OFFBOARD')
            break

        r = _mavros_call_blocking(node, mode_cli, mode_req, timeout_sec=8.0)
        sent = r is not None and getattr(r, 'mode_sent', False)
        time.sleep(0.25)
        _, _, mode = state_mon.snapshot()
        if sent and mode == 'OFFBOARD':
            print(f'[MAVROS] 模式切换成功: OFFBOARD（第 {attempt} 次）')
            break

        print(
            f'[MAVROS] 设置 OFFBOARD 未生效（第 {attempt}/{max_retry} 次） '
            f'mode_sent={sent} current_mode={mode}'
        )
        time.sleep(0.5)
    else:
        print('[MAVROS] ⚠️ 未能切换到 OFFBOARD，请检查 setpoint 频率、飞控状态与 PX4 参数')

    arm_req = CommandBool.Request()
    arm_req.value = True

    for attempt in range(1, max_retry + 1):
        _, armed, _ = state_mon.snapshot()
        if armed:
            print('[MAVROS] 当前已解锁 (armed=True)')
            break

        r = _mavros_call_blocking(node, arm_cli, arm_req, timeout_sec=8.0)
        success = r is not None and bool(getattr(r, 'success', False))
        result_code = getattr(r, 'result', None)
        time.sleep(0.25)
        _, armed, mode = state_mon.snapshot()
        if success and armed:
            print(f'[MAVROS] 解锁成功（第 {attempt} 次）')
            break
        print(
            f'[MAVROS] 解锁未生效（第 {attempt}/{max_retry} 次） '
            f'success={success} result={result_code}({_mav_result_text(result_code)}) '
            f'armed={armed} mode={mode}'
        )
        time.sleep(0.5)
    else:
        print('[MAVROS] ⚠️ 未能解锁，请检查 pre-arm、遥控器档位与 COM_RCL_EXCEPT')

    connected, armed, mode = state_mon.snapshot()
    print(f'[MAVROS] 最终状态 connected={connected} mode={mode} armed={armed}')


class _MavrosCmdVelRepeater:
    """按 min_hz 定时发布，推理慢时在两次推理之间重复上一帧 Twist。"""

    def __init__(self, node, topic: str, min_hz: float):
        from geometry_msgs.msg import Twist

        self._pub = node.create_publisher(Twist, topic, 10)
        self._lock = threading.Lock()
        self._last = Twist()
        hz = max(float(min_hz), 1.0)
        self._timer = node.create_timer(1.0 / hz, self._on_timer)

    def _on_timer(self) -> None:
        from geometry_msgs.msg import Twist

        with self._lock:
            msg = Twist()
            msg.linear.x = self._last.linear.x
            msg.linear.y = self._last.linear.y
            msg.linear.z = self._last.linear.z
            msg.angular.x = self._last.angular.x
            msg.angular.y = self._last.angular.y
            msg.angular.z = self._last.angular.z
        self._pub.publish(msg)

    def update(self, twist) -> None:
        with self._lock:
            self._last.linear.x = twist.linear.x
            self._last.linear.y = twist.linear.y
            self._last.linear.z = twist.linear.z
            self._last.angular.x = twist.angular.x
            self._last.angular.y = twist.angular.y
            self._last.angular.z = twist.angular.z


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="警戒线导航：识别警戒线 → 计算中心线 → 航行轨迹 → 偏航角",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--model',       required=True,  help='SegFormer .pth 权重路径')

    # 模型推理模式
    p.add_argument('--images',      default=None,   help='输入图像文件或目录（模型推理模式）')
    p.add_argument('--output',      default=None,   help='可视化结果保存目录（模型推理模式）')

    # 实时相机模式
    p.add_argument('--camera',      default=None,
                   help='相机设备编号或路径（实时模式），如 0 或 /dev/video0')

    p.add_argument('--show',        action='store_true', help='弹窗显示（按 Q/ESC 退出）')
    p.add_argument('--num-classes', type=int, default=3,
                   help='分割类别数，默认 3')
    p.add_argument('--input-size',  type=int, default=640,
                   help='推理输入宽度（像素），高度按 640:384 比例计算，默认 640')
    p.add_argument('--model-name',  type=str, default=None,
                   help='SegFormer 基础架构名（留空则自动从权重推断）')
    p.add_argument('--ema-alpha',   type=float, default=0.3,
                   help='偏航角 EMA 滤波系数（0~1，越小越平滑），默认 0.3')
    p.add_argument('--roi-top',     type=float, default=0.1,
                   help='天空噪声截断比例，默认 0.1')

    # MAVROS（仅实时相机模式）
    p.add_argument('--mavros', action='store_true',
                   help='向 MAVROS 发布 setpoint_velocity/cmd_vel_unstamped (Twist)')
    p.add_argument('--mavros-topic', type=str,
                   default='/mavros/setpoint_velocity/cmd_vel_unstamped',
                   help='cmd_vel 话题全名')
    p.add_argument('--mavros-state-topic', type=str,
                   default='/mavros/state',
                   help='mavros state 话题全名（用于 connected/mode/armed 监控）')
    p.add_argument('--mavros-set-mode-srv', type=str,
                   default='/mavros/set_mode',
                   help='SetMode 服务全名')
    p.add_argument('--mavros-arm-srv', type=str,
                   default='/mavros/cmd/arming',
                   help='解锁 CommandBool 服务全名')
    p.add_argument('--mavros-min-hz', type=float, default=20.0,
                   help='最低发布频率 (Hz)；推理低于此频率时重复发送上一帧指令')
    p.add_argument('--mavros-linear-x', type=float, default=0.5,
                   help='前进速度 linear.x (m/s)，默认 0.5')
    p.add_argument('--mavros-max-wz', type=float, default=1.0,
                   help='由 yaw_deg 换算的角速度限幅 |angular.z| (rad/s)')
    p.add_argument('--mavros-prepare-delay', type=float, default=2.5,
                   help='开始发 setpoint 后等待多久再请求 OFFBOARD+解锁（秒），满足 PX4 预热')
    p.add_argument('--mavros-stop-when-not-ok', action='store_true',
                   help='视觉 status≠ok 时发布零速度；默认关闭（非 ok 时仍前进 linear.x，仅角速度置 0）')
    return p.parse_args()


# ─── 模型推理模式 ──────────────────────────────────────────────────────────────

def run_image_mode(args):
    """对图片/目录进行推理，输出可视化图片 + 偏航角 CSV。"""
    if args.output is None and not args.show:
        print("⚠️  未指定 --output 也未开启 --show，结果不保存不显示。")

    img_paths = collect_images(args.images)
    print(f"[信息] 共 {len(img_paths)} 张图像")

    seg_model, device = load_segformer(
        args.model,
        num_classes=args.num_classes,
        model_name=args.model_name,
    )

    yaw_filter = YawFilter(alpha=args.ema_alpha)

    out_dir = None
    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[信息] 结果保存至: {out_dir}")

    csv_file = None
    if out_dir:
        csv_path = out_dir / 'yaw_angle.csv'
        csv_file = open(str(csv_path), 'w')
        csv_file.write('frame,yaw_deg,status\n')

    total = saved = ok = 0
    input_h = args.input_size * 384 // 640

    try:
        for img_path in img_paths:
            total += 1
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"[警告] 无法读取: {img_path}")
                continue

            mask = infer_mask(
                seg_model, device, img_bgr,
                input_width=args.input_size,
                input_height=input_h,
                num_classes=args.num_classes,
            )

            vis, result = process_frame(
                img_bgr, mask,
                yaw_filter=yaw_filter,
                roi_top_ratio=args.roi_top,
            )

            yaw_deg = result['yaw_deg']
            n_bnd_px = int(np.sum(mask == BOUNDARY_CLASS))
            print(f"  {img_path.name}  边界px={n_bnd_px:5d}  "
                  f"偏航角={yaw_deg:+6.1f}°  status={result['status']}")

            if csv_file:
                csv_file.write(f"{img_path.name},{yaw_deg:.2f},{result['status']}\n")

            if result['status'] == 'ok':
                ok += 1

            if out_dir:
                save_path = out_dir / (img_path.stem + '_nav.jpg')
                cv2.imwrite(str(save_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 94])
                saved += 1

            if args.show:
                cv2.imshow("警戒线导航 (Q/ESC 退出)", vis)
                key = cv2.waitKey(0)
                if key in (ord('q'), ord('Q'), 27):
                    print("[信息] 用户退出")
                    break
    finally:
        if csv_file:
            csv_file.close()
        if args.show:
            cv2.destroyAllWindows()

    print(f"\n{'='*60}")
    print(f"  总计 {total} 张 | 已保存 {saved} 张 | 有效导航帧 {ok} 张")
    if out_dir:
        print(f"  偏航角已导出: {out_dir / 'yaw_angle.csv'}")
    print(f"  输出格式: yaw_deg = 偏航角（度），+右偏 -左偏，0=正前方")
    print(f"{'='*60}")


# ─── 实时相机模式 ──────────────────────────────────────────────────────────────

def run_camera_mode(args):
    """从摄像头读帧，实时输出偏航角；可选 MAVROS 定频发布 cmd_vel。"""
    _rclpy = None

    # 解析相机设备
    try:
        cam_id = int(args.camera)
    except ValueError:
        cam_id = args.camera

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        sys.exit(f"❌ 无法打开相机: {args.camera}")

    print(f"[信息] 相机已打开: {args.camera}")
    print(f"[信息] 按 Q/ESC 退出")

    print("[信息] 加载分割模型（完成后若启用 MAVROS 再解锁/切 OFFBOARD）…")
    seg_model, device = load_segformer(
        args.model,
        num_classes=args.num_classes,
        model_name=args.model_name,
    )

    ros_node = None
    executor = None
    spin_thread = None
    repeater = None
    state_mon = None
    rclpy_initialized = False

    if args.mavros:
        import rclpy
        from rclpy.executors import MultiThreadedExecutor

        _rclpy = rclpy
        rclpy.init()
        rclpy_initialized = True
        ros_node = rclpy.create_node('realtime_pilot_v4')
        state_mon = _MavrosStateMonitor(ros_node, args.mavros_state_topic)
        repeater = _MavrosCmdVelRepeater(
            ros_node, args.mavros_topic, args.mavros_min_hz,
        )
        # 首帧推理前也发前进 setpoint，避免 OFFBOARD 预热阶段全是 0 速
        repeater.update(
            _twist_cmd_vel(args.mavros_linear_x, 0.0, args.mavros_max_wz),
        )
        executor = MultiThreadedExecutor()
        executor.add_node(ros_node)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()
        print(
            f"[MAVROS] 话题={args.mavros_topic}  定频≥{args.mavros_min_hz} Hz "
            f"linear.x={args.mavros_linear_x} m/s"
        )
        print(
            f"[MAVROS] state={args.mavros_state_topic} "
            f"set_mode={args.mavros_set_mode_srv} arm={args.mavros_arm_srv}"
        )
        _mavros_set_offboard_and_arm(
            ros_node,
            state_mon,
            args.mavros_prepare_delay,
            args.mavros_set_mode_srv,
            args.mavros_arm_srv,
        )

    yaw_filter = YawFilter(alpha=args.ema_alpha)
    input_h = args.input_size * 384 // 640

    frame_count = 0
    try:
        while True:
            ret, img_bgr = cap.read()
            if not ret:
                print("[警告] 读取帧失败，重试...")
                time.sleep(0.1)
                continue

            frame_count += 1

            mask = infer_mask(
                seg_model, device, img_bgr,
                input_width=args.input_size,
                input_height=input_h,
                num_classes=args.num_classes,
            )

            vis, result = process_frame(
                img_bgr, mask,
                yaw_filter=yaw_filter,
                roi_top_ratio=args.roi_top,
            )

            yaw_deg = result['yaw_deg']

            if repeater is not None:
                # 默认：视觉非 ok 仍保持前进（wz=0），否则常见于水面反光/短暂的 boundary
                # 检测失败时整船 linear.x 为 0，表现为「完全不前进」。
                if result['status'] == 'ok':
                    twist = _twist_cmd_vel(
                        args.mavros_linear_x,
                        yaw_deg,
                        args.mavros_max_wz,
                    )
                elif args.mavros_stop_when_not_ok:
                    twist = _twist_stop()
                else:
                    twist = _twist_cmd_vel(
                        args.mavros_linear_x,
                        0.0,
                        args.mavros_max_wz,
                    )
                repeater.update(twist)

            # 实时输出偏航角
            print(f"\r  帧#{frame_count:05d}  偏航角={yaw_deg:+6.1f}°  "
                  f"status={result['status']}",
                  end='', flush=True)

            if args.show:
                cv2.imshow("实时警戒线导航 (Q/ESC 退出)", vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    print("\n[信息] 用户退出")
                    break
    except KeyboardInterrupt:
        print("\n[信息] Ctrl+C 退出")
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()
        if rclpy_initialized and _rclpy is not None:
            if executor is not None:
                executor.shutdown()
            if ros_node is not None:
                ros_node.destroy_node()
            _rclpy.shutdown()

    print(f"\n{'='*60}")
    print(f"  共处理 {frame_count} 帧")
    print(f"{'='*60}")


# ─── 主入口 ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.images and args.camera:
        sys.exit("❌ --images 和 --camera 不能同时指定，请选择一种模式")
    if args.mavros and args.camera is None:
        sys.exit("❌ --mavros 仅用于实时相机模式，请指定 --camera")
    if args.mavros and args.images:
        sys.exit("❌ --mavros 不能与 --images 同时使用")

    if args.images:
        run_image_mode(args)
    elif args.camera is not None:
        run_camera_mode(args)
    else:
        sys.exit("❌ 请指定 --images（模型推理模式）或 --camera（实时相机模式）")


if __name__ == '__main__':
    main()
