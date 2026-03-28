#!/usr/bin/env python3
"""
警戒线中线导航脚本 —— 对测试集图片进行模型推理，识别两岸警戒线，
计算中线路径，输出航向角控制信号（可接入飞控 → 差速电机）。

功能:
  1. 识别左右警戒线（倒影过滤）
  2. 近端 ROI 截断（只用图像下方区域，远端噪声不参与）
  3. 基于已知航道宽度(2.5m)的像素/米比例估算
  4. 中线计算 + 样条平滑
  5. EMA 时序滤波（减少帧间抖动）
  6. Pure Pursuit 航向角控制信号输出（steering_deg）
  7. 异常帧过滤（航道宽度偏差 > 40% 的行丢弃）

输出控制信号说明（每帧 JSON 格式写入 CSV / 打印到 stdout）:
  steering_deg : 航向角偏差（度），+右偏 -左偏，0=正前方
  throttle_pct : 油门百分比 0~100
  cte_m        : 横向偏差（米），中线偏离图像中心的物理距离
  heading_deg  : 航向角（度）

用法:
python scripts/realtime_pilot_v2.py \
    --images dataset_final/images/test \
    --model models/segformer_river/best_model.pth \
    --output realtime_pilot_v2_vis/19:51


    使用旧版 B0
python scripts/realtime_pilot_v2.py \
    --images dataset_final/images/test \
    --model models/segformer_river/best_model.pth \
    --model-name nvidia/segformer-b0-finetuned-ade-512-512 \
    --output realtime_pilot_v2_vis/19:51
.
  # 同时弹窗查看
  python scripts/Boundary_identification.py \\
      --images dataset_final/images/test \\
      --model  models/segformer_river/best_model.pth \\
      --output vis_boundary \\
      --show
"""

import argparse
import json
import math
import sys
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

# ─── 航道参数 ──────────────────────────────────────────────────────────────────
CHANNEL_WIDTH_M  = 2.5    # 已知航道宽度（米）
ROI_TOP_RATIO    = 0.1    # 去除图像最顶部这个比例的行（避免天空噪声）
WIDTH_OUTLIER_TH = 0.4    # 航道宽度偏差超过此比例的行视为异常

# ─── 样式常量 ──────────────────────────────────────────────────────────────────
C_YELLOW   = (0, 255, 255)   # BGR 黄色
C_CENTER   = (0, 210, 0)     # BGR 亮绿（中线）
DASH_LEN   = 20
GAP_LEN    = 10
THICKNESS  = 3
MIN_AREA   = 30              # 连通域最少像素数，过滤噪声（降低阈值保留更多碎片）

# ─── PIL 中文字体工具（cv2.putText 不支持中文）────────────────────────────────
_CN_FONT_CACHE: Dict[int, Any] = {}

def _get_pil_font(size: int) -> Any:
    """获取指定大小的中文 PIL 字体（带缓存）。"""
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
    """返回中文文本渲染后的 (宽, 高) 像素值。"""
    font = _get_pil_font(font_size)
    try:
        bbox = font.getbbox(text)       # (left, top, right, bottom)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:             # Pillow < 9.2
        from PIL import Image, ImageDraw
        draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
        return draw.textsize(text, font=font)


def _put_cn_texts(
    canvas: np.ndarray,
    items: List[Tuple],
) -> None:
    """
    批量渲染中文文本到 canvas（in-place）。
    items 中每项: (text, xy_topleft, font_size, color_bgr)
               或 (text, xy_topleft, font_size, color_bgr, stroke_w, stroke_bgr)
    一次 BGR→RGB→BGR 转换，效率最优。
    """
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
        pass  # 静默降级，不崩溃


# ─── EMA 时序滤波器 ───────────────────────────────────────────────────────────

class CTEFilter:
    """指数滑动平均（EMA）滤波器，用于平滑舵角/CTE 帧间抖动。"""

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


# ─── 轻量级 Pure Pursuit（像素空间）─────────────────────────────────────────

class SimplePurePursuit:
    """
    像素空间 Pure Pursuit 控制器。
    船在图像底部中心，朝上（y 减小方向）行驶。
    沿路径找到 lookahead 处的目标点，计算航向角偏差。
    """

    def __init__(
        self,
        img_w: int = 640,
        img_h: int = 480,
        lookahead_ratio: float = 0.20,
        max_steering_deg: float = 90.0,
    ):
        self.img_w = img_w
        self.img_h = img_h
        self.lookahead_ratio = lookahead_ratio
        self.max_steer = math.radians(max_steering_deg)

    def compute(
        self, path_px: List[Tuple[int, int]]
    ) -> Dict[str, Any]:
        """
        输入中线路径像素点列表 [(x,y), ...]，从图像底部到顶部排列。
        输出控制信号字典。
        """
        if len(path_px) < 3:
            return self._stop("路径点不足")

        w, h = self.img_w, self.img_h
        cx = w / 2.0
        start_y = h - 1

        lookahead_px = h * self.lookahead_ratio

        # 沿路径找 lookahead 目标点
        target_x, target_y = None, None
        for px, py in path_px:
            dist_y = start_y - py
            if dist_y >= lookahead_px:
                target_x, target_y = px, py
                break

        if target_x is None:
            target_x, target_y = path_px[-1]

        dx = target_x - cx
        dy = start_y - target_y
        ld = math.sqrt(dx ** 2 + dy ** 2)

        if ld < 5:
            return self._stop("目标点距离过近")

        # 航向角 = atan2(横向偏移, 纵向深度)
        steering = math.atan2(dx, max(dy, 1.0))
        steering = max(-self.max_steer, min(self.max_steer, steering))

        # 油门：转弯越大，速度越慢
        steer_ratio = abs(steering) / self.max_steer
        throttle_pct = (1.0 - 0.4 * steer_ratio) * 100.0
        throttle_pct = max(0.0, min(100.0, throttle_pct))

        return {
            'steering_rad': steering,
            'steering_deg': math.degrees(steering),
            'throttle_pct': throttle_pct,
            'target_px': (target_x, target_y),
            'lookahead_px': ld,
            'status': 'ok',
        }

    def _stop(self, reason: str) -> Dict[str, Any]:
        return {
            'steering_rad': 0.0,
            'steering_deg': 0.0,
            'throttle_pct': 0.0,
            'target_px': (0, 0),
            'lookahead_px': 0.0,
            'status': f'stop: {reason}',
        }


# ─── 黄色虚线绘制 ──────────────────────────────────────────────────────────────

def _draw_yellow_dashed(
    canvas: np.ndarray,
    pts: List[Tuple[int, int]],
    color: Tuple[int, int, int] = C_YELLOW,
    thickness: int = THICKNESS,
    dash: int = DASH_LEN,
    gap: int = GAP_LEN,
) -> None:
    """沿折线点序列绘制黄色虚线。"""
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
    """根据折线点的平均 x 坐标判断其在图像的左半侧还是右半侧。"""
    mean_x = sum(p[0] for p in pts) / len(pts)
    return 'left' if mean_x < img_w / 2 else 'right'


def _min_y(pts: List[Tuple[int, int]]) -> int:
    """返回折线中 y 坐标的最小值（图像中最靠上的点）。"""
    return min(p[1] for p in pts)


def extract_boundary_polylines(
    mask: np.ndarray,
) -> List[List[Tuple[int, int]]]:
    """
    从分割 mask 中提取边界折线，每侧（左/右）最多保留一条。

    流程:
      1. 取 mask == BOUNDARY_CLASS 的二值图
      2. 形态学闭运算填补断裂
      3. 连通域分析，过滤小噪声块
      4. 逐连通域按行扫描取 x 中心 → 平滑 → 折线点列表
      5. 按折线平均 x 分左/右两组；同侧若有多条，保留 y_min 最小的那条
         （位置靠上 = 离相机更远 = 真实警戒线；靠下 = 水面倒影）
    """
    h, w = mask.shape[:2]

    bnd_bin = (mask == BOUNDARY_CLASS).astype(np.uint8)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bnd_bin = cv2.morphologyEx(bnd_bin, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bnd_bin)

    # 先收集所有有效折线
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

    # 按左/右分组
    groups: dict = {'left': [], 'right': []}
    for pts in candidates:
        side = _polyline_side(pts, w)
        groups[side].append(pts)

    # 同侧多条折线：合并为一条（按 y 排序取所有点的行字典），而非只取 y_min 最小的
    polylines: List[List[Tuple[int, int]]] = []
    n_filtered = 0
    for side, side_pts in groups.items():
        if not side_pts:
            continue
        if len(side_pts) == 1:
            polylines.append(side_pts[0])
        else:
            # 合并同侧所有碎片到一个行字典，每行取中值
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


# ─── 单帧可视化 + 中线计算 + 控制信号 ─────────────────────────────────────────

def _interp_dict(row_dict: dict) -> dict:
    """将稀疏 {y: x} 线性插值为连续整数 y 的密集字典，填补行间空隙。"""
    if len(row_dict) < 2:
        return dict(row_dict)
    ys = sorted(row_dict.keys())
    full: dict = {}
    for i in range(len(ys) - 1):
        y0, y1 = ys[i], ys[i + 1]
        x0, x1 = row_dict[y0], row_dict[y1]
        for y in range(y0, y1):
            t = (y - y0) / (y1 - y0)
            full[y] = x0 + t * (x1 - x0)
    full[ys[-1]] = row_dict[ys[-1]]
    return full


def _build_sailing_path(
    center_pts: List[Tuple[int, int]],
    img_w: int,
    img_h: int,
    n_interp: int = 30,
) -> List[Tuple[int, int]]:
    """
    从船头位置（图像底部中心）到中线起点之间插入一段样条过渡曲线，
    拼接成完整的"蓝色航行路径"。

    center_pts: 中线点列表，已按 y 降序（[0] = 最近端，y 最大）。
    返回: 完整路径 [(x,y), ...]，第 [0] 个是船头 (img_w//2, img_h-1)，
          然后是平滑过渡段，最后接中线原始点。
    """
    boat_x, boat_y = img_w // 2, img_h - 1
    cl_x, cl_y = center_pts[0]  # 中线最近端点

    # 如果中线已经到了船底附近，不需要额外过渡
    if abs(cl_y - boat_y) < 5 and abs(cl_x - boat_x) < 5:
        return [(boat_x, boat_y)] + center_pts

    # 用三次样条在船头和中线起点之间做平滑过渡
    # 控制点：船头 → 中间过渡 → 中线起点 → 中线第二点（保证切线连续）
    try:
        from scipy.interpolate import CubicSpline
        # 以 y（从大到小 = 从近到远）为自变量，x 为因变量
        # 取中线前几个点来保证末端切线连续
        n_cl = min(5, len(center_pts))
        ctrl_ys = [float(boat_y)] + [float(center_pts[i][1]) for i in range(n_cl)]
        ctrl_xs = [float(boat_x)] + [float(center_pts[i][0]) for i in range(n_cl)]

        # y 必须严格递减，去重
        clean_ys, clean_xs = [ctrl_ys[0]], [ctrl_xs[0]]
        for yy, xx in zip(ctrl_ys[1:], ctrl_xs[1:]):
            if yy < clean_ys[-1]:
                clean_ys.append(yy)
                clean_xs.append(xx)

        if len(clean_ys) < 3:
            # 太少控制点，线性连接
            return [(boat_x, boat_y)] + center_pts

        # y 递减 → 翻转为递增给 CubicSpline
        rev_ys = list(reversed(clean_ys))
        rev_xs = list(reversed(clean_xs))
        cs = CubicSpline(rev_ys, rev_xs, bc_type='natural')

        # 在船头到中线起点之间采样过渡段
        trans_ys = np.linspace(cl_y, boat_y, n_interp + 2)  # 递增
        trans_xs = cs(trans_ys)

        # 翻转为 y 递减（船头在前）
        transition = [
            (int(round(float(x))), int(round(float(y))))
            for x, y in zip(reversed(trans_xs), reversed(trans_ys))
        ]

        # 拼接：过渡段 + 中线（跳过重复的起点）
        sailing_path = transition + center_pts[1:]
        return sailing_path

    except Exception:
        # scipy 不可用，线性插值
        n = n_interp
        path = []
        for i in range(n + 1):
            t = i / n
            x = int(round(boat_x + t * (cl_x - boat_x)))
            y = int(round(boat_y + t * (cl_y - boat_y)))
            path.append((x, y))
        path += center_pts[1:]
        return path


def compute_centerline_and_control(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    pursuit: SimplePurePursuit,
    steer_filter: Optional[CTEFilter] = None,
    roi_top_ratio: float = ROI_TOP_RATIO,
    channel_width_m: float = CHANNEL_WIDTH_M,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    在原图上叠加警戒线与中线，并输出航向角控制信号。

    流程:
      1. 提取左右警戒线
      2. 近端 ROI 截断
      3. 异常帧过滤（航道宽度偏差 > 40%）
      4. 逐行取左右 x 中点 → 中线
      5. 样条平滑中线
      6. Pure Pursuit 计算航向角
      7. EMA 滤波平滑舵角
      8. 基于已知航道宽度计算横向偏差（米）

    返回: (可视化图像, 控制信号字典)
    """
    h, w = img_bgr.shape[:2]
    canvas = img_bgr.copy()

    empty_ctrl = {
        'heading_deg': 0.0,
        'status': 'no centerline',
        'n_filtered': 0,
        'center_pts': [], 'path_pts': [],
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

    # 绘制单侧/无边界时的警戒线（黄色虚线，仅在无法计算中线时显示）
    if len(polylines) != 2:
        for pts in polylines:
            _draw_yellow_dashed(canvas, pts)
        # 单侧或无边界 → 无法计算中线
        empty_ctrl['n_filtered'] = n_filtered
        empty_ctrl['status'] = f'boundary count={len(polylines)}, need 2'
        return canvas, empty_ctrl

    # ── 2. 区分左右 ──
    avg_x = [sum(p[0] for p in pts) / len(pts) for pts in polylines]
    if avg_x[0] <= avg_x[1]:
        left_pts, right_pts = polylines[0], polylines[1]
    else:
        left_pts, right_pts = polylines[1], polylines[0]

    # ── 3. 构建行字典：去除最顶部 roi_top_ratio 的噪声行，其余全部保留 ──
    #       注意：不强制截断下半段，因为警戒线可能主要分布在图像上半部分
    sky_cut = int(h * roi_top_ratio)   # 仅去除天空极顶部噪声
    left_dict  = {y: float(x) for x, y in left_pts  if y >= sky_cut}
    right_dict = {y: float(x) for x, y in right_pts if y >= sky_cut}

    # ── 4. 插值 + 外推 → 稠密行覆盖，消除稀疏行导致的空交集 ──
    all_ys = sorted(set(left_dict.keys()) | set(right_dict.keys()))
    if len(all_ys) < 2:
        empty_ctrl['n_filtered'] = n_filtered
        empty_ctrl['status'] = 'too few boundary rows'
        return canvas, empty_ctrl
    y_min_u, y_max_u = all_ys[0], all_ys[-1]
    left_dense  = _extrapolate_dict(left_dict,  y_min_u, y_max_u)
    right_dense = _extrapolate_dict(right_dict, y_min_u, y_max_u)
    overlap_ys  = sorted(set(left_dense.keys()) & set(right_dense.keys()))

    if len(overlap_ys) < 3:
        empty_ctrl['n_filtered'] = n_filtered
        empty_ctrl['status'] = 'overlap rows < 3'
        return canvas, empty_ctrl

    # ── 5. 异常帧过滤（航道宽度偏差 > 40% 的行丢弃）──
    width_dict = {y: right_dense[y] - left_dense[y] for y in overlap_ys}
    # 用近端（y 最大 = 最靠近船底）的行估算基准宽度
    near_rows = [y for y in overlap_ys if y >= max(overlap_ys) - (max(overlap_ys) - min(overlap_ys)) * 0.3]
    if not near_rows:
        near_rows = overlap_ys
    mean_width = float(np.mean([width_dict[y] for y in near_rows]))

    if mean_width < 10:
        empty_ctrl['n_filtered'] = n_filtered
        empty_ctrl['status'] = 'channel too narrow'
        return canvas, empty_ctrl

    valid_ys = [
        y for y in overlap_ys
        if abs(width_dict[y] - mean_width) / mean_width < WIDTH_OUTLIER_TH
    ]

    if len(valid_ys) < 3:
        empty_ctrl['n_filtered'] = n_filtered
        empty_ctrl['status'] = 'too few valid rows after width filter'
        return canvas, empty_ctrl

    # ── 6. 逐行取中点 → 中线（按 y 降序 = 从船底到远端，Pure Pursuit 正确方向）──
    center_pts = [
        (int(round((left_dense[y] + right_dense[y]) / 2.0)), y)
        for y in sorted(valid_ys, reverse=True)   # 大 y（近）→ 小 y（远）
    ]

    # ── 7. 样条平滑中线 ──
    if len(center_pts) >= 5:
        try:
            from scipy.interpolate import splprep, splev
            pts_arr = np.array(center_pts, dtype=np.float64)
            tck, _ = splprep([pts_arr[:, 0], pts_arr[:, 1]], s=len(center_pts) * 2, k=3)
            u_new = np.linspace(0, 1, len(center_pts))
            xs, ys = splev(u_new, tck)
            center_pts = [
                (int(round(float(x))), int(round(float(y))))
                for x, y in zip(xs, ys)
            ]
        except Exception:
            pass  # scipy 不可用时退回原始折线

    # ── 8. 构建蓝色航行路径：船头 → 平滑过渡 → 中线 ──
    sailing_path = _build_sailing_path(center_pts, w, h)

    # ── 9. 基于蓝色航行路径计算航向角 ──
    #  航向角 = 路径前方几个点切线相对正前方的偏转角
    #  sailing_path[0] = 船头位置 (w/2, h-1)
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
        heading_deg = math.degrees(math.atan2(dx, max(dy, 0.1)))
    else:
        heading_deg = 0.0

    # EMA 时序滤波防止舵角频繁振荡
    if steer_filter is not None:
        heading_deg = steer_filter.update(heading_deg)

    # ── 10. 基于已知航道宽度的横向偏差物理量化 ──
    channel_width_px = mean_width
    scale_ppm = channel_width_px / channel_width_m   # 像素/米
    # 用中线中段点估算横向偏差（底端受透视影响大）
    mid_idx = len(center_pts) // 2
    cte_px = center_pts[mid_idx][0] - w / 2.0
    cte_m_val = cte_px / scale_ppm if scale_ppm > 0 else 0.0

    # ── 11. 可视化：绘制导航走廊 + 中线 + 蓝色航行路径 + 控制面板 ──
    _draw_navigation_overlay(
        canvas, left_dense, right_dense, valid_ys,
        center_pts, sailing_path, heading_deg, cte_m_val,
    )

    ctrl = {
        'heading_deg': round(heading_deg, 2),
        'status': 'ok',
        'n_filtered': n_filtered,
        'center_pts': center_pts,
        'path_pts': sailing_path,
    }
    return canvas, ctrl


def _draw_navigation_overlay(
    canvas: np.ndarray,
    left_dense: dict,
    right_dense: dict,
    valid_ys: list,
    center_pts: List[Tuple[int, int]],
    sailing_path: List[Tuple[int, int]],
    heading_deg: float,
    cte_m: float,
) -> None:
    """
    直观导航叠加层：
      ① 导航走廊半透明绿色填充（两侧警戒线之间）
      ② 左右警戒线橙色实线（粗）
      ③ 中线白色虚线
      ④ 蓝色航行路径（船头→中线，粗实线+方向箭头）
      ⑤ 船位标记（底部中心三角形）
      ⑥ 转向指示条（底部居中，左/直/右）
      ⑦ 简洁 HUD（仅关键数值）
    """
    h, w = canvas.shape[:2]

    # ① 走廊填充（半透明绿）
    if len(valid_ys) >= 2:
        fill_layer = canvas.copy()
        for y in valid_ys:
            xl = max(0, int(left_dense[y]))
            xr = min(w - 1, int(right_dense[y]))
            if xr > xl:
                cv2.line(fill_layer, (xl, y), (xr, y), (0, 180, 60), 1)
        cv2.addWeighted(fill_layer, 0.25, canvas, 0.75, 0, canvas)

    # ② 左右警戒线（橙色粗实线）
    C_WARN = (0, 140, 255)   # 橙
    if valid_ys:
        left_line_pts  = [(int(left_dense[y]),  y) for y in sorted(valid_ys)]
        right_line_pts = [(int(right_dense[y]), y) for y in sorted(valid_ys)]
        for pts in [left_line_pts, right_line_pts]:
            for i in range(1, len(pts)):
                cv2.line(canvas, pts[i-1], pts[i], C_WARN, 3, cv2.LINE_AA)

    # ③ 中线白色虚线（参考线）
    C_CL = (200, 200, 200)
    if len(center_pts) >= 2:
        step = max(1, len(center_pts) // 20)
        for i in range(1, len(center_pts), step):
            cv2.line(canvas, center_pts[i-1], center_pts[i], C_CL, 1, cv2.LINE_AA)

    # ④ 蓝色航行路径（主路径，粗线+方向箭头）
    C_SAIL = (255, 180, 0)   # BGR 蓝色
    if len(sailing_path) >= 2:
        # 粗实线 + 白色描边
        for i in range(1, len(sailing_path)):
            cv2.line(canvas, sailing_path[i-1], sailing_path[i],
                     (255, 255, 255), 5, cv2.LINE_AA)
        for i in range(1, len(sailing_path)):
            cv2.line(canvas, sailing_path[i-1], sailing_path[i],
                     C_SAIL, 3, cv2.LINE_AA)
        # 方向箭头（从近→远，即 y 递减方向）
        arrow_step = max(3, len(sailing_path) // 6)
        for i in range(0, len(sailing_path) - arrow_step, arrow_step):
            p1 = sailing_path[i]
            p2 = sailing_path[i + arrow_step]
            cv2.arrowedLine(canvas, p1, p2, C_SAIL, 2, cv2.LINE_AA, tipLength=0.25)

    # ⑤ 船位标记（底部中心向上的三角形）
    boat_cx = w // 2
    boat_cy = h - 12
    boat_pts = np.array([
        [boat_cx,      boat_cy - 18],
        [boat_cx - 10, boat_cy + 4],
        [boat_cx + 10, boat_cy + 4],
    ], np.int32)
    cv2.fillPoly(canvas, [boat_pts], (0, 220, 255))
    cv2.polylines(canvas, [boat_pts], isClosed=True, color=(255, 255, 255), thickness=1)

    # ⑥ 转向指示条
    _draw_steering_bar(canvas, heading_deg, w, h, max_steer=90.0)

    # ⑦ 简洁 HUD
    _draw_compact_hud(canvas, heading_deg, cte_m, w, h)


def _draw_steering_bar(
    canvas: np.ndarray,
    steering_deg: float,
    w: int, h: int,
    max_steer: float = 90.0,
) -> None:
    """在图像底部中央绘制直观的转向指示条。"""
    bar_w, bar_h = 200, 32
    x0 = w // 2 - bar_w // 2
    y0 = h - bar_h - 4

    # 背景
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + bar_w, y0 + bar_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)
    cv2.rectangle(canvas, (x0, y0), (x0 + bar_w, y0 + bar_h), (120, 120, 120), 1)

    # 中线（直行参考线）
    mid_x = x0 + bar_w // 2
    cv2.line(canvas, (mid_x, y0 + 4), (mid_x, y0 + bar_h - 4), (180, 180, 180), 1)

    # 转向填充条
    ratio = max(-1.0, min(1.0, steering_deg / max_steer))
    fill_len = int(abs(ratio) * (bar_w // 2 - 4))
    if ratio > 0:   # 右转 → 从中线向右延伸
        col = (0, 100, 255)   # 橙红
        cv2.rectangle(canvas, (mid_x, y0 + 5),
                      (mid_x + fill_len, y0 + bar_h - 5), col, -1)
    elif ratio < 0:  # 左转 → 从中线向左延伸
        col = (255, 180, 0)   # 蓝黄
        cv2.rectangle(canvas, (mid_x - fill_len, y0 + 5),
                      (mid_x, y0 + bar_h - 5), col, -1)

    # 文字（用 PIL 渲染，cv2.putText 不支持中文）
    if abs(steering_deg) < 3:
        label = "直行"
        col_t = (0, 255, 120)
    elif steering_deg > 0:
        label = f"右转 {steering_deg:.0f}deg"
        col_t = (0, 100, 255)
    else:
        label = f"左转 {-steering_deg:.0f}deg"
        col_t = (255, 200, 0)

    fs_bar = 15
    tw, th = _cn_text_size(label, fs_bar)
    x_txt = mid_x - tw // 2
    y_txt = y0 + (bar_h - th) // 2
    _put_cn_texts(canvas, [(label, (x_txt, y_txt), fs_bar, col_t, 2, (0, 0, 0))])


def _draw_compact_hud(
    canvas: np.ndarray,
    heading_deg: float,
    cte_m: float,
    w: int, h: int,
) -> None:
    """绘制简洁 HUD：左上角显示偏航角，右上角显示状态（均用 PIL 渲染中文）。"""
    status = 'ok'

    # 左上第一行：偏航角
    yaw_col = (0, 255, 120) if abs(heading_deg) < 5 else (0, 180, 255) if abs(heading_deg) < 15 else (0, 80, 255)
    yaw_dir = "右偏" if heading_deg > 0 else "左偏" if heading_deg < 0 else "直行"
    yaw_txt = f"偏航 {yaw_dir} {abs(heading_deg):.1f}°"

    # 右上：状态
    if status == 'ok':
        s_col, s_txt = (0, 230, 80), "导航中"
    else:
        s_col, s_txt = (0, 60, 255), "无中线"

    fs1 = 18
    _, th1 = _cn_text_size(yaw_txt, fs1)
    ts_w, ts_h = _cn_text_size(s_txt, fs1)

    y1 = 6
    y_status = 6

    _put_cn_texts(canvas, [
        (yaw_txt, (8, y1),                 fs1, yaw_col,       2, (0, 0, 0)),
        (s_txt,   (w - ts_w - 8, y_status), fs1, s_col,          2, (0, 0, 0)),
    ])


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="警戒线中线导航：SegFormer 推理 → 中线计算 → 航向角控制信号",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--images',      required=True,  help='输入图像文件或目录')
    p.add_argument('--model',       required=True,  help='SegFormer .pth 权重路径')
    p.add_argument('--output',      default=None,   help='结果保存目录')
    p.add_argument('--show',        action='store_true', help='弹窗逐张显示（按 Q/ESC 退出）')
    p.add_argument('--num-classes', type=int, default=3,
                   help='分割类别数，默认 3')
    p.add_argument('--input-size',  type=int, default=640,
                   help='推理输入宽度（像素），高度按 640:384 比例计算，默认 640')
    p.add_argument('--model-name',  type=str,
                   default=None,
                   help='SegFormer 基础架构名，须与训练时一致（留空则自动推断）')
    p.add_argument('--channel-width', type=float, default=2.5,
                   help='已知航道宽度（米），默认 2.5')
    p.add_argument('--lookahead',   type=float, default=0.20,
                   help='Pure Pursuit 预瞄比例（相对图像高度 0~1），默认 0.20')
    p.add_argument('--max-steer',   type=float, default=90.0,
                   help='最大转向角（度），差速电机可实现 90 度转弯，默认 90')
    p.add_argument('--ema-alpha',   type=float, default=0.3,
                   help='EMA 滤波系数（0~1，越小越平滑），默认 0.3')
    p.add_argument('--roi-top',     type=float, default=0.1,
                   help='天空噪声截断比例，去除图像最顶部这个比例的行，默认 0.1（警戒线通常在图像上半部分，不做下截断）')
    return p.parse_args()


def main():
    args = parse_args()

    if args.output is None and not args.show:
        print("⚠️  未指定 --output 也未开启 --show，结果不保存不显示。")

    img_paths = collect_images(args.images)
    print(f"[信息] 共 {len(img_paths)} 张图像")

    seg_model, device = load_segformer(
        args.model,
        num_classes=args.num_classes,
        model_name=args.model_name,
    )

    pursuit = SimplePurePursuit(
        lookahead_ratio=args.lookahead,
        max_steering_deg=args.max_steer,
    )
    steer_filter = CTEFilter(alpha=args.ema_alpha)

    out_dir = None
    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[信息] 结果保存至: {out_dir}")

    # 偏航角 CSV 文件
    ctrl_csv_path = None
    ctrl_csv_file = None
    if out_dir:
        ctrl_csv_path = out_dir / 'yaw_angle.csv'
        ctrl_csv_file = open(str(ctrl_csv_path), 'w')
        ctrl_csv_file.write('frame,yaw_deg,status\n')

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

            vis, ctrl = compute_centerline_and_control(
                img_bgr, mask, pursuit,
                steer_filter=steer_filter,
                roi_top_ratio=args.roi_top,
                channel_width_m=args.channel_width,
            )

            n_bnd_px = int(np.sum(mask == BOUNDARY_CLASS))

            # 打印偏航角
            yaw_deg = ctrl['heading_deg']
            print(f"  {img_path.name}  边界px={n_bnd_px:5d}  "
                  f"偏航角={yaw_deg:+6.1f}°  "
                  f"status={ctrl['status']}")

            # 写入 CSV
            if ctrl_csv_file:
                ctrl_csv_file.write(
                    f"{img_path.name},"
                    f"{yaw_deg:.2f},"
                    f"{ctrl['status']}\n"
                )

            if ctrl['status'] == 'ok':
                ok += 1

            if out_dir:
                save_path = out_dir / (img_path.stem + '_boundary.jpg')
                cv2.imwrite(str(save_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 94])
                saved += 1

            if args.show:
                cv2.imshow("警戒线中线导航 (Q/ESC 退出)", vis)
                key = cv2.waitKey(0)
                if key in (ord('q'), ord('Q'), 27):
                    print("[信息] 用户退出")
                    break
    finally:
        if ctrl_csv_file:
            ctrl_csv_file.close()
        if args.show:
            cv2.destroyAllWindows()

    print(f"\n{'='*60}")
    print(f"  总计 {total} 张 | 已保存 {saved} 张 | 有效控制帧 {ok} 张")
    if ctrl_csv_path:
        print(f"  偏航角已导出: {ctrl_csv_path}")
    print(f"{'='*60}")
    print(f"\n[输出格式]")
    print(f"  yaw_deg : 偏航角（度），+右偏 -左偏，0=正前方")


if __name__ == '__main__':
    main()
