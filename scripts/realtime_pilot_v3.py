#!/usr/bin/env python3
"""
偏航角导航脚本 —— 输出小船偏航角（yaw angle）并保存可视化图片。

支持两种推理模式:
  1. 模型推理模式 (--mode model):  对测试集图片批量推理
  2. IMX390 相机实时模式 (--mode camera): 读取 GMSL IMX390 单目相机实时推理

算法流程:
  1. SegFormer 语义分割 → 识别左右警戒线
  2. 倒影过滤（每侧只保留 y 最小的边界线）
  3. 左右警戒线外推+插值补全全段 → 逐行取中点 → 中线
  4. 基于中线前方切线计算偏航角（度），+右偏 -左偏，0=正前方
  5. EMA 时序滤波防止帧间抖动
  6. 可视化：绘制双侧边界、中线、偏航角 HUD 并保存图片

修复说明 vs v2:
  - 边界分左右改为"质心 x 最小=左、最大=右"，不依赖图像中轴，修复转弯时双边界同侧问题
  - 行字典改为外推补全（最多外推 80 行），大幅增加左右重叠行数，修复 overlap rows < 3

用法:
  # 模型推理模式（批量处理图片目录）
  python scripts/realtime_pilot_v3.py --mode model \
      --images dataset_final/images/test \
      --model models/segformer_river/best_model.pth \
      --output realtime_pilot_v3_vis

  # IMX390 相机实时模式
  python scripts/realtime_pilot_v3.py --mode camera \
      --model models/segformer_river/best_model.pth \
      --camera-id 0 --camera-width 1920 --camera-height 1080 --show
"""

import argparse
import math
import sys
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
)

# ─── 参数 ──────────────────────────────────────────────────────────────────────
ROI_TOP_RATIO    = 0.1    # 去除图像最顶部比例（天空噪声）
WIDTH_OUTLIER_TH = 0.4    # 航道宽度偏差阈值
MIN_AREA         = 50     # 连通域最少像素数（降低阈值以保留小连通域）
MIN_COMP_SEP     = 0.04   # 两连通域质心 x 间距 / 图像宽，低于此视为同一边界
EXTRAP_MAX       = 80     # 外推最大行数

# ─── 可视化颜色 (BGR) ──────────────────────────────────────────────────────────
C_LEFT   = (0,  140, 255)   # 橙色：左边界
C_RIGHT  = (0,  60,  220)   # 红色：右边界
C_CENTER = (0,  210,  0)    # 亮绿：中线
C_BOAT   = (0,  220, 255)   # 黄色：船位三角
C_ARROW  = (255, 180, 0)    # 蓝色：航向箭头


# ─── EMA 滤波器 ───────────────────────────────────────────────────────────────

class YawFilter:
    """指数滑动平均（EMA）滤波器，平滑偏航角帧间抖动。"""

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


# ─── 边界提取 ──────────────────────────────────────────────────────────────────

def extract_boundary_polylines(
    mask: np.ndarray,
) -> Tuple[List[List[Tuple[int, int]]], int]:
    """
    从分割 mask 提取左右警戒线折线。

    修复：
    - 改用"质心 x 最小=左，最大=右"分组，不再依赖图像中轴 w/2，
      避免转弯时双边界都在同侧导致 boundary count=1。
    - 每侧保留 y_min 最小（最靠上 = 真实警戒线），其余视为倒影。

    返回: (polylines_dict={'left':pts,'right':pts}, n_filtered)
    """
    h, w = mask.shape[:2]

    bnd_bin = (mask == BOUNDARY_CLASS).astype(np.uint8)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bnd_bin = cv2.morphologyEx(bnd_bin, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bnd_bin)

    # 收集所有满足最小面积的连通域，按质心 x 排序
    comps = []
    for lid in range(1, num_labels):
        if int(stats[lid, cv2.CC_STAT_AREA]) < MIN_AREA:
            continue
        comps.append({
            'lid': lid,
            'cx':  float(centroids[lid, 0]),
            'cy':  float(centroids[lid, 1]),
        })

    if len(comps) < 2:
        # 合并所有有效小块仍不足 2 个，返回空
        polylines_all = []
        for c in comps:
            comp_mask = (labels == c['lid'])
            row_dict  = scan_comp_rows(comp_mask)
            row_dict  = smooth_row_dict(row_dict, window=11)
            pts = [(int(round(x)), y) for y, x in sorted(row_dict.items())]
            if len(pts) >= 2:
                polylines_all.append(pts)
        return polylines_all, 0

    comps.sort(key=lambda c: c['cx'])

    # 检查质心间距是否足够（排除两个几乎重叠的噪声块）
    sep = (comps[-1]['cx'] - comps[0]['cx']) / max(w, 1)
    if sep < MIN_COMP_SEP:
        return [], 0

    # 最左 = 左边界，最右 = 右边界
    left_lid  = comps[0]['lid']
    right_lid = comps[-1]['lid']

    def _comp_to_polyline(lid):
        comp_mask = (labels == lid)
        row_dict  = scan_comp_rows(comp_mask)
        row_dict  = smooth_row_dict(row_dict, window=11)
        return [(int(round(x)), y) for y, x in sorted(row_dict.items())]

    left_pts  = _comp_to_polyline(left_lid)
    right_pts = _comp_to_polyline(right_lid)

    # 倒影过滤：中间连通域数量作为过滤计数
    n_filtered = len(comps) - 2

    polylines = []
    if len(left_pts)  >= 2:
        polylines.append(left_pts)
    if len(right_pts) >= 2:
        polylines.append(right_pts)
    return polylines, max(0, n_filtered)


# ─── 稠密插值+外推 ────────────────────────────────────────────────────────────

def _extrap_dict(row_dict: dict, y_min: int, y_max: int) -> dict:
    """
    将稀疏 {y: x} 插值+外推到 [y_min, y_max] 整数行范围。

    修复 overlap rows < 3 的核心：左右边界覆盖行不同时，
    通过线性外推将各自延伸到对方的 y 范围，保证有足够重叠行。
    """
    if len(row_dict) == 0:
        return {}

    ys = sorted(row_dict.keys())
    xs = [row_dict[y] for y in ys]

    # 内插：填满已知范围内的空行
    full_ys = list(range(ys[0], ys[-1] + 1))
    full_xs = np.interp(full_ys, ys, xs).tolist()
    result  = {y: float(x) for y, x in zip(full_ys, full_xs)}

    # 向上外推（最多 EXTRAP_MAX 行）
    if len(ys) >= 2:
        slope_up = (xs[0] - xs[1]) / max(ys[0] - ys[1], 1)
        for step in range(1, EXTRAP_MAX + 1):
            y_ext = ys[0] - step
            if y_ext < y_min:
                break
            result[y_ext] = xs[0] + slope_up * step

    # 向下外推（最多 EXTRAP_MAX 行）
    if len(ys) >= 2:
        slope_dn = (xs[-1] - xs[-2]) / max(ys[-1] - ys[-2], 1)
        for step in range(1, EXTRAP_MAX + 1):
            y_ext = ys[-1] + step
            if y_ext > y_max:
                break
            result[y_ext] = xs[-1] + slope_dn * step

    return result


# ─── 偏航角计算 ────────────────────────────────────────────────────────────────

def compute_yaw_angle(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    yaw_filter: Optional[YawFilter] = None,
    roi_top_ratio: float = ROI_TOP_RATIO,
) -> Dict[str, Any]:
    """
    计算偏航角并生成可视化图像。

    返回:
      yaw_deg    : 偏航角（度），+右偏 -左偏，0=正前方
      status     : 'ok' 或失败原因
      n_filtered : 倒影过滤数量
      vis        : 可视化 BGR 图像（numpy ndarray）
    """
    h, w = img_bgr.shape[:2]
    canvas = img_bgr.copy()

    def _fail(reason):
        # 失败时在图上标注原因
        cv2.putText(canvas, f"NO CENTERLINE: {reason}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 60, 255), 2, cv2.LINE_AA)
        return {'yaw_deg': 0.0, 'status': reason, 'n_filtered': 0, 'vis': canvas}

    # 1. 提取左右警戒线
    polylines, n_filtered = extract_boundary_polylines(mask)
    if len(polylines) != 2:
        return _fail(f'boundary count={len(polylines)}, need 2')

    # 2. 区分左右（质心 x 最小 = 左）
    avg_x = [sum(p[0] for p in pts) / len(pts) for pts in polylines]
    if avg_x[0] <= avg_x[1]:
        left_pts, right_pts = polylines[0], polylines[1]
    else:
        left_pts, right_pts = polylines[1], polylines[0]

    # 3. 行字典，去除天空噪声
    sky_cut = int(h * roi_top_ratio)
    left_raw  = {y: float(x) for x, y in left_pts  if y >= sky_cut}
    right_raw = {y: float(x) for x, y in right_pts if y >= sky_cut}

    if len(left_raw) < 2 or len(right_raw) < 2:
        return _fail('boundary too short after sky cut')

    # 4. 插值+外推到并集 y 范围
    all_ys = sorted(set(left_raw.keys()) | set(right_raw.keys()))
    y_min_u, y_max_u = all_ys[0], all_ys[-1]

    left_dense  = _extrap_dict(left_raw,  y_min_u, y_max_u)
    right_dense = _extrap_dict(right_raw, y_min_u, y_max_u)
    overlap_ys  = sorted(set(left_dense.keys()) & set(right_dense.keys()))

    if len(overlap_ys) < 3:
        return _fail('overlap rows < 3 after extrapolation')

    # 5. 异常行过滤（航道宽度偏差 > 40%）
    width_dict = {y: right_dense[y] - left_dense[y] for y in overlap_ys}
    near_rows = [y for y in overlap_ys
                 if y >= max(overlap_ys) - (max(overlap_ys) - min(overlap_ys)) * 0.3]
    if not near_rows:
        near_rows = overlap_ys
    mean_width = float(np.mean([width_dict[y] for y in near_rows]))

    if mean_width < 10:
        return _fail('channel too narrow')

    valid_ys = [
        y for y in overlap_ys
        if abs(width_dict[y] - mean_width) / mean_width < WIDTH_OUTLIER_TH
    ]
    if len(valid_ys) < 3:
        return _fail('too few valid rows after width filter')

    # 6. 中线点（按 y 降序: 近→远）
    center_pts = [
        (int(round((left_dense[y] + right_dense[y]) / 2.0)), y)
        for y in sorted(valid_ys, reverse=True)
    ]

    # 7. 样条平滑
    if len(center_pts) >= 5:
        try:
            from scipy.interpolate import splprep, splev
            pts_arr = np.array(center_pts, dtype=np.float64)
            tck, _ = splprep([pts_arr[:, 0], pts_arr[:, 1]],
                             s=len(center_pts) * 2, k=3)
            u_new = np.linspace(0, 1, len(center_pts))
            xs, ys_s = splev(u_new, tck)
            center_pts = [
                (int(round(float(x))), int(round(float(y))))
                for x, y in zip(xs, ys_s)
            ]
        except Exception:
            pass

    # 8. 偏航角：中线前方切线相对正前方的偏转
    cx = w / 2.0
    boat_y = h - 1

    if len(center_pts) >= 5:
        look_idx = min(5, len(center_pts) // 4)
        dx = float(center_pts[look_idx][0] - cx)
        dy = float(boat_y - center_pts[look_idx][1])
    elif len(center_pts) >= 2:
        dx = float(center_pts[-1][0] - cx)
        dy = float(boat_y - center_pts[-1][1])
    else:
        dx, dy = 0.0, 1.0

    if abs(dx) > 0.1 or abs(dy) > 0.1:
        yaw_deg = math.degrees(math.atan2(dx, max(dy, 0.1)))
    else:
        yaw_deg = 0.0

    # 9. EMA 滤波
    if yaw_filter is not None:
        yaw_deg = yaw_filter.update(yaw_deg)

    yaw_deg = round(yaw_deg, 2)

    # 10. 可视化
    _draw_overlay(canvas, left_dense, right_dense, valid_ys,
                  center_pts, yaw_deg, h, w)

    return {
        'yaw_deg': yaw_deg,
        'status': 'ok',
        'n_filtered': n_filtered,
        'vis': canvas,
    }


# ─── 可视化叠加层 ──────────────────────────────────────────────────────────────

def _draw_overlay(
    canvas: np.ndarray,
    left_dense: dict,
    right_dense: dict,
    valid_ys: list,
    center_pts: List[Tuple[int, int]],
    yaw_deg: float,
    h: int,
    w: int,
) -> None:
    """在 canvas 上绘制：走廊填充、左右边界、中线、船位、航向箭头、HUD。"""

    # ① 走廊半透明绿色填充
    fill_layer = canvas.copy()
    for y in valid_ys:
        xl = max(0, int(left_dense[y]))
        xr = min(w - 1, int(right_dense[y]))
        if xr > xl:
            cv2.line(fill_layer, (xl, y), (xr, y), (0, 180, 60), 1)
    cv2.addWeighted(fill_layer, 0.22, canvas, 0.78, 0, canvas)

    # ② 左右边界实线
    left_line  = [(int(left_dense[y]),  y) for y in sorted(valid_ys)]
    right_line = [(int(right_dense[y]), y) for y in sorted(valid_ys)]
    for pts, col in [(left_line, C_LEFT), (right_line, C_RIGHT)]:
        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i-1], pts[i], col, 3, cv2.LINE_AA)

    # ③ 中线虚线
    dash, gap = 18, 8
    drawing = True
    seg_remain = dash
    cl_pts = center_pts
    for i in range(1, len(cl_pts)):
        p1 = np.array(cl_pts[i-1], dtype=np.float64)
        p2 = np.array(cl_pts[i],   dtype=np.float64)
        seg_len = float(np.linalg.norm(p2 - p1))
        if seg_len < 1:
            continue
        d   = (p2 - p1) / seg_len
        pos = 0.0
        while pos < seg_len:
            step = min(seg_remain, seg_len - pos)
            q1   = p1 + d * pos
            q2   = p1 + d * (pos + step)
            if drawing:
                cv2.line(canvas,
                         (int(round(q1[0])), int(round(q1[1]))),
                         (int(round(q2[0])), int(round(q2[1]))),
                         C_CENTER, 2, cv2.LINE_AA)
            pos        += step
            seg_remain -= step
            if seg_remain <= 0:
                drawing    = not drawing
                seg_remain = dash if drawing else gap

    # ④ 船位三角（底部中心）
    bx, by = w // 2, h - 12
    boat_pts = np.array([[bx, by-18], [bx-10, by+4], [bx+10, by+4]], np.int32)
    cv2.fillPoly(canvas, [boat_pts], C_BOAT)
    cv2.polylines(canvas, [boat_pts], isClosed=True,
                  color=(255, 255, 255), thickness=1)

    # ⑤ 航向箭头（从船头沿偏航角方向延伸 80px）
    arrow_len = 80
    angle_rad = math.radians(yaw_deg)
    ax = int(bx + arrow_len * math.sin(angle_rad))
    ay = int(by - 18 - arrow_len * math.cos(angle_rad))
    cv2.arrowedLine(canvas, (bx, by - 18), (ax, ay),
                    C_ARROW, 3, cv2.LINE_AA, tipLength=0.3)

    # ⑥ HUD 文字（英文，避免字体问题）
    if abs(yaw_deg) < 3:
        dir_str = "STRAIGHT"
        hud_col = (0, 230, 80)
    elif yaw_deg > 0:
        dir_str = f"RIGHT  {yaw_deg:+.1f}deg"
        hud_col = (0, 100, 255)
    else:
        dir_str = f"LEFT   {yaw_deg:+.1f}deg"
        hud_col = (255, 180, 0)

    # 背景框
    (tw, th), _ = cv2.getTextSize(dir_str, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(canvas, (8, 8), (16 + tw, 20 + th), (0, 0, 0), -1)
    cv2.putText(canvas, dir_str, (12, 12 + th),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, hud_col, 2, cv2.LINE_AA)

    # 第二行：valid rows 数量
    info = f"valid_rows={len(valid_ys)}  center_pts={len(center_pts)}"
    cv2.putText(canvas, info, (12, 20 + th + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


# ─── 模型推理模式 ──────────────────────────────────────────────────────────────

def run_model_mode(args):
    """批量处理图片目录，输出每帧偏航角并保存可视化图片。"""
    img_paths = collect_images(args.images)
    print(f"[信息] 共 {len(img_paths)} 张图像")

    seg_model, device = load_segformer(
        args.model, num_classes=args.num_classes, model_name=args.model_name,
    )
    yaw_filter = YawFilter(alpha=args.ema_alpha)

    out_dir = None
    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[信息] 可视化结果保存至: {out_dir}")

    input_h = args.input_size * 384 // 640
    total = ok = saved = 0

    for img_path in img_paths:
        total += 1
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[警告] 无法读取: {img_path}")
            continue

        mask = infer_mask(
            seg_model, device, img_bgr,
            input_width=args.input_size, input_height=input_h,
            num_classes=args.num_classes,
        )

        result = compute_yaw_angle(img_bgr, mask, yaw_filter, args.roi_top)

        if result['status'] == 'ok':
            ok += 1

        print(f"  {img_path.name}  yaw={result['yaw_deg']:+6.2f}°  "
              f"status={result['status']}")

        if out_dir:
            save_path = out_dir / (img_path.stem + '_yaw.jpg')
            cv2.imwrite(str(save_path), result['vis'],
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved += 1

        if args.show:
            cv2.imshow("Yaw Navigation", result['vis'])
            key = cv2.waitKey(0) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break

    if args.show:
        cv2.destroyAllWindows()

    print(f"\n[结果] 总计 {total} 张 | 有效 {ok} 张 | 已保存 {saved} 张")


# ─── IMX390 相机实时模式 ──────────────────────────────────────────────────────

def run_camera_mode(args):
    """从 IMX390 单目相机实时采集 → 推理 → 输出偏航角。"""
    seg_model, device = load_segformer(
        args.model, num_classes=args.num_classes, model_name=args.model_name,
    )
    yaw_filter = YawFilter(alpha=args.ema_alpha)
    input_h = args.input_size * 384 // 640

    # 打开相机
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        sys.exit(f"[错误] 无法打开相机 {args.camera_id}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    cap.set(cv2.CAP_PROP_FPS, args.camera_fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[相机] IMX390 已打开: {actual_w}x{actual_h} @ {actual_fps:.0f}fps")
    print(f"[信息] 按 Q/ESC 退出")

    frame_count = 0
    t_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[警告] 相机读取失败，重试...")
                time.sleep(0.1)
                continue

            frame_count += 1

            mask = infer_mask(
                seg_model, device, frame,
                input_width=args.input_size, input_height=input_h,
                num_classes=args.num_classes,
            )

            result = compute_yaw_angle(frame, mask, yaw_filter, args.roi_top)

            # 计算实时 FPS
            elapsed = time.time() - t_start
            fps = frame_count / elapsed if elapsed > 0 else 0.0

            print(f"\r  帧#{frame_count:05d}  "
                  f"yaw={result['yaw_deg']:+6.2f}°  "
                  f"status={result['status']}  "
                  f"FPS={fps:.1f}", end='', flush=True)

            # 可选：显示画面（按 Q/ESC 退出）
            if args.show:
                cv2.imshow("IMX390 Yaw Navigation (Q/ESC to quit)", result['vis'])
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

    print(f"\n[结果] 共处理 {frame_count} 帧, 平均 FPS={fps:.1f}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="偏航角导航：SegFormer 推理 → 中线 → 偏航角输出",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # 模式
    p.add_argument('--mode', choices=['model', 'camera'], default='model',
                   help='推理模式: model=图片批量推理, camera=IMX390 实时推理 (默认 model)')

    # 通用参数
    p.add_argument('--model', required=True, help='SegFormer .pth 权重路径')
    p.add_argument('--num-classes', type=int, default=3, help='分割类别数 (默认 3)')
    p.add_argument('--input-size', type=int, default=640,
                   help='推理输入宽度，高度按 640:384 比例 (默认 640)')
    p.add_argument('--model-name', type=str,
                   default='nvidia/segformer-b0-finetuned-ade-512-512',
                   help='SegFormer 架构名')
    p.add_argument('--ema-alpha', type=float, default=0.3,
                   help='EMA 滤波系数 0~1, 越小越平滑 (默认 0.3)')
    p.add_argument('--roi-top', type=float, default=0.1,
                   help='天空噪声截断比例 (默认 0.1)')
    p.add_argument('--show', action='store_true', help='显示画面')

    # 模型推理模式参数
    p.add_argument('--images',  default=None, help='[model模式] 输入图像文件或目录')
    p.add_argument('--output',  default=None, help='[model模式] 可视化图片保存目录')

    # IMX390 相机模式参数
    p.add_argument('--camera-id', type=int, default=0,
                   help='[camera模式] 相机设备 ID (默认 0)')
    p.add_argument('--camera-width', type=int, default=1920,
                   help='[camera模式] 采集宽度 (默认 1920)')
    p.add_argument('--camera-height', type=int, default=1080,
                   help='[camera模式] 采集高度 (默认 1080)')
    p.add_argument('--camera-fps', type=int, default=30,
                   help='[camera模式] 采集帧率 (默认 30)')
    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == 'model':
        if args.images is None:
            sys.exit("[错误] model 模式需要 --images 参数")
        run_model_mode(args)
    elif args.mode == 'camera':
        run_camera_mode(args)


if __name__ == '__main__':
    main()
