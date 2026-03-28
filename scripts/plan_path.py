#!/usr/bin/env python3
"""
自主路径规划脚本 —— 从图像底部中心平滑延伸至航道中线。

【三级回退策略】
  A. 双边界模式：左右边界都检测到 → 取中线 → 平滑路径
  B. 单边界模式：仅一侧边界 → 边界 + 水域对侧边缘 → 推算虚拟中线
  C. 纯水域模式：无边界 → 逐行扫描水域范围中心 → 沿水域中心前进

【用法】
  # 使用已有 mask
  python scripts/plan_path.py \
      --images dataset_final/images/val \
      --masks  dataset_final/masks/val  \
      --output vis_plan

  # 使用模型推理
  python scripts/plan_path.py \
      --images dataset_final/images/val \
      --model  models/segformer_river/best_model.pth \
      --output vis_plan

  # 弹窗查看
  python scripts/plan_path.py \
      --images dataset_final/images/val \
      --masks  dataset_final/masks/val  \
      --show
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
from scipy.interpolate import splprep, splev

sys.path.insert(0, str(Path(__file__).resolve().parent))
from visualize_centerline import (
    BOUNDARY_CLASS, WATER_CLASS,
    MIN_COMP_PX, MIN_SEPARATION,
    find_two_boundaries, scan_comp_rows, smooth_row_dict, compute_centerline,
    load_segformer, infer_mask,
    collect_images, find_mask_path,
    _draw_dashed_polyline, _draw_solid_polyline, _draw_legend,
    _label_at, _put_text_block, _medfilt1d,
    C_LEFT, C_RIGHT, C_CENTER, C_SKIP,
    TH_BOUND, TH_CENTER,
    SMOOTH_WINDOW,
)

# ─────────────────────────────────────────────────────────────────────────────
# 颜色和样式
# ─────────────────────────────────────────────────────────────────────────────
C_PATH      = (255, 180, 0)
C_START     = (0,   255, 255)
C_WATER_CL  = (180, 220, 0)   # 青绿色：水域中心线（回退模式标识）
TH_PATH     = 5
ARROW_LEN   = 18
ARROW_ANGLE = 25

MIN_WATER_ROWS = 10  # 水域扫描至少需要的有效行数


# ─────────────────────────────────────────────────────────────────────────────
# 回退策略 B：单边界 → 虚拟中线
# ─────────────────────────────────────────────────────────────────────────────

def _find_single_boundary(mask: np.ndarray):
    """
    当 find_two_boundaries 失败时，尝试找到唯一的一条边界线。

    Returns:
        (comp_mask, side, err)
        side: 'left' 或 'right'，表示这条边界线在图像的哪一侧
    """
    h, w = mask.shape[:2]
    bnd_bin = (mask == BOUNDARY_CLASS).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bnd_bin = cv2.morphologyEx(bnd_bin, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bnd_bin)

    comps = []
    for lid in range(1, num_labels):
        area = int(stats[lid, cv2.CC_STAT_AREA])
        if area < MIN_COMP_PX:
            continue
        comps.append({
            'lid': lid,
            'area': area,
            'centroid_x': float(centroids[lid, 0]),
        })

    if not comps:
        return None, None, "无边界连通域"

    best = max(comps, key=lambda c: c['area'])
    comp_mask = (labels == best['lid'])
    side = 'left' if best['centroid_x'] < w / 2 else 'right'
    return comp_mask, side, None


def _water_edge_per_row(mask: np.ndarray, side: str):
    """
    对边界对侧逐行扫描水域像素，找到水域的另一个边缘位置。

    Args:
        mask: 分割 mask
        side: 已知边界在 'left' 或 'right' 侧

    Returns:
        {y: x_edge} 字典，表示水域在对侧的边缘 x 坐标
    """
    h, w = mask.shape[:2]
    water_bin = ((mask == WATER_CLASS) | (mask == BOUNDARY_CLASS))
    edge_dict = {}

    for y in range(h):
        row = water_bin[y, :]
        water_cols = np.where(row)[0]
        if len(water_cols) < 3:
            continue
        if side == 'left':
            edge_dict[y] = float(water_cols[-1])
        else:
            edge_dict[y] = float(water_cols[0])

    return edge_dict


def compute_single_boundary_centerline(
    mask: np.ndarray, comp_mask: np.ndarray, side: str, img_w: int
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    单边界模式：用已知边界 + 水域对侧边缘 → 虚拟中线。

    Returns:
        (boundary_pts, virtual_edge_pts, center_pts)
    """
    bnd_dict = scan_comp_rows(comp_mask)
    bnd_dict = smooth_row_dict(bnd_dict)
    edge_dict = _water_edge_per_row(mask, side)
    edge_dict = smooth_row_dict(edge_dict, window=21)

    overlap_ys = sorted(set(bnd_dict.keys()) & set(edge_dict.keys()))
    if len(overlap_ys) < 3:
        return [], [], []

    bnd_pts = []
    edge_pts = []
    center_pts = []
    for y in overlap_ys:
        xb = bnd_dict[y]
        xe = edge_dict[y]
        if side == 'left':
            if xb >= xe:
                continue
        else:
            if xe >= xb:
                continue
        xc = (xb + xe) / 2.0
        if not (0 <= xc < img_w):
            continue
        bnd_pts.append((int(round(xb)), y))
        edge_pts.append((int(round(xe)), y))
        center_pts.append((int(round(xc)), y))

    return bnd_pts, edge_pts, center_pts


# ─────────────────────────────────────────────────────────────────────────────
# 回退策略 C：纯水域 → 水域中心线
# ─────────────────────────────────────────────────────────────────────────────

def compute_water_centerline(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    无边界时，逐行扫描水域像素范围，取中点作为水域中心线。

    Returns:
        center_pts: 中心线点列表
    """
    h, w = mask.shape[:2]
    water_bin = (mask == WATER_CLASS) | (mask == BOUNDARY_CLASS)

    row_centers = {}
    for y in range(h):
        water_cols = np.where(water_bin[y, :])[0]
        if len(water_cols) < 5:
            continue
        x_min = float(water_cols[0])
        x_max = float(water_cols[-1])
        if x_max - x_min < 10:
            continue
        row_centers[y] = (x_min + x_max) / 2.0

    if len(row_centers) < MIN_WATER_ROWS:
        return []

    ys = sorted(row_centers.keys())
    xs = np.array([row_centers[y] for y in ys], dtype=np.float32)
    xs_smooth = _medfilt1d(xs, min(21, len(xs)))

    center_pts = [(int(round(x)), y) for y, x in zip(ys, xs_smooth)]
    return center_pts


# ─────────────────────────────────────────────────────────────────────────────
# 核心：从底部中心生成平滑路径
# ─────────────────────────────────────────────────────────────────────────────

def plan_path_from_bottom(
    center_pts: List[Tuple[int, int]],
    img_h: int,
    img_w: int,
    mask: np.ndarray,
    num_output_pts: int = 200,
    merge_rows: int = 80,
) -> Tuple[List[Tuple[int, int]], Optional[str]]:
    """
    从图像底部水域中心点生成平滑路径，过渡到参考中线。
    起始点使用底行实际水域中心（而非图像几何中心），避免路径在
    非对称河道中向边界偏移。
    """
    if len(center_pts) < 3:
        return [], "参考中线点数不足"

    start_y = img_h - 1
    # 用底行实际水域/边界中心替代图像固定中心
    _bot_cols = np.where(
        (mask[start_y] == WATER_CLASS) | (mask[start_y] == BOUNDARY_CLASS)
    )[0]
    start_x = (
        int(round(float(_bot_cols[0] + _bot_cols[-1]) / 2))
        if len(_bot_cols) >= 2
        else img_w // 2
    )

    sorted_pts = sorted(center_pts, key=lambda p: -p[1])

    cl_bottom_x, cl_bottom_y = sorted_pts[0]

    anchor_pts = [(start_x, start_y)]

    blend_y = max(cl_bottom_y, start_y - merge_rows)
    mid_y = (start_y + blend_y) // 2
    mid_x = (start_x + cl_bottom_x) // 2
    anchor_pts.append((mid_x, mid_y))

    step = max(1, len(sorted_pts) // 30)
    for pt in sorted_pts[::step]:
        if pt[1] <= blend_y:
            anchor_pts.append(pt)
    if sorted_pts[-1] not in anchor_pts:
        anchor_pts.append(sorted_pts[-1])

    anchor_pts.sort(key=lambda p: -p[1])

    unique = []
    for p in anchor_pts:
        if not unique or p != unique[-1]:
            unique.append(p)
    anchor_pts = unique

    if len(anchor_pts) < 3:
        return anchor_pts, None

    xs = np.array([p[0] for p in anchor_pts], dtype=np.float64)
    ys = np.array([p[1] for p in anchor_pts], dtype=np.float64)

    try:
        k = min(3, len(xs) - 1)
        tck, u = splprep([xs, ys], s=len(xs) * 2, k=k)
        u_new = np.linspace(0, 1, num_output_pts)
        sx, sy = splev(u_new, tck)
    except Exception:
        sx = np.interp(np.linspace(0, len(xs) - 1, num_output_pts),
                       np.arange(len(xs)), xs)
        sy = np.interp(np.linspace(0, len(ys) - 1, num_output_pts),
                       np.arange(len(ys)), ys)

    path_pts = []
    for x, y in zip(sx, sy):
        xi, yi = int(round(x)), int(round(y))
        xi = max(0, min(img_w - 1, xi))
        yi = max(0, min(img_h - 1, yi))
        path_pts.append((xi, yi))

    path_pts = _clip_to_water(path_pts, mask, img_w)

    if len(path_pts) < 3:
        return [], "路径被裁剪后点数不足"

    return path_pts, None


def _clip_to_water(
    path_pts: List[Tuple[int, int]],
    mask: np.ndarray,
    img_w: int,
    margin: int = 15,
) -> List[Tuple[int, int]]:
    """裁剪路径使其保持在水域/边界区域内。"""
    h, w = mask.shape[:2]
    safe = []
    for x, y in path_pts:
        y_c = max(0, min(h - 1, y))
        x_c = max(0, min(w - 1, x))
        cls = mask[y_c, x_c]
        if cls in (WATER_CLASS, BOUNDARY_CLASS):
            safe.append((x, y))
        else:
            shifted = _try_shift_to_water(x, y, mask, img_w, margin)
            if shifted is not None:
                safe.append(shifted)
    return safe


def _try_shift_to_water(
    x: int, y: int, mask: np.ndarray, img_w: int, margin: int
) -> Optional[Tuple[int, int]]:
    """对超出水域的点做横向微调。"""
    h, w = mask.shape[:2]
    center_x = img_w // 2
    direction = 1 if x < center_x else -1
    for dx in range(1, margin + 1):
        nx = x + direction * dx
        if 0 <= nx < w:
            if mask[min(y, h - 1), nx] in (WATER_CLASS, BOUNDARY_CLASS):
                return (nx, y)
    for dx in range(1, margin + 1):
        nx = x - direction * dx
        if 0 <= nx < w:
            if mask[min(y, h - 1), nx] in (WATER_CLASS, BOUNDARY_CLASS):
                return (nx, y)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────────────────────────────────────

def _draw_path_with_arrows(canvas, pts, color, thickness, arrow_every=30):
    """绘制路径实线并沿路径方向画箭头。"""
    if len(pts) < 2:
        return
    pts_np = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts_np], isClosed=False,
                  color=color, thickness=thickness, lineType=cv2.LINE_AA)

    for i in range(arrow_every, len(pts), arrow_every):
        p1 = np.array(pts[i - 1], dtype=np.float64)
        p2 = np.array(pts[i], dtype=np.float64)
        d = p2 - p1
        norm = np.linalg.norm(d)
        if norm < 1:
            continue
        d = d / norm

        angle_rad = np.radians(ARROW_ANGLE)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        left  = p2 - ARROW_LEN * np.array([d[0]*cos_a - d[1]*sin_a,
                                            d[0]*sin_a + d[1]*cos_a])
        right = p2 - ARROW_LEN * np.array([d[0]*cos_a + d[1]*sin_a,
                                           -d[0]*sin_a + d[1]*cos_a])

        tip = tuple(p2.astype(int))
        cv2.line(canvas, tip, tuple(left.astype(int)), color, max(1, thickness - 1), cv2.LINE_AA)
        cv2.line(canvas, tip, tuple(right.astype(int)), color, max(1, thickness - 1), cv2.LINE_AA)


def _draw_start_marker(canvas, x, y, radius=12):
    cv2.circle(canvas, (x, y), radius, C_START, 3, cv2.LINE_AA)
    cv2.circle(canvas, (x, y), 4, C_START, -1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# 主渲染函数：三级回退
# ─────────────────────────────────────────────────────────────────────────────

def render_planned_frame(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    fname: str = "",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    h, w = img_bgr.shape[:2]
    canvas = img_bgr.copy()

    bnd_mask = (mask == BOUNDARY_CLASS)
    if bnd_mask.any():
        tint = canvas.copy()
        tint[bnd_mask] = np.clip(
            tint[bnd_mask].astype(np.int32) + np.array([0, 0, 80], np.int32), 0, 255
        ).astype(np.uint8)
        cv2.addWeighted(tint, 0.6, canvas, 0.4, 0, canvas)

    status: Dict[str, Any] = {
        'skipped': False, 'reason': '', 'mode': '',
        'n_center_pts': 0, 'n_path_pts': 0,
    }

    left_pts = right_pts = center_pts = []
    mode = ''

    # ── 策略 A：双边界 ──
    left_mask_arr, right_mask_arr, err_a = find_two_boundaries(mask)
    if left_mask_arr is not None:
        left_raw = scan_comp_rows(left_mask_arr)
        right_raw = scan_comp_rows(right_mask_arr)
        left_smooth = smooth_row_dict(left_raw)
        right_smooth = smooth_row_dict(right_raw)
        lp, rp, cp, err_cl = compute_centerline(left_smooth, right_smooth, w)
        if err_cl is None and len(cp) >= 3:
            left_pts, right_pts, center_pts = lp, rp, cp
            mode = 'dual'

    # ── 策略 B：单边界 ──
    if mode == '':
        comp_mask, side, err_b = _find_single_boundary(mask)
        if comp_mask is not None:
            bnd_pts_b, edge_pts_b, cp_b = compute_single_boundary_centerline(
                mask, comp_mask, side, w
            )
            if len(cp_b) >= 3:
                if side == 'left':
                    left_pts = bnd_pts_b
                    right_pts = edge_pts_b
                else:
                    left_pts = edge_pts_b
                    right_pts = bnd_pts_b
                center_pts = cp_b
                mode = 'single'

    # ── 策略 C：纯水域 ──
    if mode == '':
        cp_c = compute_water_centerline(mask)
        if len(cp_c) >= 3:
            center_pts = cp_c
            mode = 'water'

    # ── 全部失败 ──
    if mode == '' or len(center_pts) < 3:
        status.update(skipped=True, reason="无法检测到可用水域/边界")
        _put_text_block(canvas, [fname, "跳过: 无法检测到可用水域/边界"], color=C_SKIP)
        return canvas, status

    status['mode'] = mode
    status['n_center_pts'] = len(center_pts)

    # ── 路径规划 ──
    path_pts, path_err = plan_path_from_bottom(center_pts, h, w, mask)
    if path_err or len(path_pts) < 3:
        status.update(skipped=True, reason=path_err or "路径生成失败")
        _put_text_block(canvas, [fname, f"跳过: {path_err}"], color=C_SKIP)
        return canvas, status

    status['n_path_pts'] = len(path_pts)

    # ── 绘制 ──
    if left_pts:
        _draw_dashed_polyline(canvas, left_pts, C_LEFT, TH_BOUND)
    if right_pts:
        _draw_dashed_polyline(canvas, right_pts, C_RIGHT, TH_BOUND)

    cl_color = C_CENTER if mode == 'dual' else C_WATER_CL
    _draw_solid_polyline(canvas, center_pts, cl_color, 2)
    _draw_path_with_arrows(canvas, path_pts, C_PATH, TH_PATH)
    _draw_start_marker(canvas, w // 2, h - 1)

    def _lbl(pts, ratio=0.75):
        idx = min(int(len(pts) * ratio), len(pts) - 1)
        return pts[idx]

    if left_pts:
        _label_at(canvas, "Left Bnd",  _lbl(left_pts),  C_LEFT,  scale=0.48)
    if right_pts:
        _label_at(canvas, "Right Bnd", _lbl(right_pts), C_RIGHT, scale=0.48)

    mode_labels = {'dual': 'Centerline', 'single': 'Est. Center', 'water': 'Water Center'}
    _label_at(canvas, mode_labels[mode], _lbl(center_pts), cl_color, scale=0.50)
    _label_at(canvas, "Planned Path", _lbl(path_pts), C_PATH, scale=0.55)

    legend_items = [(C_PATH, "Planned Path"), (C_START, "Start Point")]
    if mode == 'dual':
        legend_items = [
            (C_LEFT,   "Left Boundary"),
            (C_RIGHT,  "Right Boundary"),
            (C_CENTER, "Centerline"),
        ] + legend_items
    elif mode == 'single':
        legend_items = [
            (C_LEFT,     "Boundary"),
            (C_RIGHT,    "Water Edge"),
            (C_WATER_CL, "Est. Centerline"),
        ] + legend_items
    else:
        legend_items = [(C_WATER_CL, "Water Centerline")] + legend_items
    _draw_legend(canvas, legend_items)

    mode_str = {'dual': 'Dual-Bnd', 'single': 'Single-Bnd', 'water': 'Water-Only'}
    _put_text_block(canvas, [
        fname,
        f"[{mode_str[mode]}] CL:{len(center_pts)}  Path:{len(path_pts)}",
    ], color=(220, 220, 220))

    return canvas, status


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="自主路径规划：从图像底部中心沿 centerline 方向生成平滑路径",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--images',      required=True,  help='输入图像文件或目录')
    p.add_argument('--masks',       default=None,   help='mask 文件或目录（与 --model 二选一）')
    p.add_argument('--model',       default=None,   help='SegFormer 权重 .pth 路径')
    p.add_argument('--output',      default=None,   help='结果保存目录')
    p.add_argument('--show',        action='store_true', help='弹窗显示')
    p.add_argument('--num-classes', type=int, default=3,  help='分割类别数')
    p.add_argument('--input-size',  type=int, default=512, help='推理输入尺寸')
    return p.parse_args()


def main():
    args = parse_args()

    if args.masks is None and args.model is None:
        sys.exit("❌ 请通过 --masks 或 --model 指定 mask 来源")
    if args.masks is not None and args.model is not None:
        sys.exit("❌ --masks 与 --model 互斥，只能指定一个")
    if not args.show and args.output is None:
        print("⚠️  未指定 --output 也未开启 --show，结果不保存不显示。")

    img_paths = collect_images(args.images)
    print(f"[信息] 共找到 {len(img_paths)} 张图像")

    seg_model, device = None, None
    if args.model:
        seg_model, device = load_segformer(args.model, num_classes=args.num_classes)

    out_dir = None
    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[信息] 结果保存至: {out_dir}")

    total = skipped = saved = 0
    mode_counts = {'dual': 0, 'single': 0, 'water': 0}

    for img_path in img_paths:
        total += 1
        fname = img_path.name

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[警告] 无法读取: {img_path}")
            continue

        if seg_model is not None:
            mask_arr = infer_mask(seg_model, device, img_bgr,
                                 input_size=args.input_size,
                                 num_classes=args.num_classes)
        else:
            from PIL import Image as PILImage
            mpath = find_mask_path(args.masks, img_path)
            if mpath is None:
                print(f"[警告] 找不到 mask: {img_path.stem}.png，跳过")
                skipped += 1
                continue
            mask_arr = np.array(PILImage.open(str(mpath)).convert('P'))

        vis, status = render_planned_frame(img_bgr, mask_arr, fname=fname)

        if status['skipped']:
            skipped += 1
            print(f"  [跳过] {fname}: {status['reason']}")
        else:
            m = status['mode']
            mode_counts[m] = mode_counts.get(m, 0) + 1
            mode_tag = {'dual': '双边界', 'single': '单边界', 'water': '水域'}[m]
            print(f"  [OK]   {fname}: [{mode_tag}] "
                  f"中线={status['n_center_pts']} 路径={status['n_path_pts']}")

        if out_dir is not None:
            suffix = "_skip" if status['skipped'] else "_plan"
            save_path = out_dir / (img_path.stem + suffix + '.jpg')
            cv2.imwrite(str(save_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 94])
            saved += 1

        if args.show:
            cv2.imshow(f"路径规划 - {fname}", vis)
            key = cv2.waitKey(0)
            if key in (ord('q'), ord('Q'), 27):
                print("[信息] 用户退出")
                break

    if args.show:
        cv2.destroyAllWindows()

    print("\n──────────────── 统计 ────────────────")
    print(f"  总计: {total} 张 | 跳过: {skipped} | 有效: {total - skipped}")
    for m, cnt in mode_counts.items():
        label = {'dual': '双边界', 'single': '单边界', 'water': '水域中心'}[m]
        print(f"    {label}: {cnt} 张")
    if out_dir:
        print(f"  保存: {saved} 张 → {out_dir}")
    print("──────────────────────────────────────")


if __name__ == '__main__':
    main()
