#!/usr/bin/env python3
"""
警戒线可视化脚本 —— 对测试集图片进行模型推理，将识别出的警戒线（边界线）
用黄色虚线标注叠加在原图上并保存。

用法:
  python scripts/Boundary_identification.py \
      --images dataset_final/images/test \
      --model  models/segformer_river/best_model.pth \
      --output vis_boundary

  # 同时弹窗查看
  python scripts/Boundary_identification.py \
      --images dataset_final/images/test \
      --model  models/segformer_river/best_model.pth \
      --output vis_boundary \
      --show
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from visualize_centerline import (
    BOUNDARY_CLASS, WATER_CLASS,
    load_segformer, infer_mask,
    collect_images,
    scan_comp_rows, smooth_row_dict,
)

# ─── 样式常量 ──────────────────────────────────────────────────────────────────
C_YELLOW   = (0, 255, 255)   # BGR 黄色
DASH_LEN   = 20
GAP_LEN    = 10
THICKNESS  = 3
MIN_AREA   = 80              # 连通域最少像素数，过滤噪声


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

    # 按左/右分组，每组内选 y_min 最小的折线（最靠上 = 真实警戒线）
    groups: dict = {'left': [], 'right': []}
    for pts in candidates:
        side = _polyline_side(pts, w)
        groups[side].append(pts)

    polylines: List[List[Tuple[int, int]]] = []
    n_filtered = 0
    for side, side_pts in groups.items():
        if not side_pts:
            continue
        # 按 y_min 升序排列，取第一条（最靠上）
        side_pts.sort(key=_min_y)
        polylines.append(side_pts[0])
        n_filtered += len(side_pts) - 1  # 其余视为倒影

    return polylines, n_filtered


# ─── 单帧可视化 ───────────────────────────────────────────────────────────────

def visualize_boundaries(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    在原图上叠加：
      · 边界区域半透明红色底色（让警戒带直观可见）
      · 每条边界连通域中心线用黄色虚线勾勒

    返回可视化图像（不修改输入）。
    """
    canvas = img_bgr.copy()

    # 边界区域半透明红色底色
    bnd_mask = (mask == BOUNDARY_CLASS)
    if bnd_mask.any():
        tint = canvas.copy()
        tint[bnd_mask] = np.clip(
            tint[bnd_mask].astype(np.int32) + np.array([0, 0, 100], np.int32),
            0, 255,
        ).astype(np.uint8)
        cv2.addWeighted(tint, 0.55, canvas, 0.45, 0, canvas)

    # 黄色虚线中心线（每侧最多一条，倒影已过滤）
    polylines, n_filtered = extract_boundary_polylines(mask)
    for pts in polylines:
        _draw_yellow_dashed(canvas, pts)

    return canvas, polylines, n_filtered


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="警戒线可视化：SegFormer 推理 → 黄色虚线标注边界",
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
                   default='nvidia/segformer-b0-finetuned-ade-512-512',
                   help='SegFormer 基础架构名，须与训练时一致')
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

    out_dir = None
    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[信息] 结果保存至: {out_dir}")

    total = saved = 0
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

            vis, polylines, n_filtered = visualize_boundaries(img_bgr, mask)

            n_bnd_px = int(np.sum(mask == BOUNDARY_CLASS))
            filtered_str = f"  [过滤倒影 {n_filtered}]" if n_filtered > 0 else ""
            print(f"  {img_path.name}  边界像素={n_bnd_px:5d}  "
                  f"警戒线段数={len(polylines)}{filtered_str}")

            if out_dir:
                save_path = out_dir / (img_path.stem + '_boundary.jpg')
                cv2.imwrite(str(save_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 94])
                saved += 1

            if args.show:
                cv2.imshow("警戒线识别 (Q/ESC 退出)", vis)
                key = cv2.waitKey(0)
                if key in (ord('q'), ord('Q'), 27):
                    print("[信息] 用户退出")
                    break
    finally:
        if args.show:
            cv2.destroyAllWindows()

    print(f"\n{'='*50}")
    print(f"  总计 {total} 张 | 已保存 {saved} 张")
    print(f"  说明: 每侧保留 y 最小（最靠上）的警戒线，其余视为倒影过滤")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
