#!/usr/bin/env python3
"""
可视化分割模型预测的边界线（boundary）与导航中线。

【算法】
  1. 逐行扫描 boundary 像素（类别 ID=2），以图像水平中线为分界，
     左半边取代表 x（中值），右半边取代表 x（中值）
  2. 对左/右点列分别做滑动中值平滑，去除毛刺
  3. 取左右都有数据的行，逐行求平均得到中线
  4. 绘制结果：只有 2 条边界线 + 1 条中线，干净标注名称
  仅检测到单侧边界的帧自动跳过。

【输入模式】
  A. mask 模式：直接读取已有的分割 mask PNG（无需 GPU）
  B. 模型推理模式：加载训练好的 SegFormer 权重实时推理

用法：
  # A. mask 模式
  python scripts/visualize_centerline.py \\
      --images dataset_final/images/val \\
      --masks  dataset_final/masks/val  \\
      --output vis_centerline

  # B. 模型推理
  python scripts/visualize_centerline.py \\
      --images dataset_final/images/val \\
      --model  models/segformer_river/best_model.pth \\
      --output vis_centerline

python scripts/visualize_centerline.py --images dataset_final/images/val --model models/segformer_river/best_model.pth --output vis_centerline



  # 弹窗查看（按任意键翻页，q 退出）
  python scripts/visualize_centerline.py \\
      --images dataset_final/images/val \\
      --masks  dataset_final/masks/val  \\
      --show
"""

import argparse
import sys
import os
import site
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image


def _ensure_jetson_cuda_lib_path() -> None:
    """优先使用 Jetson 真实 CUDA 库，避免误加载 stub 库。

    部分 PyTorch wheel 依赖 pip 安装的 cuSPARSELt（libcusparseLt.so.0），
    该库位于 venv 的 site-packages/nvidia/cusparselt/lib，须加入 LD_LIBRARY_PATH。
    """
    preferred = [
        '/usr/local/cuda/targets/aarch64-linux/lib',
        '/lib/aarch64-linux-gnu',
    ]
    site_dirs = list(site.getsitepackages())
    user_sp = site.getusersitepackages()
    if user_sp:
        site_dirs.append(user_sp)
    for sp in site_dirs:
        csl_lib = os.path.join(sp, 'nvidia', 'cusparselt', 'lib')
        if os.path.isfile(os.path.join(csl_lib, 'libcusparseLt.so.0')):
            preferred.insert(0, csl_lib)

    existing = os.environ.get('LD_LIBRARY_PATH', '')
    parts = [p for p in existing.split(':') if p]

    changed = False
    for p in reversed(preferred):
        if os.path.isdir(p) and p not in parts:
            parts.insert(0, p)
            changed = True

    if changed:
        os.environ['LD_LIBRARY_PATH'] = ':'.join(parts)

# ─────────────────────────────────────────────────────────────────────────────
# 参数
# ─────────────────────────────────────────────────────────────────────────────
BOUNDARY_CLASS  = 2      # mask 中边界线的类别 ID
WATER_CLASS     = 1      # mask 中水域的类别 ID

MIN_ROW_PX      = 1      # 每行至少有多少 boundary 像素才算有效
SMOOTH_WINDOW   = 11     # 滑动中值平滑窗口大小（行数，取奇数）
MIN_VALID_ROWS  = 5      # 左右各自有效行数低于此则跳过
MIN_OVERLAP_ROWS= 5      # 左右重叠行低于此则跳过

# 可视化颜色 (BGR)
C_LEFT   = (30,  160, 255)   # 橙色：左边界
C_RIGHT  = (60,  60,  220)   # 红色：右边界
C_CENTER = (0,   210,  0)    # 亮绿：中线
C_SKIP   = (0,   140, 255)   # 橙：跳过帧文字

TH_BOUND  = 2   # 边界线线宽
TH_CENTER = 4   # 中线线宽

# 虚线绘制参数（dash 段长 / gap 段长，像素）
DASH_LEN  = 20
GAP_LEN   = 10


# ─────────────────────────────────────────────────────────────────────────────
# 核心算法：连通域识别 → 逐行扫描 → 平滑 → 中线
# ─────────────────────────────────────────────────────────────────────────────

def _medfilt1d(arr, k):
    """对 1D 数组做长度为 k 的滑动中值滤波（k 自动取奇数且不超过 arr 长度）。"""
    k = min(k, len(arr))
    if k % 2 == 0:
        k -= 1
    k = max(k, 1)
    result = arr.copy()
    half = k // 2
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        result[i] = float(np.median(arr[lo:hi]))
    return result


MIN_COMP_PX    = 50     # 连通域最小面积（像素），低于此视为噪声
MIN_SEPARATION = 0.04   # 两连通域质心 x 之差 / 图像宽，低于此视为同一边界


def find_two_boundaries(mask: np.ndarray):
    """
    用连通域分析找出"左边界"和"右边界"连通域，不依赖图像水平中线。

    策略：
      1. 形态学闭合 → 连通域分析 → 过滤噪声
      2. 在所有满足最小面积的连通域中，按质心 x 排序
      3. 选质心 x 最小的为"左"，质心 x 最大的为"右"
      4. 若两者质心 x 差 / 图像宽 < MIN_SEPARATION → 视为单边界 → 返回 None

    Returns:
        (left_labels, right_labels, label_img) 或 (None, None, reason_str)
        left_labels / right_labels：该连通域对应的像素 mask（bool 数组）
    """
    h, w = mask.shape[:2]

    bnd_bin = (mask == BOUNDARY_CLASS).astype(np.uint8)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bnd_bin = cv2.morphologyEx(bnd_bin, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bnd_bin)

    comps = []
    for lid in range(1, num_labels):
        area = int(stats[lid, cv2.CC_STAT_AREA])
        if area < MIN_COMP_PX:
            continue
        comps.append({
            'lid':        lid,
            'area':       area,
            'centroid_x': float(centroids[lid, 0]),
            'centroid_y': float(centroids[lid, 1]),
        })

    if len(comps) == 0:
        return None, None, "未检测到 boundary 像素"
    if len(comps) == 1:
        return None, None, "仅检测到单条边界线，无法计算中线"

    comps.sort(key=lambda c: c['centroid_x'])
    left_comp  = comps[0]
    right_comp = comps[-1]

    sep = (right_comp['centroid_x'] - left_comp['centroid_x']) / max(w, 1)
    if sep < MIN_SEPARATION:
        return None, None, f"两条边界线质心间距过近（{sep:.1%}），视为单边界"

    left_mask  = (labels == left_comp['lid'])
    right_mask = (labels == right_comp['lid'])
    return left_mask, right_mask, None


def scan_comp_rows(comp_mask: np.ndarray):
    """
    在单个连通域 mask 内逐行取 x 中值，返回 {y: x_median} 字典。
    """
    row_dict = {}
    rows, cols = np.where(comp_mask)
    if len(rows) == 0:
        return row_dict
    for y in np.unique(rows):
        xs_in_row = cols[rows == y]
        if len(xs_in_row) >= 1:
            row_dict[int(y)] = float(np.median(xs_in_row))
    return row_dict


def smooth_row_dict(row_dict: dict, window: int = SMOOTH_WINDOW):
    """对 {y: x} 字典做滑动中值平滑，返回平滑后的 {y: x}。"""
    if len(row_dict) < 3:
        return row_dict
    ys   = sorted(row_dict.keys())
    xs   = np.array([row_dict[y] for y in ys], dtype=np.float32)
    xs_s = _medfilt1d(xs, window)
    return {y: float(x) for y, x in zip(ys, xs_s)}


def _extrapolate_dict(row_dict: dict, y_min: int, y_max: int) -> dict:
    """
    将 {y: x} 扩展到 [y_min, y_max] 范围：
    - 已有行：线性插值填充空隙
    - 超出范围的行：用端点的线性趋势外推（最多外推 50 行）
    """
    if len(row_dict) == 0:
        return {}

    ys_known = sorted(row_dict.keys())
    xs_known = [row_dict[y] for y in ys_known]

    # 内插：覆盖已知范围内的空行
    full_ys = list(range(ys_known[0], ys_known[-1] + 1))
    full_xs = np.interp(full_ys, ys_known, xs_known).tolist()
    result  = {y: float(x) for y, x in zip(full_ys, full_xs)}

    # 向上外推（y < ys_known[0]，最多 50 行）
    if len(ys_known) >= 2:
        dy = ys_known[0] - ys_known[1]  # 负值（向上）
        dx = xs_known[0] - xs_known[1]
        slope = dx / dy if dy != 0 else 0.0
        for step in range(1, 51):
            y_ext = ys_known[0] - step
            if y_ext < y_min:
                break
            result[y_ext] = xs_known[0] + slope * step

    # 向下外推（y > ys_known[-1]，最多 50 行）
    if len(ys_known) >= 2:
        dy = ys_known[-1] - ys_known[-2]  # 正值（向下）
        dx = xs_known[-1] - xs_known[-2]
        slope = dx / dy if dy != 0 else 0.0
        for step in range(1, 51):
            y_ext = ys_known[-1] + step
            if y_ext > y_max:
                break
            result[y_ext] = xs_known[-1] + slope * step

    return result


def compute_centerline(left_dict: dict, right_dict: dict, img_width: int):
    """
    由左/右边界的 {y: x} 字典计算中线点列表。
    两侧分别内插+外推到对方的 y 范围，取并集计算中线。

    Returns:
        (left_pts, right_pts, center_pts, reason)
        reason 非 None 时表示失败原因。
    """
    if len(left_dict) < MIN_VALID_ROWS:
        return [], [], [], f"左边界有效行数不足（{len(left_dict)} < {MIN_VALID_ROWS}）"
    if len(right_dict) < MIN_VALID_ROWS:
        return [], [], [], f"右边界有效行数不足（{len(right_dict)} < {MIN_VALID_ROWS}）"

    # 确定并集 y 范围
    all_ys  = sorted(set(left_dict.keys()) | set(right_dict.keys()))
    y_min_u = all_ys[0]
    y_max_u = all_ys[-1]

    # 对各自插值 + 外推到并集范围
    left_full  = _extrapolate_dict(left_dict,  y_min_u, y_max_u)
    right_full = _extrapolate_dict(right_dict, y_min_u, y_max_u)

    union_ys = sorted(set(left_full.keys()) & set(right_full.keys()))
    if len(union_ys) < MIN_OVERLAP_ROWS:
        return [], [], [], f"并集覆盖行不足（{len(union_ys)} < {MIN_OVERLAP_ROWS}）"

    w          = img_width
    left_pts   = []
    right_pts  = []
    center_pts = []

    for y in union_ys:
        xl = left_full[y]
        xr = right_full[y]
        # 两侧都必须在图像内，且左在右的左边
        if not (0 <= xl < w and 0 <= xr < w):
            continue
        if xl >= xr:
            continue
        xc = (xl + xr) / 2.0
        left_pts.append((int(round(xl)), y))
        right_pts.append((int(round(xr)), y))
        center_pts.append((int(round(xc)), y))

    if len(center_pts) < MIN_OVERLAP_ROWS:
        return [], [], [], f"有效中线点不足（{len(center_pts)} < {MIN_OVERLAP_ROWS}）"

    return left_pts, right_pts, center_pts, None


# ─────────────────────────────────────────────────────────────────────────────
# 可视化工具
# ─────────────────────────────────────────────────────────────────────────────

def _draw_dashed_polyline(canvas, pts, color, thickness, dash=DASH_LEN, gap=GAP_LEN):
    """在 canvas 上绘制虚线折线。"""
    if len(pts) < 2:
        return
    acc = 0        # 已累积的像素长度
    drawing = True # 当前是 dash 还是 gap
    seg_remain = dash

    for i in range(1, len(pts)):
        p0 = np.array(pts[i - 1], dtype=float)
        p1 = np.array(pts[i],     dtype=float)
        seg_len = float(np.linalg.norm(p1 - p0))
        if seg_len < 0.5:
            continue

        walked = 0.0
        while walked < seg_len:
            step = min(seg_remain, seg_len - walked)
            t0   = walked / seg_len
            t1   = (walked + step) / seg_len
            if drawing:
                pt_a = (int(p0[0] + t0 * (p1[0] - p0[0])),
                        int(p0[1] + t0 * (p1[1] - p0[1])))
                pt_b = (int(p0[0] + t1 * (p1[0] - p0[0])),
                        int(p0[1] + t1 * (p1[1] - p0[1])))
                cv2.line(canvas, pt_a, pt_b, color, thickness, cv2.LINE_AA)
            walked     += step
            seg_remain -= step
            if seg_remain <= 0:
                drawing    = not drawing
                seg_remain = dash if drawing else gap


def _draw_solid_polyline(canvas, pts, color, thickness):
    """在 canvas 上绘制实线折线。"""
    if len(pts) < 2:
        return
    pts_np = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts_np], isClosed=False,
                  color=color, thickness=thickness, lineType=cv2.LINE_AA)


def _label_at(canvas, text, pt, color, bg=(0, 0, 0),
               scale=0.55, thickness=1):
    """在 pt 旁绘制带黑色描边的彩色文字（居中对齐）。"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(2, pt[0] - tw // 2)
    y = max(th + 2, pt[1])
    # 描边
    for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        cv2.putText(canvas, text, (x + dx, y + dy), font, scale,
                    bg, thickness + 1, cv2.LINE_AA)
    cv2.putText(canvas, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _put_text_block(canvas, lines, org_y=26, color=(255, 255, 255), scale=0.55):
    """在左上角绘制多行文字（带黑底描边）。"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    lh = int(24 * scale) + 4
    for i, line in enumerate(lines):
        y = org_y + i * lh
        for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            cv2.putText(canvas, line, (10 + dx, y + dy), font, scale,
                        (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, line, (10, y), font, scale, color, 1, cv2.LINE_AA)


def _draw_legend(canvas, items, margin=10):
    """在右下角绘制图例框（颜色块 + 文字）。"""
    h, w = canvas.shape[:2]
    font   = cv2.FONT_HERSHEY_SIMPLEX
    scale  = 0.50
    lh     = 22
    bw, bh = 20, 12
    pad    = 6

    # 计算最大文字宽
    max_tw = max(cv2.getTextSize(label, font, scale, 1)[0][0] for _, label in items)
    box_w  = pad * 2 + bw + 6 + max_tw
    box_h  = pad * 2 + len(items) * lh

    x0 = w - box_w - margin
    y0 = h - box_h - margin

    # 半透明背景
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0 - 2, y0 - 2), (x0 + box_w, y0 + box_h),
                  (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

    for i, (color, label) in enumerate(items):
        y = y0 + pad + i * lh
        # 颜色块
        cv2.rectangle(canvas,
                      (x0 + pad, y),
                      (x0 + pad + bw, y + bh),
                      color, -1)
        # 文字
        cv2.putText(canvas, label,
                    (x0 + pad + bw + 6, y + bh - 1),
                    font, scale, (220, 220, 220), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# 主渲染函数
# ─────────────────────────────────────────────────────────────────────────────

def render_frame(img_bgr: np.ndarray, mask: np.ndarray, fname: str = ""):
    """
    对单帧图像及对应 mask 计算并渲染：左边界、右边界、中线。

    Returns:
        (vis_img, status)
        status: {'skipped': bool, 'reason': str, 'n_center_pts': int}
    """
    h, w = img_bgr.shape[:2]

    # ── 1. 在原图上轻叠 boundary 像素（仅红色半透明，不叠水域）
    canvas   = img_bgr.copy()
    bnd_mask = (mask == BOUNDARY_CLASS)
    tint     = canvas.copy()
    tint[bnd_mask] = np.clip(
        tint[bnd_mask].astype(np.int32) + np.array([0, 0, 80], np.int32),
        0, 255
    ).astype(np.uint8)
    cv2.addWeighted(tint, 0.6, canvas, 0.4, 0, canvas)

    status = {'skipped': False, 'reason': '', 'n_center_pts': 0}

    # ── 2. 连通域识别两条边界
    left_mask, right_mask, err = find_two_boundaries(mask)

    if left_mask is None:
        status.update(skipped=True, reason=err)
        _put_text_block(canvas, [fname, f"跳过: {err}"], color=C_SKIP)
        return canvas, status

    # ── 3. 各连通域逐行扫描 + 平滑
    left_raw   = scan_comp_rows(left_mask)
    right_raw  = scan_comp_rows(right_mask)
    left_smooth  = smooth_row_dict(left_raw)
    right_smooth = smooth_row_dict(right_raw)

    # ── 4. 中线
    left_pts, right_pts, center_pts, err = compute_centerline(left_smooth, right_smooth, w)

    if err:
        status.update(skipped=True, reason=err)
        _put_text_block(canvas, [fname, f"跳过: {err}"], color=C_SKIP)
        return canvas, status

    status['n_center_pts'] = len(center_pts)

    # ── 5. 绘制三条线
    # 左边界：橙色虚线
    _draw_dashed_polyline(canvas, left_pts, C_LEFT, TH_BOUND)
    # 右边界：红色虚线
    _draw_dashed_polyline(canvas, right_pts, C_RIGHT, TH_BOUND)
    # 中线：亮绿实线
    _draw_solid_polyline(canvas, center_pts, C_CENTER, TH_CENTER)

    # ── 6. 在线条底端（图像偏下方）打标签
    #    取线条中下部的点作为标签位置，避免超出边界
    def _label_pt(pts, ratio=0.80):
        """取线条从上到下 ratio 位置处的点。"""
        idx = min(int(len(pts) * ratio), len(pts) - 1)
        return pts[idx]

    _label_at(canvas, "Left Boundary",  _label_pt(left_pts),   C_LEFT,   scale=0.55)
    _label_at(canvas, "Right Boundary", _label_pt(right_pts),  C_RIGHT,  scale=0.55)
    _label_at(canvas, "Centerline",     _label_pt(center_pts), C_CENTER, scale=0.58)

    # ── 7. 图例（右下角）
    legend_items = [
        (C_LEFT,   "Left Boundary  (dashed)"),
        (C_RIGHT,  "Right Boundary (dashed)"),
        (C_CENTER, "Centerline     (solid)"),
    ]
    _draw_legend(canvas, legend_items)

    # ── 8. 左上角简洁信息
    _put_text_block(canvas, [
        fname,
        f"Left rows:{len(left_pts)}  Right rows:{len(right_pts)}  Center pts:{len(center_pts)}",
    ], color=(220, 220, 220))

    return canvas, status


# ─────────────────────────────────────────────────────────────────────────────
# 模型推理（可选）
# ─────────────────────────────────────────────────────────────────────────────

def _get_device():
    """健壮的 CUDA 设备检测，强制只用 GPU 0（与训练脚本一致）。

    除了基本的 CUDA 初始化之外，还会运行一次小型 conv2d 来验证
    当前 GPU 的 compute capability 确实被 PyTorch 内核支持。
    """
    import torch
    if 'CUDA_VISIBLE_DEVICES' not in os.environ or \
            os.environ.get('CUDA_VISIBLE_DEVICES', '') == '':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if not torch.cuda.is_available():
        return torch.device('cpu')
    try:
        torch.cuda.init()
        # 基本 tensor 拷贝
        torch.zeros(1).cuda()
        # conv2d 验证：某些 CC（如 Orin 8.7）在简单运算通过后
        # conv kernel 仍可能缺失，此处提前暴露问题
        _x = torch.randn(1, 1, 4, 4, device='cuda')
        _w = torch.randn(1, 1, 3, 3, device='cuda')
        torch.nn.functional.conv2d(_x, _w)
        return torch.device('cuda')
    except Exception as e:
        print(f"[警告] CUDA 初始化失败，回退 CPU: {e}")
        return torch.device('cpu')


def _offline_segformer_config(model_name: str, num_classes: int):
    """网络不可达时，根据模型名构造最小可用 SegFormer 配置。"""
    from transformers import SegformerConfig

    # 默认按 B0 架构构建（当前项目训练默认）。
    hidden_sizes = [32, 64, 160, 256]
    depths = [2, 2, 2, 2]

    return SegformerConfig(
        num_labels=num_classes,
        num_encoder_blocks=4,
        depths=depths,
        sr_ratios=[8, 4, 2, 1],
        hidden_sizes=hidden_sizes,
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        num_attention_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        hidden_act='gelu',
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        decoder_hidden_size=256,
        semantic_loss_ignore_index=255,
    )


def _detect_model_name_from_sd(sd: dict) -> str:
    """从 state_dict 的张量形状自动推断 SegFormer 架构。"""
    import re
    key0 = 'segformer.encoder.patch_embeddings.0.proj.weight'
    if key0 not in sd:
        return None
    first_dim = sd[key0].shape[0]
    if first_dim == 32:
        return 'nvidia/segformer-b0-finetuned-ade-512-512'
    # first_dim == 64: B1/B2/B4/B5，通过 decoder hidden dim 区分
    dec_key = 'decode_head.linear_c.0.proj.weight'
    if dec_key not in sd:
        return None
    dec_dim = sd[dec_key].shape[0]
    if dec_dim == 256:
        return 'nvidia/segformer-b1-finetuned-ade-512-512'
    # dec_dim == 768: B2/B4/B5，通过 stage-2 block 数量区分
    stage2 = {int(m.group(1))
               for k in sd
               for m in [re.match(r'segformer\.encoder\.block\.2\.(\d+)\.', k)] if m}
    n2 = len(stage2)
    if n2 <= 6:
        return 'nvidia/segformer-b2-finetuned-ade-512-512'
    elif n2 <= 27:
        return 'nvidia/segformer-b4-finetuned-ade-512-512'
    else:
        return 'nvidia/segformer-b5-finetuned-ade-512-512'


def load_segformer(model_path: str, num_classes: int = 3,
                   model_name: str = None):
    """加载 SegFormer 权重，返回 (model, device)。

    model_name 留空时自动从 checkpoint 权重形状推断架构，
    也可手动指定（须与训练时一致）：
        - B0: nvidia/segformer-b0-finetuned-ade-512-512
        - B2: nvidia/segformer-b2-finetuned-ade-512-512
    """
    _ensure_jetson_cuda_lib_path()

    try:
        import torch
        from transformers import SegformerForSemanticSegmentation
    except ImportError:
        sys.exit("❌ 请安装 torch 和 transformers：pip install torch transformers")

    device = _get_device()
    print(f"[模型] 使用设备: {device}")

    # ── 先读取权重，推断架构 ──────────────────────────────────────────────────
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict):
        sd = ckpt.get('model_state_dict', ckpt)
        # checkpoint 中若保存了 model_name，优先使用
        if model_name is None and 'model_name' in ckpt:
            model_name = ckpt['model_name']
    else:
        sd = ckpt

    if model_name is None:
        model_name = _detect_model_name_from_sd(sd)
        if model_name is None:
            model_name = 'nvidia/segformer-b0-finetuned-ade-512-512'
            print(f"[警告] 无法推断架构，默认使用 B0")
        else:
            print(f"[模型] 自动推断架构: {model_name}")

    # ── 加载架构 ─────────────────────────────────────────────────────────────
    try:
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )
        print(f"[模型] 使用本地缓存架构: {model_name}")
    except Exception as e:
        print(f"[警告] 无法从缓存加载架构，启用离线配置: {e}")
        cfg = _offline_segformer_config(model_name, num_classes)
        model = SegformerForSemanticSegmentation(cfg)

    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    print(f"[模型] 权重加载成功: {model_path} (架构: {model_name})")
    return model, device


def infer_mask(model, device, img_bgr: np.ndarray,
               input_size: int = 640, num_classes: int = 3,
               input_width: int = 640, input_height: int = 384) -> np.ndarray:
    """对单张 BGR 图推理，返回 H×W 分割 mask (uint8)。
    
    输入分辨率默认 640×384（16:9），与训练时保持一致。
    input_size 参数保留用于向后兼容，若传入则同时覆盖 input_width/input_height。
    """
    import torch
    import torch.nn.functional as F

    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std  = np.array([0.229, 0.224, 0.225], np.float32)

    oh, ow = img_bgr.shape[:2]
    rgb    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rsz    = cv2.resize(rgb, (input_width, input_height))
    norm   = (rsz.astype(np.float32) / 255.0 - mean) / std
    t      = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = model(pixel_values=t).logits
        logits = F.interpolate(logits, size=(oh, ow), mode='bilinear', align_corners=False)
        pred   = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred


# ─────────────────────────────────────────────────────────────────────────────
# 文件收集工具
# ─────────────────────────────────────────────────────────────────────────────

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


def collect_images(path: str):
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        files = sorted([f for f in p.iterdir() if f.suffix.lower() in IMG_EXTS])
        if not files:
            sys.exit(f"❌ 目录 {path} 中没有图像文件")
        return files
    sys.exit(f"❌ 路径不存在: {path}")


def find_mask_path(mask_source: str, img_path: Path):
    ms = Path(mask_source)
    if ms.is_file():
        return ms
    for name in (img_path.stem + '.png', img_path.name):
        c = ms / name
        if c.exists():
            return c
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="可视化 boundary 中线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--images',      required=True,  help='输入图像文件或目录')
    p.add_argument('--masks',       default=None,   help='mask 文件或目录（与 --model 二选一）')
    p.add_argument('--model',       default=None,   help='SegFormer 权重 .pth 路径')
    p.add_argument('--output',      default=None,   help='结果保存目录')
    p.add_argument('--show',        action='store_true', help='弹窗显示')
    p.add_argument('--num-classes', type=int, default=3,  help='分割类别数（默认 3）')
    p.add_argument('--input-size',  type=int, default=512, help='推理输入尺寸（默认 512）')
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

    for img_path in img_paths:
        total += 1
        fname = img_path.name

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[警告] 无法读取: {img_path}")
            continue

        # 获取 mask
        if seg_model is not None:
            mask = infer_mask(seg_model, device, img_bgr,
                              input_size=args.input_size,
                              num_classes=args.num_classes)
        else:
            mpath = find_mask_path(args.masks, img_path)
            if mpath is None:
                print(f"[警告] 找不到 mask: {img_path.stem}.png，跳过")
                skipped += 1
                continue
            mask = np.array(Image.open(str(mpath)).convert('P'))

        vis, status = render_frame(img_bgr, mask, fname=fname)

        if status['skipped']:
            skipped += 1
            print(f"  [跳过] {fname}: {status['reason']}")
        else:
            print(f"  [OK]   {fname}: 中线点={status['n_center_pts']}")

        if out_dir is not None:
            suffix    = "_skip" if status['skipped'] else "_center"
            save_path = out_dir / (img_path.stem + suffix + '.jpg')
            cv2.imwrite(str(save_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 94])
            saved += 1

        if args.show:
            cv2.imshow(f"中线可视化 - {fname}", vis)
            key = cv2.waitKey(0)
            if key in (ord('q'), ord('Q'), 27):
                print("[信息] 用户退出")
                break

    if args.show:
        cv2.destroyAllWindows()

    print("\n──────────────── 统计 ────────────────")
    print(f"  总计: {total} 张 | 跳过: {skipped} | 有效: {total - skipped}")
    if out_dir:
        print(f"  保存: {saved} 张 → {out_dir}")
    print("──────────────────────────────────────")


if __name__ == '__main__':
    main()
