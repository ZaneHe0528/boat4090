#!/usr/bin/env python3
"""
警戒线导航脚本 —— 识别两岸警戒线，计算中心线，连接船头生成平滑航行轨迹，
输出小船偏航角。支持 PyTorch 和 TensorRT 两种推理后端。
通过 px4_msgs 直接与 PX4 通信，Offboard 速度控制，20Hz 定频发布。

【直接跑相机（TensorRT）】
  python scripts/realtime_pilot_jsh.py \
      --camera /dev/video2 \
      --engine models/segformer_river/segformer_b0_640x384_fp16.engine

【PX4 Offboard 控制】0.3 m/s 前进，20Hz 发布 setpoint，1 秒预热后自动解锁：
  python scripts/realtime_pilot_jsh.py \
      --camera /dev/video3 \
      --engine models/segformer_river/segformer_b0_640x384_fp16.engine \
      --px4

速度限制，加速度限制，偏航角限制
python scripts/realtime_pilot_jsh.py --camera /dev/video3 --engine models/segformer_river/segformer_b0_640x384_fp16.engine --px4 --px4-forward-speed 0.2 --px4-max-accel 0.08 --px4-max-yaw-rate-deg 12

python scripts/realtime_pilot_jsh.py --camera /dev/video3 --engine models/segformer_river/segformer_b0_640x384_fp16.engine --px4 --px4-forward-speed 0.25 --px4-max-accel 0.08 --px4-yaw-gain 0.2 --px4-max-yaw-offset-deg 6 --px4-max-yaw-rate-deg 5


记录日志
python scripts/realtime_pilot_jsh.py \
  --camera /dev/video3 \
  --engine models/segformer_river/segformer_b0_640x384_fp16.engine \
  --px4 \
  --yaw-log debug/yaw_run_01.csv

python scripts/realtime_pilot_jsh.py \
  --camera /dev/video3 \
  --engine models/segformer_river/segformer_b0_640x384_fp16.engine \
  --px4 \
  --yaw-log debug/yaw_run_02.csv \
  --save-large-yaw-frames auto \
  --large-yaw-threshold-deg 8

【模型推理模式】对图片/目录进行推理：
  python scripts/realtime_pilot_jsh.py \
      --images dataset_final/images/test \
      --model  models/segformer_river/best_model.pth \
      --output vis_boundary/test

【实时相机模式】从摄像头读帧，实时输出偏航角：
  python scripts/realtime_pilot_jsh.py \
      --camera /dev/video3 \
      --engine models/segformer_river/segformer_b0_640x384_fp16.engine

【离线视频测试】读取视频文件，逐帧推理，输出警戒线数量和偏航角：
python scripts/realtime_pilot_jsh.py \
    --video video528.mp4 \
    --engine models/segformer_river/segformer_b0_640x384_fp16.engine \
    --save-video auto
"""

import argparse
from contextlib import nullcontext
import csv
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
    BOUNDARY_CLASS,
    load_segformer, infer_mask,
    collect_images,
    smooth_row_dict,
    _extrapolate_dict,
)

MIN_AREA   = 30              # 连通域最少像素数，过滤噪声

ROI_TOP_RATIO    = 0.1       # 去除图像最顶部比例（天空噪声）
WIDTH_OUTLIER_TH = 0.4       # 航道宽度偏差超过此比例的行视为异常
BOUNDARY_SMOOTH_WINDOW = 11  # 边界逐行平滑窗口（从 5 → 11，减少源头噪声）
CENTER_SMOOTH_WINDOW   = 31  # 中线 x 坐标平滑窗口（从 7 → 31，更强的平滑）
POSTPROC_MASK_WIDTH    = 640 # 后处理计算宽度上限


def _create_yaw_log_writer(log_path_arg: Optional[str], mode_name: str):
    """创建偏航角日志 CSV，返回 (file_handle, csv_writer, path)。"""
    if not log_path_arg:
        return None, None, None

    if log_path_arg.lower() == 'auto':
        log_dir = Path('logs')
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        log_path = log_dir / f'yaw_log_{mode_name}_{timestamp}.csv'
    else:
        log_path = Path(log_path_arg)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, 'w', newline='')
    writer = csv.writer(log_file)
    writer.writerow([
        'wall_time',
        'frame',
        'yaw_deg',
        'status',
        'n_filtered',
        'center_count',
        'path_count',
        'px4_status_ok',
        'px4_target_speed',
        'px4_command_speed',
        'px4_raw_yaw_deg',
        'px4_target_yaw_deg',
        'px4_command_yaw_deg',
        'px4_absolute_yaw_deg',
        'px4_velocity_x',
        'px4_velocity_y',
    ])
    print(f'[信息] 偏航角日志已开启: {log_path}')
    return log_file, writer, log_path


def _write_yaw_log_row(
    writer,
    frame_index: int,
    result: Dict[str, Any],
    control_state: Optional[Dict[str, Any]] = None,
) -> None:
    if writer is None:
        return
    control_state = control_state or {}
    writer.writerow([
        f'{time.time():.3f}',
        frame_index,
        f"{float(result['yaw_deg']):.2f}",
        result['status'],
        int(result.get('n_filtered', 0)),
        len(result.get('center_pts', [])),
        len(result.get('path_pts', [])),
        control_state.get('status_ok', ''),
        _format_optional_float(control_state.get('target_speed')),
        _format_optional_float(control_state.get('command_speed')),
        _format_optional_float(control_state.get('raw_yaw_deg')),
        _format_optional_float(control_state.get('target_yaw_deg')),
        _format_optional_float(control_state.get('command_yaw_deg')),
        _format_optional_float(control_state.get('absolute_yaw_deg')),
        _format_optional_float(control_state.get('velocity_x')),
        _format_optional_float(control_state.get('velocity_y')),
    ])


def _format_optional_float(value: Optional[float]) -> str:
    if value is None:
        return ''
    return f'{float(value):.4f}'


def _resolve_capture_dir(save_dir_arg: Optional[str], mode_name: str) -> Optional[Path]:
    if not save_dir_arg:
        return None
    if save_dir_arg.lower() == 'auto':
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        save_dir = Path('debug') / f'yaw_frames_{mode_name}_{timestamp}'
    else:
        save_dir = Path(save_dir_arg)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f'[信息] 大偏航图像保存已开启: {save_dir}')
    return save_dir


def _draw_navigation_overlay(
    img_bgr: np.ndarray,
    polylines: List[List[Tuple[int, int]]],
    result: Dict[str, Any],
    boundary_scale_x: float = 1.0,
    boundary_scale_y: float = 1.0,
    pts_scale_x: float = 1.0,
    pts_scale_y: float = 1.0,
) -> np.ndarray:
    """在图像上绘制边界线、中心线、航行路径和偏航角指示，用于离线验证。

    Args:
        boundary_scale_x/y: polylines 坐标（proc_mask 空间）→ 原图的缩放系数
        pts_scale_x/y:      center_pts / path_pts（mask 空间）→ 原图的缩放系数
    """
    vis = img_bgr.copy()
    h, w = vis.shape[:2]

    def _to_img_boundary(pts):
        return [(int(round(x * boundary_scale_x)),
                 int(round(y * boundary_scale_y))) for x, y in pts]

    def _to_img_pts(pts):
        return [(int(round(x * pts_scale_x)),
                 int(round(y * pts_scale_y))) for x, y in pts]

    colors_boundary = [(0, 200, 255), (255, 200, 0)]  # 左=橙黄, 右=青蓝
    for i, pts in enumerate(polylines):
        if len(pts) < 2:
            continue
        scaled = _to_img_boundary(pts)
        color = colors_boundary[i % 2]
        arr = np.array(scaled, dtype=np.int32)
        cv2.polylines(vis, [arr], False, color, 2, cv2.LINE_AA)

    center_pts = result.get('center_pts', [])
    if len(center_pts) >= 2:
        scaled = _to_img_pts(center_pts)
        arr = np.array(scaled, dtype=np.int32)
        cv2.polylines(vis, [arr], False, (0, 255, 0), 2, cv2.LINE_AA)

    path_pts = result.get('path_pts', [])
    if len(path_pts) >= 2:
        scaled = _to_img_pts(path_pts)
        arr = np.array(scaled, dtype=np.int32)
        cv2.polylines(vis, [arr], False, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.circle(vis, scaled[0], 6, (0, 0, 255), -1)

    yaw_deg = float(result.get('yaw_deg', 0.0))
    status = result.get('status', '')
    boat_x, boat_y = w // 2, h - 1
    arrow_len = int(h * 0.25)
    angle_rad = math.radians(yaw_deg)
    tip_x = int(boat_x + arrow_len * math.sin(angle_rad))
    tip_y = int(boat_y - arrow_len * math.cos(angle_rad))
    cv2.arrowedLine(vis, (boat_x, boat_y), (tip_x, tip_y),
                    (255, 255, 255), 3, cv2.LINE_AA, tipLength=0.15)
    ref_tip_y = boat_y - arrow_len
    cv2.arrowedLine(vis, (boat_x, boat_y), (boat_x, ref_tip_y),
                    (128, 128, 128), 1, cv2.LINE_AA, tipLength=0.15)

    label = f"Yaw: {yaw_deg:+.2f} deg  Status: {status}"
    cv2.putText(vis, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    n_bnd = len(polylines)
    cv2.putText(vis, f"Boundaries: {n_bnd}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)

    return vis


def _maybe_save_yaw_frame(
    save_dir: Optional[Path],
    frame_index: int,
    img_bgr: np.ndarray,
    result: Dict[str, Any],
    threshold_deg: float,
    mode_name: str,
) -> None:
    if save_dir is None:
        return
    yaw_deg = float(result.get('yaw_deg', 0.0))
    if abs(yaw_deg) < threshold_deg:
        return

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    yaw_tag = f'{yaw_deg:+06.2f}'.replace('+', 'p').replace('-', 'm')
    status = str(result.get('status', 'unknown')).replace(' ', '_')
    file_name = f'{mode_name}_frame_{frame_index:06d}_yaw_{yaw_tag}_{status}_{timestamp}.jpg'
    cv2.imwrite(str(save_dir / file_name), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])


def _load_inference_backend(args):
    if bool(args.model) == bool(args.engine):
        sys.exit('❌ 请二选一指定 --model 或 --engine')

    if args.engine:
        try:
            from tensorrt_runtime import TensorRTRunner
        except Exception as exc:
            sys.exit(f'❌ TensorRT 运行时加载失败: {exc}')

        try:
            import torch
        except Exception as exc:
            sys.exit(f'❌ TensorRT 模式需要可用的 PyTorch CUDA 环境: {exc}')

        if not torch.cuda.is_available():
            sys.exit('❌ TensorRT 模式需要 CUDA，但当前环境未启用 CUDA')

        runner = TensorRTRunner(args.engine)
        print(f'[信息] 已加载 TensorRT 引擎: {args.engine}')
        return {
            'kind': 'tensorrt',
            'runner': runner,
            'device': 'cuda',
        }

    seg_model, device = load_segformer(
        args.model,
        num_classes=args.num_classes,
        model_name=args.model_name,
    )
    print(f'[信息] 已加载 PyTorch 权重: {args.model}')
    return {
        'kind': 'pytorch',
        'model': seg_model,
        'device': device,
    }


def _infer_mask_tensorrt(
    runner,
    img_bgr: np.ndarray,
    input_width: int,
    input_height: int,
) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_width, input_height))
    normalized = (resized.astype(np.float32) / 255.0 - mean) / std
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).contiguous()
    tensor = tensor.to(device='cuda', dtype=torch.float32)

    with torch.no_grad():
        outputs = runner.infer(tensor)
        logits = outputs[runner.get_primary_output_name()]
        logits = F.interpolate(logits, size=(input_height, input_width), mode='bilinear', align_corners=False)
        return logits.argmax(1).squeeze(0).to('cpu').numpy().astype(np.uint8)


def _infer_mask_pytorch(
    model,
    device,
    img_bgr: np.ndarray,
    input_width: int,
    input_height: int,
) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_width, input_height))
    normalized = (resized.astype(np.float32) / 255.0 - mean) / std
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = model(pixel_values=tensor).logits
        logits = F.interpolate(logits, size=(input_height, input_width), mode='bilinear', align_corners=False)
        return logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)


def _infer_mask_with_backend(
    backend: Dict[str, Any],
    args,
    img_bgr: np.ndarray,
    input_h: int,
) -> np.ndarray:
    if backend['kind'] == 'tensorrt':
        return _infer_mask_tensorrt(
            backend['runner'],
            img_bgr,
            input_width=args.input_size,
            input_height=input_h,
        )

    return _infer_mask_pytorch(
        backend['model'], backend['device'], img_bgr,
        input_width=args.input_size,
        input_height=input_h,
    )


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


# ─── 边界折线提取 ──────────────────────────────────────────────────────────────

def _polyline_side(pts: List[Tuple[int, int]], img_w: int) -> str:
    mean_x = sum(p[0] for p in pts) / len(pts)
    return 'left' if mean_x < img_w / 2 else 'right'


def _smooth_centerline_points(
    center_pts: List[Tuple[int, int]],
    window: int = CENTER_SMOOTH_WINDOW,
) -> List[Tuple[int, int]]:
    """用高斯加权平滑 x 坐标，比 box filter 有更好的平滑效果且不会产生鬼影。"""
    if len(center_pts) < 3:
        return center_pts

    xs = np.array([pt[0] for pt in center_pts], dtype=np.float32)
    ys = [pt[1] for pt in center_pts]

    k = min(window, len(center_pts))
    if k % 2 == 0:
        k -= 1
    if k <= 1:
        return center_pts

    pad = k // 2
    gaussian_kernel = cv2.getGaussianKernel(k, 0).flatten().astype(np.float32)
    xs_pad = np.pad(xs, (pad, pad), mode='edge')
    xs_smooth = np.convolve(xs_pad, gaussian_kernel, mode='valid')
    return [(int(round(float(x))), y) for x, y in zip(xs_smooth, ys)]


def _prepare_postprocess_mask(
    mask: np.ndarray,
    max_width: int = POSTPROC_MASK_WIDTH,
) -> Tuple[np.ndarray, float, float]:
    """将 mask 缩到较低分辨率用于后处理，并返回回映射缩放系数。"""
    h, w = mask.shape[:2]
    proc_w = min(int(max_width), int(w))
    if proc_w <= 0 or proc_w == w:
        return mask, 1.0, 1.0

    proc_h = max(1, int(round(h * proc_w / w)))
    proc_mask = cv2.resize(mask, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST)
    scale_x = float(w) / float(proc_w)
    scale_y = float(h) / float(proc_h)
    return proc_mask, scale_x, scale_y


def _scale_points_to_image(
    pts: List[Tuple[int, int]],
    scale_x: float,
    scale_y: float,
) -> List[Tuple[int, int]]:
    if scale_x == 1.0 and scale_y == 1.0:
        return pts
    return [
        (int(round(x * scale_x)), int(round(y * scale_y)))
        for x, y in pts
    ]


def extract_boundary_polylines(
    mask: np.ndarray,
) -> Tuple[List[List[Tuple[int, int]]], int]:
    """
    从分割 mask 中直接逐行提取左右边界折线。

    这里不再对每个连通域单独做逐行扫描，而是在低分辨率 mask 上
    直接按行聚合左半边 / 右半边的 boundary 像素，显著降低 CPU 开销。

    返回: (polylines, n_filtered)
    """
    _, w = mask.shape[:2]

    bnd_bin = (mask == BOUNDARY_CLASS).astype(np.uint8)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bnd_bin = cv2.morphologyEx(bnd_bin, cv2.MORPH_CLOSE, kernel)

    ys_all, xs_all = np.nonzero(bnd_bin)
    if ys_all.size == 0:
        return [], 0

    center_x = w // 2

    def _build_side_polyline(side_mask: np.ndarray) -> List[Tuple[int, int]]:
        if not np.any(side_mask):
            return []

        side_rows = ys_all[side_mask].astype(np.int32, copy=False)
        side_cols = xs_all[side_mask].astype(np.float32, copy=False)
        counts = np.bincount(side_rows, minlength=mask.shape[0]).astype(np.float32, copy=False)
        sums = np.bincount(side_rows, weights=side_cols, minlength=mask.shape[0])
        valid_rows = np.flatnonzero(counts > 0)
        if valid_rows.size < 2:
            return []

        row_dict = {
            int(y): float(sums[y] / counts[y])
            for y in valid_rows.tolist()
        }
        row_dict = smooth_row_dict(row_dict, window=BOUNDARY_SMOOTH_WINDOW)
        return [(int(round(x)), y) for y, x in sorted(row_dict.items())]

    polylines: List[List[Tuple[int, int]]] = []

    left_pts = _build_side_polyline(xs_all < center_x)
    if left_pts:
        polylines.append(left_pts)

    right_pts = _build_side_polyline(xs_all >= center_x)
    if right_pts:
        polylines.append(right_pts)

    return polylines, 0


# ─── 构建航行路径：船头 → 平滑过渡 → 中线 ──────────────────────────────────────

def _build_sailing_path(
    center_pts: List[Tuple[int, int]],
    img_w: int,
    img_h: int,
    n_interp: int = 30,
) -> List[Tuple[int, int]]:
    """
    从船头位置（图像底部中心）到中线起点做线性过渡，
    拼接成完整航行路径。
    center_pts: 中线点列表，按 y 降序（[0]=最近端, y 最大）。
    """
    boat_x, boat_y = img_w // 2, img_h - 1
    cl_x, cl_y = center_pts[0]

    if abs(cl_y - boat_y) < 5 and abs(cl_x - boat_x) < 5:
        return [(boat_x, boat_y)] + center_pts

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
    stage_timers: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    处理单帧：
      1. 提取左右警戒线
      2. 计算中心线
      3. 构建船头→中线的平滑航行轨迹
      4. 计算偏航角
    返回: (原始图像, 控制信号字典)
    """
    h, w = img_bgr.shape[:2]

    def _stage(name: str):
        if stage_timers is None:
            return nullcontext()
        timer = stage_timers.get(name)
        return timer if timer is not None else nullcontext()

    with _stage('prepare_mask'):
        proc_mask, scale_x, scale_y = _prepare_postprocess_mask(mask)
        proc_h, proc_w = proc_mask.shape[:2]

    empty_result = {
        'yaw_deg': 0.0,
        'status': 'no centerline',
        'n_filtered': 0,
        'center_pts': [],
        'path_pts': [],
    }

    # ── 1. 提取左右警戒线 ──
    with _stage('extract_boundary'):
        polylines, n_filtered = extract_boundary_polylines(proc_mask)

    if len(polylines) != 2:
        empty_result['n_filtered'] = n_filtered
        empty_result['status'] = f'boundary count={len(polylines)}, need 2'
        return img_bgr, empty_result

    # ── 2. 区分左右 ──
    avg_x = [sum(p[0] for p in pts) / len(pts) for pts in polylines]
    if avg_x[0] <= avg_x[1]:
        left_pts, right_pts = polylines[0], polylines[1]
    else:
        left_pts, right_pts = polylines[1], polylines[0]

    # ── 3. 构建行字典 ──
    with _stage('dense_rows'):
        sky_cut = int(proc_h * roi_top_ratio)
        left_dict  = {y: float(x) for x, y in left_pts  if y >= sky_cut}
        right_dict = {y: float(x) for x, y in right_pts if y >= sky_cut}

        # ── 4. 插值 + 外推 → 稠密行覆盖 ──
        all_ys = sorted(set(left_dict.keys()) | set(right_dict.keys()))
    if len(all_ys) < 2:
        empty_result['n_filtered'] = n_filtered
        empty_result['status'] = 'too few boundary rows'
        return img_bgr, empty_result

    with _stage('dense_rows'):
        y_min_u, y_max_u = all_ys[0], all_ys[-1]
        left_dense  = _extrapolate_dict(left_dict,  y_min_u, y_max_u)
        right_dense = _extrapolate_dict(right_dict, y_min_u, y_max_u)
        overlap_ys  = sorted(set(left_dense.keys()) & set(right_dense.keys()))

    if len(overlap_ys) < 3:
        empty_result['n_filtered'] = n_filtered
        empty_result['status'] = 'overlap rows < 3'
        return img_bgr, empty_result

    # ── 5. 异常宽度过滤 ──
    with _stage('width_filter'):
        width_dict = {y: right_dense[y] - left_dense[y] for y in overlap_ys}
        near_rows = [y for y in overlap_ys
                     if y >= max(overlap_ys) - (max(overlap_ys) - min(overlap_ys)) * 0.3]
        if not near_rows:
            near_rows = overlap_ys
        mean_width = float(np.mean([width_dict[y] for y in near_rows]))

    if mean_width < 10:
        empty_result['n_filtered'] = n_filtered
        empty_result['status'] = 'channel too narrow'
        return img_bgr, empty_result

    with _stage('width_filter'):
        valid_ys = [
            y for y in overlap_ys
            if abs(width_dict[y] - mean_width) / mean_width < WIDTH_OUTLIER_TH
        ]

    if len(valid_ys) < 3:
        empty_result['n_filtered'] = n_filtered
        empty_result['status'] = 'too few valid rows after width filter'
        return img_bgr, empty_result

    # ── 6. 逐行取中点 → 中线（按 y 降序 = 从近到远）──
    with _stage('center_path'):
        center_pts = [
            (int(round((left_dense[y] + right_dense[y]) / 2.0)), y)
            for y in sorted(valid_ys, reverse=True)
        ]

        # ── 6.5. 对 center_pts 的间隙用线性插值填充，使其成为逐行连续 ──
        if len(center_pts) >= 2:
            ys_asc = np.array([pt[1] for pt in reversed(center_pts)], dtype=np.int32)
            xs_asc = np.array([pt[0] for pt in reversed(center_pts)], dtype=np.float32)
            all_ys_dense = np.arange(ys_asc[0], ys_asc[-1] + 1, dtype=np.int32)
            xs_interp = np.interp(all_ys_dense, ys_asc, xs_asc)
            center_pts = [(int(round(float(x))), int(y))
                          for x, y in zip(xs_interp[::-1], all_ys_dense[::-1])]

        # ── 7. 轻量平滑中线 ──
        center_pts = _smooth_centerline_points(center_pts)

        # ── 8. 构建航行路径：船头 → 平滑过渡 → 中线 ──
        sailing_path = _build_sailing_path(center_pts, proc_w, proc_h)

        # ── 9. 回映射到原图坐标系，再计算偏航角 ──
        center_pts = _scale_points_to_image(center_pts, scale_x, scale_y)
        sailing_path = _scale_points_to_image(sailing_path, scale_x, scale_y)

    # ── 10. 计算偏航角 ──
    with _stage('yaw'):
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

    result = {
        'yaw_deg': round(yaw_deg, 2),
        'status': 'ok',
        'n_filtered': n_filtered,
        'center_pts': center_pts,
        'path_pts': sailing_path,
    }
    return img_bgr, result


# ─── PX4 Offboard 控制器（20Hz 定频发布 setpoint）──────────────────────────────

class _PX4OffboardController:
    """
    通过 px4_msgs 直接与 PX4 通信的 Offboard 控制器。
    以 20Hz 定频发布 OffboardControlMode + TrajectorySetpoint，
    推理帧率低时在两帧之间重复上一次指令，保证 Offboard 不掉线。
    发送 20 帧 setpoint 后自动切 Offboard 模式并解锁。
    """

    CONTROL_HZ = 20
    DEFAULT_FORWARD_SPEED = 0.3
    DEFAULT_MAX_ACCEL = 0.15
    DEFAULT_MAX_YAW_RATE_DEG = 25.0
    DEFAULT_YAW_GAIN = 0.35
    DEFAULT_MAX_YAW_OFFSET_DEG = 12.0

    def __init__(
        self,
        node,
        stop_when_not_ok: bool = False,
        forward_speed: float = DEFAULT_FORWARD_SPEED,
        max_accel: float = DEFAULT_MAX_ACCEL,
        max_yaw_rate_deg: float = DEFAULT_MAX_YAW_RATE_DEG,
        yaw_gain: float = DEFAULT_YAW_GAIN,
        max_yaw_offset_deg: float = DEFAULT_MAX_YAW_OFFSET_DEG,
    ):
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
        from px4_msgs.msg import (
            OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleAttitude,
        )

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._node = node
        self._lock = threading.Lock()
        self._yaw_deg = 0.0
        self._current_yaw = 0.0
        self._status_ok = True
        self._stop_when_not_ok = stop_when_not_ok
        self._target_forward_speed = max(0.0, float(forward_speed))
        self._max_accel = max(0.0, float(max_accel))
        self._max_yaw_rate_deg = max(0.0, float(max_yaw_rate_deg))
        self._yaw_gain = max(0.0, float(yaw_gain))
        self._max_yaw_offset_deg = max(0.0, float(max_yaw_offset_deg))
        self._command_speed = 0.0
        self._target_absolute_yaw = None
        self._command_absolute_yaw = None
        self._last_control_state = {
            'status_ok': True,
            'target_speed': 0.0,
            'command_speed': 0.0,
            'raw_yaw_deg': 0.0,
            'target_yaw_deg': 0.0,
            'command_yaw_deg': 0.0,
            'absolute_yaw_deg': 0.0,
            'velocity_x': 0.0,
            'velocity_y': 0.0,
        }

        self._offboard_setpoint_counter = 0
        self._armed = False
        self._offboard_mode = False

        self._offboard_mode_pub = node.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos)
        self._trajectory_pub = node.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos)
        self._vehicle_cmd_pub = node.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos)

        self._attitude_sub = node.create_subscription(
            VehicleAttitude, '/fmu/out/vehicle_attitude', self._attitude_cb, qos)

        self._timer = node.create_timer(1.0 / self.CONTROL_HZ, self._timer_cb)

    # ── 姿态回调：从四元数提取当前偏航角 ──

    def _attitude_cb(self, msg):
        w, x, y, z = msg.q
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        with self._lock:
            self._current_yaw = math.atan2(siny_cosp, cosy_cosp)

    # ── 20Hz 定时回调 ──

    def _timer_cb(self):
        self._publish_offboard_control_mode()
        self._publish_trajectory_setpoint()

        if self._offboard_setpoint_counter < 20:
            self._offboard_setpoint_counter += 1

        if self._offboard_setpoint_counter == 20:
            if not self._offboard_mode:
                self._set_offboard_mode()
                self._offboard_mode = True
            if not self._armed:
                self._arm()
                self._armed = True

    # ── 发布 OffboardControlMode ──

    def _publish_offboard_control_mode(self):
        from px4_msgs.msg import OffboardControlMode
        msg = OffboardControlMode()
        msg.timestamp = int(self._node.get_clock().now().nanoseconds / 1000)
        msg.position = False
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.thrust_and_torque = False
        msg.direct_actuator = False
        self._offboard_mode_pub.publish(msg)

    # ── 发布 TrajectorySetpoint（速度 + 偏航角）──

    def _publish_trajectory_setpoint(self):
        from px4_msgs.msg import TrajectorySetpoint
        msg = TrajectorySetpoint()
        msg.timestamp = int(self._node.get_clock().now().nanoseconds / 1000)

        with self._lock:
            current_yaw = self._current_yaw
            yaw_deg = self._yaw_deg
            status_ok = self._status_ok
            target_abs_yaw = self._target_absolute_yaw

        if self._command_absolute_yaw is None:
            self._command_absolute_yaw = current_yaw

        if not status_ok and self._stop_when_not_ok:
            target_speed = 0.0
            target_abs_yaw = current_yaw
        elif not status_ok:
            target_speed = self._target_forward_speed
            target_abs_yaw = current_yaw
        else:
            target_speed = self._target_forward_speed

        if target_abs_yaw is None:
            target_abs_yaw = current_yaw

        speed = self._slew_value(
            current_value=self._command_speed,
            target_value=target_speed,
            max_delta=self._max_accel / self.CONTROL_HZ,
        )
        self._command_speed = speed

        max_yaw_step = math.radians(self._max_yaw_rate_deg) / self.CONTROL_HZ
        absolute_yaw = self._slew_angle(
            self._command_absolute_yaw, target_abs_yaw, max_yaw_step,
        )
        self._command_absolute_yaw = absolute_yaw

        target_offset_deg = math.degrees(math.atan2(
            math.sin(target_abs_yaw - current_yaw),
            math.cos(target_abs_yaw - current_yaw),
        ))
        command_offset_deg = math.degrees(math.atan2(
            math.sin(absolute_yaw - current_yaw),
            math.cos(absolute_yaw - current_yaw),
        ))

        velocity_x = speed * math.cos(absolute_yaw)
        velocity_y = speed * math.sin(absolute_yaw)

        msg.velocity[0] = velocity_x
        msg.velocity[1] = velocity_y
        msg.velocity[2] = 0.0
        msg.position[0] = float('nan')
        msg.position[1] = float('nan')
        msg.position[2] = float('nan')
        msg.acceleration[0] = float('nan')
        msg.acceleration[1] = float('nan')
        msg.acceleration[2] = float('nan')
        msg.yaw = absolute_yaw
        msg.yawspeed = float('nan')

        with self._lock:
            self._last_control_state = {
                'status_ok': status_ok,
                'target_speed': target_speed,
                'command_speed': speed,
                'raw_yaw_deg': yaw_deg,
                'target_yaw_deg': target_offset_deg,
                'command_yaw_deg': command_offset_deg,
                'absolute_yaw_deg': math.degrees(absolute_yaw),
                'velocity_x': velocity_x,
                'velocity_y': velocity_y,
            }

        self._trajectory_pub.publish(msg)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def _slew_value(current_value: float, target_value: float, max_delta: float) -> float:
        if max_delta <= 0.0:
            return target_value
        delta = target_value - current_value
        if delta > max_delta:
            return current_value + max_delta
        if delta < -max_delta:
            return current_value - max_delta
        return target_value

    @staticmethod
    def _slew_angle(current: float, target: float, max_delta: float) -> float:
        """Rate-limit an angle value, correctly handling ±pi wrapping."""
        if max_delta <= 0.0:
            return target
        diff = math.atan2(math.sin(target - current), math.cos(target - current))
        if abs(diff) <= max_delta:
            return target
        return _PX4OffboardController._normalize_angle(
            current + math.copysign(max_delta, diff)
        )

    # ── VehicleCommand 发布 ──

    def _publish_vehicle_command(self, command, **kwargs):
        from px4_msgs.msg import VehicleCommand
        msg = VehicleCommand()
        msg.timestamp = int(self._node.get_clock().now().nanoseconds / 1000)
        msg.command = command
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        for field, value in kwargs.items():
            setattr(msg, field, value)
        self._vehicle_cmd_pub.publish(msg)

    def _arm(self):
        from px4_msgs.msg import VehicleCommand
        self._publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self._node.get_logger().info('发送解锁指令')

    def _set_offboard_mode(self):
        from px4_msgs.msg import VehicleCommand
        self._publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0, param2=6.0)
        self._node.get_logger().info('切换到 Offboard 模式')

    # ── 外部接口：推理线程更新偏航角 ──

    def update(self, yaw_deg: float, status_ok: bool) -> None:
        with self._lock:
            self._yaw_deg = yaw_deg
            self._status_ok = status_ok
            offset_deg = yaw_deg * self._yaw_gain
            offset_deg = max(-self._max_yaw_offset_deg,
                             min(self._max_yaw_offset_deg, offset_deg))
            self._target_absolute_yaw = self._normalize_angle(
                self._current_yaw + math.radians(offset_deg)
            )

    def get_debug_state(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._last_control_state)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="警戒线导航：识别警戒线 → 计算中心线 → 航行轨迹 → 偏航角",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--model',       default=None,   help='SegFormer .pth 权重路径（PyTorch 模式）')
    p.add_argument('--engine',      default=None,   help='TensorRT engine 路径（TensorRT 模式）')

    # 模型推理模式
    p.add_argument('--images',      default=None,   help='输入图像文件或目录（模型推理模式）')
    p.add_argument('--output',      default=None,   help='可视化结果保存目录（模型推理模式）')

    # 实时相机模式
    p.add_argument('--camera',      default=None,
                   help='相机设备编号或路径（实时模式），如 0 或 /dev/video0')

    # 离线视频测试模式
    p.add_argument('--video',       default=None,
                   help='离线视频文件路径（视频测试模式），如 video528.mp4')

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
    p.add_argument('--yaw-log', type=str, default=None,
                   help='记录偏航角日志到 CSV；传 auto 自动保存到 logs/，也可直接给文件路径')
    p.add_argument('--save-large-yaw-frames', type=str, default=None,
                   help='保存大偏航原始图像；传 auto 自动保存到 debug/，也可直接给目录路径')
    p.add_argument('--large-yaw-threshold-deg', type=float, default=8.0,
                   help='触发保存原始图像的偏航角阈值 deg，默认 8')
    p.add_argument('--save-video', type=str, default=None,
                   help='将带标注的可视化视频保存到指定路径（仅离线视频模式），'
                        '如 output_vis.mp4；传 auto 则自动命名')

    # PX4 Offboard 控制（仅实时相机模式）
    p.add_argument('--px4', action='store_true',
                   help='启用 PX4 Offboard 控制（通过 px4_msgs 直接通信，0.3 m/s 前进，20Hz 发布）')
    p.add_argument('--px4-stop-when-not-ok', action='store_true',
                   help='视觉 status≠ok 时发布零速度；默认关闭（非 ok 时仍前进，仅偏航角置 0）')
    p.add_argument('--px4-forward-speed', type=float,
                   default=_PX4OffboardController.DEFAULT_FORWARD_SPEED,
                   help='PX4 前进目标速度 m/s，默认 0.3')
    p.add_argument('--px4-max-accel', type=float,
                   default=_PX4OffboardController.DEFAULT_MAX_ACCEL,
                   help='PX4 速度变化上限 m/s^2，默认 0.15')
    p.add_argument('--px4-max-yaw-rate-deg', type=float,
                   default=_PX4OffboardController.DEFAULT_MAX_YAW_RATE_DEG,
                   help='PX4 偏航角指令变化上限 deg/s，默认 25')
    p.add_argument('--px4-yaw-gain', type=float,
                   default=_PX4OffboardController.DEFAULT_YAW_GAIN,
                   help='视觉偏航角缩放系数，默认 0.35；越小转向越柔和')
    p.add_argument('--px4-max-yaw-offset-deg', type=float,
                   default=_PX4OffboardController.DEFAULT_MAX_YAW_OFFSET_DEG,
                   help='PX4 允许的最大偏航修正角 deg，默认 12')
    return p.parse_args()


# ─── 模型推理模式 ──────────────────────────────────────────────────────────────

def run_image_mode(args):
    """对图片/目录进行推理，输出可视化图片 + 偏航角 CSV。"""
    if args.output is None and not args.show:
        print("⚠️  未指定 --output 也未开启 --show，结果不保存不显示。")

    img_paths = collect_images(args.images)
    print(f"[信息] 共 {len(img_paths)} 张图像")

    backend = _load_inference_backend(args)

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

    yaw_log_file, yaw_log_writer, _ = _create_yaw_log_writer(args.yaw_log, 'images')
    yaw_capture_dir = _resolve_capture_dir(args.save_large_yaw_frames, 'images')

    total = saved = ok = 0
    input_h = args.input_size * 384 // 640

    try:
        for img_path in img_paths:
            total += 1
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"[警告] 无法读取: {img_path}")
                continue

            mask = _infer_mask_with_backend(backend, args, img_bgr, input_h)

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

            _write_yaw_log_row(yaw_log_writer, total, result)
            _maybe_save_yaw_frame(
                yaw_capture_dir,
                total,
                img_bgr,
                result,
                args.large_yaw_threshold_deg,
                'images',
            )

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
        if yaw_log_file:
            yaw_log_file.close()
        if args.show:
            cv2.destroyAllWindows()

    print(f"\n{'='*60}")
    print(f"  总计 {total} 张 | 已保存 {saved} 张 | 有效导航帧 {ok} 张")
    if out_dir:
        print(f"  偏航角已导出: {out_dir / 'yaw_angle.csv'}")
    print(f"  输出格式: yaw_deg = 偏航角（度），+右偏 -左偏，0=正前方")
    print(f"{'='*60}")


# ─── 离线视频测试模式 ──────────────────────────────────────────────────────────

def run_video_mode(args):
    """读取离线视频文件，逐帧推理，在终端输出警戒线数量和偏航角。"""
    video_path = args.video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"❌ 无法打开视频文件: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"{'='*60}")
    print(f"  离线视频测试模式")
    print(f"  视频文件: {video_path}")
    print(f"  分辨率: {width}x{height}  帧率: {fps:.1f}  总帧数: {total_frames}")
    print(f"{'='*60}")

    backend = _load_inference_backend(args)
    yaw_filter = YawFilter(alpha=args.ema_alpha)
    input_h = args.input_size * 384 // 640

    frame_count = 0
    ok_count = 0
    yaw_log_file, yaw_log_writer, _ = _create_yaw_log_writer(args.yaw_log, 'video')
    yaw_capture_dir = _resolve_capture_dir(args.save_large_yaw_frames, 'video')

    video_writer = None
    save_video_path = None
    if args.save_video:
        if args.save_video.lower() == 'auto':
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            src_name = Path(video_path).stem
            save_video_path = f'debug/{src_name}_vis_{timestamp}.mp4'
        else:
            save_video_path = args.save_video
        Path(save_video_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(save_video_path, fourcc, fps, (width, height))
        print(f"[信息] 标注视频将保存至: {save_video_path}")

    need_vis = args.show or video_writer is not None

    mask_w = args.input_size
    mask_h = input_h
    img_to_mask_x = float(width) / float(mask_w)
    img_to_mask_y = float(height) / float(mask_h)

    print(f"\n{'帧号':>6s}  {'警戒线数':>8s}  {'偏航角':>10s}  {'状态'}")
    print(f"{'-'*6}  {'-'*8}  {'-'*10}  {'-'*30}")

    try:
        while True:
            ret, img_bgr = cap.read()
            if not ret:
                break

            frame_count += 1

            mask = _infer_mask_with_backend(backend, args, img_bgr, input_h)

            proc_mask, sx, sy = _prepare_postprocess_mask(mask)
            polylines, _ = extract_boundary_polylines(proc_mask)
            n_boundary_lines = len(polylines)

            _, result = process_frame(
                img_bgr, mask,
                yaw_filter=yaw_filter,
                roi_top_ratio=args.roi_top,
            )

            yaw_deg = result['yaw_deg']
            status = result['status']

            if status == 'ok':
                ok_count += 1

            _write_yaw_log_row(yaw_log_writer, frame_count, result)
            _maybe_save_yaw_frame(
                yaw_capture_dir,
                frame_count,
                img_bgr,
                result,
                args.large_yaw_threshold_deg,
                'video',
            )

            print(f"  {frame_count:5d}  {n_boundary_lines:8d}  {yaw_deg:+9.2f}°  {status}")

            if need_vis:
                vis = _draw_navigation_overlay(
                    img_bgr, polylines, result,
                    boundary_scale_x=sx * img_to_mask_x,
                    boundary_scale_y=sy * img_to_mask_y,
                    pts_scale_x=img_to_mask_x,
                    pts_scale_y=img_to_mask_y,
                )
                frame_label = f"Frame: {frame_count}/{total_frames}"
                cv2.putText(vis, frame_label, (10, vis.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

                if video_writer is not None:
                    video_writer.write(vis)

                if args.show:
                    cv2.imshow("离线视频测试 (Q/ESC 退出)", vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), ord('Q'), 27):
                        print("\n[信息] 用户退出")
                        break

    except KeyboardInterrupt:
        print("\n[信息] Ctrl+C 退出")
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if yaw_log_file:
            yaw_log_file.close()
        if args.show:
            cv2.destroyAllWindows()

    print(f"\n{'='*60}")
    print(f"  总帧数: {frame_count}")
    print(f"  有效导航帧(status=ok): {ok_count}")
    print(f"  有效率: {ok_count/max(frame_count,1)*100:.1f}%")
    if save_video_path:
        print(f"  标注视频已保存: {save_video_path}")
    print(f"{'='*60}")


# ─── 实时相机模式 ──────────────────────────────────────────────────────────────

def run_camera_mode(args):
    """从摄像头读帧，实时输出偏航角；可选 PX4 Offboard 20Hz 定频发布 setpoint。"""
    _rclpy = None

    try:
        cam_id = int(args.camera)
    except ValueError:
        cam_id = args.camera

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        sys.exit(f"❌ 无法打开相机: {args.camera}")

    print(f"[信息] 相机已打开: {args.camera}")
    print(f"[信息] 按 Q/ESC 退出")

    print("[信息] 加载分割模型…")
    backend = _load_inference_backend(args)

    ros_node = None
    executor = None
    spin_thread = None
    px4_ctrl = None
    rclpy_initialized = False

    if args.px4:
        import rclpy
        from rclpy.executors import MultiThreadedExecutor

        _rclpy = rclpy
        rclpy.init()
        rclpy_initialized = True
        ros_node = rclpy.create_node('realtime_pilot_px4')

        px4_ctrl = _PX4OffboardController(
            ros_node,
            stop_when_not_ok=args.px4_stop_when_not_ok,
            forward_speed=args.px4_forward_speed,
            max_accel=args.px4_max_accel,
            max_yaw_rate_deg=args.px4_max_yaw_rate_deg,
            yaw_gain=args.px4_yaw_gain,
            max_yaw_offset_deg=args.px4_max_yaw_offset_deg,
        )

        executor = MultiThreadedExecutor()
        executor.add_node(ros_node)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()
        print(
            f"[PX4] Offboard 控制已启动  "
            f"速度={args.px4_forward_speed} m/s  "
            f"最大加速度={args.px4_max_accel} m/s^2  "
            f"偏航增益={args.px4_yaw_gain}  "
            f"最大偏航修正={args.px4_max_yaw_offset_deg} deg  "
            f"最大偏航变化率={args.px4_max_yaw_rate_deg} deg/s  "
            f"发布频率={_PX4OffboardController.CONTROL_HZ} Hz"
        )
        print("[PX4] 发送 setpoint 中，1 秒后自动切 Offboard 并解锁…")

    yaw_filter = YawFilter(alpha=args.ema_alpha)
    input_h = args.input_size * 384 // 640
    yaw_log_file, yaw_log_writer, _ = _create_yaw_log_writer(args.yaw_log, 'camera')
    yaw_capture_dir = _resolve_capture_dir(args.save_large_yaw_frames, 'camera')

    frame_count = 0
    try:
        while True:
            ret, img_bgr = cap.read()
            if not ret:
                print("[警告] 读取帧失败，重试...")
                time.sleep(0.1)
                continue

            frame_count += 1

            mask = _infer_mask_with_backend(backend, args, img_bgr, input_h)

            vis, result = process_frame(
                img_bgr, mask,
                yaw_filter=yaw_filter,
                roi_top_ratio=args.roi_top,
            )

            yaw_deg = result['yaw_deg']

            if px4_ctrl is not None:
                px4_ctrl.update(yaw_deg, result['status'] == 'ok')

            control_state = px4_ctrl.get_debug_state() if px4_ctrl is not None else None
            _write_yaw_log_row(yaw_log_writer, frame_count, result, control_state=control_state)
            _maybe_save_yaw_frame(
                yaw_capture_dir,
                frame_count,
                img_bgr,
                result,
                args.large_yaw_threshold_deg,
                'camera',
            )

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
        if yaw_log_file:
            yaw_log_file.close()
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

    mode_count = sum([
        bool(args.images),
        bool(args.camera),
        bool(args.video),
    ])
    if mode_count > 1:
        sys.exit("❌ --images / --camera / --video 只能指定一种模式")
    if bool(args.model) == bool(args.engine):
        sys.exit("❌ 请二选一指定 --model 或 --engine")
    if args.px4 and args.camera is None:
        sys.exit("❌ --px4 仅用于实时相机模式，请指定 --camera")
    if args.px4 and (args.images or args.video):
        sys.exit("❌ --px4 不能与 --images / --video 同时使用")

    if args.images:
        run_image_mode(args)
    elif args.video:
        run_video_mode(args)
    elif args.camera is not None:
        run_camera_mode(args)
    else:
        sys.exit("❌ 请指定 --images（模型推理）/ --camera（实时相机）/ --video（离线视频测试）")


if __name__ == '__main__':
    main()
