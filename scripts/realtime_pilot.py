#!/usr/bin/env python3
"""
端到端实时自主导航 —— 从摄像头图像到控制信号输出。

【完整数据流】
  摄像头/图片 → SegFormer分割 → 三级回退路径规划 → Pure Pursuit跟踪
  → PID控制 → 控制信号输出（steering_deg, throttle_pct, speed_mps）

【用法】
  # 1. 图片处理测试
  python scripts/realtime_pilot.py \
      --images dataset_final/images/val \
      --masks  dataset_final/masks/val \
      --output vis_pilot

进入项目目录 /home/yanmin/boat5
执行：
source .venv/bin/activate
运行：
python realtime_pilot.py --images dataset_final/images/val --masks dataset_final/masks/val --output vis_pilot

  # 2. 模型推理
  python scripts/realtime_pilot.py \
      --images dataset_final/images/val \
      --model models/segformer_river/best_model.pth

python scripts/realtime_pilot.py --images dataset_final/images/test --model models/segformer_river/best_model.pth --output vis_pilot_model


自定义停车距离（如 1.5 米）：
python scripts/realtime_pilot.py --images dataset_final/images/val --masks dataset_final/masks/val --shore-stop-dist 1.5



  # 3. 实时相机处理
  python scripts/realtime_pilot.py \
      --camera 0 \
      --model models/segformer_river/best_model.pth \
      --show
"""

import argparse
import sys
import os
import time
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plan_path import (
    plan_path_from_bottom,
    render_planned_frame,
    compute_water_centerline,
    compute_single_boundary_centerline,
    _find_single_boundary,
    _clip_to_water,
)
from visualize_centerline import (
    BOUNDARY_CLASS, WATER_CLASS,
    load_segformer, infer_mask,
    collect_images, find_mask_path,
)
from scripts.realtime_pilot_v2 import extract_boundary_polylines

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

# ─── 航道参数 ──────────────────────────────────────────────────────────────────
CHANNEL_WIDTH_M = 2.5    # 已知航道宽度（米）
ROI_TOP_RATIO   = 0.5    # 中线计算只用图像下方这个比例的行


# ─────────────────────────────────────────────────────────────────────────────
# EMA 时序滤波器
# ─────────────────────────────────────────────────────────────────────────────

class CTEFilter:
    """指数滑动平均（EMA）滤波器，用于平滑舵角帧间抖动。"""

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


# ─────────────────────────────────────────────────────────────────────────────
# 轻量级 Pure Pursuit（像素空间，不依赖 ROS / config_loader）
# ─────────────────────────────────────────────────────────────────────────────

class SimplePurePursuit:
    """
    像素空间 Pure Pursuit 控制器。
    直接在像素坐标系下计算转向角，避免依赖不可靠的像素/米标定。

    核心思路：
      - 船在图像底部中心，朝上（y减小方向）行驶
      - 沿路径找到 lookahead_ratio * 图像高度 处的目标点
      - 目标点相对底部中心的横向偏移归一化后映射为舵角
    """

    def __init__(
        self,
        img_w: int = 640,
        img_h: int = 480,
        lookahead_ratio: float = 0.20,
        target_speed: float = 1.5,
        max_speed: float = 3.0,
        max_steering_deg: float = 30.0,
    ):
        self.img_w = img_w
        self.img_h = img_h
        self.lookahead_ratio = lookahead_ratio
        self.target_speed = target_speed
        self.max_speed = max_speed
        self.max_steer = math.radians(max_steering_deg)

    def compute(
        self, path_px: List[Tuple[int, int]]
    ) -> Dict[str, Any]:
        if len(path_px) < 3:
            return self._stop("路径点不足")

        w, h = self.img_w, self.img_h
        cx = w / 2.0
        start_y = h - 1

        lookahead_px = h * self.lookahead_ratio

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

        lateral_norm = dx / (w / 2.0)
        curvature = 2.0 * dx / (ld ** 2) if ld > 0 else 0.0

        steering = math.atan2(dx, max(dy, 1.0))
        steering = max(-self.max_steer, min(self.max_steer, steering))

        steer_ratio = abs(steering) / self.max_steer
        speed_factor = 1.0 - 0.4 * steer_ratio
        speed = self.target_speed * speed_factor
        throttle_pct = (speed / self.max_speed) * 100.0
        throttle_pct = max(0.0, min(100.0, throttle_pct))

        return {
            'steering_rad': steering,
            'steering_deg': math.degrees(steering),
            'throttle_pct': throttle_pct,
            'target_px': (target_x, target_y),
            'lateral_norm': lateral_norm,
            'lookahead_px': ld,
            'curvature': curvature,
            'speed_mps': speed,
            'status': 'ok',
        }

    def _stop(self, reason: str) -> Dict[str, Any]:
        return {
            'steering_rad': 0.0,
            'steering_deg': 0.0,
            'throttle_pct': 0.0,
            'target_px': (0, 0),
            'lateral_norm': 0.0,
            'lookahead_px': 0.0,
            'curvature': 0.0,
            'speed_mps': 0.0,
            'status': f'stop: {reason}',
        }


# ─────────────────────────────────────────────────────────────────────────────
# IMX390-GMSL2 相机地面投影（像素→世界坐标）
# ─────────────────────────────────────────────────────────────────────────────

class IMX390Projector:
    """
    基于 IMX390-GMSL2 传感器参数的针孔相机地面投影器。

    将图像像素坐标通过透视投影映射到水面平面，得到真实世界坐标（米）。

    IMX390 传感器参数（固定）：
      - 有效像素: 1936 × 1100
      - 输出分辨率: 1920 × 1080
      - 像素尺寸: 3.0 µm × 3.0 µm（方形像素）
      - 传感器有效面积: 5.81mm × 3.30mm
      - 光学格式: 1/2.7"

    可选镜头（HFOV）:
      - H60:  63.9°  (F1.6)
      - H120: 120.6° (F1.4)
      - H190: 186°   (F2.4, 鱼眼, 本模型不适用)

    世界坐标系（船体坐标系）：
      原点 = 相机正下方水面点
      X = 右(+)  Y = 前(+)  Z = 上(+)
    """

    SENSOR_W_MM = 5.81
    SENSOR_H_MM = 3.30
    NATIVE_W = 1920
    NATIVE_H = 1080

    def __init__(
        self,
        camera_height: float = 0.5,
        camera_pitch_deg: float = 10.0,
        hfov_deg: float = 120.0,
    ):
        self.camera_height = camera_height
        self.pitch = math.radians(abs(camera_pitch_deg))
        self.hfov_deg = hfov_deg

        self.native_fx = self.NATIVE_W / (2.0 * math.tan(math.radians(hfov_deg / 2.0)))
        self.native_fy = self.native_fx

        self._img_w = self.NATIVE_W
        self._img_h = self.NATIVE_H
        self._update_intrinsics()

    def _update_intrinsics(self):
        sx = self._img_w / self.NATIVE_W
        sy = self._img_h / self.NATIVE_H
        self.fx = self.native_fx * sx
        self.fy = self.native_fy * sy
        self.cx = self._img_w / 2.0
        self.cy = self._img_h / 2.0
        self.sin_p = math.sin(self.pitch)
        self.cos_p = math.cos(self.pitch)

    def update_resolution(self, img_w: int, img_h: int):
        if img_w != self._img_w or img_h != self._img_h:
            self._img_w = img_w
            self._img_h = img_h
            self._update_intrinsics()

    def pixel_to_ground(self, u: float, v: float) -> Optional[Tuple[float, float]]:
        """
        像素 (u, v) → 水面坐标 (x_m, y_m)。

        相机坐标系: X 右, Y 下, Z 前（光轴）
        世界坐标系: X 右, Y 前（水平）, Z 上

        相机俯仰角 θ > 0 表示光轴向下倾斜 θ 度。
        旋转后世界方向:
            d_X_w = dx
            d_Y_w = cos(θ) - dy·sin(θ)   (前向)
            d_Z_w = -(sin(θ) + dy·cos(θ)) (竖直, 负=向下)

        地面交点 (Z=0): t = H / (sin(θ) + dy·cos(θ))
        """
        dx = (u - self.cx) / self.fx
        dy = (v - self.cy) / self.fy

        denom = self.sin_p + dy * self.cos_p
        if denom < 1e-6:
            return None

        t = self.camera_height / denom
        x_m = t * (self.cos_p - dy * self.sin_p)    # X = 正前方
        y_m = t * dx                                  # Y = 横向（+右 -左）
        return (x_m, y_m)

    def info(self) -> str:
        vfov = 2.0 * math.degrees(math.atan(self.NATIVE_H / (2.0 * self.native_fy)))
        return (
            f"IMX390-GMSL2 | HFOV={self.hfov_deg:.1f}° VFOV={vfov:.1f}° | "
            f"fx={self.native_fx:.1f}px | "
            f"安装: h={self.camera_height}m pitch={math.degrees(self.pitch):.1f}°"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 综合：单帧处理流水线
# ─────────────────────────────────────────────────────────────────────────────

def _water_width_at_row(mask: np.ndarray, row: int) -> Optional[int]:
    """测量 mask 中指定行的河道宽度（像素），取水域+边界的最大跨度。"""
    row = max(0, min(mask.shape[0] - 1, row))
    cols = np.where((mask[row] == WATER_CLASS) | (mask[row] == BOUNDARY_CLASS))[0]
    if len(cols) < 2:
        return None
    return int(cols[-1] - cols[0])


def _truncate_path_at_shore(
    path_pts: List[Tuple[int, int]],
    mask: np.ndarray,
    projector: Optional[IMX390Projector] = None,
    river_width_m: float = 2.5,
    shore_stop_dist: float = 1.0,
) -> Tuple[List[Tuple[int, int]], float]:
    """
    截断路径在距岸边 shore_stop_dist 处。
    
    使用 cv2.distanceTransform 计算像素到边界的距离，然后逐点检查：
    - 无 projector：使用同行水平像素距离 * 该行比例尺
    - 有 projector：投影路径点和最近边界像素到地面，计算两者真实距离
    
    返回: (截断后的路径, 路径终点处的岸距米)
    """
    if not path_pts:
        return path_pts, float('inf')

    h, w = mask.shape[:2]
    boundary_mask = (mask == BOUNDARY_CLASS)
    if not boundary_mask.any():
        return path_pts, float('inf')

    if projector is not None:
        projector.update_resolution(w, h)

    # 预计算每一行边界像素列，加速逐点查询
    boundary_cols_by_row = [np.where(boundary_mask[y])[0] for y in range(h)]

    def _distance_to_shore_m(px_x: int, px_y: int) -> float:
        cols = boundary_cols_by_row[px_y]

        # 当前行无边界像素时，向上（远距离方向）搜索最近的有边界行
        # 不使用 distanceTransform 避免垂直/斜向距离误触发截断
        if len(cols) == 0:
            for dy in range(1, 60):
                look_y = px_y - dy
                if look_y < 0:
                    break
                look_cols = boundary_cols_by_row[look_y]
                if len(look_cols) > 0:
                    cols = look_cols
                    break

        # 仍无边界：用水域横向边缘估算距岸
        if len(cols) == 0:
            wr = np.where((mask[px_y] == WATER_CLASS) | (mask[px_y] == BOUNDARY_CLASS))[0]
            if len(wr) < 2:
                return 0.0
            hor_e = min(abs(px_x - float(wr[0])), abs(px_x - float(wr[-1])))
            rw_e = int(wr[-1] - wr[0])
            if rw_e < 10:
                return 0.0
            return float(hor_e * river_width_m / rw_e)

        nearest_idx = int(np.argmin(np.abs(cols - px_x)))
        bnd_x = int(cols[nearest_idx])
        horiz_px = abs(px_x - bnd_x)

        # projector 模式优先用真实地面距离（路径点与同一行最近边界点）
        if projector is not None:
            gnd_pt = projector.pixel_to_ground(float(px_x), float(px_y))
            gnd_bnd = projector.pixel_to_ground(float(bnd_x), float(px_y))
            if gnd_pt is not None and gnd_bnd is not None:
                return float(math.hypot(gnd_pt[0] - gnd_bnd[0], gnd_pt[1] - gnd_bnd[1]))

        # 后备：同行水平像素距离 * 行比例尺
        rw_px = _water_width_at_row(mask, px_y)
        if rw_px is None or rw_px < 10:
            rw_px = _water_width_at_row(mask, h // 2)
        if rw_px is None or rw_px < 10:
            return float('inf')
        scale = river_width_m / rw_px
        return float(horiz_px * scale)

    truncated: List[Tuple[int, int]] = []
    for x, y in path_pts:
        px_x = max(0, min(w - 1, int(x)))
        px_y = max(0, min(h - 1, int(y)))
        dist_m = _distance_to_shore_m(px_x, px_y)
        if dist_m <= shore_stop_dist:
            break
        truncated.append((px_x, px_y))

    last_dist = float('inf')
    if truncated:
        lx, ly = truncated[-1]
        last_dist = _distance_to_shore_m(lx, ly)
    return truncated, last_dist


def _fill_heading(mileage_list: List[Dict[str, Any]], window: int = 5) -> None:
    """
    基于地面坐标 (_gx=前向, _gy=横向) 为每个点计算航向角 heading_deg。
    自适应滑窗：先看前方 window 点，若前向步长过小则扩大窗口直到覆盖足够深度差。

    heading = atan2(Δy_lateral, Δx_forward)
    0° = 正前方, +右偏, -左偏
    """
    n = len(mileage_list)
    if n < 2:
        for pt in mileage_list:
            pt['heading_deg'] = 0.0
            pt.pop('_gx', None)
            pt.pop('_gy', None)
        return

    MIN_FWD = 0.01  # 最小前向步长(m)，低于此值扩大窗口
    prev_heading = 0.0

    for i in range(n):
        found = False
        for w in range(window, n, window):
            j = min(i + w, n - 1)
            k = i if j > i else max(i - w, 0)

            d_fwd = mileage_list[j]['_gx'] - mileage_list[k]['_gx']
            d_lat = mileage_list[j]['_gy'] - mileage_list[k]['_gy']

            if abs(d_fwd) >= MIN_FWD:
                heading = math.degrees(math.atan2(d_lat, d_fwd))
                mileage_list[i]['heading_deg'] = round(heading, 2)
                prev_heading = mileage_list[i]['heading_deg']
                found = True
                break

            if j == n - 1 and k == 0:
                break

        if not found:
            mileage_list[i]['heading_deg'] = prev_heading

    for pt in mileage_list:
        pt.pop('_gx', None)
        pt.pop('_gy', None)


def _compute_path_mileage(
    path_pts: List[Tuple[int, int]],
    mask: np.ndarray,
    river_width_m: float = 2.5,
    projector: Optional[IMX390Projector] = None,
    boat_speed: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    计算路径上每个点的世界坐标 (x_m, y_m)、航向角和预计到达时间。

    两种模式:
      1. 相机投影模式 (projector != None): 基于 IMX390 针孔模型做透视逆投影。
      2. 河宽比例模式 (后备): 用每行河道像素宽度和已知河宽做线性比例。

    坐标系（船体坐标系）：
      原点 = 图像底部中心（船的位置）
      X = 正前方，x_m: 累计前进里程（米）
      Y = 横向，  y_m: 横向偏移（+右 -左，米）
      heading_deg: 航向角（0°=正前方, +右偏, -左偏）
    """
    h, w = mask.shape[:2]

    # ── 模式1: 相机投影 ──
    if projector is not None:
        projector.update_resolution(w, h)
        mileage_list = []
        cumulative_m = 0.0
        prev_ground = None

        for i, (px, py) in enumerate(path_pts):
            gnd = projector.pixel_to_ground(float(px), float(py))
            if gnd is None:
                gnd = prev_ground if prev_ground else (0.0, 0.0)

            if i > 0 and prev_ground is not None:
                seg_m = math.sqrt(
                    (gnd[0] - prev_ground[0]) ** 2 +
                    (gnd[1] - prev_ground[1]) ** 2
                )
                cumulative_m += seg_m

            # gnd[0] = X（正前方距离），gnd[1] = Y（横向偏移）
            mileage_list.append({
                'x_px': int(px), 'y_px': int(py),
                'x_m': round(cumulative_m, 3),          # X：累计路径里程
                'y_m': round(gnd[1], 3),                 # Y：横向偏移
                'time_s': round(cumulative_m / boat_speed, 2) if boat_speed > 0 else 0.0,
                '_gx': gnd[0], '_gy': gnd[1],
            })
            prev_ground = gnd

        _fill_heading(mileage_list, window=max(5, len(mileage_list) // 20))
        return mileage_list

    # ── 模式2: 河宽比例法（后备） ──
    img_cx = w / 2.0

    _heading_proj = IMX390Projector(camera_height=0.5, camera_pitch_deg=10.0, hfov_deg=120.0)
    _heading_proj.update_resolution(w, h)

    fallback_widths = []
    for row_sample in range(0, h, max(1, h // 20)):
        rw = _water_width_at_row(mask, row_sample)
        if rw and rw > 10:
            fallback_widths.append(rw)
    fallback_width = int(np.median(fallback_widths)) if fallback_widths else w

    mileage_list = []
    cumulative_m = 0.0

    for i, (px, py) in enumerate(path_pts):
        row = max(0, min(h - 1, int(py)))
        rw_px = _water_width_at_row(mask, row)
        if rw_px is None or rw_px < 10:
            rw_px = fallback_width

        scale = river_width_m / rw_px
        y_m = (px - img_cx) * scale                     # Y：横向偏移

        if i > 0:
            dx_px = px - path_pts[i - 1][0]
            dy_px = py - path_pts[i - 1][1]
            seg_px = math.sqrt(dx_px * dx_px + dy_px * dy_px)
            prev_row = max(0, min(h - 1, int(path_pts[i - 1][1])))
            prev_rw = _water_width_at_row(mask, prev_row)
            if prev_rw is None or prev_rw < 10:
                prev_rw = fallback_width
            avg_scale = (scale + river_width_m / prev_rw) / 2.0
            cumulative_m += seg_px * avg_scale

        # 地面坐标（透视修正后），仅用于航向角计算
        gnd = _heading_proj.pixel_to_ground(float(px), float(py))
        if gnd is not None:
            _gx, _gy = gnd
        elif i > 0:
            # 超出地平线：从上一个点的地面坐标逐点外推
            prev_entry = mileage_list[i - 1]
            dpx = px - path_pts[i - 1][0]
            dpy = py - path_pts[i - 1][1]
            ext_scale = river_width_m / max(rw_px, 10)
            seg = math.sqrt(dpx * dpx + dpy * dpy)
            _gx = prev_entry['_gx'] + max(seg, 1.0) * ext_scale * 3.0
            _gy = prev_entry['_gy'] + dpx * ext_scale
        else:
            _gx, _gy = 0.0, y_m

        mileage_list.append({
            'x_px': int(px), 'y_px': int(py),
            'x_m': round(cumulative_m, 3),               # X：累计路径里程
            'y_m': round(y_m, 3),                        # Y：横向偏移
            'time_s': round(cumulative_m / boat_speed, 2) if boat_speed > 0 else 0.0,
            '_gx': _gx, '_gy': _gy,
        })

    _fill_heading(mileage_list, window=max(5, len(mileage_list) // 20))
    return mileage_list


def process_frame(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    pursuit: SimplePurePursuit,
    river_width_m: float = 2.5,
    projector: Optional[IMX390Projector] = None,
    boat_speed: float = 0.4,
    shore_stop_dist: float = 1.0,
    steer_filter: Optional[CTEFilter] = None,
    roi_top_ratio: float = ROI_TOP_RATIO,
) -> Dict[str, Any]:
    """
    单帧全流水线：mask → 路径规划 → 岸边截断 → Pure Pursuit → 控制信号输出。
    输出包含世界坐标 (x_m, y_m)、航向角、里程和到达时间。
    路径在距岸边 shore_stop_dist 处截断，船自然沿截断后的路径行驶到终点。
    """
    h, w = img_bgr.shape[:2]

    _empty_result = {
        'steering_deg': 0.0, 'throttle_pct': 0.0,
        'speed_mps': 0.0,
        'path_pts': 0, 'mode': 'none', 'pp_status': '',
        'target_x_m': 0.0,
        'target_y_m': 0.0,
        'heading_deg': 0.0,
        'total_mileage_m': 0.0,
        'mileage_coords': [],
        'shore_dist_m': float('inf'),
        'cte_m': 0.0,
    }

    center_pts: List[Tuple[int, int]] = []
    mode = ''
    channel_width_px = None  # 近端航道像素宽度

    # ── 双边界检测（Boundary_identification 方法：形态学闭运算 + 倒影过滤）──
    polylines, n_filtered = extract_boundary_polylines(mask)
    if len(polylines) == 2:
        # 按平均 x 区分左右
        avg_x = [sum(p[0] for p in pts) / len(pts) for pts in polylines]
        if avg_x[0] <= avg_x[1]:
            left_pts, right_pts = polylines[0], polylines[1]
        else:
            left_pts, right_pts = polylines[1], polylines[0]

        # ROI 截断：只用图像下部，远端噪声不参与中线计算
        roi_top = int(h * roi_top_ratio)
        left_roi = [(x, y) for x, y in left_pts if y >= roi_top]
        right_roi = [(x, y) for x, y in right_pts if y >= roi_top]

        left_dict = {y: float(x) for x, y in left_roi}
        right_dict = {y: float(x) for x, y in right_roi}
        overlap_ys = sorted(set(left_dict.keys()) & set(right_dict.keys()))

        if len(overlap_ys) >= 3:
            center_pts = [
                (int(round((left_dict[y] + right_dict[y]) / 2.0)), y)
                for y in overlap_ys
            ]
            mode = 'dual'

            # 近端航道像素宽度，用于物理量化
            bottom_ys = [y for y in overlap_ys if y >= h * 0.7]
            if bottom_ys:
                channel_width_px = float(np.mean(
                    [right_dict[y] - left_dict[y] for y in bottom_ys]
                ))

    # ── 回退：单边界 ──
    if not mode:
        comp, side, _ = _find_single_boundary(mask)
        if comp is not None:
            _, _, cp_b = compute_single_boundary_centerline(mask, comp, side, w)
            if len(cp_b) >= 3:
                center_pts, mode = cp_b, 'single'

    # ── 回退：纯水域 ──
    if not mode:
        cp_c = compute_water_centerline(mask)
        if len(cp_c) >= 3:
            center_pts, mode = cp_c, 'water'

    if not center_pts:
        r = dict(_empty_result)
        r.update(pp_status='no centerline')
        return r

    path_pts, _ = plan_path_from_bottom(center_pts, h, w, mask)
    if len(path_pts) < 3:
        r = dict(_empty_result)
        r.update(mode=mode, pp_status='path too short')
        return r

    path_pts, shore_dist_m = _truncate_path_at_shore(
        path_pts, mask, projector=projector,
        river_width_m=river_width_m, shore_stop_dist=shore_stop_dist
    )
    if len(path_pts) < 3:
        r = dict(_empty_result)
        r.update(mode=mode, pp_status='path too short (shore truncated)')
        return r

    mileage_coords = _compute_path_mileage(
        path_pts, mask, river_width_m, projector=projector,
        boat_speed=boat_speed,
    )
    total_mileage_m = mileage_coords[-1]['x_m'] if mileage_coords else 0.0   # X = 前进里程

    pursuit.img_w = w
    pursuit.img_h = h
    pp = pursuit.compute(path_pts)

    # EMA 时序滤波：平滑舵角，减少帧间抖动
    if steer_filter is not None and pp['status'] == 'ok':
        pp['steering_deg'] = steer_filter.update(pp['steering_deg'])
        pp['steering_rad'] = math.radians(pp['steering_deg'])

    target_px_x, target_px_y = pp['target_px']

    # 航向角：基于路径前几个点的切线方向
    if projector is not None:
        _hp = projector
        _hp.update_resolution(w, h)
        _tg = _hp.pixel_to_ground(float(target_px_x), float(target_px_y))
        if _tg and _tg[0] > 0.01:
            heading_deg = math.degrees(math.atan2(_tg[1], _tg[0]))
        else:
            heading_deg = 0.0
    else:
        if len(path_pts) >= 2:
            dx = float(path_pts[-1][0] - path_pts[0][0])
            dy = float(h - 1 - path_pts[-1][1] if len(path_pts) > 1 else 0)
            if len(path_pts) >= 5:
                look_idx = min(3, len(path_pts) // 4)
                dx = float(path_pts[look_idx][0] - path_pts[0][0])
                dy = float(h - 1 - path_pts[look_idx][1])
            if abs(dx) > 0.1 or abs(dy) > 0.1:
                heading_deg = math.degrees(math.atan2(dx, dy))
            else:
                heading_deg = 0.0
        else:
            heading_deg = 0.0

    # 目标点世界坐标：从里程列表中找最近点
    target_x_m = 0.0
    target_y_m = 0.0
    best_dist = float('inf')
    for mc in mileage_coords:
        d = (mc['x_px'] - int(target_px_x)) ** 2 + (mc['y_px'] - int(target_px_y)) ** 2
        if d < best_dist:
            best_dist = d
            target_x_m = mc['x_m']
            target_y_m = mc['y_m']

    if projector is not None and best_dist > 100:
        projector.update_resolution(w, h)
        tg = projector.pixel_to_ground(float(target_px_x), float(target_px_y))
        if tg:
            target_x_m, target_y_m = tg

    # 基于已知航道宽度的横向偏差物理量化
    cte_m = 0.0
    if channel_width_px is not None and channel_width_px > 10 and path_pts:
        scale_ppm = channel_width_px / river_width_m
        cte_px = path_pts[0][0] - w / 2.0
        cte_m = cte_px / scale_ppm

    return {
        'steering_deg': pp['steering_deg'],
        'throttle_pct': pp['throttle_pct'],
        'speed_mps': pp['speed_mps'],
        'path_pts': len(path_pts),
        'mode': mode,
        'pp_status': pp['status'],
        'target_x_m': round(target_x_m, 3),
        'target_y_m': round(target_y_m, 3),
        'heading_deg': round(heading_deg, 2),
        'total_mileage_m': round(total_mileage_m, 3),
        'mileage_coords': mileage_coords,
        'shore_dist_m': round(shore_dist_m, 3),
        'cte_m': round(cte_m, 3),
        'path_pts_list': path_pts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 可视化叠加控制信息
# ─────────────────────────────────────────────────────────────────────────────

def draw_control_hud(canvas: np.ndarray, result: Dict[str, Any]):
    """在画面右上角绘制控制信息 HUD（含目标点坐标、里程和岸距）。"""
    h, w = canvas.shape[:2]
    tx = result.get('target_x_m', 0.0)
    ty = result.get('target_y_m', 0.0)
    hdg = result.get('heading_deg', 0.0)
    mile = result.get('total_mileage_m', 0.0)
    shore_dist = result.get('shore_dist_m', float('inf'))

    cte = result.get('cte_m', 0.0)
    shore_str = f"{shore_dist:.2f} m" if shore_dist < 100 else "N/A"
    lines = [
        f"Mode: {result['mode']}",
        f"Heading: {hdg:+.1f} deg",
        f"Steer: {result['steering_deg']:+.1f} deg",
        f"Throttle: {result['throttle_pct']:.0f}%",
        f"Speed: {result['speed_mps']:.2f} m/s",
        f"X(fwd): {tx:.2f} m  Y(lat): {ty:+.2f} m",
        f"Mileage: {mile:.2f} m",
        f"CTE: {cte:+.3f} m",
        f"Shore: {shore_str}",
        f"PP: {result['pp_status']}",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.50
    lh = 20
    pad = 8
    max_tw = max(cv2.getTextSize(l, font, scale, 1)[0][0] for l in lines)
    box_w = max_tw + pad * 2
    box_h = len(lines) * lh + pad * 2

    x0 = w - box_w - 10
    y0 = 10
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)

    for i, line in enumerate(lines):
        text_y = y0 + pad + (i + 1) * lh - 4
        cv2.putText(canvas, line, (x0 + pad, text_y), font, scale, (0, 255, 200), 1, cv2.LINE_AA)


def overlay_truncated_path(canvas: np.ndarray, path_pts: List[Tuple[int, int]]):
    """在可视化图上用浅蓝色线条覆盖截断后的路径。"""
    if not path_pts or len(path_pts) < 2:
        return
    pts_np = np.array(path_pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts_np], isClosed=False,
                  color=(255, 200, 100), thickness=3, lineType=cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# CLI 主入口
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="端到端自主导航：图像→路径→控制信号",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--camera', type=int, default=None, help='摄像头设备ID（如 0）')
    src.add_argument('--images', default=None, help='输入图像文件/目录')

    mask_src = p.add_mutually_exclusive_group()
    mask_src.add_argument('--model', default=None, help='SegFormer .pth 权重')
    mask_src.add_argument('--masks', default=None, help='mask 文件/目录')

    p.add_argument('--target-speed', type=float, default=1.5, help='目标速度 m/s（Pure Pursuit油门计算用）')
    p.add_argument('--max-speed',    type=float, default=3.0, help='最大速度 m/s')
    p.add_argument('--lookahead',    type=float, default=0.20, help='Pure Pursuit 预瞄比例（相对图像高度 0-1）')
    p.add_argument('--boat-speed',   type=float, default=0.4, help='实际船速 m/s（用于里程时间估算）')

    p.add_argument('--river-width', type=float, default=2.5,
                   help='河道实际宽度（米），用于河宽比例法后备换算')
    p.add_argument('--shore-stop-dist', type=float, default=1.0,
                   help='岸边停车距离阈值（米），检测到离岸 ≤ 此值时停止电机，默认 1.0')

    cam = p.add_argument_group('IMX390-GMSL2 相机参数（提供后启用精确投影模型）')
    cam.add_argument('--camera-height', type=float, default=None,
                     help='相机离水面高度（米），例如 0.5')
    cam.add_argument('--camera-pitch', type=float, default=10.0,
                     help='相机俯仰角（度，>0 表示向下看），默认 10°')
    cam.add_argument('--camera-hfov', type=float, default=120.0,
                     help='水平视场角（度）。H60=63.9 / H120=120.6 / H190=186(鱼眼)')

    p.add_argument('--output', default=None, help='可视化保存目录')
    p.add_argument('--show', action='store_true', help='弹窗显示')
    p.add_argument('--num-classes', type=int, default=3)
    p.add_argument('--input-size',  type=int, default=640,
                   help='推理输入宽度（像素），高度自动按16:9计算，默认640')
    p.add_argument('--model-name',  type=str,
                   default='nvidia/segformer-b0-finetuned-ade-512-512',
                   help='SegFormer基础架构名，需与训练时一致')
    return p.parse_args()


def run_on_images(args):
    """离线图片模式"""
    if args.masks is None and args.model is None:
        sys.exit("❌ 图片模式需要 --masks 或 --model")

    img_paths = collect_images(args.images)
    print(f"[信息] 共 {len(img_paths)} 张图像")

    seg_model, device = None, None
    if args.model:
        seg_model, device = load_segformer(args.model, num_classes=args.num_classes,
                                           model_name=args.model_name)

    pursuit = SimplePurePursuit(
        lookahead_ratio=args.lookahead, target_speed=args.target_speed, max_speed=args.max_speed,
    )
    steer_filter = CTEFilter(alpha=0.3)
    projector = args._projector

    out_dir = None
    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

    total = ok = 0
    try:
        for img_path in img_paths:
            total += 1
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue

            if seg_model:
                input_h = args.input_size * 384 // 640
                mask = infer_mask(seg_model, device, img_bgr,
                                  input_width=args.input_size, input_height=input_h,
                                  num_classes=args.num_classes)
            else:
                from PIL import Image as PILImage
                mp = find_mask_path(args.masks, img_path)
                if mp is None:
                    continue
                mask = np.array(PILImage.open(str(mp)).convert('P'))

            result = process_frame(img_bgr, mask, pursuit,
                                   river_width_m=args.river_width,
                                   projector=projector,
                                   boat_speed=args.boat_speed,
                                   shore_stop_dist=args.shore_stop_dist,
                                   steer_filter=steer_filter)

            vis, _ = render_planned_frame(img_bgr, mask, fname=img_path.name)
            overlay_truncated_path(vis, result.get('path_pts_list', []))
            draw_control_hud(vis, result)

            shore_info = (f"  shore={result['shore_dist_m']:.2f}m"
                          if result['shore_dist_m'] < 100 else "")
            tag = result['mode'] or 'skip'
            print(f"\n  [{tag:>6s}] {img_path.name}  "
                  f"heading={result['heading_deg']:+6.1f}°  "
                  f"steer={result['steering_deg']:+5.1f}°  "
                  f"throttle={result['throttle_pct']:.0f}%  "
                  f"X(fwd)={result['target_x_m']:.2f}m  "
                  f"Y(lat)={result['target_y_m']:+.2f}m  "
                  f"mileage={result['total_mileage_m']:.2f}m"
                  f"{shore_info}")

            mc = result['mileage_coords']
            if mc:
                n_show = min(len(mc), 15)
                step = max(1, len(mc) // n_show)
                samples = mc[::step]
                if mc[-1] not in samples:
                    samples.append(mc[-1])
                print(f"  {'idx':>4s}  {'x_px':>5s} {'y_px':>5s}  "
                      f"{'X/fwd(m)':>9s} {'Y/lat(m)':>9s}  "
                      f"{'heading':>8s}  {'time(s)':>7s}")
                print(f"  {'----':>4s}  {'-----':>5s} {'-----':>5s}  "
                      f"{'--------':>9s} {'--------':>9s}  "
                      f"{'--------':>8s}  {'-------':>7s}")
                for j, pt in enumerate(samples):
                    idx = mc.index(pt)
                    print(f"  {idx:4d}  {pt['x_px']:5d} {pt['y_px']:5d}  "
                          f"{pt['x_m']:9.3f} {pt['y_m']:+9.3f}  "
                          f"{pt.get('heading_deg', 0.0):+8.2f}°  "
                          f"{pt.get('time_s', 0.0):7.2f}")

            if out_dir and mc:
                csv_path = out_dir / (img_path.stem + '_mileage.csv')
                with open(str(csv_path), 'w') as f:
                    f.write('idx,x_px,y_px,x_m,y_m,heading_deg,time_s\n')
                    for j, pt in enumerate(mc):
                        f.write(f"{j},{pt['x_px']},{pt['y_px']},"
                                f"{pt['x_m']},{pt['y_m']},"
                                f"{pt.get('heading_deg', 0.0)},"
                                f"{pt.get('time_s', 0.0)}\n")

            if result['path_pts'] > 0:
                ok += 1

            if out_dir:
                cv2.imwrite(str(out_dir / (img_path.stem + '_pilot.jpg')), vis,
                            [cv2.IMWRITE_JPEG_QUALITY, 94])

            if args.show:
                cv2.imshow("Pilot", vis)
                key = cv2.waitKey(0)
                if key in (ord('q'), ord('Q'), 27):
                    break
    finally:
        if args.show:
            cv2.destroyAllWindows()

    print(f"\n{'='*50}")
    print(f"  总计 {total} | 成功 {ok} | 跳过 {total - ok}")
    print(f"{'='*50}")


def run_on_camera(args):
    """实时相机模式"""
    if args.model is None:
        sys.exit("❌ 相机模式需要 --model")

    seg_model, device = load_segformer(args.model, num_classes=args.num_classes,
                                       model_name=args.model_name)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"❌ 无法打开摄像头 {args.camera}")
    print(f"[相机] 已打开设备 {args.camera}")

    pursuit = SimplePurePursuit(
        lookahead_ratio=args.lookahead, target_speed=args.target_speed, max_speed=args.max_speed,
    )
    steer_filter = CTEFilter(alpha=0.3)
    projector = args._projector

    out_dir = None
    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    fps_t0 = time.time()

    try:
        while True:
            ret, img_bgr = cap.read()
            if not ret:
                print("[相机] 读取失败，重试 ...")
                time.sleep(0.1)
                continue

            t0 = time.time()

            input_h = args.input_size * 384 // 640
            mask = infer_mask(seg_model, device, img_bgr,
                              input_width=args.input_size, input_height=input_h,
                              num_classes=args.num_classes)
            result = process_frame(img_bgr, mask, pursuit,
                                   river_width_m=args.river_width,
                                   projector=projector,
                                   boat_speed=args.boat_speed,
                                   shore_stop_dist=args.shore_stop_dist,
                                   steer_filter=steer_filter)

            dt = time.time() - t0
            fps = 1.0 / max(dt, 1e-6)

            shore_info = (f"shore={result['shore_dist_m']:.2f}m "
                          if result['shore_dist_m'] < 100 else "")
            tag = result['mode'] or 'skip'
            print(f"\r[{frame_idx:05d}] [{tag}] "
                  f"hdg={result['heading_deg']:+5.1f}°  "
                  f"steer={result['steering_deg']:+5.1f}°  "
                  f"throttle={result['throttle_pct']:.0f}%  "
                  f"X={result['target_x_m']:.2f}m Y={result['target_y_m']:+.2f}m  "
                  f"mile={result['total_mileage_m']:.2f}m  "
                  f"{shore_info}"
                  f"{fps:.1f}fps  {result['pp_status']}", end='', flush=True)

            if args.show or out_dir:
                vis, _ = render_planned_frame(img_bgr, mask)
                overlay_truncated_path(vis, result.get('path_pts_list', []))
                draw_control_hud(vis, result)
                cv2.putText(vis, f"{fps:.1f} FPS", (10, vis.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if args.show:
                    cv2.imshow("Pilot", vis)
                    key = cv2.waitKey(1)
                    if key in (ord('q'), ord('Q'), 27):
                        print("\n[信息] 用户退出")
                        break

                if out_dir and frame_idx % 10 == 0:
                    cv2.imwrite(str(out_dir / f"frame_{frame_idx:05d}.jpg"), vis,
                                [cv2.IMWRITE_JPEG_QUALITY, 85])

            frame_idx += 1

    except KeyboardInterrupt:
        print("\n[信息] Ctrl+C 退出")
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()
        print(f"\n[信息] 共处理 {frame_idx} 帧")


def main():
    args = parse_args()

    projector = None
    if args.camera_height is not None:
        projector = IMX390Projector(
            camera_height=args.camera_height,
            camera_pitch_deg=args.camera_pitch,
            hfov_deg=args.camera_hfov,
        )
    args._projector = projector

    print("="*60)
    print("  River Lane Pilot — 端到端自主导航")
    print(f"  船速: {args.boat_speed} m/s  预瞄比例: {args.lookahead}")
    print(f"  避障: 岸距 ≤ {args.shore_stop_dist}m 时停车")
    if projector:
        print(f"  坐标模式: 相机投影 | {projector.info()}")
    else:
        print(f"  坐标模式: 河宽比例法 | 河道宽度={args.river_width}m")
        print(f"  提示: 提供 --camera-height 可启用 IMX390 精确投影")
    print("="*60)

    if args.camera is not None:
        run_on_camera(args)
    else:
        run_on_images(args)


if __name__ == '__main__':
    main()
