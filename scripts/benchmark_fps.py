#!/usr/bin/env python3
"""
FPS 基准测试脚本 —— 测量完整推理流水线在当前硬件上的吞吐量。

流水线各阶段分别计时：
  ① 预处理（resize + normalize）
  ② 模型推理（SegFormer forward pass）
  ③ 后处理（插值恢复原分辨率 + argmax）
  ④ process_frame（边界提取 + 中线 + 偏航角）
  ⑤ 【全流程】= ①②③④ 合计

用法:
  python scripts/benchmark_fps.py \
      --model models/segformer_river/best_model.pth \
      --num-frames 100

  # 同时测试多个输入分辨率
  python scripts/benchmark_fps.py \
      --model models/segformer_river/best_model.pth \
      --input-sizes 640 512 384 \
      --num-frames 100

  # 使用真实图片（更准确的 process_frame 耗时）
  python scripts/benchmark_fps.py \
      --model models/segformer_river/best_model.pth \
      --images dataset_final/images/test \
      --num-frames 50
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from visualize_centerline import (
    BOUNDARY_CLASS,
    load_segformer,
    collect_images,
)
from realtime_pilot_v4 import process_frame, YawFilter


# ─── 计时工具 ─────────────────────────────────────────────────────────────────

class Timer:
    def __init__(self):
        self.records: List[float] = []

    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.records.append(time.perf_counter() - self._t)

    def mean_ms(self) -> float:
        return float(np.mean(self.records)) * 1000.0

    def p95_ms(self) -> float:
        return float(np.percentile(self.records, 95)) * 1000.0

    def fps(self) -> float:
        m = float(np.mean(self.records))
        return 1.0 / m if m > 0 else 0.0


# ─── GPU 同步工具 ─────────────────────────────────────────────────────────────

def _cuda_sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


# ─── 分阶段推理 ───────────────────────────────────────────────────────────────

def infer_staged(
    model, device, img_bgr: np.ndarray,
    input_width: int, input_height: int, num_classes: int,
    t_pre: Timer, t_infer: Timer, t_post: Timer,
) -> np.ndarray:
    """分阶段计时推理，返回 mask。"""
    import torch
    import torch.nn.functional as F

    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std  = np.array([0.229, 0.224, 0.225], np.float32)
    oh, ow = img_bgr.shape[:2]

    # ① 预处理
    _cuda_sync()
    with t_pre:
        rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        rsz  = cv2.resize(rgb, (input_width, input_height))
        norm = (rsz.astype(np.float32) / 255.0 - mean) / std
        t    = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).float().to(device)
    _cuda_sync()

    # ② 模型推理
    _cuda_sync()
    with t_infer:
        with torch.no_grad():
            logits = model(pixel_values=t).logits
    _cuda_sync()

    # ③ 后处理
    _cuda_sync()
    with t_post:
        with torch.no_grad():
            logits = F.interpolate(
                logits, size=(oh, ow), mode='bilinear', align_corners=False,
            )
            pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    _cuda_sync()

    return pred


# ─── 打印分割线 ───────────────────────────────────────────────────────────────

def _sep(char='─', n=62):
    print(char * n)


# ─── 单分辨率基准测试 ─────────────────────────────────────────────────────────

def run_benchmark(
    model, device,
    frames: List[np.ndarray],
    input_width: int,
    input_height: int,
    num_classes: int,
    warmup: int = 10,
    label: str = '',
) -> None:
    import torch

    yaw_filter = YawFilter(alpha=0.3)

    t_pre   = Timer()
    t_infer = Timer()
    t_post  = Timer()
    t_proc  = Timer()
    t_total = Timer()

    n = len(frames)
    print(f"\n[分辨率] 输入 {input_width}×{input_height}  "
          f"({'×'.join(str(s) for s in frames[0].shape[1::-1])} 原图)  "
          f"测试帧数={n}  预热={warmup}")

    # ── 预热（避免第一批帧JIT/内存分配计入统计）──
    _w = min(warmup, n)
    for f in frames[:_w]:
        infer_staged(model, device, f, input_width, input_height, num_classes,
                     Timer(), Timer(), Timer())

    # ── 正式计时 ──
    for img in frames:
        with t_total:
            mask = infer_staged(
                model, device, img,
                input_width, input_height, num_classes,
                t_pre, t_infer, t_post,
            )
            with t_proc:
                _, _ = process_frame(img, mask, yaw_filter=yaw_filter)
        _cuda_sync()

    # ── 打印结果 ──
    _sep()
    print(f"{'阶段':<20} {'均值(ms)':>10} {'P95(ms)':>10}")
    _sep('·')
    stages = [
        ('① 预处理',      t_pre),
        ('② 模型推理',    t_infer),
        ('③ 后处理',      t_post),
        ('④ process_frame', t_proc),
        ('⑤ 全流程合计',  t_total),
    ]
    for name, tm in stages:
        marker = ' ◀' if name.startswith('⑤') else ''
        print(f"  {name:<18} {tm.mean_ms():>9.1f}  {tm.p95_ms():>9.1f}{marker}")
    _sep()

    fps = t_total.fps()
    target = 20.0
    status = '✅ 达标' if fps >= target else f'❌ 不达标（需提速 {target/fps:.1f}×）'
    print(f"  全流程 FPS = {fps:.1f}   目标 {target} FPS  →  {status}")

    # 推理瓶颈分析
    infer_ratio = t_infer.mean_ms() / t_total.mean_ms() * 100
    proc_ratio  = t_proc.mean_ms()  / t_total.mean_ms() * 100
    print(f"  瓶颈占比：模型推理 {infer_ratio:.0f}%  |  后处理算法 {proc_ratio:.0f}%")
    _sep()

    if fps < target:
        print("  【优化建议】")
        if infer_ratio > 50:
            print("    • 模型是瓶颈 → 推荐转换为 TensorRT (deployment/convert_tensorrt.py)")
            print("    • 或减小 --input-size（如 512→384）")
        else:
            print("    • process_frame 是瓶颈 → 可精简边界提取像素操作")
        print("    • Jetson 上先执行: sudo jetson_clocks  (锁定最高频率)")
        print("    • 确保使用 MAXN 或 15W 功耗模式: sudo nvpmodel -m 0")


# ─── 构造随机帧（无图片时使用）────────────────────────────────────────────────

def _make_fake_frames(n: int, h: int = 480, w: int = 640) -> List[np.ndarray]:
    rng = np.random.default_rng(42)
    return [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(n)]


# ─── GPU / 设备信息打印 ───────────────────────────────────────────────────────

def _print_device_info(device) -> None:
    _sep('═')
    print("  硬件信息")
    _sep('─')
    try:
        import torch
        if str(device) == 'cpu':
            import platform
            print(f"  CPU 模式  ({platform.processor() or platform.machine()})")
        else:
            idx = device.index if hasattr(device, 'index') else 0
            prop = torch.cuda.get_device_properties(idx)
            mem_gb = prop.total_memory / 1024**3
            print(f"  GPU: {prop.name}")
            print(f"  显存: {mem_gb:.1f} GB")
            print(f"  SM 数量: {prop.multi_processor_count}")
            print(f"  CUDA Capability: {prop.major}.{prop.minor}")
    except Exception as e:
        print(f"  (无法获取设备信息: {e})")
    _sep('═')


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="FPS 基准测试：测量推理流水线各阶段耗时",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--model',        required=True,  help='SegFormer .pth 权重路径')
    p.add_argument('--images',       default=None,   help='真实图片目录（可选，不指定则用随机帧）')
    p.add_argument('--num-frames',   type=int, default=100, help='测试帧数，默认 100')
    p.add_argument('--warmup',       type=int, default=10,  help='预热帧数，默认 10')
    p.add_argument('--input-sizes',  type=int, nargs='+', default=[640],
                   help='推理输入宽度（可指定多个对比），默认 640')
    p.add_argument('--num-classes',  type=int, default=3)
    p.add_argument('--model-name',   type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    model, device = load_segformer(
        args.model,
        num_classes=args.num_classes,
        model_name=args.model_name,
    )

    _print_device_info(device)

    # 准备帧
    if args.images:
        img_paths = collect_images(args.images)
        frames_all: List[np.ndarray] = []
        for p in img_paths:
            f = cv2.imread(str(p))
            if f is not None:
                frames_all.append(f)
        if not frames_all:
            print("[警告] 未能读取任何图片，改用随机帧")
            frames_all = _make_fake_frames(args.num_frames)
        # 循环截取所需帧数
        n = args.num_frames
        frames = [frames_all[i % len(frames_all)] for i in range(n)]
        print(f"[信息] 使用真实图片 {len(frames_all)} 张，循环组成 {n} 帧测试集")
    else:
        frames = _make_fake_frames(args.num_frames)
        print(f"[信息] 使用随机合成帧 {args.num_frames} 张（480×640）")

    # 逐分辨率测试
    for iw in args.input_sizes:
        ih = iw * 384 // 640
        run_benchmark(
            model, device, frames,
            input_width=iw, input_height=ih,
            num_classes=args.num_classes,
            warmup=args.warmup,
            label=f'{iw}x{ih}',
        )

    print()
    print("  提示: 在 Jetson 上部署前，建议先运行：")
    print("    sudo nvpmodel -m 0          # MAXN 功耗模式（最高性能）")
    print("    sudo jetson_clocks          # 锁定 CPU/GPU/EMC 最高频率")
    print("    python deployment/convert_tensorrt.py  # 转 TensorRT 可大幅提速")


if __name__ == '__main__':
    main()
