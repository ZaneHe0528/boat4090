#!/usr/bin/env python3
"""Benchmark a TensorRT engine with trtexec and estimate end-to-end FPS."""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark TensorRT engine with trtexec',
    )
    parser.add_argument('--engine', required=True, help='TensorRT engine path')
    parser.add_argument('--input-name', default='pixel_values', help='Input tensor name')
    parser.add_argument('--input-width', type=int, default=640, help='Input width')
    parser.add_argument('--input-height', type=int, default=384, help='Input height')
    parser.add_argument('--use-shapes', action='store_true', help='Pass --shapes for dynamic-shape engines')
    parser.add_argument('--warmup-ms', type=int, default=1000, help='trtexec warmup time in ms')
    parser.add_argument('--duration-s', type=int, default=10, help='trtexec duration in seconds')
    parser.add_argument('--pre-ms', type=float, default=10.2, help='Measured preprocess time in ms')
    parser.add_argument('--post-ms', type=float, default=9.4, help='Measured logits postprocess time in ms')
    parser.add_argument('--proc-ms', type=float, default=37.4, help='Measured process_frame time in ms')
    return parser.parse_args()


def _find_trtexec() -> str:
    candidates = [
        shutil.which('trtexec'),
        '/usr/src/tensorrt/bin/trtexec',
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError('未找到 trtexec，请确认 TensorRT 已安装')


def _extract(pattern: str, text: str):
    match = re.search(pattern, text, re.MULTILINE)
    return float(match.group(1)) if match else None


def main():
    args = parse_args()
    engine_path = Path(args.engine)
    if not engine_path.exists():
        sys.exit(f'❌ TensorRT engine 不存在: {engine_path}')

    try:
        trtexec = _find_trtexec()
    except FileNotFoundError as exc:
        sys.exit(f'❌ {exc}')

    cmd = [
        trtexec,
        f'--loadEngine={engine_path}',
        f'--warmUp={int(args.warmup_ms)}',
        f'--duration={int(args.duration_s)}',
        '--useCudaGraph',
        '--noDataTransfers',
    ]
    if args.use_shapes:
        shape = f'{args.input_name}:1x3x{args.input_height}x{args.input_width}'
        cmd.append(f'--shapes={shape}')

    print('[TensorRT] 基准命令:')
    print('  ' + ' '.join(cmd))

    result = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = result.stdout
    print(output)
    if result.returncode != 0:
        sys.exit(f'❌ trtexec 运行失败，退出码: {result.returncode}')

    throughput = _extract(r'Throughput:\s*([0-9.]+)\s*qps', output)
    latency = _extract(r'GPU Compute Time:.*mean = ([0-9.]+) ms', output)
    if latency is None:
        latency = _extract(r'Latency:.*mean = ([0-9.]+) ms', output)

    print('─' * 62)
    if throughput is not None:
        print(f'  TensorRT 吞吐: {throughput:.2f} FPS')
    if latency is not None:
        print(f'  TensorRT 推理: {latency:.2f} ms')

    if latency is not None:
        total_ms = float(args.pre_ms) + float(latency) + float(args.post_ms) + float(args.proc_ms)
        fps = 1000.0 / total_ms if total_ms > 0 else 0.0
        print(f'  估算端到端耗时: {total_ms:.2f} ms')
        print(f'  估算端到端 FPS: {fps:.2f}')
    print('─' * 62)


if __name__ == '__main__':
    main()