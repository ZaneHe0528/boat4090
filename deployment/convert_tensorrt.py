#!/usr/bin/env python3
"""Build a TensorRT engine from ONNX using trtexec."""

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Build TensorRT engine from ONNX with trtexec',
    )
    parser.add_argument('--onnx', required=True, help='Input ONNX path')
    parser.add_argument('--engine', required=True, help='Output TensorRT engine path')
    parser.add_argument('--input-name', default='pixel_values', help='ONNX input tensor name')
    parser.add_argument('--input-width', type=int, default=640, help='Input width')
    parser.add_argument('--input-height', type=int, default=384, help='Input height')
    parser.add_argument('--use-shapes', action='store_true', help='Pass --shapes for dynamic-shape ONNX models')
    parser.add_argument('--workspace-mb', type=int, default=2048, help='Workspace size in MiB')
    parser.add_argument('--fp16', action='store_true', help='Enable FP16 build')
    parser.add_argument('--int8', action='store_true', help='Enable INT8 build')
    parser.add_argument('--skip-inference', action='store_true', help='Build engine only')
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


def main():
    args = parse_args()

    try:
        trtexec = _find_trtexec()
    except FileNotFoundError as exc:
        sys.exit(f'❌ {exc}')

    onnx_path = Path(args.onnx)
    engine_path = Path(args.engine)
    if not onnx_path.exists():
        sys.exit(f'❌ ONNX 文件不存在: {onnx_path}')
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        trtexec,
        f'--onnx={onnx_path}',
        f'--saveEngine={engine_path}',
        f'--memPoolSize=workspace:{int(args.workspace_mb)}M',
    ]
    if args.use_shapes:
        shape = f'{args.input_name}:1x3x{args.input_height}x{args.input_width}'
        cmd.append(f'--shapes={shape}')
    if args.fp16:
        cmd.append('--fp16')
    if args.int8:
        cmd.append('--int8')
    if args.skip_inference:
        cmd.append('--skipInference')

    print('[TensorRT] 执行命令:')
    print('  ' + ' '.join(shlex.quote(part) for part in cmd))

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        sys.exit(f'❌ TensorRT 引擎构建失败，退出码: {result.returncode}')

    print(f'✓ TensorRT 引擎已保存到: {engine_path}')


if __name__ == '__main__':
    main()