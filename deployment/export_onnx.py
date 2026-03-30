#!/usr/bin/env python3
"""Export the trained SegFormer checkpoint to a static-shape ONNX model."""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / 'scripts'
sys.path.insert(0, str(SCRIPTS_DIR))

from visualize_centerline import load_segformer


class _SegformerLogitsWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).logits


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export SegFormer checkpoint to ONNX',
    )
    parser.add_argument('--model', required=True, help='SegFormer .pth checkpoint path')
    parser.add_argument(
        '--output',
        default='models/segformer_river/segformer_b0_640x384.onnx',
        help='Output ONNX path',
    )
    parser.add_argument('--input-width', type=int, default=640, help='Static ONNX input width')
    parser.add_argument('--input-height', type=int, default=384, help='Static ONNX input height')
    parser.add_argument('--num-classes', type=int, default=3, help='Segmentation class count')
    parser.add_argument('--model-name', type=str, default=None, help='Optional SegFormer backbone name')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset version')
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        import onnx  # noqa: F401
    except ImportError:
        sys.exit('❌ 缺少 onnx 包，请先执行: conda run -n boat pip install onnx')

    model, _ = load_segformer(
        args.model,
        num_classes=args.num_classes,
        model_name=args.model_name,
    )
    model = model.to('cpu').eval()
    wrapper = _SegformerLogitsWrapper(model).eval()

    dummy = torch.randn(
        1,
        3,
        int(args.input_height),
        int(args.input_width),
        dtype=torch.float32,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f'[ONNX] 导出静态模型: 1x3x{args.input_height}x{args.input_width} '
        f'-> {output_path}'
    )
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(output_path),
            export_params=True,
            opset_version=int(args.opset),
            do_constant_folding=True,
            input_names=['pixel_values'],
            output_names=['logits'],
            dynamic_axes=None,
            dynamo=False,
        )

    print(f'✓ ONNX 模型已保存到: {output_path}')


if __name__ == '__main__':
    main()