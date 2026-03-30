#!/usr/bin/env python3
"""TensorRT runtime helper compatible with the boat conda environment."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

SYSTEM_TRT_SITE_PACKAGES = Path('/usr/lib/python3.10/dist-packages')

if str(SYSTEM_TRT_SITE_PACKAGES) not in sys.path and SYSTEM_TRT_SITE_PACKAGES.exists():
    sys.path.insert(0, str(SYSTEM_TRT_SITE_PACKAGES))

import tensorrt as trt
import torch


def _torch_dtype_from_trt(dtype) -> torch.dtype:
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }
    if dtype not in mapping:
        raise TypeError(f'Unsupported TensorRT dtype: {dtype}')
    return mapping[dtype]


class TensorRTRunner:
    def __init__(self, engine_path: str):
        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(f'Engine not found: {self.engine_path}')

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(self.engine_path, 'rb') as handle:
            self.engine = self.runtime.deserialize_cuda_engine(handle.read())
        if self.engine is None:
            raise RuntimeError(f'Failed to deserialize TensorRT engine: {self.engine_path}')

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError('Failed to create TensorRT execution context')

        self.input_name = None
        self.output_names = []
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_names.append(name)
        if self.input_name is None or not self.output_names:
            raise RuntimeError('TensorRT engine I/O tensors are incomplete')

        self._output_buffers: Dict[str, torch.Tensor] = {}

    def infer(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not input_tensor.is_cuda:
            raise ValueError('TensorRT input tensor must live on CUDA device')
        if not input_tensor.is_contiguous():
            input_tensor = input_tensor.contiguous()

        self.context.set_input_shape(self.input_name, tuple(int(v) for v in input_tensor.shape))
        self.context.set_tensor_address(self.input_name, int(input_tensor.data_ptr()))

        outputs: Dict[str, torch.Tensor] = {}
        for name in self.output_names:
            shape = tuple(int(v) for v in self.context.get_tensor_shape(name))
            dtype = _torch_dtype_from_trt(self.engine.get_tensor_dtype(name))
            buf = self._output_buffers.get(name)
            if buf is None or tuple(buf.shape) != shape or buf.dtype != dtype:
                buf = torch.empty(shape, dtype=dtype, device=input_tensor.device)
                self._output_buffers[name] = buf
            self.context.set_tensor_address(name, int(buf.data_ptr()))
            outputs[name] = buf

        stream = torch.cuda.current_stream(device=input_tensor.device).cuda_stream
        if not self.context.execute_async_v3(stream_handle=stream):
            raise RuntimeError('TensorRT execution failed')
        return outputs

    def get_primary_output_name(self) -> str:
        return self.output_names[0]
