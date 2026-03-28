"""
SegFormer模型封装 - SegFormer Model Wrapper
负责语义分割模型的加载、推理和后处理
"""

import cv2
import numpy as np
import onnxruntime as ort
from typing import Tuple, Optional, Dict, Any
import time
from pathlib import Path

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

from ..utils.logger import get_logger
from ..utils.config_loader import get_config


class SegFormerModel:
    """SegFormer语义分割模型包装器"""
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 use_tensorrt: bool = True):
        """
        初始化SegFormer模型
        
        Args:
            config: 模型配置字典
            use_tensorrt: 是否使用TensorRT加速
        """
        self.logger = get_logger("SegFormer")
        self.config = config or get_config().get_segmentation_config()
        
        # 模型参数
        self.model_path = self.config.get("model_path", "models/segformer_river_lane.onnx")
        self.engine_path = self.config.get("engine_path", "models/segformer_river_lane.engine")
        self.input_height = self.config.get("input_height", 512)
        self.input_width = self.config.get("input_width", 1024)
        self.num_classes = self.config.get("num_classes", 3)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.5)
        
        # TensorRT配置
        self.use_tensorrt = use_tensorrt and TRT_AVAILABLE
        if self.use_tensorrt and not TRT_AVAILABLE:
            self.logger.warning("TensorRT不可用，回退到ONNX Runtime")
            self.use_tensorrt = False
        
        # 模型会话
        self.session = None
        self.trt_context = None
        self.trt_engine = None
        
        # 性能统计
        self.inference_times = []
        self.preprocess_times = []
        self.postprocess_times = []
        
        # 初始化模型
        self._load_model()
    
    def _load_model(self) -> None:
        """加载模型"""
        try:
            if self.use_tensorrt:
                self._load_tensorrt_model()
            else:
                self._load_onnx_model()
            
            self.logger.info(f"✓ 模型加载成功: {'TensorRT' if self.use_tensorrt else 'ONNX'}")
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            # 尝试回退到ONNX
            if self.use_tensorrt:
                self.logger.info("尝试回退到ONNX Runtime...")
                self.use_tensorrt = False
                self._load_onnx_model()
    
    def _load_onnx_model(self) -> None:
        """加载ONNX模型"""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"ONNX模型文件不存在: {self.model_path}")
        
        # 配置ONNX Runtime
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=session_options,
            providers=providers
        )
        
        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        self.logger.info(f"ONNX模型输入: {self.input_name}")
        self.logger.info(f"ONNX模型输出: {self.output_name}")
    
    def _load_tensorrt_model(self) -> None:
        """加载TensorRT模型"""
        if not Path(self.engine_path).exists():
            raise FileNotFoundError(f"TensorRT引擎文件不存在: {self.engine_path}")
        
        # 加载TensorRT引擎
        with open(self.engine_path, 'rb') as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self.trt_engine = runtime.deserialize_cuda_engine(f.read())
        
        # 创建执行上下文
        self.trt_context = self.trt_engine.create_execution_context()
        
        # 分配GPU内存
        self._allocate_buffers()
    
    def _allocate_buffers(self) -> None:
        """为TensorRT分配缓冲区"""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for binding in self.trt_engine:
            size = trt.volume(self.trt_engine.get_binding_shape(binding)) * \
                   self.trt_engine.max_batch_size
            dtype = trt.nptype(self.trt_engine.get_binding_dtype(binding))
            
            # 分配主机和设备内存
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.trt_engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            预处理后的张量
        """
        start_time = time.time()
        
        # 转换为RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 调整尺寸
        resized = cv2.resize(rgb_image, (self.input_width, self.input_height))
        
        # 归一化到[0,1]并转换为float32
        normalized = resized.astype(np.float32) / 255.0
        
        # 标准化 (ImageNet均值和标准差)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std
        
        # 转换维度 (H, W, C) -> (1, C, H, W)
        tensor = np.transpose(normalized, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        
        self.preprocess_times.append(time.time() - start_time)
        
        return tensor
    
    def postprocess(self, output: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
        """
        后处理输出结果
        
        Args:
            output: 模型输出张量
            original_shape: 原始图像尺寸 (height, width)
            
        Returns:
            分割掩码 (单通道，值为类别ID)
        """
        start_time = time.time()
        
        # 如果输出是概率分布，取argmax获取类别
        if output.shape[1] > 1:  # 多类别输出
            mask = np.argmax(output, axis=1)
        else:  # 二值分割
            mask = (output > self.confidence_threshold).astype(np.uint8)
        
        # 去除批次维度
        mask = mask.squeeze(0)
        
        # 调整回原始尺寸
        original_height, original_width = original_shape
        mask_resized = cv2.resize(
            mask.astype(np.uint8), 
            (original_width, original_height), 
            interpolation=cv2.INTER_NEAREST
        )
        
        self.postprocess_times.append(time.time() - start_time)
        
        return mask_resized
    
    def infer_onnx(self, input_tensor: np.ndarray) -> np.ndarray:
        """ONNX推理"""
        start_time = time.time()
        
        output = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )[0]
        
        self.inference_times.append(time.time() - start_time)
        return output
    
    def infer_tensorrt(self, input_tensor: np.ndarray) -> np.ndarray:
        """TensorRT推理"""
        start_time = time.time()
        
        # 复制输入数据到GPU
        np.copyto(self.inputs[0]['host'], input_tensor.ravel())
        cuda.memcpy_htod(self.inputs[0]['device'], self.inputs[0]['host'])
        
        # 执行推理
        self.trt_context.execute_v2(bindings=self.bindings)
        
        # 复制输出数据到CPU
        cuda.memcpy_dtoh(self.outputs[0]['host'], self.outputs[0]['device'])
        
        # 重塑输出张量
        output_shape = (1, self.num_classes, self.input_height, self.input_width)
        output = self.outputs[0]['host'].reshape(output_shape)
        
        self.inference_times.append(time.time() - start_time)
        return output
    
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        执行完整的预测流程
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            分割掩码 (单通道，值为类别ID)
        """
        if self.session is None and self.trt_context is None:
            raise RuntimeError("模型未加载")
        
        original_shape = image.shape[:2]
        
        # 预处理
        input_tensor = self.preprocess(image)
        
        # 推理
        if self.use_tensorrt:
            output = self.infer_tensorrt(input_tensor)
        else:
            output = self.infer_onnx(input_tensor)
        
        # 后处理
        mask = self.postprocess(output, original_shape)
        
        return mask
    
    def get_performance_stats(self) -> Dict[str, float]:
        """获取性能统计信息"""
        if not self.inference_times:
            return {}
        
        return {
            'avg_preprocess_time': np.mean(self.preprocess_times),
            'avg_inference_time': np.mean(self.inference_times),
            'avg_postprocess_time': np.mean(self.postprocess_times),
            'avg_total_time': np.mean(self.preprocess_times) + 
                             np.mean(self.inference_times) + 
                             np.mean(self.postprocess_times),
            'avg_fps': 1.0 / (np.mean(self.preprocess_times) + 
                             np.mean(self.inference_times) + 
                             np.mean(self.postprocess_times))
        }
    
    def reset_stats(self) -> None:
        """重置性能统计"""
        self.inference_times.clear()
        self.preprocess_times.clear()
        self.postprocess_times.clear()
    
    def __del__(self):
        """清理资源"""
        if self.session:
            del self.session
        if self.trt_context:
            self.trt_context.destroy()
        if self.trt_engine:
            del self.trt_engine


if __name__ == "__main__":
    # 测试SegFormer模型
    model = SegFormerModel(use_tensorrt=False)
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    try:
        # 执行预测
        mask = model.predict(test_image)
        print(f"预测完成，输出掩码形状: {mask.shape}")
        print(f"掩码中的唯一值: {np.unique(mask)}")
        
        # 显示性能统计
        stats = model.get_performance_stats()
        for key, value in stats.items():
            print(f"{key}: {value:.4f}")
            
    except Exception as e:
        print(f"模型测试失败: {e}")