import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def build_engine(onnx_path, engine_path, fp16=True):
    """构建TensorRT引擎"""
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # 解析ONNX
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # 配置
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("启用FP16加速")
    
    # 构建引擎
    print("正在构建TensorRT引擎（可能需要几分钟）...")
    engine = builder.build_engine(network, config)
    
    # 保存引擎
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    print(f"✓ TensorRT引擎已保存到: {engine_path}")
    return engine

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', required=True)
    parser.add_argument('--engine', required=True)
    parser.add_argument('--fp16', action='store_true')
    args = parser.parse_args()
    
    build_engine(args.onnx, args.engine, args.fp16)