import torch
from transformers import SegformerForSemanticSegmentation

def export_to_onnx(model_path, output_path, input_size=(1, 3, 512, 512)):
    """导出为ONNX格式"""
    
    # 加载模型
    model = SegformerForSemanticSegmentation.from_pretrained(
        'nvidia/segformer-b0-finetuned-ade-512-512',
        num_labels=3
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 创建示例输入
    dummy_input = torch.randn(*input_size)
    
    # 导出ONNX
    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )
    
    print(f"✓ ONNX模型已保存到: {output_path}")

if __name__ == "__main__":
    export_to_onnx(
        model_path='models/segformer_river/best_model.pth',
        output_path='models/segformer_river.onnx',
        input_size=(1, 3, 512, 512)
    )