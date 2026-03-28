#!/usr/bin/env python3
"""
将Labelme JSON标注转换为PNG分割掩码
"""
import json
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse
from skimage.draw import polygon as fill_polygon

def labelme_to_mask(json_path, output_path, label_map):
    """将单个Labelme JSON转换为分割掩码"""
    
    try:
        # 读取JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 读取失败 {json_path}: {e}")
        return False
    
    # 获取图像尺寸
    height = data['imageHeight']
    width = data['imageWidth']
    
    # 创建空白掩码（默认为background=0）
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # 按顺序填充多边形（后面的会覆盖前面的）
    # 建议标注顺序：background -> water -> boundary
    for shape in data['shapes']:
        label = shape['label']
        label_id = label_map.get(label, 0)
        
        points = shape['points']
        
        # 创建单个形状的掩码
        if len(points) >= 3:  # 多边形至少需要3个点
            # 提取x和y坐标
            y_coords = np.array([p[1] for p in points], dtype=np.int32)
            x_coords = np.array([p[0] for p in points], dtype=np.int32)
            
            # 使用skimage填充多边形
            rr, cc = fill_polygon(y_coords, x_coords, (height, width))
            
            # 填充到总掩码
            mask[rr, cc] = label_id
    
    # 保存为PNG（palette模式）
    mask_img = Image.fromarray(mask, mode='P')
    
    # 设置调色板（用于可视化）
    palette = [
        0, 0, 0,       # background - 黑色
        0, 128, 255,   # water - 蓝色
        255, 0, 0      # boundary - 红色
    ]
    # 填充剩余的调色板
    palette += [0] * (256 * 3 - len(palette))
    mask_img.putpalette(palette)
    
    mask_img.save(output_path)
    return True

def convert_all_annotations(input_dir, output_dir, image_dir=None):
    """批量转换所有标注"""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"❌ 输入目录不存在: {input_dir}")
        return
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 类别映射
    label_map = {
        'background': 0,
        'water': 1,
        'boundary': 2
    }
    
    # 获取所有JSON文件
    json_files = list(input_path.glob("*.json"))
    
    if not json_files:
        print(f"❌ 未找到JSON文件: {input_dir}")
        return
    
    print(f"找到 {len(json_files)} 个标注文件")
    print(f"输出目录: {output_path}")
    print()
    
    # 转换
    success_count = 0
    for json_file in tqdm(json_files, desc="转换标注"):
        output_file = output_path / f"{json_file.stem}.png"
        
        if labelme_to_mask(json_file, output_file, label_map):
            success_count += 1
    
    print()
    print(f"✅ 转换完成！")
    print(f"   成功: {success_count}/{len(json_files)}")
    print(f"   失败: {len(json_files) - success_count}")
    
    # 如果提供了图像目录，复制对应的图像
    if image_dir:
        image_path = Path(image_dir)
        image_output = output_path.parent / 'images'
        image_output.mkdir(parents=True, exist_ok=True)
        
        print(f"\n复制对应图像到: {image_output}")
        from shutil import copy2
        
        copied = 0
        for json_file in json_files:
            # 尝试多种图像格式
            for ext in ['.jpg', '.png', '.jpeg']:
                img_file = image_path / f"{json_file.stem}{ext}"
                if img_file.exists():
                    copy2(img_file, image_output / img_file.name)
                    copied += 1
                    break
        
        print(f"   复制了 {copied} 张图像")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将Labelme标注转换为PNG掩码')
    parser.add_argument(
        '--input',
        default='dataset/annotations',
        help='Labelme JSON文件目录'
    )
    parser.add_argument(
        '--output',
        default='dataset/masks',
        help='输出掩码目录'
    )
    parser.add_argument(
        '--images',
        default='dataset',
        help='原始图像目录（可选，用于同时复制图像）'
    )
    
    args = parser.parse_args()
    
    convert_all_annotations(args.input, args.output, args.images)
