#!/usr/bin/env python3
"""
划分数据集为训练集、验证集、测试集
"""
import os
import random
from pathlib import Path
from shutil import copy2
from tqdm import tqdm
import argparse

def split_dataset(image_dir, mask_dir, output_dir, split_ratio=(0.7, 0.15, 0.15), seed=42):
    """
    划分数据集
    
    Args:
        image_dir: 图像目录
        mask_dir: 掩码目录
        output_dir: 输出目录
        split_ratio: (训练, 验证, 测试)比例
        seed: 随机种子
    """
    
    train_ratio, val_ratio, test_ratio = split_ratio
    assert abs(sum(split_ratio) - 1.0) < 1e-6, f"比例之和必须为1，当前为{sum(split_ratio)}"
    
    # 设置随机种子
    random.seed(seed)
    
    # 获取所有图像文件
    image_path = Path(image_dir)
    mask_path = Path(mask_dir)
    
    image_files = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        image_files.extend(list(image_path.glob(ext)))
    
    # 过滤出有对应掩码的图像
    valid_files = []
    for img_file in image_files:
        mask_file = mask_path / f"{img_file.stem}.png"
        if mask_file.exists():
            valid_files.append(img_file)
        else:
            print(f"⚠️  跳过（无掩码）: {img_file.name}")
    
    if not valid_files:
        print("❌ 未找到有效的图像-掩码对")
        return
    
    # 随机打乱
    random.shuffle(valid_files)
    
    # 计算划分点
    n_total = len(valid_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # 划分
    train_files = valid_files[:n_train]
    val_files = valid_files[n_train:n_train+n_val]
    test_files = valid_files[n_train+n_val:]
    
    print("=" * 60)
    print("数据集划分")
    print("=" * 60)
    print(f"总数: {n_total}")
    print(f"训练集: {len(train_files)} ({len(train_files)/n_total*100:.1f}%)")
    print(f"验证集: {len(val_files)} ({len(val_files)/n_total*100:.1f}%)")
    print(f"测试集: {len(test_files)} ({len(test_files)/n_total*100:.1f}%)")
    print()
    
    # 创建目录结构
    output_path = Path(output_dir)
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'masks' / split).mkdir(parents=True, exist_ok=True)
    
    # 复制文件
    def copy_split(files, split_name):
        print(f"复制 {split_name} 集...")
        for img_file in tqdm(files):
            # 复制图像
            dst_img = output_path / 'images' / split_name / img_file.name
            copy2(img_file, dst_img)
            
            # 复制掩码
            mask_file = mask_path / f"{img_file.stem}.png"
            dst_mask = output_path / 'masks' / split_name / f"{img_file.stem}.png"
            copy2(mask_file, dst_mask)
    
    copy_split(train_files, 'train')
    copy_split(val_files, 'val')
    copy_split(test_files, 'test')
    
    # 生成文件列表
    splits_dir = output_path / 'splits'
    splits_dir.mkdir(exist_ok=True)
    
    def write_file_list(files, list_path):
        with open(list_path, 'w') as f:
            for img_file in files:
                f.write(f"{img_file.stem}\n")
    
    write_file_list(train_files, splits_dir / 'train.txt')
    write_file_list(val_files, splits_dir / 'val.txt')
    write_file_list(test_files, splits_dir / 'test.txt')
    
    print()
    print("✅ 数据集划分完成！")
    print(f"   输出目录: {output_path}")
    print(f"   文件列表: {splits_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='划分数据集')
    parser.add_argument(
        '--images',
        default='dataset/images',
        help='图像目录'
    )
    parser.add_argument(
        '--masks',
        default='dataset/masks',
        help='掩码目录'
    )
    parser.add_argument(
        '--output',
        default='dataset_final',
        help='输出目录'
    )
    parser.add_argument(
        '--train',
        type=float,
        default=0.7,
        help='训练集比例（默认0.7）'
    )
    parser.add_argument(
        '--val',
        type=float,
        default=0.15,
        help='验证集比例（默认0.15）'
    )
    parser.add_argument(
        '--test',
        type=float,
        default=0.15,
        help='测试集比例（默认0.15）'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（默认42）'
    )
    
    args = parser.parse_args()
    
    split_dataset(
        image_dir=args.images,
        mask_dir=args.masks,
        output_dir=args.output,
        split_ratio=(args.train, args.val, args.test),
        seed=args.seed
    )
