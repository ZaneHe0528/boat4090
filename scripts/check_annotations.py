#!/usr/bin/env python3
"""
检查数据标注质量
"""
import os
import json
from pathlib import Path
from collections import defaultdict

def check_annotation_quality(annotation_dir):
    """检查标注质量"""
    annotation_dir = Path(annotation_dir)
    
    if not annotation_dir.exists():
        print(f"❌ 目录不存在: {annotation_dir}")
        return
    
    json_files = list(annotation_dir.glob("*.json"))
    
    if not json_files:
        print(f"❌ 未找到JSON标注文件: {annotation_dir}")
        return
    
    issues = []
    stats = defaultdict(int)
    valid_labels = {'background', 'water', 'boundary'}
    
    print(f"正在检查 {len(json_files)} 个标注文件...\n")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            issues.append(f"{json_file.name}: 无法读取文件 - {e}")
            continue
        
        # 检查是否有标注
        shapes = data.get('shapes', [])
        if not shapes:
            issues.append(f"{json_file.name}: 没有标注任何区域")
            continue
        
        # 检查类别
        labels = {s['label'] for s in shapes}
        invalid_labels = labels - valid_labels
        if invalid_labels:
            issues.append(f"{json_file.name}: 包含无效类别 {invalid_labels}")
        
        # 检查是否标注完整（至少应该有水域和边界）
        if len(shapes) < 2:
            issues.append(f"{json_file.name}: 标注区域过少（仅{len(shapes)}个）")
        
        # 统计类别分布
        for label in labels:
            stats[label] += 1
        
        stats['total_files'] += 1
    
    # 打印统计信息
    print("=" * 60)
    print("标注统计")
    print("=" * 60)
    print(f"总文件数: {stats['total_files']}")
    print(f"Background标注: {stats.get('background', 0)}")
    print(f"Water标注: {stats.get('water', 0)}")
    print(f"Boundary标注: {stats.get('boundary', 0)}")
    print()
    
    # 打印问题
    if issues:
        print("=" * 60)
        print(f"发现 {len(issues)} 个问题")
        print("=" * 60)
        for issue in issues:
            print(f"  ⚠️  {issue}")
        print()
        return False
    else:
        print("✅ 所有标注检查通过！")
        print()
        return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='检查数据标注质量')
    parser.add_argument(
        '--dir',
        default='dataset/annotations',
        help='标注文件目录（默认: dataset/annotations）'
    )
    
    args = parser.parse_args()
    
    success = check_annotation_quality(args.dir)
    exit(0 if success else 1)
