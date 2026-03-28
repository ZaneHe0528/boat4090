import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation


CLASS_NAMES = ["background", "water", "boundary"]

# 若未显式设置，默认绑定到可见设备 0，避免历史 ckpt 设备索引造成歧义
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class RiverDataset(Dataset):
    """和训练脚本一致的数据读取方式，确保评估口径一致。"""

    def __init__(self, images_dir, masks_dir, dilate_boundary_px=0):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.dilate_boundary_px = dilate_boundary_px
        all_images = sorted(list(self.images_dir.glob("*.jpg")))
        self.image_files = [p for p in all_images if (self.masks_dir / f"{p.stem}.png").exists()]
        skipped = len(all_images) - len(self.image_files)
        if skipped > 0:
            print(f"[Dataset] 跳过 {skipped} 张无对应掩码的图像")

        if not self.image_files:
            raise FileNotFoundError(f"未在 {self.images_dir} 找到 .jpg 图像")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        from scipy import ndimage
        
        img_path = self.image_files[idx]
        mask_path = self.masks_dir / f"{img_path.stem}.png"

        if not mask_path.exists():
            raise FileNotFoundError(f"找不到对应掩码: {mask_path}")

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))

        # 与训练脚本保持一致的 boundary 膨胀
        if self.dilate_boundary_px > 0:
            boundary_bin = (mask == 2).astype(np.uint8)
            dilated = ndimage.binary_dilation(boundary_bin, iterations=self.dilate_boundary_px)
            mask[dilated == 1] = 2

        # 手动实现 Resize 和 Normalize（与训练保持一致：384x640，保持 16:9 宽高比）
        image = Image.fromarray(image).resize((640, 384), Image.Resampling.BILINEAR)
        mask = Image.fromarray(mask, mode='L').resize((640, 384), Image.Resampling.NEAREST)
        
        image = np.array(image, dtype=np.float32)
        mask = np.array(mask, dtype=np.int64)
        
        # ImageNet 归一化
        image = image / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        return image, mask


def calculate_miou(preds, masks, num_classes):
    """计算整体 mIoU 和各类别 IoU。"""
    ious = []
    for cls in range(num_classes):
        pred_cls = preds == cls
        mask_cls = masks == cls

        intersection = (pred_cls & mask_cls).sum()
        union = (pred_cls | mask_cls).sum()

        if union == 0:
            iou = float("nan")
        else:
            iou = float(intersection) / float(union)

        ious.append(iou)

    return float(np.nanmean(ious)), ious


def _row_center_from_mask(mask_row):
    """按行估计可通行区域中心点。"""
    cols = np.where(mask_row)[0]
    if cols.size < 2:
        return None
    return 0.5 * (float(cols[0]) + float(cols[-1]))


def calculate_centerline_error(preds_2d, masks_2d, water_class=1):
    """计算中线偏差（像素和归一化）及覆盖率。"""
    per_row_errors_px = []
    used_rows = 0
    gt_rows = 0

    n, h, w = preds_2d.shape
    for i in range(n):
        pred_water = preds_2d[i] == water_class
        gt_water = masks_2d[i] == water_class

        for r in range(h):
            gt_center = _row_center_from_mask(gt_water[r])
            if gt_center is not None:
                gt_rows += 1

            pred_center = _row_center_from_mask(pred_water[r])
            if gt_center is None or pred_center is None:
                continue

            used_rows += 1
            per_row_errors_px.append(abs(pred_center - gt_center))

    if not per_row_errors_px:
        return {
            "centerline_mae_px": float("nan"),
            "centerline_mae_norm": float("nan"),
            "centerline_p95_px": float("nan"),
            "centerline_coverage": 0.0,
        }

    err_px = np.array(per_row_errors_px, dtype=np.float32)
    return {
        "centerline_mae_px": float(np.mean(err_px)),
        "centerline_mae_norm": float(np.mean(err_px) / max(w, 1)),
        "centerline_p95_px": float(np.percentile(err_px, 95)),
        "centerline_coverage": float(used_rows / max(gt_rows, 1)),
    }


def load_model(model_name, num_classes, model_path, device):
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )

    # 先统一映射到 CPU，再整体迁移到目标设备，规避 ckpt 内保存的 cuda:N 索引问题
    ckpt = torch.load(model_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    return model


def evaluate_model(model, dataloader, device, num_classes=3, save_dir="models/segformer_river"):
    """全面评估模型并保存可视化结果。"""
    model.eval()

    all_preds = []
    all_masks = []

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(pixel_values=images)
            logits = outputs.logits

            logits = nn.functional.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_masks.append(masks.numpy())

    all_preds_2d = np.concatenate(all_preds, axis=0)
    all_masks_2d = np.concatenate(all_masks, axis=0)

    flat_preds = all_preds_2d.flatten()
    flat_masks = all_masks_2d.flatten()

    miou, class_ious = calculate_miou(flat_preds, flat_masks, num_classes)
    cm = confusion_matrix(flat_masks, flat_preds, labels=list(range(num_classes)))

    report = classification_report(
        flat_masks,
        flat_preds,
        labels=list(range(num_classes)),
        target_names=CLASS_NAMES[:num_classes],
        digits=4,
    )

    centerline = calculate_centerline_error(all_preds_2d, all_masks_2d, water_class=1)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES[:num_classes], yticklabels=CLASS_NAMES[:num_classes])
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = save_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()

    print("=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"mIoU: {miou:.4f}")
    print("类别 IoU:")
    for idx, iou in enumerate(class_ious):
        print(f"  {CLASS_NAMES[idx]}: {iou:.4f}")

    print("\n中线相关指标:")
    print(f"  Centerline MAE (px): {centerline['centerline_mae_px']:.2f}")
    print(f"  Centerline MAE (norm): {centerline['centerline_mae_norm']:.4f}")
    print(f"  Centerline P95 (px): {centerline['centerline_p95_px']:.2f}")
    print(f"  Centerline Coverage: {centerline['centerline_coverage']:.4f}")

    print("\n分类报告:")
    print(report)
    print(f"混淆矩阵已保存: {cm_path}")

    return {
        "miou": miou,
        "class_ious": class_ious,
        "centerline": centerline,
        "classification_report": report,
        "confusion_matrix": cm,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="评估 SegFormer 河道分割模型")
    parser.add_argument("--model-name", default="nvidia/segformer-b2-finetuned-ade-512-512")
    parser.add_argument("--model-path", default="models/segformer_river/best_model.pth")
    parser.add_argument("--images-dir", default="dataset_final/images/val")
    parser.add_argument("--masks-dir", default="dataset_final/masks/val")
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-dir", default="models/segformer_river")
    parser.add_argument("--dilate-boundary-px", type=int, default=3,
                        help="GT掩码boundary膨胀半径，必须与训练时boundary_dilate_px一致（新训练默认3）")
    return parser.parse_args()


def get_device():
    """健壮设备检测：CUDA 不可用或初始化失败时自动回退 CPU。"""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        torch.cuda.init()
        _ = torch.zeros(1).cuda()
        return torch.device("cuda")
    except Exception as e:
        print(f"[警告] CUDA 初始化失败，回退到 CPU: {e}")
        return torch.device("cpu")


def main():
    args = parse_args()

    device = get_device()
    print(f"使用设备: {device}")

    dataset = RiverDataset(args.images_dir, args.masks_dir,
                           dilate_boundary_px=args.dilate_boundary_px)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # 禁用多进程以避免 numpy/cv2 兼容性问题
        pin_memory=(device.type == "cuda"),
    )

    model = load_model(
        model_name=args.model_name,
        num_classes=args.num_classes,
        model_path=args.model_path,
        device=device,
    )

    evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        num_classes=args.num_classes,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()