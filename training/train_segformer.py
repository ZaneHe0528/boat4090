import sys
import os

# 若未设置 CUDA_VISIBLE_DEVICES，默认使用 GPU 0，避免 PyTorch 懒加载时设备枚举失败
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import SegformerForSemanticSegmentation
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
try:
    import wandb
except ImportError:
    wandb = None

class RiverDataset(Dataset):
    """河道分割数据集"""
    def __init__(self, images_dir, masks_dir, transform=None, dilate_boundary_px=0, use_training_augmentation=False):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.dilate_boundary_px = dilate_boundary_px
        self.use_training_augmentation = use_training_augmentation
        all_images = sorted(list(self.images_dir.glob("*.jpg")))
        self.image_files = [p for p in all_images if (self.masks_dir / f"{p.stem}.png").exists()]
        skipped = len(all_images) - len(self.image_files)
        if skipped > 0:
            print(f"[Dataset] 跳过 {skipped} 张无对应掩码的图像")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        from scipy import ndimage
        
        # 延迟初始化 augmentation 以避免启动时 cv2 导入错误
        if self.transform is None:
            from augmentation import get_training_augmentation, get_validation_augmentation
            if self.use_training_augmentation:
                self.transform = get_training_augmentation()
            else:
                self.transform = get_validation_augmentation()
        
        # 读取图像
        img_path = self.image_files[idx]
        image = np.array(Image.open(img_path).convert('RGB'))
        
        # 读取掩码
        mask_path = self.masks_dir / f"{img_path.stem}.png"
        mask = np.array(Image.open(mask_path))
        
        # 边界膨胀：补偿 SegFormer 1/4 降采样对细线 boundary 的洗失
        if self.dilate_boundary_px > 0:
            boundary_bin = (mask == 2).astype(np.uint8)
            dilated = ndimage.binary_dilation(boundary_bin, iterations=self.dilate_boundary_px)
            mask[dilated == 1] = 2
        
        # 数据增强
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # 转换为Tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        
        return image, mask

class DiceLoss(nn.Module):
    """Dice Loss：对每个类别按比例优化，天然缓解类别不平衡。"""
    def __init__(self, num_classes=3, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, inputs, targets):
        probs = torch.softmax(inputs, dim=1)
        dice = 0.0
        for cls in range(self.num_classes):
            pred = probs[:, cls]
            tgt = (targets == cls).float()
            intersection = (pred * tgt).sum()
            dice += (2.0 * intersection + self.smooth) / (pred.sum() + tgt.sum() + self.smooth)
        return 1.0 - dice / self.num_classes


class LovaszSoftmaxLoss(nn.Module):
    """Lovász-Softmax Loss：直接优化 IoU 的凸代理损失，对 boundary 等稀疏类特别有效。
    
    参考: Berman et al. "The Lovász-Softmax loss: A tractable surrogate for 
    the optimization of the intersection-over-union measure in neural networks" (CVPR 2018)
    """
    def __init__(self, num_classes=3, per_image=False):
        super().__init__()
        self.num_classes = num_classes
        self.per_image = per_image
    
    def forward(self, inputs, targets):
        probas = torch.softmax(inputs, dim=1)
        if self.per_image:
            loss = sum(self._lovasz_softmax_flat(
                probas[i:i+1].reshape(self.num_classes, -1).transpose(0, 1),
                targets[i:i+1].reshape(-1)
            ) for i in range(inputs.shape[0])) / inputs.shape[0]
        else:
            # (B, C, H, W) -> (B*H*W, C)
            vprobas = probas.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            vtargets = targets.reshape(-1)
            loss = self._lovasz_softmax_flat(vprobas, vtargets)
        return loss
    
    def _lovasz_softmax_flat(self, probas, labels):
        """Lovász-Softmax 在展平后的张量上的实现"""
        C = probas.shape[1]
        losses = []
        for c in range(C):
            fg = (labels == c).float()  # 前景 mask
            if fg.sum() == 0 and c != 0:
                continue
            errors = (fg - probas[:, c]).abs()
            errors_sorted, perm = torch.sort(errors, descending=True)
            fg_sorted = fg[perm]
            losses.append(torch.dot(errors_sorted, self._lovasz_grad(fg_sorted)))
        return sum(losses) / max(len(losses), 1)
    
    @staticmethod
    def _lovasz_grad(gt_sorted):
        """计算 Lovász 扩展的梯度"""
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        if p > 1:
            jaccard[1:] = jaccard[1:] - jaccard[:-1]
        return jaccard


class DiceFocalLoss(nn.Module):
    """Focal Loss + Dice Loss + Lovász-Softmax + Boundary Dice Loss 组合：
    Focal Loss 聚焦难样本，Dice Loss 保证每类均衡优化，
    Lovász-Softmax 直接优化 IoU，额外 Boundary Dice Loss 强化边界。"""
    def __init__(self, alpha=None, gamma=2.0, dice_weight=0.5, focal_weight=0.5, 
                 num_classes=3, boundary_dice_weight=0.3, lovasz_weight=0.3):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice = DiceLoss(num_classes=num_classes)
        self.lovasz = LovaszSoftmaxLoss(num_classes=num_classes)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_dice_weight = boundary_dice_weight
        self.lovasz_weight = lovasz_weight

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma * ce).mean()
        dice = self.dice(inputs, targets)
        
        # Lovász-Softmax: 直接优化 IoU
        lovasz = self.lovasz(inputs, targets)
        
        # 额外的 boundary-specific Dice Loss
        boundary_dice = self._compute_boundary_dice(inputs, targets)
        
        return (self.focal_weight * focal + self.dice_weight * dice 
                + self.lovasz_weight * lovasz + self.boundary_dice_weight * boundary_dice)
    
    def _compute_boundary_dice(self, inputs, targets, boundary_class=2, smooth=1.0):
        """计算只针对 boundary 类的 Dice Loss"""
        probs = torch.softmax(inputs, dim=1)
        boundary_prob = probs[:, boundary_class]
        boundary_target = (targets == boundary_class).float()
        
        intersection = (boundary_prob * boundary_target).sum()
        union = boundary_prob.sum() + boundary_target.sum()
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice


class Trainer:
    """训练器"""

    @staticmethod
    def _compute_sample_weights(dataset, boundary_class=2):
        """为每张训练图计算采样权重：boundary 像素越多，被采到的概率越高。"""
        weights = []
        for img_path in dataset.image_files:
            mask_path = dataset.masks_dir / f"{img_path.stem}.png"
            mask = np.array(Image.open(mask_path))
            boundary_ratio = (mask == boundary_class).mean()
            # 基础权重 1.0，boundary 丰富的图像额外加权（系数可调）
            weights.append(1.0 + boundary_ratio * 20.0)
        return torch.DoubleTensor(weights)

    @staticmethod
    def _get_device():
        """健壮的 CUDA 设备检测，防止驱动异常导致崩溃"""
        if not torch.cuda.is_available():
            return torch.device('cpu')
        try:
            # 真正触发 CUDA 上下文以验证设备可用
            torch.cuda.init()
            _ = torch.zeros(1).cuda()
            return torch.device('cuda')
        except Exception as e:
            print(f"[警告] CUDA 初始化失败，回退到 CPU: {e}")
            return torch.device('cpu')

    def __init__(self, config):
        self.config = config
        # 健壮的设备检测：验证CUDA实际可用
        self.device = self._get_device()
        print(f"[Trainer] 使用设备: {self.device}")
        
        # 初始化模型
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            config['model_name'],
            num_labels=config['num_classes'],
            ignore_mismatched_sizes=True
        ).to(self.device)
        
        # 损失函数：Focal + Dice + Lovász + Boundary Dice 组合，有效应对 boundary 严重不平衡
        self.criterion = DiceFocalLoss(
            alpha=torch.tensor(config['class_weights'], dtype=torch.float32).to(self.device),
            gamma=config.get('focal_gamma', 2.0),
            dice_weight=config.get('dice_weight', 0.5),
            focal_weight=config.get('focal_weight', 0.5),
            num_classes=config['num_classes'],
            boundary_dice_weight=config.get('boundary_dice_weight', 0.3),
            lovasz_weight=config.get('lovasz_weight', 0.3)
        )
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs']
        )
        
        # 数据加载器（延迟导入 augmentation 避免 cv2 导入问题）
        self.train_dataset = RiverDataset(
            config['train_images'],
            config['train_masks'],
            transform=None,
            dilate_boundary_px=config.get('boundary_dilate_px', 0),
            use_training_augmentation=True
        )
        self.val_dataset = RiverDataset(
            config['val_images'],
            config['val_masks'],
            transform=None,
            dilate_boundary_px=config.get('boundary_dilate_px', 0),
            use_training_augmentation=False
        )
        
        use_cuda = (self.device.type == 'cuda')
        # 边界感知过采样：boundary 丰富的图像被更频繁采样
        sample_weights = self._compute_sample_weights(self.train_dataset, boundary_class=2)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['batch_size'],
            sampler=sampler,
            num_workers=0,  # 禁用多进程以避免 albumentations/cv2 导入问题
            pin_memory=use_cuda
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=0,  # 禁用多进程以避免 albumentations/cv2 导入问题
            pin_memory=use_cuda
        )
        
        # 初始化wandb（可选）
        if config.get('use_wandb') and wandb is not None:
            wandb.init(project="river-lane-pilot", config=config)
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # 前向传播
            outputs = self.model(pixel_values=images)
            logits = outputs.logits
            
            # 调整logits尺寸以匹配mask
            logits = nn.functional.interpolate(
                logits,
                size=masks.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            
            # 计算损失
            loss = self.criterion(logits, masks)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_masks = []
        
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(pixel_values=images)
                logits = outputs.logits
                
                logits = nn.functional.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
                
                loss = self.criterion(logits, masks)
                total_loss += loss.item()
                
                # 计算准确率和收集预测用于 IoU
                preds = torch.argmax(logits, dim=1)
                correct += (preds == masks).sum().item()
                total += masks.numel()
                
                all_preds.append(preds.cpu().numpy())
                all_masks.append(masks.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        # 计算 per-class IoU
        all_preds_flat = np.concatenate(all_preds, axis=0).flatten()
        all_masks_flat = np.concatenate(all_masks, axis=0).flatten()
        
        class_ious = []
        for cls in range(self.config['num_classes']):
            pred_cls = all_preds_flat == cls
            mask_cls = all_masks_flat == cls
            intersection = (pred_cls & mask_cls).sum()
            union = (pred_cls | mask_cls).sum()
            if union == 0:
                iou = float('nan')
            else:
                iou = float(intersection) / float(union)
            class_ious.append(iou)
        
        boundary_iou = class_ious[2] if len(class_ious) > 2 else float('nan')
        
        return avg_loss, accuracy, boundary_iou, class_ious
    
    def train(self):
        """完整训练流程"""
        best_boundary_iou = float('-inf')
        best_epoch = 0
        patience = 15  # early stopping 耐心值
        epochs_without_improvement = 0
        
        for epoch in range(1, self.config['epochs'] + 1):
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc, boundary_iou, class_ious = self.validate()
            
            # 更新学习率
            self.scheduler.step()
            
            # 打印信息
            print(f"Epoch {epoch}/{self.config['epochs']}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Acc: {val_acc:.4f}")
            print(f"  Class IoUs: bg={class_ious[0]:.4f}, water={class_ious[1]:.4f}, boundary={boundary_iou:.4f}")
            
            # 记录到wandb
            if self.config.get('use_wandb') and wandb is not None:
                wandb.log({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'boundary_iou': boundary_iou,
                    'lr': self.optimizer.param_groups[0]['lr']
                })
            
            # 按 boundary IoU 选择最佳模型（而非按总体 loss）
            if boundary_iou > best_boundary_iou and not np.isnan(boundary_iou):
                best_boundary_iou = boundary_iou
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(
                    {
                        'model_state_dict': self.model.state_dict(),
                        'model_name': self.config['model_name'],
                    },
                    Path(self.config['output_dir']) / 'best_model.pth'
                )
                print(f"  ✓ 保存最佳模型 (boundary IoU={boundary_iou:.4f})")
            else:
                epochs_without_improvement += 1
            
            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"\n早停：boundary IoU 在 {patience} 个 epoch 内无改进，停止训练")
                print(f"最佳模型在 epoch {best_epoch}，boundary IoU={best_boundary_iou:.4f}")
                break
            
            # 定期保存检查点
            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_loss,
                }, Path(self.config['output_dir']) / f'checkpoint_epoch_{epoch}.pth')

if __name__ == "__main__":
    config = {
        'model_name': 'nvidia/segformer-b2-finetuned-ade-512-512',  # 升级到 B2（25M参数）提升 boundary 特征表达
        'num_classes': 3,
        'class_weights': [1.0, 2.0, 30.0],  # background, water, boundary - 大幅提高boundary权重
        'focal_gamma': 2.0,
        'dice_weight': 0.4,
        'focal_weight': 0.3,
        'lovasz_weight': 0.3,          # Lovász-Softmax 直接优化 IoU
        'boundary_dice_weight': 0.3,
        'boundary_dilate_px': 3,       # 增大膨胀到 3px 缓解细线匹配困难
        'learning_rate': 6e-5,         # B2 参数更多，使用稍低学习率
        'weight_decay': 0.01,
        'batch_size': 8,               # B2 更大，降低 batch size
        'epochs': 150,
        'train_images': 'dataset_final/images/train',
        'train_masks': 'dataset_final/masks/train',
        'val_images': 'dataset_final/images/val',
        'val_masks': 'dataset_final/masks/val',
        'output_dir': 'models/segformer_river',
        'use_wandb': False
    }
    
    # 创建输出目录
    Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
    
    # 开始训练
    trainer = Trainer(config)
    trainer.train()