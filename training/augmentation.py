import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def get_training_augmentation(height=384, width=640):
    """训练集数据增强（保持 16:9 宽高比，避免 albumentations cv2 依赖）
    
    包含几何增强（旋转、缩放、弹性变形）以提升 boundary 学习效果。
    """
    class TrainingAugmentation:
        def __init__(self, h, w):
            self.h = h
            self.w = w
        
        def __call__(self, image, mask):
            # 1. Resize
            image = Image.fromarray(image).resize((self.w, self.h), Image.Resampling.BILINEAR)
            mask = Image.fromarray(mask, mode='L').resize((self.w, self.h), Image.Resampling.NEAREST)
            
            image = np.array(image, dtype=np.float32)
            mask = np.array(mask, dtype=np.int64)
            
            # 2. 随机水平翻转
            if np.random.rand() < 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
            
            # 3. 随机旋转 (-10°~+10°)：增加 boundary 角度多样性
            if np.random.rand() < 0.5:
                angle = np.random.uniform(-10, 10)
                image, mask = self._rotate(image, mask, angle)
            
            # 4. 随机缩放裁剪 (0.8~1.2)：让模型看到不同粗细的 boundary
            if np.random.rand() < 0.5:
                scale = np.random.uniform(0.8, 1.2)
                image, mask = self._random_scale_crop(image, mask, scale)
            
            # 5. 随机亮度对比度调整
            if np.random.rand() < 0.5:
                brightness = np.random.uniform(0.7, 1.3)
                image = image * brightness
                image = np.clip(image, 0, 255)
            
            # 6. 随机对比度
            if np.random.rand() < 0.3:
                contrast = np.random.uniform(0.8, 1.2)
                mean_val = image.mean()
                image = (image - mean_val) * contrast + mean_val
                image = np.clip(image, 0, 255)
            
            # 7. 随机色调饱和度调整
            if np.random.rand() < 0.4:
                hsv = self._rgb2hsv(image / 255.0)
                hsv[:, :, 0] = (hsv[:, :, 0] + np.random.uniform(-0.04, 0.04)) % 1.0
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(0.8, 1.2), 0, 1)
                image = self._hsv2rgb(hsv) * 255
            
            # 8. 随机高斯模糊：模拟不同清晰度
            if np.random.rand() < 0.2:
                sigma = np.random.uniform(0.5, 1.5)
                for c in range(3):
                    image[:, :, c] = ndi.gaussian_filter(image[:, :, c], sigma=sigma)
            
            # 9. 归一化
            image = image / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = (image - mean) / std
            
            return {'image': image, 'mask': mask}
        
        def _rotate(self, image, mask, angle):
            """旋转图像和掩码（使用 OpenCV 避免 scipy 版本 bug）"""
            import cv2
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            rotated_img = cv2.warpAffine(image.astype(np.float32), M, (w, h),
                                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            rotated_mask = cv2.warpAffine(mask.astype(np.uint8), M, (w, h),
                                          flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
            return np.clip(rotated_img, 0, 255), rotated_mask.astype(np.int64)
        
        def _random_scale_crop(self, image, mask, scale):
            """随机缩放后中心裁剪回原尺寸"""
            h, w = image.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            
            img_pil = Image.fromarray(image.astype(np.uint8)).resize((new_w, new_h), Image.Resampling.BILINEAR)
            mask_pil = Image.fromarray(mask.astype(np.uint8), mode='L').resize((new_w, new_h), Image.Resampling.NEAREST)
            
            img_scaled = np.array(img_pil, dtype=np.float32)
            mask_scaled = np.array(mask_pil, dtype=np.int64)
            
            if scale > 1.0:
                # 随机裁剪
                y0 = np.random.randint(0, new_h - h + 1)
                x0 = np.random.randint(0, new_w - w + 1)
                img_scaled = img_scaled[y0:y0+h, x0:x0+w]
                mask_scaled = mask_scaled[y0:y0+h, x0:x0+w]
            else:
                # 零填充
                pad_y = (h - new_h) // 2
                pad_x = (w - new_w) // 2
                img_out = np.zeros((h, w, 3), dtype=np.float32)
                mask_out = np.zeros((h, w), dtype=np.int64)
                img_out[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = img_scaled
                mask_out[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = mask_scaled
                img_scaled = img_out
                mask_scaled = mask_out
            
            return img_scaled, mask_scaled
        
        @staticmethod
        def _rgb2hsv(rgb):
            r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
            max_c = np.maximum(np.maximum(r, g), b)
            min_c = np.minimum(np.minimum(r, g), b)
            delta = max_c - min_c
            
            h = np.zeros_like(max_c)
            mask = delta != 0
            h[mask & (max_c == r)] = (60 * ((g[mask & (max_c == r)] - b[mask & (max_c == r)]) / delta[mask & (max_c == r)]) + 360) % 360
            h[mask & (max_c == g)] = (60 * ((b[mask & (max_c == g)] - r[mask & (max_c == g)]) / delta[mask & (max_c == g)]) + 120) % 360
            h[mask & (max_c == b)] = (60 * ((r[mask & (max_c == b)] - g[mask & (max_c == b)]) / delta[mask & (max_c == b)]) + 240) % 360
            h = h / 360.0
            
            s = np.zeros_like(max_c)
            s[max_c != 0] = delta[max_c != 0] / max_c[max_c != 0]
            
            v = max_c
            
            return np.stack([h, s, v], axis=-1)
        
        @staticmethod
        def _hsv2rgb(hsv):
            h, s, v = hsv[:, :, 0] * 360, hsv[:, :, 1], hsv[:, :, 2]
            c = v * s
            x = c * (1 - np.abs((h / 60) % 2 - 1))
            m = v - c
            
            r, g, b = np.zeros_like(h), np.zeros_like(h), np.zeros_like(h)
            
            mask = (h >= 0) & (h < 60)
            r[mask] = c[mask]
            g[mask] = x[mask]
            
            mask = (h >= 60) & (h < 120)
            r[mask] = x[mask]
            g[mask] = c[mask]
            
            mask = (h >= 120) & (h < 180)
            g[mask] = c[mask]
            b[mask] = x[mask]
            
            mask = (h >= 180) & (h < 240)
            g[mask] = x[mask]
            b[mask] = c[mask]
            
            mask = (h >= 240) & (h < 300)
            r[mask] = x[mask]
            b[mask] = c[mask]
            
            mask = (h >= 300) & (h <= 360)
            r[mask] = c[mask]
            b[mask] = x[mask]
            
            return np.stack([r + m, g + m, b + m], axis=-1)
    
    return TrainingAugmentation(height, width)


def get_validation_augmentation(height=384, width=640):
    """验证集数据预处理（无随机增强，保持 16:9 宽高比，避免 albumentations cv2 依赖）"""
    class ValidationAugmentation:
        def __init__(self, h, w):
            self.h = h
            self.w = w
        
        def __call__(self, image, mask):
            # Resize + Normalize only
            image = Image.fromarray(image).resize((self.w, self.h), Image.Resampling.BILINEAR)
            mask = Image.fromarray(mask, mode='L').resize((self.w, self.h), Image.Resampling.NEAREST)
            
            image = np.array(image, dtype=np.float32)
            mask = np.array(mask, dtype=np.int64)
            
            # 归一化
            image = image / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = (image - mean) / std
            
            return {'image': image, 'mask': mask}
    
    return ValidationAugmentation(height, width)
