"""
路径处理器 - Path Processor
负责将感知结果转换为可导航的路径
"""

import numpy as np
import cv2
import math
from typing import List, Tuple, Optional, Dict, Any
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist

from ..utils.logger import get_logger
from ..utils.config_loader import get_config


class PathProcessor:
    """路径处理和转换器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化路径处理器
        
        Args:
            config: 配置字典
        """
        self.logger = get_logger("PathProcessor")
        self.config = config or get_config()
        
        # 图像到世界坐标转换参数
        self.pixel_per_meter = 100  # 像素每米，需要根据实际情况标定
        self.image_height = 480     # 图像高度
        self.image_width = 640      # 图像宽度
        
        # 路径处理参数
        self.path_smoothing_factor = 0.3   # 样条插值平滑因子
        self.min_path_points = 5           # 最小路径点数
        self.max_path_length = 20.0        # 最大路径长度 (米)
        self.path_resolution = 0.2         # 路径分辨率 (米)
        
        # 安全检查参数
        self.max_curvature = 0.5           # 最大曲率 (1/米)
        self.min_turn_radius = 2.0         # 最小转弯半径 (米)
        
        # 历史路径用于平滑
        self.path_history = []
        self.history_window = 3
        
        # 坐标转换矩阵（单应性变换）
        self.homography_matrix = None
        self.inverse_homography = None
        
    def set_camera_calibration(self, 
                              homography_matrix: Optional[np.ndarray] = None,
                              pixel_per_meter: Optional[float] = None) -> None:
        """
        设置相机标定参数
        
        Args:
            homography_matrix: 单应性变换矩阵 (3x3)
            pixel_per_meter: 像素每米比例
        """
        if homography_matrix is not None:
            self.homography_matrix = homography_matrix
            try:
                self.inverse_homography = np.linalg.inv(homography_matrix)
            except np.linalg.LinAlgError:
                self.logger.warning("单应性矩阵不可逆，使用默认转换")
                self.homography_matrix = None
        
        if pixel_per_meter is not None:
            self.pixel_per_meter = pixel_per_meter
            
        self.logger.info(f"相机标定参数更新: pixel_per_meter={self.pixel_per_meter}")
    
    def pixels_to_world(self, pixel_points: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
        """
        将像素坐标转换为世界坐标
        
        Args:
            pixel_points: 像素坐标点列表 [(x_pixel, y_pixel), ...]
            
        Returns:
            世界坐标点列表 [(x_world, y_world), ...]
        """
        if not pixel_points:
            return []
        
        world_points = []
        
        if self.homography_matrix is not None and self.inverse_homography is not None:
            # 使用单应性变换
            for x_pixel, y_pixel in pixel_points:
                # 齐次坐标
                pixel_homogeneous = np.array([[x_pixel, y_pixel, 1]], dtype=np.float32).T
                
                # 变换到世界坐标
                world_homogeneous = self.inverse_homography @ pixel_homogeneous
                
                # 归一化
                if world_homogeneous[2, 0] != 0:
                    x_world = world_homogeneous[0, 0] / world_homogeneous[2, 0]
                    y_world = world_homogeneous[1, 0] / world_homogeneous[2, 0]
                    world_points.append((x_world, y_world))
        
        else:
            # 简化变换：假设相机垂直向下，使用像素-米换算
            for x_pixel, y_pixel in pixel_points:
                # 将图像坐标原点移到底部中心
                x_centered = x_pixel - self.image_width / 2
                y_from_bottom = self.image_height - y_pixel
                
                # 转换为世界坐标 (米)
                x_world = x_centered / self.pixel_per_meter
                y_world = y_from_bottom / self.pixel_per_meter
                
                world_points.append((x_world, y_world))
        
        return world_points
    
    def world_to_pixels(self, world_points: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
        """
        将世界坐标转换为像素坐标
        
        Args:
            world_points: 世界坐标点列表
            
        Returns:
            像素坐标点列表
        """
        if not world_points:
            return []
        
        pixel_points = []
        
        if self.homography_matrix is not None:
            # 使用单应性变换
            for x_world, y_world in world_points:
                world_homogeneous = np.array([[x_world, y_world, 1]], dtype=np.float32).T
                pixel_homogeneous = self.homography_matrix @ world_homogeneous
                
                if pixel_homogeneous[2, 0] != 0:
                    x_pixel = int(pixel_homogeneous[0, 0] / pixel_homogeneous[2, 0])
                    y_pixel = int(pixel_homogeneous[1, 0] / pixel_homogeneous[2, 0])
                    pixel_points.append((x_pixel, y_pixel))
        
        else:
            # 简化逆变换
            for x_world, y_world in world_points:
                x_pixel = int(x_world * self.pixel_per_meter + self.image_width / 2)
                y_pixel = int(self.image_height - y_world * self.pixel_per_meter)
                pixel_points.append((x_pixel, y_pixel))
        
        return pixel_points
    
    def smooth_path_spline(self, path_points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        使用样条插值平滑路径
        
        Args:
            path_points: 原始路径点
            
        Returns:
            平滑后的路径点
        """
        if len(path_points) < 3:
            return path_points
        
        try:
            # 提取x, y坐标
            x_coords = np.array([p[0] for p in path_points])
            y_coords = np.array([p[1] for p in path_points])
            
            # 参数化样条插值
            tck, u = splprep([x_coords, y_coords], 
                           s=self.path_smoothing_factor, 
                           k=min(3, len(path_points)-1))
            
            # 生成平滑路径
            u_new = np.linspace(0, 1, len(path_points) * 2)  # 增加点密度
            smooth_coords = splev(u_new, tck)
            
            smooth_points = list(zip(smooth_coords[0], smooth_coords[1]))
            
            return smooth_points
        
        except Exception as e:
            self.logger.warning(f"样条平滑失败: {e}")
            return path_points
    
    def resample_path(self, path_points: List[Tuple[float, float]], 
                     target_resolution: float) -> List[Tuple[float, float]]:
        """
        重新采样路径以获得均匀分辨率
        
        Args:
            path_points: 原始路径点
            target_resolution: 目标分辨率 (米)
            
        Returns:
            重新采样的路径点
        """
        if len(path_points) < 2:
            return path_points
        
        resampled_points = [path_points[0]]  # 起始点
        current_distance = 0.0
        
        for i in range(1, len(path_points)):
            x1, y1 = path_points[i-1]
            x2, y2 = path_points[i]
            
            segment_length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            # 在这个线段上插入点
            while current_distance + target_resolution <= segment_length:
                current_distance += target_resolution
                ratio = current_distance / segment_length
                
                # 线性插值
                x_new = x1 + ratio * (x2 - x1)
                y_new = y1 + ratio * (y2 - y1)
                resampled_points.append((x_new, y_new))
            
            # 更新当前距离
            current_distance = current_distance + target_resolution - segment_length
            if current_distance < 0:
                current_distance = 0
        
        # 添加终点
        if path_points[-1] not in resampled_points:
            resampled_points.append(path_points[-1])
        
        return resampled_points
    
    def validate_path_curvature(self, path_points: List[Tuple[float, float]]) -> Tuple[bool, List[float]]:
        """
        验证路径曲率是否在安全范围内
        
        Args:
            path_points: 路径点列表
            
        Returns:
            (是否安全, 曲率列表)
        """
        if len(path_points) < 3:
            return True, []
        
        curvatures = []
        is_safe = True
        
        for i in range(1, len(path_points) - 1):
            p1 = path_points[i-1]
            p2 = path_points[i]
            p3 = path_points[i+1]
            
            # 计算曲率 (使用三点法)
            # K = |det(A)| / ||AB||^3, 其中A是向量矩阵
            dx1, dy1 = p2[0] - p1[0], p2[1] - p1[1]
            dx2, dy2 = p3[0] - p2[0], p3[1] - p2[1]
            
            # 向量叉积
            cross_product = dx1 * dy2 - dy1 * dx2
            
            # 距离
            dist1 = math.sqrt(dx1**2 + dy1**2)
            dist2 = math.sqrt(dx2**2 + dy2**2)
            
            if dist1 > 1e-6 and dist2 > 1e-6:
                # 曲率计算
                curvature = abs(cross_product) / (dist1 * dist2 * (dist1 + dist2) / 2)
                curvatures.append(curvature)
                
                if curvature > self.max_curvature:
                    is_safe = False
            else:
                curvatures.append(0.0)
        
        return is_safe, curvatures
    
    def limit_path_length(self, path_points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        限制路径长度
        
        Args:
            path_points: 原始路径点
            
        Returns:
            限制长度后的路径点
        """
        if not path_points:
            return []
        
        limited_points = [path_points[0]]
        total_length = 0.0
        
        for i in range(1, len(path_points)):
            x1, y1 = path_points[i-1]
            x2, y2 = path_points[i]
            
            segment_length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            if total_length + segment_length <= self.max_path_length:
                limited_points.append(path_points[i])
                total_length += segment_length
            else:
                # 添加部分线段以达到最大长度
                remaining_length = self.max_path_length - total_length
                if remaining_length > 0:
                    ratio = remaining_length / segment_length
                    x_final = x1 + ratio * (x2 - x1)
                    y_final = y1 + ratio * (y2 - y1)
                    limited_points.append((x_final, y_final))
                break
        
        return limited_points
    
    def process_lane_to_path(self, 
                           center_line: List[Tuple[int, int]],
                           enable_smoothing: bool = True) -> Dict[str, Any]:
        """
        将车道线中心线转换为可导航路径
        
        Args:
            center_line: 像素坐标的中心线点列表
            enable_smoothing: 是否启用平滑
            
        Returns:
            处理结果字典
        """
        import time
        start_time = time.time()
        
        if not center_line or len(center_line) < self.min_path_points:
            return {
                'world_path': [],
                'pixel_path': center_line,
                'is_valid': False,
                'error_message': f'中心线点数不足: {len(center_line)} < {self.min_path_points}',
                'processing_time': time.time() - start_time
            }
        
        # 1. 转换到世界坐标
        world_points = self.pixels_to_world(center_line)
        
        if not world_points:
            return {
                'world_path': [],
                'pixel_path': center_line,
                'is_valid': False,
                'error_message': '坐标转换失败',
                'processing_time': time.time() - start_time
            }
        
        # 2. 去除重复和过近的点
        filtered_points = self._remove_close_points(world_points, min_distance=0.1)
        
        # 3. 限制路径长度
        length_limited_points = self.limit_path_length(filtered_points)
        
        # 4. 平滑处理
        if enable_smoothing and len(length_limited_points) >= 3:
            smoothed_points = self.smooth_path_spline(length_limited_points)
        else:
            smoothed_points = length_limited_points
        
        # 5. 重新采样
        resampled_points = self.resample_path(smoothed_points, self.path_resolution)
        
        # 6. 验证曲率
        is_curvature_safe, curvatures = self.validate_path_curvature(resampled_points)
        
        # 7. 计算路径统计信息
        path_length = self._calculate_path_length(resampled_points)
        max_curvature = max(curvatures) if curvatures else 0.0
        
        # 8. 转换回像素坐标用于可视化
        final_pixel_path = self.world_to_pixels(resampled_points)
        
        processing_time = time.time() - start_time
        
        result = {
            'world_path': resampled_points,
            'pixel_path': final_pixel_path,
            'is_valid': is_curvature_safe and len(resampled_points) >= self.min_path_points,
            'path_length': path_length,
            'path_points_count': len(resampled_points),
            'max_curvature': max_curvature,
            'curvatures': curvatures,
            'is_curvature_safe': is_curvature_safe,
            'smoothing_applied': enable_smoothing and len(length_limited_points) >= 3,
            'processing_time': processing_time,
            'error_message': '' if is_curvature_safe else f'路径曲率过大: {max_curvature:.3f} > {self.max_curvature}'
        }
        
        return result
    
    def _remove_close_points(self, points: List[Tuple[float, float]], 
                           min_distance: float = 0.1) -> List[Tuple[float, float]]:
        """移除过近的点"""
        if len(points) <= 1:
            return points
        
        filtered = [points[0]]
        
        for point in points[1:]:
            last_point = filtered[-1]
            distance = math.sqrt((point[0] - last_point[0])**2 + (point[1] - last_point[1])**2)
            
            if distance >= min_distance:
                filtered.append(point)
        
        return filtered
    
    def _calculate_path_length(self, points: List[Tuple[float, float]]) -> float:
        """计算路径总长度"""
        if len(points) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(points)):
            x1, y1 = points[i-1]
            x2, y2 = points[i]
            total_length += math.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        return total_length
    
    def smooth_path_history(self, current_path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """使用历史路径平滑当前路径"""
        self.path_history.append(current_path)
        
        if len(self.path_history) > self.history_window:
            self.path_history.pop(0)
        
        # 简单实现：返回当前路径
        # 更复杂的实现可以对历史路径进行加权平均
        return current_path


if __name__ == "__main__":
    # 测试路径处理器
    processor = PathProcessor()
    
    print("路径处理器测试...")
    
    # 创建测试中心线（像素坐标）
    test_center_line = [
        (320, 400),  # 图像底部中心
        (325, 350),
        (330, 300),
        (335, 250),
        (340, 200),
        (345, 150),
        (350, 100)   # 向前延伸
    ]
    
    # 处理路径
    result = processor.process_lane_to_path(test_center_line, enable_smoothing=True)
    
    print("处理结果:")
    print(f"- 路径有效: {result['is_valid']}")
    print(f"- 路径长度: {result['path_length']:.2f}m")
    print(f"- 路径点数: {result['path_points_count']}")
    print(f"- 最大曲率: {result['max_curvature']:.4f}")
    print(f"- 曲率安全: {result['is_curvature_safe']}")
    print(f"- 处理时间: {result['processing_time']*1000:.2f}ms")
    
    if result['world_path']:
        print("前几个世界坐标点:")
        for i, (x, y) in enumerate(result['world_path'][:5]):
            print(f"  Point {i}: ({x:.2f}, {y:.2f})")
    
    # 测试坐标转换
    print("\\n测试坐标转换:")
    test_pixel = [(320, 400), (400, 300)]
    world_coords = processor.pixels_to_world(test_pixel)
    back_to_pixel = processor.world_to_pixels(world_coords)
    
    print(f"原始像素坐标: {test_pixel}")
    print(f"世界坐标: {world_coords}")
    print(f"转换回像素坐标: {back_to_pixel}")