"""
车道线检测器 - Lane Detector
从语义分割结果中提取和处理车道线信息
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from scipy import ndimage
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import time

from ..utils.logger import get_logger
from ..utils.config_loader import get_config


class LaneDetector:
    """车道线检测与处理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化车道线检测器
        
        Args:
            config: 配置字典
        """
        self.logger = get_logger("LaneDetector")
        self.config = config or get_config()
        
        # 车道线处理参数
        lane_config = self.config.get("lane_processing", {})
        self.roi_height_ratio = lane_config.get("roi_height_ratio", 0.6)
        self.roi_width_ratio = lane_config.get("roi_width_ratio", 1.0)
        self.min_lane_width = lane_config.get("min_lane_width", 50)
        self.max_lane_width = lane_config.get("max_lane_width", 800)
        self.smoothing_window = lane_config.get("lane_smoothing_window", 5)
        self.poly_degree = lane_config.get("polynomial_degree", 2)
        
        # 颜色检测参数
        color_config = self.config.get("color_detection", {})
        self.red_lower = np.array(color_config.get("red_line", {}).get("lower_hsv", [0, 50, 50]))
        self.red_upper = np.array(color_config.get("red_line", {}).get("upper_hsv", [10, 255, 255]))
        self.red_alt_lower = np.array(color_config.get("red_line", {}).get("alt_lower_hsv", [170, 50, 50]))
        self.red_alt_upper = np.array(color_config.get("red_line", {}).get("alt_upper_hsv", [180, 255, 255]))
        self.min_area = color_config.get("red_line", {}).get("min_area", 500)
        self.kernel_size = color_config.get("red_line", {}).get("morphology_kernel_size", 5)
        
        # 历史数据用于平滑
        self.left_lane_history = []
        self.right_lane_history = []
        self.center_line_history = []
        
        # 性能统计
        self.processing_times = []
    
    def set_roi(self, image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        设置感兴趣区域 (ROI)
        
        Args:
            image_shape: 图像尺寸 (height, width)
            
        Returns:
            ROI区域 (x, y, width, height)
        """
        height, width = image_shape
        
        # ROI从图像下半部分开始
        roi_height = int(height * self.roi_height_ratio)
        roi_width = int(width * self.roi_width_ratio)
        
        x = (width - roi_width) // 2
        y = height - roi_height
        
        return x, y, roi_width, roi_height
    
    def extract_red_lines_from_mask(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """
        从语义分割掩码中提取红线像素点
        
        Args:
            mask: 语义分割掩码 (单通道)
            
        Returns:
            红线像素点列表 [(x, y), ...]
        """
        # 假设红线的类别ID为2
        red_mask = (mask == 2).astype(np.uint8)
        
        # 形态学операций cleaning
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤小面积轮廓
        valid_contours = [c for c in contours if cv2.contourArea(c) > self.min_area]
        
        # 提取所有点
        points = []
        for contour in valid_contours:
            for point in contour:
                x, y = point[0]
                points.append((x, y))
        
        return points
    
    def extract_red_lines_from_color(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """
        使用颜色阈值提取红线像素点 (备用方法)
        
        Args:
            image: BGR图像
            
        Returns:
            红线像素点列表
        """
        # 转换为HSV色彩空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 创建红色掩码 (两个红色范围)
        mask1 = cv2.inRange(hsv, self.red_lower, self.red_upper)
        mask2 = cv2.inRange(hsv, self.red_alt_lower, self.red_alt_upper)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # 形态学处理
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓并提取点
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        points = []
        for contour in contours:
            if cv2.contourArea(contour) > self.min_area:
                for point in contour:
                    x, y = point[0]
                    points.append((x, y))
        
        return points
    
    def cluster_lane_points(self, points: List[Tuple[int, int]], 
                          image_width: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        将检测到的点聚类为左右车道线
        
        Args:
            points: 检测到的像素点列表
            image_width: 图像宽度
            
        Returns:
            (左车道线点, 右车道线点)
        """
        if not points:
            return [], []
        
        # 将点按x坐标分组
        left_points = []
        right_points = []
        
        center_x = image_width // 2
        
        for x, y in points:
            if x < center_x:
                left_points.append((x, y))
            else:
                right_points.append((x, y))
        
        return left_points, right_points
    
    def fit_lane_polynomial(self, points: List[Tuple[int, int]], 
                           degree: int = 2) -> Optional[np.ndarray]:
        """
        拟合车道线多项式
        
        Args:
            points: 车道线点列表
            degree: 多项式阶数
            
        Returns:
            多项式系数数组，如果拟合失败返回None
        """
        if len(points) < degree + 1:
            return None
        
        try:
            # 提取x, y坐标
            x_coords = np.array([p[0] for p in points])
            y_coords = np.array([p[1] for p in points])
            
            # 使用y作为自变量拟合x = f(y) (因为车道线可能垂直)
            coeffs = np.polyfit(y_coords, x_coords, degree)
            
            return coeffs
            
        except np.RankWarning:
            self.logger.warning("多项式拟合：矩阵秩不足")
            return None
        except Exception as e:
            self.logger.error(f"多项式拟合失败: {e}")
            return None
    
    def generate_lane_points(self, coeffs: np.ndarray, 
                           y_start: int, y_end: int, 
                           step: int = 10) -> List[Tuple[int, int]]:
        """
        根据多项式系数生成车道线点
        
        Args:
            coeffs: 多项式系数
            y_start: 起始y坐标
            y_end: 结束y坐标
            step: y坐标步长
            
        Returns:
            生成的车道线点列表
        """
        points = []
        
        for y in range(y_start, y_end, step):
            try:
                x = np.polyval(coeffs, y)
                points.append((int(x), y))
            except:
                continue
        
        return points
    
    def calculate_center_line(self, left_lane: List[Tuple[int, int]], 
                            right_lane: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        计算中心线
        
        Args:
            left_lane: 左车道线点列表
            right_lane: 右车道线点列表
            
        Returns:
            中心线点列表
        """
        if not left_lane or not right_lane:
            return []
        
        # 拟合左右车道线
        left_coeffs = self.fit_lane_polynomial(left_lane, self.poly_degree)
        right_coeffs = self.fit_lane_polynomial(right_lane, self.poly_degree)
        
        if left_coeffs is None or right_coeffs is None:
            return []
        
        # 计算中心线系数 (两车道线的均值)
        center_coeffs = (left_coeffs + right_coeffs) / 2
        
        # 生成中心线点
        y_min = min(min(p[1] for p in left_lane), min(p[1] for p in right_lane))
        y_max = max(max(p[1] for p in left_lane), max(p[1] for p in right_lane))
        
        center_points = self.generate_lane_points(center_coeffs, y_min, y_max, 10)
        
        return center_points
    
    def smooth_lane_history(self, current_lane: List[Tuple[int, int]], 
                          history: List[List[Tuple[int, int]]]) -> List[Tuple[int, int]]:
        """
        使用历史数据平滑车道线
        
        Args:
            current_lane: 当前检测到的车道线
            history: 历史车道线数据
            
        Returns:
            平滑后的车道线
        """
        # 添加当前车道线到历史记录
        history.append(current_lane)
        
        # 保持历史记录长度
        if len(history) > self.smoothing_window:
            history.pop(0)
        
        # 如果历史数据不足，直接返回当前车道线
        if len(history) < 2:
            return current_lane
        
        # 计算平均车道线 (简化实现)
        if not current_lane:
            return current_lane
        
        # 这里可以实现更复杂的平滑算法
        # 暂时返回当前车道线
        return current_lane
    
    def validate_lanes(self, left_lane: List[Tuple[int, int]], 
                      right_lane: List[Tuple[int, int]],
                      image_width: int) -> Tuple[bool, str]:
        """
        验证检测到的车道线是否合理
        
        Args:
            left_lane: 左车道线
            right_lane: 右车道线  
            image_width: 图像宽度
            
        Returns:
            (是否有效, 错误信息)
        """
        if not left_lane and not right_lane:
            return False, "未检测到任何车道线"
        
        if not left_lane:
            return False, "未检测到左车道线"
        
        if not right_lane:
            return False, "未检测到右车道线"
        
        # 检查车道宽度
        left_x_avg = np.mean([p[0] for p in left_lane])
        right_x_avg = np.mean([p[0] for p in right_lane])
        lane_width = abs(right_x_avg - left_x_avg)
        
        if lane_width < self.min_lane_width:
            return False, f"车道宽度过小: {lane_width:.1f} < {self.min_lane_width}"
        
        if lane_width > self.max_lane_width:
            return False, f"车道宽度过大: {lane_width:.1f} > {self.max_lane_width}"
        
        # 检查车道线位置合理性
        if left_x_avg > right_x_avg:
            return False, "左右车道线位置颠倒"
        
        return True, "车道线检测有效"
    
    def process_frame(self, image: np.ndarray, 
                     mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        处理单帧图像，检测车道线
        
        Args:
            image: 输入图像 (BGR)
            mask: 语义分割掩码 (可选)
            
        Returns:
            检测结果字典
        """
        start_time = time.time()
        
        height, width = image.shape[:2]
        roi = self.set_roi((height, width))
        
        # 提取红线点
        if mask is not None:
            points = self.extract_red_lines_from_mask(mask)
        else:
            points = self.extract_red_lines_from_color(image)
        
        # 聚类为左右车道线
        left_points, right_points = self.cluster_lane_points(points, width)
        
        # 拟合车道线
        left_lane = []
        right_lane = []
        
        if left_points:
            left_coeffs = self.fit_lane_polynomial(left_points, self.poly_degree)
            if left_coeffs is not None:
                left_lane = self.generate_lane_points(left_coeffs, roi[1], roi[1] + roi[3])
        
        if right_points:
            right_coeffs = self.fit_lane_polynomial(right_points, self.poly_degree)
            if right_coeffs is not None:
                right_lane = self.generate_lane_points(right_coeffs, roi[1], roi[1] + roi[3])
        
        # 平滑处理
        left_lane = self.smooth_lane_history(left_lane, self.left_lane_history)
        right_lane = self.smooth_lane_history(right_lane, self.right_lane_history)
        
        # 计算中心线
        center_line = self.calculate_center_line(left_lane, right_lane)
        center_line = self.smooth_lane_history(center_line, self.center_line_history)
        
        # 验证结果
        is_valid, message = self.validate_lanes(left_lane, right_lane, width)
        
        # 记录处理时间
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        result = {
            'left_lane': left_lane,
            'right_lane': right_lane,
            'center_line': center_line,
            'raw_points': points,
            'roi': roi,
            'is_valid': is_valid,
            'validation_message': message,
            'processing_time': processing_time,
            'total_points': len(points),
            'left_points': len(left_points),
            'right_points': len(right_points)
        }
        
        return result
    
    def get_performance_stats(self) -> Dict[str, float]:
        """获取性能统计"""
        if not self.processing_times:
            return {}
        
        return {
            'avg_processing_time': np.mean(self.processing_times),
            'max_processing_time': np.max(self.processing_times),
            'min_processing_time': np.min(self.processing_times),
            'processing_fps': 1.0 / np.mean(self.processing_times)
        }
    
    def reset_history(self) -> None:
        """重置历史数据"""
        self.left_lane_history.clear()
        self.right_lane_history.clear()
        self.center_line_history.clear()
        self.processing_times.clear()


if __name__ == "__main__":
    # 测试车道线检测器
    detector = LaneDetector()
    
    # 创建测试图像和掩码
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_mask = np.zeros((480, 640), dtype=np.uint8)
    
    # 添加一些模拟的红线区域
    test_mask[200:400, 100:120] = 2  # 左红线
    test_mask[200:400, 520:540] = 2  # 右红线
    
    # 处理帧
    result = detector.process_frame(test_image, test_mask)
    
    print("车道线检测结果:")
    print(f"- 检测有效: {result['is_valid']}")
    print(f"- 验证信息: {result['validation_message']}")
    print(f"- 左车道线点数: {len(result['left_lane'])}")
    print(f"- 右车道线点数: {len(result['right_lane'])}")
    print(f"- 中心线点数: {len(result['center_line'])}")
    print(f"- 处理时间: {result['processing_time']:.4f}s")
    
    # 性能统计
    stats = detector.get_performance_stats()
    for key, value in stats.items():
        print(f"- {key}: {value:.4f}")