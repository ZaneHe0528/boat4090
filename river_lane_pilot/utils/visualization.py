"""
可视化工具 - Visualization Tools
用于调试和结果展示的可视化功能
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from typing import List, Tuple, Optional, Dict, Any
import seaborn as sns
from pathlib import Path


class Visualizer:
    """可视化工具类"""
    
    def __init__(self, enable_display: bool = True, save_path: Optional[str] = None):
        """
        初始化可视化器
        
        Args:
            enable_display: 是否显示窗口
            save_path: 图像保存路径
        """
        self.enable_display = enable_display
        self.save_path = Path(save_path) if save_path else None
        
        if self.save_path:
            self.save_path.mkdir(parents=True, exist_ok=True)
        
        # 设置matplotlib样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 颜色定义
        self.colors = {
            'red_line': (0, 0, 255),      # 红线 (BGR)
            'center_line': (0, 255, 0),   # 中心线 (绿色)
            'left_lane': (255, 0, 0),     # 左车道线 (蓝色)
            'right_lane': (255, 255, 0),  # 右车道线 (青色)
            'waypoint': (255, 0, 255),    # 航点 (紫色)
            'vehicle': (0, 255, 255),     # 车辆位置 (黄色)
            'roi': (128, 128, 128),       # 感兴趣区域 (灰色)
        }
    
    def draw_segmentation_overlay(self, 
                                image: np.ndarray, 
                                mask: np.ndarray, 
                                alpha: float = 0.6) -> np.ndarray:
        """
        绘制语义分割结果叠加图
        
        Args:
            image: 原始图像 (BGR)
            mask: 分割掩码 (单通道，值为类别ID)
            alpha: 叠加透明度
            
        Returns:
            叠加后的图像
        """
        overlay = image.copy()
        
        # 定义类别颜色 (BGR)
        class_colors = {
            0: (0, 0, 0),      # 背景 - 黑色
            1: (128, 64, 0),   # 水面 - 深蓝色  
            2: (0, 0, 255),    # 红线 - 红色
        }
        
        # 创建彩色掩码
        colored_mask = np.zeros_like(image)
        for class_id, color in class_colors.items():
            colored_mask[mask == class_id] = color
        
        # 叠加掩码
        result = cv2.addWeighted(overlay, 1-alpha, colored_mask, alpha, 0)
        
        return result
    
    def draw_lane_detection(self, 
                          image: np.ndarray,
                          left_lane: Optional[List[Tuple[int, int]]] = None,
                          right_lane: Optional[List[Tuple[int, int]]] = None,
                          center_line: Optional[List[Tuple[int, int]]] = None,
                          roi: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        绘制车道线检测结果
        
        Args:
            image: 原始图像
            left_lane: 左车道线点列表 [(x, y), ...]
            right_lane: 右车道线点列表  
            center_line: 中心线点列表
            roi: 感兴趣区域 (x, y, w, h)
            
        Returns:
            绘制结果图像
        """
        result = image.copy()
        
        # 绘制ROI
        if roi is not None:
            x, y, w, h = roi
            cv2.rectangle(result, (x, y), (x+w, y+h), self.colors['roi'], 2)
            cv2.putText(result, 'ROI', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['roi'], 2)
        
        # 绘制左车道线
        if left_lane and len(left_lane) > 1:
            points = np.array(left_lane, dtype=np.int32)
            cv2.polylines(result, [points], False, self.colors['left_lane'], 3)
            
        # 绘制右车道线
        if right_lane and len(right_lane) > 1:
            points = np.array(right_lane, dtype=np.int32)
            cv2.polylines(result, [points], False, self.colors['right_lane'], 3)
            
        # 绘制中心线
        if center_line and len(center_line) > 1:
            points = np.array(center_line, dtype=np.int32)
            cv2.polylines(result, [points], False, self.colors['center_line'], 4)
            
            # 在中心线上绘制方向箭头
            if len(center_line) >= 2:
                p1 = center_line[-2]
                p2 = center_line[-1]
                cv2.arrowedLine(result, p1, p2, self.colors['center_line'], 4, tipLength=0.3)
        
        return result
    
    def draw_navigation_info(self, 
                           image: np.ndarray,
                           steering_angle: float,
                           speed: float,
                           target_point: Optional[Tuple[int, int]] = None,
                           vehicle_position: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        绘制导航信息
        
        Args:
            image: 图像
            steering_angle: 舵角 (度)
            speed: 速度 (m/s)
            target_point: 目标点
            vehicle_position: 车辆位置
            
        Returns:
            绘制结果图像
        """
        result = image.copy()
        h, w = result.shape[:2]
        
        # 绘制目标点
        if target_point:
            cv2.circle(result, target_point, 8, self.colors['waypoint'], -1)
            cv2.circle(result, target_point, 12, self.colors['waypoint'], 2)
        
        # 绘制车辆位置（通常在图像底部中心）
        if vehicle_position is None:
            vehicle_position = (w // 2, h - 30)
        
        cv2.circle(result, vehicle_position, 6, self.colors['vehicle'], -1)
        
        # 绘制转向指示
        if target_point and vehicle_position:
            cv2.arrowedLine(result, vehicle_position, target_point, 
                           self.colors['waypoint'], 3, tipLength=0.2)
        
        # 添加文本信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # 创建信息面板背景
        panel_height = 100
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        panel[:] = (50, 50, 50)  # 深灰色背景
        
        # 绘制信息文本
        texts = [
            f"Steering: {steering_angle:.1f}°",
            f"Speed: {speed:.1f} m/s",
            f"Target: {target_point if target_point else 'None'}"
        ]
        
        for i, text in enumerate(texts):
            y_pos = 25 + i * 25
            cv2.putText(panel, text, (10, y_pos), font, font_scale, (255, 255, 255), thickness)
        
        # 将信息面板添加到图像顶部
        result = np.vstack([panel, result])
        
        return result
    
    def plot_trajectory(self, 
                       trajectory: List[Tuple[float, float]], 
                       title: str = "Trajectory Plot") -> plt.Figure:
        """
        绘制轨迹图
        
        Args:
            trajectory: 轨迹点列表 [(x, y), ...]
            title: 图表标题
            
        Returns:
            matplotlib图形对象
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if trajectory:
            x_coords, y_coords = zip(*trajectory)
            ax.plot(x_coords, y_coords, 'b-', linewidth=2, label='Trajectory')
            ax.scatter(x_coords[0], y_coords[0], color='green', s=100, label='Start', zorder=5)
            ax.scatter(x_coords[-1], y_coords[-1], color='red', s=100, label='End', zorder=5)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        return fig
    
    def plot_control_signals(self, 
                           time_stamps: List[float],
                           steering_angles: List[float],
                           speeds: List[float]) -> plt.Figure:
        """
        绘制控制信号图
        
        Args:
            time_stamps: 时间戳列表
            steering_angles: 舵角列表
            speeds: 速度列表
            
        Returns:
            matplotlib图形对象
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 绘制舵角
        ax1.plot(time_stamps, steering_angles, 'r-', linewidth=2)
        ax1.set_ylabel('Steering Angle (°)')
        ax1.set_title('Control Signals')
        ax1.grid(True, alpha=0.3)
        
        # 绘制速度
        ax2.plot(time_stamps, speeds, 'b-', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Speed (m/s)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def show_image(self, image: np.ndarray, window_name: str = "Image", wait_key: bool = True) -> None:
        """
        显示图像
        
        Args:
            image: 要显示的图像
            window_name: 窗口名称
            wait_key: 是否等待按键
        """
        if self.enable_display:
            cv2.imshow(window_name, image)
            if wait_key:
                cv2.waitKey(1)
    
    def save_image(self, image: np.ndarray, filename: str) -> None:
        """
        保存图像
        
        Args:
            image: 要保存的图像
            filename: 文件名
        """
        if self.save_path:
            filepath = self.save_path / filename
            cv2.imwrite(str(filepath), image)
    
    def save_plot(self, fig: plt.Figure, filename: str) -> None:
        """
        保存matplotlib图形
        
        Args:
            fig: matplotlib图形对象
            filename: 文件名
        """
        if self.save_path:
            filepath = self.save_path / filename
            fig.savefig(str(filepath), dpi=300, bbox_inches='tight')
    
    def cleanup(self) -> None:
        """清理资源"""
        cv2.destroyAllWindows()
        plt.close('all')


# 实时可视化类
class RealTimeVisualizer:
    """实时可视化器"""
    
    def __init__(self):
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.axes = self.axes.flatten()
        
        plt.ion()  # 开启交互模式
        
    def update_plots(self, data: Dict[str, Any]) -> None:
        """更新所有图表"""
        for ax in self.axes:
            ax.clear()
        
        # 更新各种数据图表
        # 这里可以根据需要添加具体的更新逻辑
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


if __name__ == "__main__":
    # 测试可视化器
    visualizer = Visualizer(enable_display=False)
    
    # 创建测试图像
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:] = (100, 100, 100)
    
    # 测试车道线绘制
    left_lane = [(100, 400), (150, 300), (200, 200)]
    right_lane = [(500, 400), (450, 300), (400, 200)]
    center_line = [(300, 400), (300, 300), (300, 200)]
    
    result = visualizer.draw_lane_detection(
        test_image, left_lane, right_lane, center_line
    )
    
    # 测试导航信息
    result = visualizer.draw_navigation_info(
        result, steering_angle=15.0, speed=2.5, 
        target_point=(300, 200)
    )
    
    print("可视化工具测试完成")