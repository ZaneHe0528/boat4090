"""
Pure Pursuit算法实现 - Pure Pursuit Algorithm
基于目标点的路径跟踪控制算法
"""

import numpy as np
import math
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from ..utils.logger import get_logger
from ..utils.config_loader import get_config


@dataclass
class VehicleState:
    """车辆状态"""
    x: float          # x坐标 (米)  
    y: float          # y坐标 (米)
    yaw: float        # 航向角 (弧度)
    speed: float      # 速度 (m/s)
    steering: float   # 当前舵角 (弧度)


@dataclass
class TargetPoint:
    """目标点"""
    x: float          # x坐标 (米)
    y: float          # y坐标 (米)
    speed: float      # 目标速度 (m/s)
    distance: float   # 距离当前位置的距离 (米)


class PurePursuit:
    """Pure Pursuit路径跟踪控制器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化Pure Pursuit控制器
        
        Args:
            config: 配置字典
        """
        self.logger = get_logger("PurePursuit")
        self.config = config or get_config().get_pure_pursuit_config()
        
        # Pure Pursuit参数
        self.lookahead_distance = self.config.get("lookahead_distance", 2.0)
        self.wheelbase = self.config.get("wheelbase", 1.5)  # 船体轴距
        self.min_lookahead = self.config.get("min_lookahead", 1.0)
        self.max_lookahead = self.config.get("max_lookahead", 5.0)
        self.speed_lookahead_ratio = self.config.get("speed_lookahead_ratio", 0.5)
        
        # 安全参数
        safety_config = get_config().get_safety_config()
        self.max_steering_angle = math.radians(safety_config.get("max_deviation_angle", 45.0))
        
        # 控制历史（用于平滑）
        self.steering_history = []
        self.target_history = []
        self.smoothing_window = 3
        
        # 性能统计
        self.computation_times = []
        
    def calculate_dynamic_lookahead(self, speed: float) -> float:
        """
        根据速度计算动态预瞄距离
        
        Args:
            speed: 当前速度 (m/s)
            
        Returns:
            预瞄距离 (米)
        """
        # 基础预瞄距离 + 速度相关部分
        dynamic_distance = self.lookahead_distance + speed * self.speed_lookahead_ratio
        
        # 限制在合理范围内
        dynamic_distance = max(self.min_lookahead, 
                              min(self.max_lookahead, dynamic_distance))
        
        return dynamic_distance
    
    def find_target_point(self, 
                         path: List[Tuple[float, float]], 
                         vehicle_state: VehicleState) -> Optional[TargetPoint]:
        """
        在路径上寻找目标点
        
        Args:
            path: 路径点列表 [(x, y), ...]
            vehicle_state: 当前车辆状态
            
        Returns:
            目标点，如果未找到返回None
        """
        if not path:
            return None
        
        lookahead = self.calculate_dynamic_lookahead(vehicle_state.speed)
        
        # 寻找距离车辆位置最近的路径点
        min_distance = float('inf')
        closest_index = 0
        
        for i, (x, y) in enumerate(path):
            distance = math.sqrt((x - vehicle_state.x)**2 + (y - vehicle_state.y)**2)
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        
        # 从最近点开始，寻找预瞄距离处的目标点
        for i in range(closest_index, len(path)):
            x, y = path[i]
            distance = math.sqrt((x - vehicle_state.x)**2 + (y - vehicle_state.y)**2)
            
            if distance >= lookahead:
                # 找到目标点，可以进行插值优化
                if i > 0:
                    # 线性插值获得精确的预瞄距离点
                    x_prev, y_prev = path[i-1]
                    
                    # 计算插值比例
                    d_prev = math.sqrt((x_prev - vehicle_state.x)**2 + (y_prev - vehicle_state.y)**2)
                    d_curr = distance
                    
                    if d_curr != d_prev:
                        ratio = (lookahead - d_prev) / (d_curr - d_prev)
                        ratio = max(0, min(1, ratio))  # 限制在[0,1]
                        
                        target_x = x_prev + ratio * (x - x_prev)
                        target_y = y_prev + ratio * (y - y_prev)
                    else:
                        target_x, target_y = x, y
                else:
                    target_x, target_y = x, y
                
                return TargetPoint(
                    x=target_x,
                    y=target_y, 
                    speed=vehicle_state.speed,  # 暂时使用当前速度
                    distance=lookahead
                )
        
        # 如果没找到合适的点，使用路径末尾点
        if path:
            x, y = path[-1]
            distance = math.sqrt((x - vehicle_state.x)**2 + (y - vehicle_state.y)**2)
            return TargetPoint(x=x, y=y, speed=vehicle_state.speed, distance=distance)
        
        return None
    
    def calculate_steering_angle(self, 
                               vehicle_state: VehicleState, 
                               target_point: TargetPoint) -> float:
        """
        计算Pure Pursuit舵角
        
        Args:
            vehicle_state: 当前车辆状态
            target_point: 目标点
            
        Returns:
            舵角 (弧度)
        """
        # 计算目标点在车辆坐标系中的位置
        dx = target_point.x - vehicle_state.x
        dy = target_point.y - vehicle_state.y
        
        # 转换到车辆坐标系
        cos_yaw = math.cos(vehicle_state.yaw)
        sin_yaw = math.sin(vehicle_state.yaw)
        
        # 目标点在车辆坐标系中的y坐标（横向距离）
        lateral_distance = -dx * sin_yaw + dy * cos_yaw
        
        # 预瞄距离
        lookahead = target_point.distance
        
        # Pure Pursuit公式
        # 曲率 κ = 2 * lateral_distance / lookahead^2
        if lookahead > 0:
            curvature = 2.0 * lateral_distance / (lookahead ** 2)
        else:
            curvature = 0
        
        # 舵角 δ = arctan(wheelbase * curvature)
        steering_angle = math.atan(self.wheelbase * curvature)
        
        # 限制舵角范围
        steering_angle = max(-self.max_steering_angle, 
                           min(self.max_steering_angle, steering_angle))
        
        return steering_angle
    
    def smooth_steering(self, steering_angle: float) -> float:
        """
        平滑舵角输出
        
        Args:
            steering_angle: 当前计算的舵角
            
        Returns:
            平滑后的舵角
        """
        # 添加到历史记录
        self.steering_history.append(steering_angle)
        
        # 保持历史记录长度
        if len(self.steering_history) > self.smoothing_window:
            self.steering_history.pop(0)
        
        # 简单移动平均
        smoothed_angle = sum(self.steering_history) / len(self.steering_history)
        
        return smoothed_angle
    
    def control(self, 
               path: List[Tuple[float, float]], 
               vehicle_state: VehicleState,
               enable_smoothing: bool = True) -> Dict[str, Any]:
        """
        执行Pure Pursuit控制
        
        Args:
            path: 目标路径
            vehicle_state: 当前车辆状态
            enable_smoothing: 是否启用平滑
            
        Returns:
            控制结果字典
        """
        import time
        start_time = time.time()
        
        # 寻找目标点
        target_point = self.find_target_point(path, vehicle_state)
        
        if target_point is None:
            return {
                'steering_angle': 0.0,
                'target_point': None,
                'lookahead_distance': self.lookahead_distance,
                'status': 'no_target',
                'error_message': '未找到有效目标点'
            }
        
        # 计算舵角
        raw_steering = self.calculate_steering_angle(vehicle_state, target_point)
        
        # 平滑处理
        if enable_smoothing:
            steering_angle = self.smooth_steering(raw_steering)
        else:
            steering_angle = raw_steering
        
        # 计算一些有用的调试信息
        dx = target_point.x - vehicle_state.x
        dy = target_point.y - vehicle_state.y
        cross_track_error = abs(-dx * math.sin(vehicle_state.yaw) + dy * math.cos(vehicle_state.yaw))
        heading_error = math.atan2(dy, dx) - vehicle_state.yaw
        
        # 标准化航向误差到[-π, π]
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi
        
        # 记录计算时间
        computation_time = time.time() - start_time
        self.computation_times.append(computation_time)
        
        result = {
            'steering_angle': steering_angle,           # 输出舵角 (弧度)
            'steering_angle_deg': math.degrees(steering_angle),  # 舵角 (度)
            'raw_steering_angle': raw_steering,         # 未平滑的舵角
            'target_point': {
                'x': target_point.x,
                'y': target_point.y,
                'distance': target_point.distance
            },
            'lookahead_distance': target_point.distance,
            'cross_track_error': cross_track_error,     # 横向偏差
            'heading_error': heading_error,             # 航向误差 (弧度)
            'heading_error_deg': math.degrees(heading_error),  # 航向误差 (度)
            'curvature': 2.0 * (-dx * math.sin(vehicle_state.yaw) + dy * math.cos(vehicle_state.yaw)) / (target_point.distance ** 2) if target_point.distance > 0 else 0,
            'status': 'ok',
            'computation_time': computation_time,
            'smoothing_enabled': enable_smoothing
        }
        
        return result
    
    def get_performance_stats(self) -> Dict[str, float]:
        """获取性能统计"""
        if not self.computation_times:
            return {}
        
        return {
            'avg_computation_time': np.mean(self.computation_times),
            'max_computation_time': np.max(self.computation_times),
            'min_computation_time': np.min(self.computation_times),
            'control_frequency': 1.0 / np.mean(self.computation_times)
        }
    
    def reset_history(self) -> None:
        """重置历史数据"""
        self.steering_history.clear()
        self.target_history.clear()
        self.computation_times.clear()
    
    def set_parameters(self, **kwargs) -> None:
        """
        动态设置参数
        
        Args:
            **kwargs: 参数字典，可包含lookahead_distance, wheelbase等
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                self.logger.info(f"参数更新: {key} = {value}")
    
    def validate_path(self, path: List[Tuple[float, float]]) -> Tuple[bool, str]:
        """
        验证路径有效性
        
        Args:
            path: 待验证的路径
            
        Returns:
            (是否有效, 错误信息)
        """
        if not path:
            return False, "路径为空"
        
        if len(path) < 2:
            return False, "路径点数量不足"
        
        # 检查路径点是否有效
        for i, (x, y) in enumerate(path):
            if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
                return False, f"路径点 {i} 包含无效坐标"
            
            if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
                return False, f"路径点 {i} 包含NaN或Inf值"
        
        # 检查路径长度
        total_length = 0
        for i in range(1, len(path)):
            x1, y1 = path[i-1]
            x2, y2 = path[i]
            segment_length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
            total_length += segment_length
        
        if total_length < self.min_lookahead:
            return False, f"路径总长度过短: {total_length:.2f} < {self.min_lookahead}"
        
        return True, "路径有效"


if __name__ == "__main__":
    # 测试Pure Pursuit算法
    controller = PurePursuit()
    
    # 创建测试路径（直线）
    test_path = [(i * 0.5, 0) for i in range(20)]  # 10米直线
    
    # 创建车辆状态
    vehicle = VehicleState(
        x=0.0, y=0.0, yaw=0.0, speed=2.0, steering=0.0
    )
    
    print("Pure Pursuit算法测试...")
    
    # 验证路径
    is_valid, message = controller.validate_path(test_path)
    print(f"路径有效性: {is_valid} - {message}")
    
    if is_valid:
        # 执行控制
        result = controller.control(test_path, vehicle)
        
        print("控制结果:")
        print(f"- 状态: {result['status']}")
        print(f"- 舵角: {result['steering_angle_deg']:.2f}°")
        print(f"- 目标点: ({result['target_point']['x']:.2f}, {result['target_point']['y']:.2f})")
        print(f"- 预瞄距离: {result['lookahead_distance']:.2f}m")
        print(f"- 横向误差: {result['cross_track_error']:.3f}m")
        print(f"- 航向误差: {result['heading_error_deg']:.2f}°")
        print(f"- 计算时间: {result['computation_time']*1000:.2f}ms")
    
    # 测试偏离路径的情况
    print("\\n测试偏离路径情况:")
    vehicle_offset = VehicleState(
        x=1.0, y=1.0, yaw=math.radians(30), speed=1.5, steering=0.0
    )
    
    result = controller.control(test_path, vehicle_offset)
    print(f"- 舵角: {result['steering_angle_deg']:.2f}°")
    print(f"- 横向误差: {result['cross_track_error']:.3f}m")
    print(f"- 航向误差: {result['heading_error_deg']:.2f}°")
    
    # 性能统计
    stats = controller.get_performance_stats()
    print("\\n性能统计:")
    for key, value in stats.items():
        print(f"- {key}: {value:.6f}"))