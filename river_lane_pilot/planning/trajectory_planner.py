"""
轨迹规划器 - Trajectory Planner
整合路径处理和Pure Pursuit控制的高级规划器
"""

import numpy as np
import math
import time
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum

from .pure_pursuit import PurePursuit, VehicleState, TargetPoint
from .path_processor import PathProcessor
from ..utils.logger import get_logger
from ..utils.config_loader import get_config


class PlannerState(Enum):
    """规划器状态"""
    IDLE = "idle"
    PLANNING = "planning"
    FOLLOWING = "following"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class TrajectoryPlan:
    """轨迹规划结果"""
    world_path: List[Tuple[float, float]]
    pixel_path: List[Tuple[int, int]]
    steering_angle: float
    target_speed: float
    target_point: Optional[TargetPoint]
    lookahead_distance: float
    cross_track_error: float
    heading_error: float
    path_length: float
    status: str
    timestamp: float


class TrajectoryPlanner:
    """轨迹规划器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化轨迹规划器
        
        Args:
            config: 配置字典
        """
        self.logger = get_logger("TrajectoryPlanner")
        self.config = config or get_config()
        
        # 初始化子组件
        self.pure_pursuit = PurePursuit(config)
        self.path_processor = PathProcessor(config)
        
        # 规划器状态
        self.state = PlannerState.IDLE
        self.last_plan = None
        self.planning_frequency = 10.0  # Hz
        self.last_planning_time = 0.0
        
        # 车辆参数
        vehicle_config = self.config.get("vehicle", {})
        self.target_speed = vehicle_config.get("target_speed", 1.5)
        self.max_speed = vehicle_config.get("max_speed", 3.0)
        self.min_speed = vehicle_config.get("min_speed", 0.5)
        
        # 安全参数
        safety_config = self.config.get("safety", {})
        self.enable_failsafe = safety_config.get("enable_failsafe", True)
        self.max_deviation = math.radians(safety_config.get("max_deviation_angle", 45.0))
        self.emergency_brake_distance = safety_config.get("emergency_brake_distance", 2.0)
        
        # 路径跟踪参数
        self.path_timeout = 2.0  # 路径超时时间 (秒)
        self.min_path_quality = 0.7  # 最小路径质量分数
        
        # 统计和历史
        self.planning_times = []
        self.error_count = 0
        self.success_count = 0
        self.trajectory_history = []
        
    def set_vehicle_parameters(self, target_speed: Optional[float] = None,
                             max_speed: Optional[float] = None) -> None:
        """
        设置车辆参数
        
        Args:
            target_speed: 目标速度
            max_speed: 最大速度
        """
        if target_speed is not None:
            self.target_speed = max(self.min_speed, min(max_speed or self.max_speed, target_speed))
            
        if max_speed is not None:
            self.max_speed = max(self.min_speed, max_speed)
            
        self.logger.info(f"车辆参数更新: target_speed={self.target_speed}, max_speed={self.max_speed}")
    
    def estimate_vehicle_state(self, 
                             last_state: Optional[VehicleState] = None,
                             gps_position: Optional[Tuple[float, float]] = None,
                             imu_heading: Optional[float] = None,
                             speed_estimate: Optional[float] = None) -> VehicleState:
        """
        估算当前车辆状态
        
        Args:
            last_state: 上一次的车辆状态
            gps_position: GPS位置 (x, y)
            imu_heading: IMU航向角 (弧度)
            speed_estimate: 速度估算 (m/s)
            
        Returns:
            估算的车辆状态
        """
        # 如果有精确位置信息
        if gps_position and imu_heading is not None:
            return VehicleState(
                x=gps_position[0],
                y=gps_position[1],
                yaw=imu_heading,
                speed=speed_estimate or self.target_speed,
                steering=last_state.steering if last_state else 0.0
            )
        
        # 使用图像坐标系的简化状态（船在图像底部中心）
        # 这是一个简化的假设，实际应该使用定位系统
        image_width = self.path_processor.image_width
        image_height = self.path_processor.image_height
        
        # 将图像坐标转换为世界坐标
        vehicle_pixel = (image_width // 2, image_height - 30)  # 图像底部中心
        vehicle_world = self.path_processor.pixels_to_world([vehicle_pixel])
        
        if vehicle_world:
            x, y = vehicle_world[0]
        else:
            x, y = 0.0, 0.0
        
        # 估算航向角（假设初始向前）
        yaw = 0.0
        if last_state:
            yaw = last_state.yaw  # 保持上次的航向角
        
        return VehicleState(
            x=x,
            y=y,
            yaw=yaw,
            speed=speed_estimate or self.target_speed,
            steering=last_state.steering if last_state else 0.0
        )
    
    def evaluate_path_quality(self, path_result: Dict[str, Any]) -> float:
        """
        评估路径质量
        
        Args:
            path_result: 路径处理结果
            
        Returns:
            质量分数 [0, 1]
        """
        if not path_result['is_valid']:
            return 0.0
        
        quality_score = 1.0
        
        # 1. 路径长度评分
        path_length = path_result.get('path_length', 0)
        if path_length < 1.0:  # 路径太短
            quality_score *= 0.3
        elif path_length < 2.0:
            quality_score *= 0.7
        
        # 2. 曲率评分
        max_curvature = path_result.get('max_curvature', 0)
        curvature_threshold = self.path_processor.max_curvature
        if max_curvature > curvature_threshold:
            quality_score *= 0.2  # 曲率过大严重降分
        elif max_curvature > curvature_threshold * 0.8:
            quality_score *= 0.6
        
        # 3. 点数评分
        point_count = path_result.get('path_points_count', 0)
        if point_count < 5:
            quality_score *= 0.4
        elif point_count < 10:
            quality_score *= 0.8
        
        # 4. 处理时间评分
        processing_time = path_result.get('processing_time', 0)
        if processing_time > 0.1:  # 超过100ms
            quality_score *= 0.8
        
        return max(0.0, min(1.0, quality_score))
    
    def check_safety_conditions(self, 
                               vehicle_state: VehicleState,
                               control_result: Dict[str, Any]) -> Tuple[bool, str]:
        """
        检查安全条件
        
        Args:
            vehicle_state: 当前车辆状态
            control_result: 控制结果
            
        Returns:
            (是否安全, 错误信息)
        """
        if not self.enable_failsafe:
            return True, "安全检查已禁用"
        
        # 1. 检查舵角是否过大
        steering_angle = abs(control_result.get('steering_angle', 0))
        if steering_angle > self.max_deviation:
            return False, f"舵角过大: {math.degrees(steering_angle):.1f}° > {math.degrees(self.max_deviation):.1f}°"
        
        # 2. 检查横向偏差
        cross_track_error = control_result.get('cross_track_error', 0)
        if cross_track_error > 3.0:  # 3米偏差
            return False, f"横向偏差过大: {cross_track_error:.1f}m > 3.0m"
        
        # 3. 检查速度
        if vehicle_state.speed > self.max_speed:
            return False, f"速度过快: {vehicle_state.speed:.1f} > {self.max_speed:.1f}m/s"
        
        # 4. 检查目标点距离
        target_point = control_result.get('target_point')
        if target_point:
            target_distance = target_point.get('distance', 0)
            if target_distance < 0.5:  # 目标点过近
                return False, f"目标点过近: {target_distance:.1f}m < 0.5m"
        
        return True, "安全检查通过"
    
    def adaptive_speed_control(self, 
                             control_result: Dict[str, Any],
                             path_quality: float) -> float:
        """
        自适应速度控制
        
        Args:
            control_result: 控制结果
            path_quality: 路径质量
            
        Returns:
            调整后的目标速度
        """
        base_speed = self.target_speed
        
        # 1. 根据路径质量调整
        speed_factor = 0.5 + 0.5 * path_quality  # [0.5, 1.0]
        
        # 2. 根据曲率调整
        steering_angle = abs(control_result.get('steering_angle', 0))
        if steering_angle > math.radians(15):  # 大转弯
            speed_factor *= 0.7
        elif steering_angle > math.radians(30):  # 急转弯
            speed_factor *= 0.5
        
        # 3. 根据横向偏差调整
        cross_track_error = control_result.get('cross_track_error', 0)
        if cross_track_error > 1.0:
            speed_factor *= 0.8
        
        # 计算最终速度
        adapted_speed = base_speed * speed_factor
        adapted_speed = max(self.min_speed, min(self.max_speed, adapted_speed))
        
        return adapted_speed
    
    def plan_trajectory(self, 
                       center_line: List[Tuple[int, int]],
                       vehicle_state: Optional[VehicleState] = None,
                       force_replan: bool = False) -> Optional[TrajectoryPlan]:
        """
        规划轨迹
        
        Args:
            center_line: 检测到的中心线 (像素坐标)
            vehicle_state: 当前车辆状态
            force_replan: 强制重新规划
            
        Returns:
            轨迹规划结果
        """
        planning_start_time = time.time()
        
        # 检查规划频率
        if not force_replan and (planning_start_time - self.last_planning_time) < (1.0 / self.planning_frequency):
            return self.last_plan
        
        try:
            self.state = PlannerState.PLANNING
            
            # 1. 估算车辆状态
            if vehicle_state is None:
                vehicle_state = self.estimate_vehicle_state(
                    self.last_plan.vehicle_state if hasattr(self.last_plan, 'vehicle_state') else None
                )
            
            # 2. 处理路径
            path_result = self.path_processor.process_lane_to_path(
                center_line, enable_smoothing=True
            )
            
            if not path_result['is_valid']:
                self.state = PlannerState.ERROR
                self.error_count += 1
                self.logger.warning(f"路径处理失败: {path_result.get('error_message', 'Unknown')}")
                return None
            
            # 3. 评估路径质量
            path_quality = self.evaluate_path_quality(path_result)
            
            if path_quality < self.min_path_quality:
                self.state = PlannerState.ERROR
                self.error_count += 1
                self.logger.warning(f"路径质量过低: {path_quality:.2f} < {self.min_path_quality}")
                return None
            
            # 4. Pure Pursuit控制
            world_path = path_result['world_path']
            control_result = self.pure_pursuit.control(world_path, vehicle_state)
            
            if control_result['status'] != 'ok':
                self.state = PlannerState.ERROR
                self.error_count += 1
                self.logger.warning(f"Pure Pursuit控制失败: {control_result.get('error_message', 'Unknown')}")
                return None
            
            # 5. 安全检查
            is_safe, safety_message = self.check_safety_conditions(vehicle_state, control_result)
            
            if not is_safe:
                self.state = PlannerState.EMERGENCY_STOP
                self.error_count += 1
                self.logger.error(f"安全检查失败: {safety_message}")
                return None
            
            # 6. 自适应速度控制
            target_speed = self.adaptive_speed_control(control_result, path_quality)
            
            # 7. 创建轨迹规划结果
            trajectory_plan = TrajectoryPlan(
                world_path=world_path,
                pixel_path=path_result['pixel_path'],
                steering_angle=control_result['steering_angle'],
                target_speed=target_speed,
                target_point=TargetPoint(
                    x=control_result['target_point']['x'],
                    y=control_result['target_point']['y'],
                    speed=target_speed,
                    distance=control_result['target_point']['distance']
                ),
                lookahead_distance=control_result['lookahead_distance'],
                cross_track_error=control_result['cross_track_error'],
                heading_error=control_result['heading_error'],
                path_length=path_result['path_length'],
                status='success',
                timestamp=planning_start_time
            )
            
            # 8. 更新状态
            self.state = PlannerState.FOLLOWING
            self.last_plan = trajectory_plan
            self.last_planning_time = planning_start_time
            self.success_count += 1
            
            # 9. 记录统计信息
            planning_time = time.time() - planning_start_time
            self.planning_times.append(planning_time)
            
            # 10. 添加到历史记录
            self.trajectory_history.append(trajectory_plan)
            if len(self.trajectory_history) > 50:  # 保持历史记录长度
                self.trajectory_history.pop(0)
            
            return trajectory_plan
            
        except Exception as e:
            self.state = PlannerState.ERROR
            self.error_count += 1
            self.logger.error(f"轨迹规划异常: {e}")
            return None
    
    def get_emergency_command(self) -> Dict[str, Any]:
        """获取紧急停车命令"""
        return {
            'steering_angle': 0.0,
            'target_speed': 0.0,
            'status': 'emergency_stop',
            'timestamp': time.time()
        }
    
    def get_planner_status(self) -> Dict[str, Any]:
        """获取规划器状态信息"""
        return {
            'state': self.state.value,
            'success_rate': self.success_count / max(1, self.success_count + self.error_count),
            'total_plans': self.success_count + self.error_count,
            'avg_planning_time': np.mean(self.planning_times) if self.planning_times else 0.0,
            'last_plan_time': self.last_planning_time,
            'trajectory_history_length': len(self.trajectory_history)
        }
    
    def reset_statistics(self) -> None:
        """重置统计信息"""
        self.planning_times.clear()
        self.trajectory_history.clear()
        self.success_count = 0
        self.error_count = 0
        self.last_planning_time = 0.0


if __name__ == "__main__":
    # 测试轨迹规划器
    planner = TrajectoryPlanner()
    
    print("轨迹规划器测试...")
    
    # 创建测试中心线
    test_center_line = [
        (320, 400),
        (322, 350),
        (324, 300),
        (326, 250),
        (328, 200),
        (330, 150),
        (332, 100)
    ]
    
    # 创建测试车辆状态
    test_vehicle_state = VehicleState(
        x=0.0, y=0.0, yaw=0.0, speed=1.5, steering=0.0
    )
    
    # 执行轨迹规划
    trajectory = planner.plan_trajectory(test_center_line, test_vehicle_state)
    
    if trajectory:
        print("✓ 轨迹规划成功")
        print(f"- 状态: {trajectory.status}")
        print(f"- 舵角: {math.degrees(trajectory.steering_angle):.2f}°")
        print(f"- 目标速度: {trajectory.target_speed:.2f}m/s")
        print(f"- 路径长度: {trajectory.path_length:.2f}m")
        print(f"- 横向误差: {trajectory.cross_track_error:.3f}m")
        print(f"- 航向误差: {math.degrees(trajectory.heading_error):.2f}°")
        print(f"- 预瞄距离: {trajectory.lookahead_distance:.2f}m")
    else:
        print("✗ 轨迹规划失败")
    
    # 显示规划器状态
    status = planner.get_planner_status()
    print("\\n规划器状态:")
    for key, value in status.items():
        print(f"- {key}: {value}")