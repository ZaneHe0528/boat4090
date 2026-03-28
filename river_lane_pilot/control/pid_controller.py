"""
PID控制器 - PID Controller
用于精确控制舵角和速度的PID算法实现
"""

import time
import math
from typing import Dict, Any, Optional
from dataclasses import dataclass

from ..utils.logger import get_logger
from ..utils.config_loader import get_config


@dataclass
class PIDState:
    """PID控制器内部状态"""
    previous_error: float = 0.0
    integral: float = 0.0
    derivative: float = 0.0
    previous_time: float = 0.0
    
    
class PIDController:
    """PID控制器类"""
    
    def __init__(self, 
                 kp: float = 1.0,
                 ki: float = 0.0,
                 kd: float = 0.0,
                 max_integral: float = 10.0,
                 max_output: float = 100.0,
                 min_output: float = -100.0,
                 sample_time: float = 0.01):
        """
        初始化PID控制器
        
        Args:
            kp: 比例增益
            ki: 积分增益  
            kd: 微分增益
            max_integral: 积分饱和限制
            max_output: 最大输出
            min_output: 最小输出
            sample_time: 采样时间
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_integral = max_integral
        self.max_output = max_output
        self.min_output = min_output
        self.sample_time = sample_time
        
        # 内部状态
        self.state = PIDState()
        
        # 统计信息
        self.computation_count = 0
        self.computation_times = []
        
        # 日志
        self.logger = get_logger("PID")
        
    def compute(self, setpoint: float, measurement: float, dt: Optional[float] = None) -> float:
        """
        计算PID输出
        
        Args:
            setpoint: 设定值
            measurement: 测量值
            dt: 时间间隔，如果None则自动计算
            
        Returns:
            PID输出值
        """
        start_time = time.time()
        
        # 计算时间间隔
        if dt is None:
            current_time = time.time()
            if self.state.previous_time > 0:
                dt = current_time - self.state.previous_time
            else:
                dt = self.sample_time
            self.state.previous_time = current_time
        
        # 避免除零和负时间间隔
        dt = max(dt, 1e-6)
        
        # 计算误差
        error = setpoint - measurement
        
        # 比例项
        proportional = self.kp * error
        
        # 积分项
        self.state.integral += error * dt
        # 积分饱和限制
        self.state.integral = max(-self.max_integral, 
                                 min(self.max_integral, self.state.integral))
        integral = self.ki * self.state.integral
        
        # 微分项
        if self.computation_count > 0:  # 第一次计算时跳过微分项
            derivative_error = (error - self.state.previous_error) / dt
            self.state.derivative = derivative_error
        else:
            derivative_error = 0.0
            self.state.derivative = 0.0
        
        derivative = self.kd * derivative_error
        
        # PID输出
        output = proportional + integral + derivative
        
        # 输出限制
        output = max(self.min_output, min(self.max_output, output))
        
        # 更新状态
        self.state.previous_error = error
        self.computation_count += 1
        
        # 记录计算时间
        computation_time = time.time() - start_time
        self.computation_times.append(computation_time)
        
        return output
    
    def reset(self) -> None:
        """重置PID控制器状态"""
        self.state = PIDState()
        self.computation_count = 0
        self.computation_times.clear()
        
    def set_gains(self, kp: Optional[float] = None, 
                  ki: Optional[float] = None, 
                  kd: Optional[float] = None) -> None:
        """
        设置PID增益
        
        Args:
            kp: 比例增益
            ki: 积分增益
            kd: 微分增益
        """
        if kp is not None:
            self.kp = kp
        if ki is not None:
            self.ki = ki
        if kd is not None:
            self.kd = kd
            
        self.logger.info(f"PID增益更新: Kp={self.kp}, Ki={self.ki}, Kd={self.kd}")
    
    def set_output_limits(self, min_output: float, max_output: float) -> None:
        """设置输出限制"""
        self.min_output = min_output
        self.max_output = max_output
        
        # 重新限制积分项
        if abs(self.state.integral) > self.max_integral:
            self.state.integral = math.copysign(self.max_integral, self.state.integral)
    
    def get_components(self) -> Dict[str, float]:
        """获取PID各项分量"""
        return {
            'proportional': self.kp * self.state.previous_error,
            'integral': self.ki * self.state.integral,
            'derivative': self.kd * self.state.derivative,
            'error': self.state.previous_error
        }
    
    def get_statistics(self) -> Dict[str, float]:
        """获取统计信息"""
        if not self.computation_times:
            return {}
            
        import numpy as np
        return {
            'computation_count': self.computation_count,
            'avg_computation_time': np.mean(self.computation_times),
            'max_computation_time': np.max(self.computation_times),
            'integral_value': self.state.integral,
            'last_error': self.state.previous_error
        }


class DualPIDController:
    """双路PID控制器 - 用于舵角和速度控制"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化双路PID控制器
        
        Args:
            config: 配置字典
        """
        self.logger = get_logger("DualPID")
        self.config = config or get_config().get_pid_config()
        
        # 舵角PID控制器
        steering_config = self.config.get('steering', {})
        self.steering_pid = PIDController(
            kp=steering_config.get('kp', 1.2),
            ki=steering_config.get('ki', 0.1),
            kd=steering_config.get('kd', 0.05),
            max_integral=steering_config.get('max_integral', 10.0),
            max_output=steering_config.get('max_output', 30.0),
            min_output=-steering_config.get('max_output', 30.0)
        )
        
        # 速度PID控制器  
        speed_config = self.config.get('speed', {})
        self.speed_pid = PIDController(
            kp=speed_config.get('kp', 0.8),
            ki=speed_config.get('ki', 0.05),
            kd=speed_config.get('kd', 0.02),
            max_integral=speed_config.get('max_integral', 5.0),
            max_output=speed_config.get('max_output', 100.0),
            min_output=0.0  # 速度不能为负
        )
        
        # 控制模式
        self.steering_mode = 'angle'  # 'angle' 或 'rate'
        self.speed_mode = 'velocity'  # 'velocity' 或 'throttle'
        
        # 安全限制
        vehicle_config = get_config().get("vehicle", {})
        self.max_steering_angle = math.radians(vehicle_config.get("max_steering_angle", 30.0))
        self.max_speed = vehicle_config.get("max_speed", 3.0)
        
    def update_steering(self, 
                       target_angle: float, 
                       current_angle: float,
                       dt: Optional[float] = None) -> float:
        """
        更新舵角控制
        
        Args:
            target_angle: 目标舵角 (弧度)
            current_angle: 当前舵角 (弧度)
            dt: 时间间隔
            
        Returns:
            舵角控制输出
        """
        # 角度限制
        target_angle = max(-self.max_steering_angle, 
                          min(self.max_steering_angle, target_angle))
        
        # PID计算
        steering_output = self.steering_pid.compute(target_angle, current_angle, dt)
        
        return steering_output
    
    def update_speed(self, 
                    target_speed: float, 
                    current_speed: float,
                    dt: Optional[float] = None) -> float:
        """
        更新速度控制
        
        Args:
            target_speed: 目标速度 (m/s)
            current_speed: 当前速度 (m/s)  
            dt: 时间间隔
            
        Returns:
            速度控制输出 (油门百分比)
        """
        # 速度限制
        target_speed = max(0.0, min(self.max_speed, target_speed))
        
        # PID计算
        speed_output = self.speed_pid.compute(target_speed, current_speed, dt)
        
        return speed_output
    
    def compute_control(self, 
                       target_steering: float,
                       current_steering: float,
                       target_speed: float,
                       current_speed: float,
                       dt: Optional[float] = None) -> Dict[str, float]:
        """
        计算双路控制输出
        
        Args:
            target_steering: 目标舵角 (弧度)
            current_steering: 当前舵角 (弧度)
            target_speed: 目标速度 (m/s)
            current_speed: 当前速度 (m/s)
            dt: 时间间隔
            
        Returns:
            控制输出字典
        """
        # 舵角控制
        steering_output = self.update_steering(target_steering, current_steering, dt)
        
        # 速度控制
        speed_output = self.update_speed(target_speed, current_speed, dt)
        
        return {
            'steering_output': steering_output,      # 舵角输出 (度)
            'throttle_output': speed_output,        # 油门输出 (百分比)
            'target_steering_deg': math.degrees(target_steering),
            'current_steering_deg': math.degrees(current_steering),
            'target_speed': target_speed,
            'current_speed': current_speed,
            'steering_error': target_steering - current_steering,
            'speed_error': target_speed - current_speed
        }
    
    def get_tuning_info(self) -> Dict[str, Any]:
        """获取调参信息"""
        steering_components = self.steering_pid.get_components()
        speed_components = self.speed_pid.get_components()
        
        return {
            'steering_pid': {
                'gains': {'kp': self.steering_pid.kp, 'ki': self.steering_pid.ki, 'kd': self.steering_pid.kd},
                'components': steering_components,
                'statistics': self.steering_pid.get_statistics()
            },
            'speed_pid': {
                'gains': {'kp': self.speed_pid.kp, 'ki': self.speed_pid.ki, 'kd': self.speed_pid.kd},
                'components': speed_components,
                'statistics': self.speed_pid.get_statistics()
            }
        }
    
    def reset_controllers(self) -> None:
        """重置所有PID控制器"""
        self.steering_pid.reset()
        self.speed_pid.reset()
        self.logger.info("PID控制器已重置")
    
    def tune_gains(self, 
                  steering_gains: Optional[Dict[str, float]] = None,
                  speed_gains: Optional[Dict[str, float]] = None) -> None:
        """
        调整PID增益
        
        Args:
            steering_gains: 舵角PID增益字典 {'kp': x, 'ki': y, 'kd': z}
            speed_gains: 速度PID增益字典
        """
        if steering_gains:
            self.steering_pid.set_gains(**steering_gains)
            
        if speed_gains:
            self.speed_pid.set_gains(**speed_gains)


if __name__ == "__main__":
    # 测试PID控制器
    print("PID控制器测试...")
    
    # 单个PID控制器测试
    pid = PIDController(kp=1.0, ki=0.1, kd=0.05)
    
    # 模拟阶跃响应
    setpoint = 10.0
    measurement = 0.0
    dt = 0.01
    
    print("阶跃响应测试:")
    for i in range(100):
        output = pid.compute(setpoint, measurement, dt)
        
        # 简单的系统模拟 (一阶惯性环节)
        measurement += (output - measurement) * 0.1 * dt
        
        if i % 20 == 0:
            print(f"时间 {i*dt:.2f}s: 设定值={setpoint}, 测量值={measurement:.2f}, 输出={output:.2f}")
    
    # 显示PID各项分量
    components = pid.get_components()
    print("\\nPID各项分量:")
    for key, value in components.items():
        print(f"- {key}: {value:.3f}")
    
    # 测试双路PID控制器
    print("\\n双路PID控制器测试:")
    dual_pid = DualPIDController()
    
    control_result = dual_pid.compute_control(
        target_steering=math.radians(15),   # 15度舵角
        current_steering=math.radians(5),   # 当前5度
        target_speed=2.0,                   # 目标2m/s
        current_speed=1.5                   # 当前1.5m/s
    )
    
    print("控制结果:")
    for key, value in control_result.items():
        print(f"- {key}: {value:.3f}")
    
    # 调参信息
    tuning_info = dual_pid.get_tuning_info()
    print("\\n调参信息:")
    print(f"舵角PID增益: {tuning_info['steering_pid']['gains']}")
    print(f"速度PID增益: {tuning_info['speed_pid']['gains']}"))