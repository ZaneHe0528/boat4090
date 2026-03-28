"""
日志系统 - Logger System
统一的日志管理和输出系统
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from loguru import logger


class Logger:
    """统一日志管理器"""
    
    def __init__(self, 
                 name: str = "RiverLanePilot",
                 log_level: str = "INFO",
                 log_dir: Optional[str] = None,
                 enable_file_logging: bool = True,
                 enable_console_logging: bool = True):
        """
        初始化日志器
        
        Args:
            name: 日志器名称
            log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: 日志文件目录
            enable_file_logging: 是否启用文件日志
            enable_console_logging: 是否启用控制台日志
        """
        self.name = name
        self.log_level = log_level.upper()
        
        # 设置日志目录
        if log_dir is None:
            project_root = Path(__file__).parents[2]
            log_dir = project_root / "logs"
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # 清除默认处理器
        logger.remove()
        
        # 设置控制台日志
        if enable_console_logging:
            logger.add(
                sys.stdout,
                colorize=True,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                       "<level>{level: <8}</level> | "
                       "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                       "<level>{message}</level>",
                level=self.log_level
            )
        
        # 设置文件日志
        if enable_file_logging:
            # 主日志文件
            main_log_file = self.log_dir / f"{name.lower()}.log"
            logger.add(
                str(main_log_file),
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
                       "{name}:{function}:{line} | {message}",
                level=self.log_level,
                rotation="10 MB",
                retention="7 days",
                compression="zip",
                encoding="utf-8"
            )
            
            # 错误专用日志文件
            error_log_file = self.log_dir / f"{name.lower()}_error.log"
            logger.add(
                str(error_log_file),
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
                       "{name}:{function}:{line} | {message} | {extra}",
                level="ERROR",
                rotation="5 MB",
                retention="30 days",
                compression="zip",
                encoding="utf-8"
            )
        
        # 绑定日志器名称
        self.logger = logger.bind(name=name)
    
    def debug(self, message: str, **kwargs) -> None:
        """调试日志"""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """信息日志"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """警告日志"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """错误日志"""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """严重错误日志"""
        self.logger.critical(message, **kwargs)
    
    def exception(self, message: str, **kwargs) -> None:
        """异常日志（包含堆栈跟踪）"""
        self.logger.exception(message, **kwargs)
    
    def log_performance(self, function_name: str, execution_time: float) -> None:
        """记录性能指标"""
        self.logger.info(f"Performance | {function_name} | {execution_time:.4f}s")
    
    def log_ros_topic(self, topic_name: str, message_type: str, frequency: float) -> None:
        """记录ROS话题信息"""
        self.logger.debug(f"ROS Topic | {topic_name} | {message_type} | {frequency:.2f}Hz")
    
    def log_system_status(self, component: str, status: str, details: str = "") -> None:
        """记录系统组件状态"""
        self.logger.info(f"System | {component} | {status} | {details}")
    
    def log_algorithm_result(self, algorithm: str, result: dict) -> None:
        """记录算法执行结果"""
        result_str = ", ".join([f"{k}={v}" for k, v in result.items()])
        self.logger.info(f"Algorithm | {algorithm} | {result_str}")


class PerformanceLogger:
    """性能监控日志器"""
    
    def __init__(self, logger: Logger):
        self.logger = logger
        self.start_times = {}
    
    def start_timer(self, name: str) -> None:
        """开始计时"""
        self.start_times[name] = datetime.now()
    
    def end_timer(self, name: str) -> float:
        """结束计时并记录"""
        if name not in self.start_times:
            self.logger.warning(f"Timer '{name}' was not started")
            return 0.0
        
        end_time = datetime.now()
        duration = (end_time - self.start_times[name]).total_seconds()
        self.logger.log_performance(name, duration)
        
        del self.start_times[name]
        return duration
    
    def time_function(self, func_name: str = None):
        """装饰器：自动记录函数执行时间"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                timer_name = func_name or f"{func.__module__}.{func.__name__}"
                self.start_timer(timer_name)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    self.end_timer(timer_name)
            return wrapper
        return decorator


# 全局日志实例
_global_logger = None
_performance_logger = None

def get_logger(name: str = "RiverLanePilot") -> Logger:
    """获取全局日志实例"""
    global _global_logger
    if _global_logger is None:
        _global_logger = Logger(name)
    return _global_logger

def get_performance_logger() -> PerformanceLogger:
    """获取性能日志实例"""
    global _performance_logger
    if _performance_logger is None:
        _performance_logger = PerformanceLogger(get_logger())
    return _performance_logger


if __name__ == "__main__":
    # 测试日志系统
    logger = Logger("TestLogger")
    
    logger.info("系统启动")
    logger.debug("调试信息")
    logger.warning("警告信息")
    logger.error("错误信息")
    
    # 测试性能日志
    perf_logger = PerformanceLogger(logger)
    
    perf_logger.start_timer("test_operation")
    import time
    time.sleep(0.1)
    perf_logger.end_timer("test_operation")
    
    # 测试装饰器
    @perf_logger.time_function("test_function")
    def test_func():
        time.sleep(0.05)
        return "完成"
    
    result = test_func()
    logger.info(f"函数结果: {result}")