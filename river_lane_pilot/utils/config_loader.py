"""
配置加载器 - Configuration Loader
负责加载和管理系统配置参数
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigLoader:
    """配置文件加载器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置加载器
        
        Args:
            config_path: 配置文件路径，默认为项目根目录下的config/settings.yaml
        """
        if config_path is None:
            # 获取项目根目录
            project_root = Path(__file__).parents[2]
            config_path = project_root / "config" / "settings.yaml"
        
        self.config_path = Path(config_path)
        self._config = None
        self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """
        加载配置文件
        
        Returns:
            配置字典
        
        Raises:
            FileNotFoundError: 配置文件不存在
            yaml.YAMLError: YAML格式错误
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
            print(f"✓ 成功加载配置文件: {self.config_path}")
            return self._config
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"配置文件格式错误: {e}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置项值，支持嵌套键值访问
        
        Args:
            key_path: 配置项路径，如 "camera.width" 或 "pid_controller.steering.kp"
            default: 默认值
            
        Returns:
            配置值
        """
        if self._config is None:
            self.load_config()
        
        keys = key_path.split('.')
        value = self._config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """
        设置配置项值
        
        Args:
            key_path: 配置项路径
            value: 新值
        """
        if self._config is None:
            self.load_config()
        
        keys = key_path.split('.')
        config_dict = self._config
        
        # 导航到父级字典
        for key in keys[:-1]:
            if key not in config_dict:
                config_dict[key] = {}
            config_dict = config_dict[key]
        
        # 设置最终值
        config_dict[keys[-1]] = value
    
    def save_config(self, output_path: Optional[str] = None) -> None:
        """
        保存配置到文件
        
        Args:
            output_path: 输出路径，默认覆盖原文件
        """
        if output_path is None:
            output_path = self.config_path
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, default_flow_style=False, 
                     allow_unicode=True, indent=2)
        
        print(f"✓ 配置已保存至: {output_path}")
    
    def get_camera_config(self) -> Dict[str, Any]:
        """获取相机配置"""
        return self.get("camera", {})
    
    def get_segmentation_config(self) -> Dict[str, Any]:
        """获取语义分割配置"""
        return self.get("segmentation", {})
    
    def get_pure_pursuit_config(self) -> Dict[str, Any]:
        """获取Pure Pursuit算法配置"""
        return self.get("pure_pursuit", {})
    
    def get_pid_config(self) -> Dict[str, Any]:
        """获取PID控制配置"""  
        return self.get("pid_controller", {})
    
    def get_safety_config(self) -> Dict[str, Any]:
        """获取安全配置"""
        return self.get("safety", {})
    
    def get_debug_config(self) -> Dict[str, Any]:
        """获取调试配置"""
        return self.get("debug", {})
    
    def is_debug_enabled(self) -> bool:
        """检查是否启用调试模式"""
        return self.get("debug.enable_visualization", False)
    
    def is_simulation_mode(self) -> bool:
        """检查是否为仿真模式"""
        return self.get("simulation.enabled", False)
    
    @property
    def config(self) -> Dict[str, Any]:
        """获取完整配置字典"""
        if self._config is None:
            self.load_config()
        return self._config
    
    def __getitem__(self, key: str) -> Any:
        """支持字典风格访问"""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """支持字典风格设置"""
        self.set(key, value)


# 全局配置实例
_global_config = None

def get_config() -> ConfigLoader:
    """获取全局配置实例"""
    global _global_config
    if _global_config is None:
        _global_config = ConfigLoader()
    return _global_config


if __name__ == "__main__":
    # 测试配置加载器
    config = ConfigLoader()
    
    print("相机配置:", config.get_camera_config())
    print("分割模型配置:", config.get_segmentation_config())
    print("PID配置:", config.get_pid_config())
    print("调试模式:", config.is_debug_enabled())