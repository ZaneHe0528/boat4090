"""
River Lane Pilot - 无人船河道航行系统
核心Python包
"""

__version__ = "1.0.0"
__author__ = "River Lane Pilot Team"
__email__ = "contact@riverlane-pilot.com"

from .perception import SegFormerModel, LaneDetector, CameraInterface
from .planning import PurePursuit, PathProcessor, TrajectoryPlanner  
from .control import PIDController
from .utils import ConfigLoader, Logger, Visualizer

__all__ = [
    # Perception
    "SegFormerModel", 
    "LaneDetector", 
    "CameraInterface",
    
    # Planning
    "PurePursuit", 
    "PathProcessor", 
    "TrajectoryPlanner",
    
    # Control
    "PIDController", 
    
    # Utils
    "ConfigLoader", 
    "Logger", 
    "Visualizer"
]