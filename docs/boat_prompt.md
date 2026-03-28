Role: 你是一个资深的自动驾驶系统架构师，专注于无人船（USV）开发。

Goal: 在boat文件夹中为我构建一个基于深度学习的视觉导航 Python 项目。

Hardware Context:
Compute: NVIDIA Jetson Orin Nano (JetPack 6.2)
Sensor: 一路的GMSL IMX390 Camera
Actuator: 双路无刷电机（由外部飞控接收控制信号）

Algorithm Pipeline:
Perception: 读取摄像头 → SegFormer 模型 (TensorRT加速) → 输出语义分割 Mask (水面+警戒线)。
Planning: Mask 后处理 → 提取左右警戒线坐标 → 拟合中线 → 纯追踪算法 (Pure Pursuit) 计算目标舵角。
Control: 计算 PID 控制信号 → 输出 steering_deg / throttle_pct / speed_mps 等控制信号字段。

Task Requirements:
Directory Structure: 项目文件夹结构应包含：
- config/: 存放参数文件 (settings.yaml)
- models/: 存放 ONNX 和 TensorRT engine 文件
- river_lane_pilot/: 核心 Python 源码包（perception, planning, control, utils 子模块）
- scripts/: 存放可执行脚本 (realtime_pilot.py 等)
- training/: 存放模型训练脚本

Modular Design:
代码结构应是模块化的。感知、规划、控制部分应该是独立的 Class，可以独立于 ROS2 运行（方便在笔记本上用视频测试算法）。

Output: 项目输出为控制信号数据，包括 steering_deg, throttle_pct, speed_mps, heading_deg, target_x_m, target_y_m, total_mileage_m 等字段，由外部飞控系统接收处理。