"""
摄像头接口 - Camera Interface
处理GMSL IMX390相机输入和视频流管理
"""

import cv2
import numpy as np
import threading
import time
from typing import Optional, Dict, Any, Callable
from queue import Queue, Empty
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.config_loader import get_config


class CameraInterface:
    """相机接口类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化相机接口
        
        Args:
            config: 相机配置字典
        """
        self.logger = get_logger("Camera")
        self.config = config or get_config().get_camera_config()
        
        # 相机参数
        self.device_id = self.config.get("device_id", 0)
        self.width = self.config.get("width", 1920)
        self.height = self.config.get("height", 1080)
        self.fps = self.config.get("fps", 30)
        self.exposure = self.config.get("exposure", -1)  # -1表示自动
        self.gain = self.config.get("gain", -1)  # -1表示自动
        
        # 相机对象和状态
        self.camera = None
        self.is_opened = False
        self.is_streaming = False
        self.frame_count = 0
        
        # 线程安全的帧队列
        self.frame_queue = Queue(maxsize=10)
        self.capture_thread = None
        self.stop_event = threading.Event()
        
        # 统计信息
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.actual_fps = 0.0
        self.dropped_frames = 0
        
        # 回调函数
        self.frame_callback = None
        
        # 测试模式
        self.test_mode = False
        self.test_video_path = None
        self.test_image_folder = None
        
    def set_test_video(self, video_path: str) -> None:
        """
        设置测试视频模式
        
        Args:
            video_path: 视频文件路径
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"测试视频文件不存在: {video_path}")
        
        self.test_mode = True
        self.test_video_path = video_path
        self.logger.info(f"设置测试视频: {video_path}")
    
    def set_test_images(self, image_folder: str) -> None:
        """
        设置测试图像序列模式
        
        Args:
            image_folder: 图像文件夹路径
        """
        folder_path = Path(image_folder)
        if not folder_path.exists():
            raise FileNotFoundError(f"测试图像文件夹不存在: {image_folder}")
        
        self.test_mode = True
        self.test_image_folder = image_folder
        self.logger.info(f"设置测试图像文件夹: {image_folder}")
    
    def open_camera(self) -> bool:
        """
        打开相机
        
        Returns:
            是否成功打开
        """
        try:
            if self.test_mode:
                return self._open_test_source()
            else:
                return self._open_hardware_camera()
                
        except Exception as e:
            self.logger.error(f"打开相机失败: {e}")
            return False
    
    def _open_hardware_camera(self) -> bool:
        """打开硬件相机"""
        # 尝试不同的相机后端
        backends = [
            cv2.CAP_V4L2,      # Linux V4L2
            cv2.CAP_GSTREAMER,  # GStreamer (NVIDIA推荐)
            cv2.CAP_ANY         # 默认后端
        ]
        
        for backend in backends:
            try:
                self.logger.info(f"尝试使用后端: {backend}")
                self.camera = cv2.VideoCapture(self.device_id, backend)
                
                if self.camera.isOpened():
                    # 设置相机参数
                    self._configure_camera()
                    
                    # 测试读取一帧
                    ret, frame = self.camera.read()
                    if ret and frame is not None:
                        self.is_opened = True
                        self.logger.info(f"✓ 相机打开成功 (后端: {backend})")
                        self.logger.info(f"分辨率: {frame.shape[1]}x{frame.shape[0]}")
                        return True
                    else:
                        self.camera.release()
                        
            except Exception as e:
                self.logger.warning(f"后端 {backend} 打开失败: {e}")
                continue
        
        self.logger.error("所有后端都打开失败")
        return False
    
    def _open_test_source(self) -> bool:
        """打开测试源（视频文件或图像序列）"""
        if self.test_video_path:
            self.camera = cv2.VideoCapture(self.test_video_path)
            if self.camera.isOpened():
                self.is_opened = True
                self.logger.info(f"✓ 测试视频打开成功: {self.test_video_path}")
                return True
        
        elif self.test_image_folder:
            # 图像序列模式暂时不实现详细功能
            self.is_opened = True
            self.logger.info(f"✓ 测试图像文件夹设置成功: {self.test_image_folder}")
            return True
        
        return False
    
    def _configure_camera(self) -> None:
        """配置相机参数"""
        if not self.camera:
            return
        
        try:
            # 设置分辨率
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # 设置帧率
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)
            
            # 设置曝光（如果不是自动）
            if self.exposure != -1:
                self.camera.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
                self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 手动曝光
            
            # 设置增益（如果不是自动）
            if self.gain != -1:
                self.camera.set(cv2.CAP_PROP_GAIN, self.gain)
            
            # 设置缓冲区大小（减少延迟）
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # 获取实际设置的参数
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f"相机配置完成:")
            self.logger.info(f"- 分辨率: {actual_width}x{actual_height}")
            self.logger.info(f"- 帧率: {actual_fps}")
            
        except Exception as e:
            self.logger.warning(f"设置相机参数失败: {e}")
    
    def start_streaming(self, callback: Optional[Callable] = None) -> bool:
        """
        开始视频流
        
        Args:
            callback: 帧回调函数 callback(frame, timestamp)
            
        Returns:
            是否成功开始
        """
        if not self.is_opened:
            self.logger.error("相机未打开，无法开始流")
            return False
        
        if self.is_streaming:
            self.logger.warning("视频流已在运行")
            return True
        
        self.frame_callback = callback
        self.stop_event.clear()
        
        # 启动捕获线程
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        self.is_streaming = True
        self.logger.info("✓ 视频流已开始")
        return True
    
    def _capture_loop(self) -> None:
        """相机捕获循环（在独立线程中运行）"""
        self.fps_counter = 0
        self.last_fps_time = time.time()
        
        while not self.stop_event.is_set():
            try:
                ret, frame = self.camera.read()
                
                if not ret or frame is None:
                    self.logger.warning("读取帧失败")
                    if self.test_mode and self.test_video_path:
                        # 视频结束，重新开始
                        self.camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break
                
                timestamp = time.time()
                self.frame_count += 1
                
                # 更新FPS计数
                self._update_fps()
                
                # 将帧加入队列
                try:
                    self.frame_queue.put_nowait((frame.copy(), timestamp))
                except:
                    # 队列满，丢弃旧帧
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait((frame.copy(), timestamp))
                        self.dropped_frames += 1
                    except Empty:
                        pass
                
                # 调用回调函数
                if self.frame_callback:
                    try:
                        self.frame_callback(frame, timestamp)
                    except Exception as e:
                        self.logger.error(f"帧回调函数执行失败: {e}")
                
                # 控制帧率
                if not self.test_mode:
                    time.sleep(1.0 / self.fps * 0.1)  # 稍微减慢以防CPU过载
                
            except Exception as e:
                self.logger.error(f"捕获循环异常: {e}")
                break
        
        self.logger.info("相机捕获循环结束")
    
    def _update_fps(self) -> None:
        """更新FPS统计"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.actual_fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        获取最新帧（阻塞方式）
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            图像帧，如果超时返回None
        """
        try:
            frame, timestamp = self.frame_queue.get(timeout=timeout)
            return frame
        except Empty:
            return None
    
    def get_frame_with_timestamp(self, timeout: float = 1.0) -> Optional[tuple]:
        """
        获取最新帧和时间戳
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            (frame, timestamp) 或 None
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def read_frame(self) -> Optional[np.ndarray]:
        """
        直接读取一帧（非阻塞，用于同步模式）
        
        Returns:
            图像帧或None
        """
        if not self.is_opened:
            return None
        
        try:
            ret, frame = self.camera.read()
            if ret and frame is not None:
                self.frame_count += 1
                return frame
            return None
        except Exception as e:
            self.logger.error(f"直接读取帧失败: {e}")
            return None
    
    def stop_streaming(self) -> None:
        """停止视频流"""
        if not self.is_streaming:
            return
        
        self.stop_event.set()
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        self.is_streaming = False
        self.logger.info("视频流已停止")
    
    def close_camera(self) -> None:
        """关闭相机"""
        self.stop_streaming()
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        self.is_opened = False
        self.logger.info("相机已关闭")
    
    def get_camera_info(self) -> Dict[str, Any]:
        """获取相机信息"""
        info = {
            'device_id': self.device_id,
            'is_opened': self.is_opened,
            'is_streaming': self.is_streaming,
            'frame_count': self.frame_count,
            'actual_fps': self.actual_fps,
            'dropped_frames': self.dropped_frames,
            'test_mode': self.test_mode
        }
        
        if self.camera and self.is_opened:
            info.update({
                'width': int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': self.camera.get(cv2.CAP_PROP_FPS),
                'exposure': self.camera.get(cv2.CAP_PROP_EXPOSURE),
                'gain': self.camera.get(cv2.CAP_PROP_GAIN),
            })
        
        return info
    
    def reset_statistics(self) -> None:
        """重置统计信息"""
        self.frame_count = 0
        self.dropped_frames = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.actual_fps = 0.0
    
    def __enter__(self):
        """上下文管理器入口"""
        self.open_camera()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close_camera()
    
    def __del__(self):
        """析构函数"""
        self.close_camera()


if __name__ == "__main__":
    # 测试相机接口
    camera = CameraInterface()
    
    print("相机接口测试...")
    
    # 设置测试模式（因为可能没有实际的GMSL相机）
    camera.test_mode = True
    camera.device_id = 0  # 使用默认相机或视频文件
    
    try:
        # 打开相机
        if camera.open_camera():
            print("✓ 相机打开成功")
            
            # 获取相机信息
            info = camera.get_camera_info()
            print("相机信息:")
            for key, value in info.items():
                print(f"  {key}: {value}")
            
            # 测试单帧读取
            frame = camera.read_frame()
            if frame is not None:
                print(f"✓ 读取帧成功: {frame.shape}")
            
            # 测试回调函数
            def frame_callback(frame, timestamp):
                print(f"回调: 帧大小 {frame.shape}, 时间戳 {timestamp:.3f}")
            
            # 开始流（测试3秒）
            camera.start_streaming(frame_callback)
            time.sleep(3)
            
            # 获取统计信息
            info = camera.get_camera_info()
            print(f"最终统计: FPS={info['actual_fps']:.1f}, "
                  f"总帧数={info['frame_count']}, "
                  f"丢弃帧数={info['dropped_frames']}")
            
        else:
            print("✗ 相机打开失败")
    
    finally:
        camera.close_camera()
        print("相机测试完成")