#!/usr/bin/env python3
"""
GPU Memory Monitor for IndexTTS
"""

import gc
import os
import time
import logging
import threading
import torch
import psutil
import GPUtil
from typing import Optional

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/gpu_monitor.log'),
        logging.StreamHandler()
    ]
)

class GPUMemoryMonitor:
    def __init__(self, 
                 threshold_percent: float = 80,  # 降低阈值到80%
                 check_interval: int = 300,
                 force_gc_threshold: float = 90,  # 降低强制GC阈值到90%
                 aggressive_gc: bool = True):     # 添加激进清理选项
        """
        初始化显存监控器
        
        Args:
            threshold_percent: GPU使用率阈值，超过则触发清理
            check_interval: 检查间隔（秒）
            force_gc_threshold: 强制GC阈值，超过则强制执行GC
            aggressive_gc: 是否使用激进的清理策略
        """
        self.threshold = threshold_percent
        self.interval = check_interval
        self.force_gc_threshold = force_gc_threshold
        self.aggressive_gc = aggressive_gc
        self.running = True
        self.last_clean_time = 0
        self.clean_cooldown = 60  # 清理冷却时间（秒）
        
    def clean_memory(self, force: bool = False):
        """
        执行内存清理
        
        Args:
            force: 是否强制清理，忽略冷却时间
        """
        current_time = time.time()
        if not force and (current_time - self.last_clean_time) < self.clean_cooldown:
            logging.info(f"清理操作在冷却中，还需等待 {self.clean_cooldown - (current_time - self.last_clean_time):.1f} 秒")
            return
            
        try:
            usage = self.get_memory_usage()
            logging.info(f"开始内存清理. 当前使用率: {usage}%")
            
            # 1. Python GC
            gc.collect()
            logging.info("Python GC完成")
            
            # 2. PyTorch CUDA缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if self.aggressive_gc:
                    # 强制同步GPU
                    torch.cuda.synchronize()
                logging.info("PyTorch CUDA缓存已清理")
                
            # 3. 如果使用激进清理
            if self.aggressive_gc and (force or usage > self.force_gc_threshold):
                # 重置所有GPU设备
                for device in range(torch.cuda.device_count()):
                    with torch.cuda.device(device):
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                logging.info("执行了激进清理")
                
            # 更新最后清理时间
            self.last_clean_time = current_time
            
            # 检查清理效果
            new_usage = self.get_memory_usage()
            logging.info(f"清理后使用率: {new_usage}%")
            
            # 如果清理效果不明显且是强制清理，发出警告
            if force and (new_usage > usage - 5):
                logging.warning("清理效果不明显，可能需要重启服务")
                
        except Exception as e:
            logging.error(f"内存清理失败: {e}")
            
    def get_memory_usage(self) -> float:
        """获取显存使用率"""
        try:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                allocated = torch.cuda.memory_allocated(device)
                reserved = torch.cuda.memory_reserved(device)
                total = torch.cuda.get_device_properties(device).total_memory
                return (allocated + reserved) / total * 100
        except Exception as e:
            logging.error(f"获取显存信息失败: {e}")
        return 0.0
    
    def monitor_loop(self) -> None:
        """
        监控循环
        """
        logging.info("GPU显存监控启动")
        last_warning_time = 0
        warning_cooldown = 300  # 警告冷却时间（秒）
        
        while self.running:
            try:
                usage = self.get_memory_usage()
                current_time = time.time()
                
                # 超过强制GC阈值，立即清理
                if usage > self.force_gc_threshold:
                    logging.warning(f"显存使用率({usage:.2f}%)超过强制GC阈值({self.force_gc_threshold}%)")
                    self.clean_memory(force=True)
                    
                # 超过普通阈值，执行常规清理
                elif usage > self.threshold:
                    # 检查警告冷却时间
                    if (current_time - last_warning_time) > warning_cooldown:
                        logging.warning(f"显存使用率({usage:.2f}%)超过阈值({self.threshold}%)")
                        last_warning_time = current_time
                    self.clean_memory()
                else:
                    logging.debug(f"当前显存使用率: {usage:.2f}%")
                    
            except Exception as e:
                logging.error(f"监控循环错误: {e}")
                
            time.sleep(self.interval)
    
    def start(self) -> None:
        """
        启动监控线程
        """
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()
        logging.info("显存监控线程已启动")
    
    def stop(self) -> None:
        """
        停止监控
        """
        self.running = False
        logging.info("显存监控已停止")
        
    def get_memory_info(self) -> dict:
        """获取详细的显存信息"""
        info = {}
        try:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                info.update({
                    'gpu_memory_allocated': f"{torch.cuda.memory_allocated(device) / 1024**3:.2f} GB",
                    'gpu_memory_reserved': f"{torch.cuda.memory_reserved(device) / 1024**3:.2f} GB",
                    'gpu_memory_total': f"{torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB",
                    'gpu_utilization': f"{self.get_memory_usage():.2f}%"
                })
        except Exception as e:
            logging.error(f"获取显存详细信息失败: {e}")
        return info 