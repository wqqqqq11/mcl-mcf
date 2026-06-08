# -*- coding: utf-8 -*-
import os
import sys
from datetime import datetime


class Logger:
    """训练日志记录器"""
    
    def __init__(self, log_dir='outputs/logs'):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'train_{timestamp}.log')
        
        # 同时输出到文件和控制台
        self.terminal = sys.stdout
        self.log = open(self.log_file, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


def init_logger(log_dir='outputs/logs'):
    """初始化日志记录，重定向 stdout"""
    logger = Logger(log_dir)
    sys.stdout = logger
    return logger


def log_metrics(metrics, epoch=None, phase='train'):
    """记录训练指标
    
    Args:
        metrics: dict, 包含各项指标
        epoch: int, 当前轮数
        phase: str, 阶段 (train/valid/test)
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    if epoch is not None:
        print(f"[{timestamp}] [{phase.upper()}] Epoch {epoch}")
    else:
        print(f"[{timestamp}] [{phase.upper()}]")
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    print()


def log_config(config):
    """记录训练配置"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] [CONFIG] Training Configuration:")
    
    if isinstance(config, dict):
        for key, value in config.items():
            print(f"  {key}: {value}")
    else:
        # 处理 argparse Namespace
        for key in dir(config):
            if not key.startswith('_'):
                print(f"  {key}: {getattr(config, key)}")
    print()
