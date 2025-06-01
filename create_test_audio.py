#!/usr/bin/env python3
"""
创建测试音频文件
"""

import torch
import torchaudio
import numpy as np

def create_test_audio():
    # 创建一个简单的测试音频（1秒的正弦波）
    sample_rate = 22050
    duration = 1.0
    t = torch.linspace(0, duration, int(sample_rate * duration))
    frequency = 440  # A4 音符
    audio = 0.3 * torch.sin(2 * torch.pi * frequency * t)
    audio = audio.unsqueeze(0)  # 添加通道维度

    # 保存为测试文件
    torchaudio.save('test_data/input.wav', audio, sample_rate)
    print('测试音频文件已创建：test_data/input.wav')

if __name__ == "__main__":
    create_test_audio() 