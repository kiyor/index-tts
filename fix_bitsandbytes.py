#!/usr/bin/env python3
"""
修复 bitsandbytes 兼容性问题的脚本
通过设置环境变量来禁用 bitsandbytes 的 CUDA 功能
"""

import os
import sys

# 设置环境变量来禁用 bitsandbytes 的 CUDA 功能
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 临时隐藏 GPU，让 bitsandbytes 使用 CPU 版本
os.environ['BNB_CUDA_VERSION'] = ''
os.environ['DISABLE_BNB_CUDA_CHECK'] = '1'

# 导入前先设置环境
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 现在重新设置 CUDA 可见性
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    del os.environ['CUDA_VISIBLE_DEVICES']

# 导入 IndexTTS
try:
    from indextts.infer import IndexTTS
    print("✅ IndexTTS 导入成功！")
    
    # 测试基本功能
    tts = IndexTTS(model_dir="checkpoints", cfg_path="checkpoints/config.yaml")
    print("✅ IndexTTS 初始化成功！")
    
    # 运行测试
    voice = "test_data/input.wav"
    text = "大家好，这是一个测试。"
    output_path = "test_output.wav"
    
    if os.path.exists(voice):
        tts.infer(voice, text, output_path)
        print(f"✅ 推理完成！输出文件：{output_path}")
    else:
        print(f"⚠️  参考音频文件不存在：{voice}")
        print("请将参考音频文件放在 test_data/input.wav")
        
except Exception as e:
    print(f"❌ 错误：{e}")
    sys.exit(1) 