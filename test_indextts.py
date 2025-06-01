#!/usr/bin/env python3
"""
IndexTTS 测试脚本
"""

import sys
import os

# 修复 bitsandbytes 兼容性问题
sys.modules['bitsandbytes'] = None

from indextts.infer import IndexTTS

def test_indextts():
    """测试 IndexTTS 基本功能"""
    print("🚀 开始测试 IndexTTS...")
    
    # 检查模型文件
    if not os.path.exists("checkpoints/config.yaml"):
        print("❌ 错误：找不到配置文件 checkpoints/config.yaml")
        return False
    
    if not os.path.exists("checkpoints/gpt.pth"):
        print("❌ 错误：找不到 GPT 模型文件 checkpoints/gpt.pth")
        return False
    
    if not os.path.exists("checkpoints/bigvgan_generator.pth"):
        print("❌ 错误：找不到 BigVGAN 模型文件 checkpoints/bigvgan_generator.pth")
        return False
    
    # 检查测试音频
    if not os.path.exists("test_data/input.wav"):
        print("⚠️  创建测试音频文件...")
        os.makedirs("test_data", exist_ok=True)
        import torch
        import torchaudio
        
        # 创建测试音频
        sample_rate = 22050
        duration = 1.0
        t = torch.linspace(0, duration, int(sample_rate * duration))
        frequency = 440
        audio = 0.3 * torch.sin(2 * torch.pi * frequency * t)
        audio = audio.unsqueeze(0)
        torchaudio.save('test_data/input.wav', audio, sample_rate)
        print("✅ 测试音频文件已创建")
    
    try:
        # 初始化 IndexTTS
        print("📦 初始化 IndexTTS...")
        tts = IndexTTS(
            cfg_path="checkpoints/config.yaml", 
            model_dir="checkpoints",
            is_fp16=True
        )
        print("✅ IndexTTS 初始化成功！")
        
        # 测试中文
        print("🗣️  测试中文语音合成...")
        chinese_text = "大家好，这是一个中文语音合成测试。"
        tts.infer(
            audio_prompt="test_data/input.wav",
            text=chinese_text,
            output_path="test_chinese.wav",
            verbose=False
        )
        print("✅ 中文测试完成：test_chinese.wav")
        
        # 测试英文
        print("🗣️  测试英文语音合成...")
        english_text = "Hello, this is an English text-to-speech test."
        tts.infer(
            audio_prompt="test_data/input.wav",
            text=english_text,
            output_path="test_english.wav",
            verbose=False
        )
        print("✅ 英文测试完成：test_english.wav")
        
        # 测试快速推理
        print("⚡ 测试快速推理模式...")
        long_text = "这是一个长文本测试。IndexTTS 是一个工业级可控高效的零样本文本转语音系统。它支持中文拼音纠音，支持标点符号控制停顿。"
        tts.infer_fast(
            audio_prompt="test_data/input.wav",
            text=long_text,
            output_path="test_fast.wav",
            verbose=False
        )
        print("✅ 快速推理测试完成：test_fast.wav")
        
        print("\n🎉 所有测试通过！IndexTTS 运行正常！")
        print("\n📁 生成的文件：")
        for file in ["test_chinese.wav", "test_english.wav", "test_fast.wav"]:
            if os.path.exists(file):
                size = os.path.getsize(file) / 1024
                print(f"   {file} ({size:.1f} KB)")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_indextts()
    sys.exit(0 if success else 1) 