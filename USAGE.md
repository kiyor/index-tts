# IndexTTS 使用说明

## 问题解决

### bitsandbytes 兼容性问题已解决 ✅

原问题：Tesla P4 GPU (计算能力 6.1) 与 CUDA 12.4 环境下 bitsandbytes 库不兼容。

**解决方案：**
1. 在 `indextts/infer.py` 开头添加了 bitsandbytes 修复代码
2. 通过 `sys.modules['bitsandbytes'] = None` 禁用 bitsandbytes 导入
3. 系统自动回退到标准推理模式

## 快速开始

### 1. 激活环境
```bash
conda activate index-tts
```

### 2. 运行测试
```bash
python test_indextts.py
```

### 3. 基本使用
```python
import sys
sys.modules['bitsandbytes'] = None  # 修复兼容性

from indextts.infer import IndexTTS

# 初始化
tts = IndexTTS(
    cfg_path="checkpoints/config.yaml", 
    model_dir="checkpoints",
    is_fp16=True
)

# 语音合成
tts.infer(
    audio_prompt="test_data/input.wav",  # 参考音频
    text="你好，这是一个测试。",        # 要合成的文本
    output_path="output.wav"            # 输出文件
)
```

### 4. 快速推理（长文本）
```python
# 对于长文本，使用快速推理模式
tts.infer_fast(
    audio_prompt="test_data/input.wav",
    text="这是一个很长的文本...",
    output_path="output_fast.wav",
    max_text_tokens_per_sentence=100,    # 分句最大token数
    sentences_bucket_max_size=4          # 批处理大小
)
```

## 性能信息

- **GPU**: Tesla P4 (6.1 计算能力)
- **CUDA**: 12.4
- **推理模式**: 标准推理 (无 DeepSpeed)
- **BigVGAN**: 回退到 torch 实现 (无自定义 CUDA 内核)
- **RTF**: ~1.5-1.8 (实时因子，越小越快)

## 注意事项

1. **DeepSpeed 未安装**: 系统自动回退到标准推理，性能略有影响但功能正常
2. **BigVGAN CUDA 内核**: 未编译自定义内核，使用 torch 实现，性能略有影响
3. **bitsandbytes**: 已禁用，避免兼容性问题
4. **内存使用**: 建议使用快速推理模式处理长文本

## 文件说明

- `test_indextts.py`: 完整测试脚本
- `create_test_audio.py`: 创建测试音频
- `fix_bitsandbytes.py`: 兼容性修复脚本
- `indextts/fix_imports.py`: 导入修复模块

## 故障排除

如果遇到问题，请检查：
1. 模型文件是否存在于 `checkpoints/` 目录
2. 环境是否正确激活
3. 参考音频文件是否存在

运行测试脚本可以快速诊断问题：
```bash
python test_indextts.py
``` 