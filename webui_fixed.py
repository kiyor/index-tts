#!/usr/bin/env python3
"""
IndexTTS WebUI - 修复版本
包含 bitsandbytes 兼容性修复和 Docker 支持
"""

# 修复 bitsandbytes 兼容性问题
import sys
sys.modules['bitsandbytes'] = None

import json
import os
import sys
import threading
import time

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import argparse
parser = argparse.ArgumentParser(description="IndexTTS WebUI (Fixed)")
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
parser.add_argument("--port", type=int, default=7860, help="Port to run the web UI on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the web UI on (0.0.0.0 for Docker)")
parser.add_argument("--model_dir", type=str, default="checkpoints", help="Model checkpoints directory")
parser.add_argument("--share", action="store_true", default=False, help="Create a publicly shareable link")
cmd_args = parser.parse_args()

print("🚀 IndexTTS WebUI (修复版本)")
print("=" * 50)

# 检查模型文件
if not os.path.exists(cmd_args.model_dir):
    print(f"❌ 模型目录 {cmd_args.model_dir} 不存在。请先下载模型。")
    sys.exit(1)

required_files = [
    "bigvgan_generator.pth",
    "bpe.model", 
    "gpt.pth",
    "config.yaml",
]

missing_files = []
for file in required_files:
    file_path = os.path.join(cmd_args.model_dir, file)
    if not os.path.exists(file_path):
        missing_files.append(file_path)

if missing_files:
    print("❌ 缺少必需文件:")
    for file in missing_files:
        print(f"   - {file}")
    print("\n请确保所有模型文件都已下载到 checkpoints 目录")
    sys.exit(1)

print("✅ 模型文件检查通过")

# 检查 GPU
try:
    import torch
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🎮 检测到 {gpu_count} 个 GPU: {gpu_name}")
    else:
        print("⚠️  未检测到 GPU，将使用 CPU 模式")
except Exception as e:
    print(f"⚠️  GPU 检查失败: {e}")

import gradio as gr

try:
    from indextts.infer import IndexTTS
    print("✅ IndexTTS 导入成功")
except ImportError as e:
    print(f"❌ IndexTTS 导入失败: {e}")
    sys.exit(1)

# 尝试导入 i18n，如果失败则使用简单的替代方案
try:
    from tools.i18n.i18n import I18nAuto
    i18n = I18nAuto(language="zh_CN")
    print("✅ i18n 模块加载成功")
except ImportError:
    print("⚠️  i18n 模块导入失败，使用默认语言")
    class SimpleI18n:
        def __call__(self, text):
            return text
    i18n = SimpleI18n()

print("🔧 正在初始化 IndexTTS...")
try:
    tts = IndexTTS(
        model_dir=cmd_args.model_dir, 
        cfg_path=os.path.join(cmd_args.model_dir, "config.yaml")
    )
    print("✅ IndexTTS 初始化成功")
except Exception as e:
    print(f"❌ IndexTTS 初始化失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 创建输出目录
os.makedirs("outputs", exist_ok=True)
os.makedirs("prompts", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# 加载示例案例（如果存在）
example_cases = []
if os.path.exists("tests/cases.jsonl"):
    try:
        with open("tests/cases.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                example = json.loads(line)
                example_cases.append([
                    os.path.join("tests", example.get("prompt_audio", "sample_prompt.wav")),
                    example.get("text"), 
                    ["普通推理", "批次推理"][example.get("infer_mode", 0)]
                ])
        print(f"✅ 加载了 {len(example_cases)} 个示例案例")
    except Exception as e:
        print(f"⚠️  加载示例案例失败: {e}")

def gen_single(prompt, text, infer_mode, max_text_tokens_per_sentence=120, sentences_bucket_max_size=4,
                *args, progress=gr.Progress()):
    """生成语音的主函数"""
    if not prompt:
        gr.Warning("请上传参考音频文件")
        return gr.update(value=None, visible=True)
    
    if not text or not text.strip():
        gr.Warning("请输入要合成的文本")
        return gr.update(value=None, visible=True)
    
    output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")
    
    # 设置进度条
    tts.gr_progress = progress
    
    # 解析参数
    do_sample, top_p, top_k, temperature, \
        length_penalty, num_beams, repetition_penalty, max_mel_tokens = args
    
    kwargs = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": int(num_beams),
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
    }
    
    try:
        start_time = time.time()
        if infer_mode == "普通推理":
            output = tts.infer(
                prompt, text, output_path, 
                verbose=cmd_args.verbose,
                max_text_tokens_per_sentence=int(max_text_tokens_per_sentence),
                **kwargs
            )
        else:
            # 批次推理
            output = tts.infer_fast(
                prompt, text, output_path, 
                verbose=cmd_args.verbose,
                max_text_tokens_per_sentence=int(max_text_tokens_per_sentence),
                sentences_bucket_max_size=int(sentences_bucket_max_size),
                **kwargs
            )
        
        end_time = time.time()
        gr.Info(f"✅ 生成完成！耗时 {end_time - start_time:.2f} 秒")
        return gr.update(value=output, visible=True)
        
    except Exception as e:
        error_msg = f"生成失败: {str(e)}"
        print(f"❌ {error_msg}")
        gr.Error(error_msg)
        return gr.update(value=None, visible=True)

def update_prompt_audio():
    """更新提示音频按钮状态"""
    return gr.update(interactive=True)

def on_input_text_change(text, max_tokens_per_sentence):
    """文本输入变化时的回调函数"""
    if text and len(text.strip()) > 0:
        try:
            text_tokens_list = tts.tokenizer.tokenize(text)
            sentences = tts.tokenizer.split_sentences(
                text_tokens_list, 
                max_tokens_per_sentence=int(max_tokens_per_sentence)
            )
            
            data = []
            for i, s in enumerate(sentences):
                sentence_str = ''.join(s)
                tokens_count = len(s)
                data.append([i + 1, sentence_str, tokens_count])
            
            return gr.update(value=data, visible=True)
        except Exception as e:
            print(f"⚠️  文本处理失败: {e}")
            return gr.update(value=[], visible=True)
    else:
        return gr.update(value=[], visible=True)

# 创建 Gradio 界面
with gr.Blocks(
    title="IndexTTS Demo - Fixed", 
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 1200px !important;
    }
    .header {
        text-align: center;
        margin-bottom: 20px;
    }
    """
) as demo:
    gr.HTML('''
    <div class="header">
        <h1>🎤 IndexTTS: 工业级零样本文本转语音系统</h1>
        <h3>IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot TTS System</h3>
        <p>
            <a href='https://arxiv.org/abs/2502.05512'><img src='https://img.shields.io/badge/ArXiv-2502.05512-red'></a>
            <img src='https://img.shields.io/badge/Status-Fixed-green'>
            <img src='https://img.shields.io/badge/Docker-Supported-blue'>
        </p>
        <p><strong>✅ 已修复 bitsandbytes 兼容性问题 | 支持 Docker 部署</strong></p>
    </div>
    ''')
    
    with gr.Tab("🎵 音频生成"):
        with gr.Row():
            with gr.Column(scale=1):
                prompt_audio = gr.Audio(
                    label="📎 参考音频", 
                    sources=["upload", "microphone"],
                    type="filepath"
                )
                
            with gr.Column(scale=2):
                input_text_single = gr.TextArea(
                    label="📝 目标文本",
                    placeholder="请输入要合成的文本...",
                    info=f"当前模型版本: {getattr(tts, 'model_version', None) or '1.0'}",
                    lines=3
                )
                
                with gr.Row():
                    infer_mode = gr.Radio(
                        choices=["普通推理", "批次推理"], 
                        label="⚡ 推理模式",
                        info="批次推理：更适合长句，性能翻倍",
                        value="普通推理"
                    )
                    gen_button = gr.Button("🎯 生成语音", variant="primary", size="lg")
        
        output_audio = gr.Audio(label="🎵 生成结果", visible=True)
        
        # 高级参数设置
        with gr.Accordion("⚙️ 高级参数设置", open=False):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**🎛️ GPT2 采样设置**")
                    with gr.Row():
                        do_sample = gr.Checkbox(label="启用采样", value=True)
                        temperature = gr.Slider(
                            label="温度", minimum=0.1, maximum=2.0, value=1.0, step=0.1
                        )
                    with gr.Row():
                        top_p = gr.Slider(
                            label="Top-p", minimum=0.0, maximum=1.0, value=0.8, step=0.01
                        )
                        top_k = gr.Slider(
                            label="Top-k", minimum=0, maximum=100, value=30, step=1
                        )
                    with gr.Row():
                        num_beams = gr.Slider(
                            label="束搜索数量", value=3, minimum=1, maximum=10, step=1
                        )
                        repetition_penalty = gr.Number(
                            label="重复惩罚", value=10.0, minimum=0.1, maximum=20.0, step=0.1
                        )
                    with gr.Row():
                        length_penalty = gr.Number(
                            label="长度惩罚", value=0.0, minimum=-2.0, maximum=2.0, step=0.1
                        )
                        max_mel_tokens = gr.Slider(
                            label="最大Mel token数", 
                            value=600, 
                            minimum=50, 
                            maximum=getattr(tts.cfg.gpt, 'max_mel_tokens', 1000), 
                            step=10
                        )
                
                with gr.Column(scale=1):
                    gr.Markdown("**✂️ 分句设置**")
                    max_text_tokens_per_sentence = gr.Slider(
                        label="分句最大Token数", 
                        value=120, 
                        minimum=20, 
                        maximum=getattr(tts.cfg.gpt, 'max_text_tokens', 300), 
                        step=2,
                        info="建议80~200之间"
                    )
                    sentences_bucket_max_size = gr.Slider(
                        label="分句分桶大小（批次推理）", 
                        value=4, 
                        minimum=1, 
                        maximum=16, 
                        step=1,
                        info="建议2-8之间"
                    )
                    
                    with gr.Accordion("📋 分句预览", open=True):
                        sentences_preview = gr.Dataframe(
                            headers=["序号", "分句内容", "Token数"],
                            wrap=True,
                            interactive=False
                        )
        
        # 示例案例
        if len(example_cases) > 0:
            gr.Examples(
                examples=example_cases,
                inputs=[prompt_audio, input_text_single, infer_mode],
                label="📚 示例案例"
            )
    
    with gr.Tab("ℹ️ 系统信息"):
        with gr.Row():
            with gr.Column():
                gr.Markdown(f"""
                ### 🖥️ 系统信息
                - **Python 版本**: {sys.version}
                - **工作目录**: {os.getcwd()}
                - **模型目录**: {cmd_args.model_dir}
                - **GPU 可用**: {torch.cuda.is_available()}
                - **GPU 数量**: {torch.cuda.device_count() if torch.cuda.is_available() else 0}
                - **CUDA 版本**: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}
                
                ### 📊 模型信息
                - **模型版本**: {getattr(tts, 'model_version', None) or '1.0'}
                - **设备**: {getattr(tts, 'device', 'unknown')}
                - **FP16**: {getattr(tts, 'is_fp16', False)}
                - **CUDA 内核**: {getattr(tts, 'use_cuda_kernel', False)}
                
                ### 🔧 修复状态
                - **bitsandbytes**: ✅ 已禁用
                - **DeepSpeed**: ⚠️ 未安装 (自动回退)
                - **BigVGAN CUDA**: ⚠️ 回退到 torch 实现
                """)
    
    # 高级参数列表
    advanced_params = [
        do_sample, top_p, top_k, temperature,
        length_penalty, num_beams, repetition_penalty, max_mel_tokens,
    ]
    
    # 事件绑定
    input_text_single.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_sentence],
        outputs=[sentences_preview]
    )
    
    max_text_tokens_per_sentence.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_sentence],
        outputs=[sentences_preview]
    )
    
    prompt_audio.upload(
        update_prompt_audio,
        inputs=[],
        outputs=[gen_button]
    )
    
    gen_button.click(
        gen_single,
        inputs=[
            prompt_audio, input_text_single, infer_mode,
            max_text_tokens_per_sentence, sentences_bucket_max_size,
            *advanced_params,
        ],
        outputs=[output_audio]
    )

if __name__ == "__main__":
    print("\n🌐 启动 IndexTTS WebUI...")
    print(f"   地址: http://{cmd_args.host}:{cmd_args.port}")
    print(f"   模型目录: {cmd_args.model_dir}")
    print(f"   详细模式: {cmd_args.verbose}")
    print(f"   分享链接: {cmd_args.share}")
    print("   按 Ctrl+C 停止服务")
    print("=" * 50)
    
    demo.queue(max_size=20)
    demo.launch(
        server_name=cmd_args.host,
        server_port=cmd_args.port,
        share=cmd_args.share,
        inbrowser=not cmd_args.share,  # 如果不分享则自动打开浏览器
        show_error=True
    ) 