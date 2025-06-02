#!/usr/bin/env python3
"""
IndexTTS WebUI - 统一版本
包含 bitsandbytes 兼容性修复、Docker 支持、Demos 音频选择功能、系统信息和队列管理
"""

# 修复 bitsandbytes 兼容性问题
import sys
sys.modules['bitsandbytes'] = None

import json
import os
import sys
import threading
import time
import glob
import queue
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from collections import deque
import statistics

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import argparse
parser = argparse.ArgumentParser(description="IndexTTS WebUI (统一版本)")
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
parser.add_argument("--port", type=int, default=7860, help="Port to run the web UI on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the web UI on (0.0.0.0 for Docker)")
parser.add_argument("--model_dir", type=str, default="checkpoints", help="Model checkpoints directory")
parser.add_argument("--share", action="store_true", default=False, help="Create a publicly shareable link")
cmd_args = parser.parse_args()

print("🚀 IndexTTS WebUI (统一版本)")
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

# 任务状态枚举
@dataclass
class TaskStatus:
    QUEUED = "queued"      # 排队中
    RUNNING = "running"    # 执行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"      # 失败

@dataclass
class Task:
    """任务数据结构"""
    id: str
    prompt: str
    text: str
    infer_mode: str
    params: Dict[str, Any]
    status: str = TaskStatus.QUEUED
    created_time: float = 0
    start_time: float = 0
    end_time: float = 0
    result_path: Optional[str] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.created_time == 0:
            self.created_time = time.time()

class TaskQueue:
    """任务队列管理器"""
    
    def __init__(self, max_history=50):
        self.queue = queue.Queue()
        self.current_task: Optional[Task] = None
        self.task_history: deque = deque(maxlen=max_history)
        self.execution_times: deque = deque(maxlen=20)  # 保存最近20次执行时间用于预估
        self.worker_thread = None
        self.is_running = False
        self.lock = threading.Lock()
        
    def start_worker(self, tts_instance):
        """启动队列处理工作线程"""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.is_running = True
            self.worker_thread = threading.Thread(target=self._worker, args=(tts_instance,), daemon=True)
            self.worker_thread.start()
            
    def stop_worker(self):
        """停止队列处理工作线程"""
        self.is_running = False
        
    def add_task(self, prompt: str, text: str, infer_mode: str, params: Dict[str, Any]) -> str:
        """添加任务到队列"""
        task = Task(
            id=str(uuid.uuid4())[:8],
            prompt=prompt,
            text=text,
            infer_mode=infer_mode,
            params=params
        )
        
        with self.lock:
            self.queue.put(task)
            
        print(f"📝 任务 {task.id} 已加入队列")
        return task.id
        
    def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态信息"""
        with self.lock:
            queue_size = self.queue.qsize()
            current_task_info = None
            
            if self.current_task:
                elapsed = time.time() - self.current_task.start_time
                estimated_remaining = self._estimate_remaining_time()
                current_task_info = {
                    'id': self.current_task.id,
                    'text_preview': self.current_task.text[:50] + ('...' if len(self.current_task.text) > 50 else ''),
                    'elapsed_time': elapsed,
                    'estimated_remaining': estimated_remaining
                }
                
            return {
                'queue_size': queue_size,
                'current_task': current_task_info,
                'total_completed': len([t for t in self.task_history if t.status == TaskStatus.COMPLETED]),
                'average_execution_time': statistics.mean(self.execution_times) if self.execution_times else 0,
                'estimated_wait_time': self._estimate_wait_time(queue_size)
            }
            
    def get_task_result(self, task_id: str) -> Optional[Task]:
        """根据任务ID获取任务结果"""
        with self.lock:
            # 检查当前任务
            if self.current_task and self.current_task.id == task_id:
                return self.current_task
                
            # 检查历史任务
            for task in self.task_history:
                if task.id == task_id:
                    return task
                    
        return None
        
    def _worker(self, tts_instance):
        """队列处理工作线程"""
        print("🔄 队列处理器已启动")
        
        while self.is_running:
            try:
                # 等待任务，超时检查是否需要停止
                task = self.queue.get(timeout=1)
                
                with self.lock:
                    self.current_task = task
                    task.status = TaskStatus.RUNNING
                    task.start_time = time.time()
                
                print(f"🚀 开始执行任务 {task.id}")
                
                try:
                    # 执行任务
                    output_path = os.path.join("outputs", f"task_{task.id}_{int(time.time())}.wav")
                    
                    # 根据推理模式过滤参数
                    if task.infer_mode == "普通推理":
                        # 普通推理不支持 sentences_bucket_max_size 参数
                        infer_params = {k: v for k, v in task.params.items() 
                                      if k != 'sentences_bucket_max_size'}
                        result = tts_instance.infer(
                            task.prompt, 
                            task.text, 
                            output_path,
                            verbose=cmd_args.verbose,
                            **infer_params
                        )
                    else:
                        # 批次推理支持所有参数
                        result = tts_instance.infer_fast(
                            task.prompt, 
                            task.text, 
                            output_path,
                            verbose=cmd_args.verbose,
                            **task.params
                        )
                    
                    # 任务完成
                    with self.lock:
                        task.status = TaskStatus.COMPLETED
                        task.end_time = time.time()
                        task.result_path = result
                        
                        # 记录执行时间用于预估
                        execution_time = task.end_time - task.start_time
                        self.execution_times.append(execution_time)
                        
                        # 移动到历史记录
                        self.task_history.append(task)
                        self.current_task = None
                        
                    print(f"✅ 任务 {task.id} 完成，耗时 {execution_time:.2f} 秒")
                    
                except Exception as e:
                    # 任务失败
                    with self.lock:
                        task.status = TaskStatus.FAILED
                        task.end_time = time.time()
                        task.error_message = str(e)
                        
                        # 移动到历史记录
                        self.task_history.append(task)
                        self.current_task = None
                        
                    print(f"❌ 任务 {task.id} 失败: {e}")
                    
                finally:
                    self.queue.task_done()
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ 队列处理器错误: {e}")
                
        print("🛑 队列处理器已停止")
        
    def _estimate_remaining_time(self) -> float:
        """预估当前任务剩余时间"""
        if not self.current_task or not self.execution_times:
            return 0
            
        avg_time = statistics.mean(self.execution_times)
        elapsed = time.time() - self.current_task.start_time
        return max(0, avg_time - elapsed)
        
    def _estimate_wait_time(self, queue_size: int) -> float:
        """预估排队等待时间"""
        if queue_size == 0 or not self.execution_times:
            return 0
            
        avg_time = statistics.mean(self.execution_times)
        current_remaining = self._estimate_remaining_time()
        return current_remaining + (queue_size * avg_time)

# 全局队列实例
task_queue = TaskQueue()

print("🔧 正在初始化 IndexTTS...")
try:
    tts = IndexTTS(
        model_dir=cmd_args.model_dir, 
        cfg_path=os.path.join(cmd_args.model_dir, "config.yaml")
    )
    print("✅ IndexTTS 初始化成功")
    
    # 启动队列处理器
    task_queue.start_worker(tts)
    
except Exception as e:
    print(f"❌ IndexTTS 初始化失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 创建输出目录
os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/tasks", exist_ok=True)
os.makedirs("prompts", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# 扫描demos目录获取音频文件
def scan_demos_directory():
    """扫描demos目录，返回分类和音频文件的字典"""
    demos_dir = "demos"
    if not os.path.exists(demos_dir):
        if cmd_args.verbose:
            print(f"⚠️  demos目录不存在: {demos_dir}")
        return {}
    
    demos_dict = {}
    
    try:
        # 遍历一级目录（分类）
        for category in os.listdir(demos_dir):
            category_path = os.path.join(demos_dir, category)
            if not os.path.isdir(category_path) or category.startswith('.'):
                continue
                
            if cmd_args.verbose:
                print(f"📁 扫描分类: {category}")
            demos_dict[category] = {}
            
            # 遍历二级目录（子分类）
            for subcategory in os.listdir(category_path):
                subcategory_path = os.path.join(category_path, subcategory)
                if not os.path.isdir(subcategory_path) or subcategory.startswith('.'):
                    continue
                    
                if cmd_args.verbose:
                    print(f"  📂 扫描子分类: {category}/{subcategory}")
                
                # 查找wav文件
                wav_files = glob.glob(os.path.join(subcategory_path, "*.wav"))
                if wav_files:
                    demos_dict[category][subcategory] = []
                    for wav_file in sorted(wav_files):
                        filename = os.path.basename(wav_file)
                        if cmd_args.verbose:
                            print(f"    🎵 发现音频: {filename}")
                        demos_dict[category][subcategory].append({
                            'name': filename,
                            'path': wav_file
                        })
                else:
                    if cmd_args.verbose:
                        print(f"    ⚠️  {category}/{subcategory} 目录中未找到WAV文件")
    
    except Exception as e:
        print(f"❌ 扫描demos目录失败: {e}")
        if cmd_args.verbose:
            import traceback
            traceback.print_exc()
    
    return demos_dict

def get_demo_categories():
    """获取demos分类列表（动态扫描）"""
    demos_dict = scan_demos_directory()
    return list(demos_dict.keys())

def get_demo_subcategories(category):
    """根据分类获取子分类列表（动态扫描）"""
    demos_dict = scan_demos_directory()
    if category in demos_dict:
        return list(demos_dict[category].keys())
    return []

def get_demo_audio_files(category, subcategory):
    """根据分类和子分类获取音频文件列表（动态扫描）"""
    demos_dict = scan_demos_directory()
    if category in demos_dict and subcategory in demos_dict[category]:
        return [audio['name'] for audio in demos_dict[category][subcategory]]
    return []

def get_demo_audio_path(category, subcategory, filename):
    """获取指定音频文件的完整路径（动态扫描）"""
    demos_dict = scan_demos_directory()
    if category in demos_dict and subcategory in demos_dict[category]:
        for audio in demos_dict[category][subcategory]:
            if audio['name'] == filename:
                return audio['path']
    return None

def get_demos_statistics():
    """获取demos统计信息（动态扫描）"""
    demos_dict = scan_demos_directory()
    total_categories = len(demos_dict)
    total_files = sum(len(files) for subcats in demos_dict.values() for files in subcats.values())
    return total_categories, total_files, demos_dict

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
                # 移除参考音频，只保留文本和推理模式
                example_cases.append([
                    example.get("text"), 
                    ["普通推理", "批次推理"][example.get("infer_mode", 0)]
                ])
        print(f"✅ 加载了 {len(example_cases)} 个示例案例")
    except Exception as e:
        print(f"⚠️  加载示例案例失败: {e}")

def gen_single_with_queue(prompt, text, infer_mode, max_text_tokens_per_sentence=120, sentences_bucket_max_size=4,
                         *args, progress=gr.Progress()):
    """使用队列的语音生成函数"""
    if not prompt:
        gr.Warning("请上传参考音频文件或选择预设音频")
        return gr.update(value=None, visible=True), "请上传参考音频"
    
    if not text or not text.strip():
        gr.Warning("请输入要合成的文本")
        return gr.update(value=None, visible=True), "请输入要合成的文本"
    
    # 解析参数
    do_sample, top_p, top_k, temperature, \
        length_penalty, num_beams, repetition_penalty, max_mel_tokens = args
    
    params = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": int(num_beams),
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
        "max_text_tokens_per_sentence": int(max_text_tokens_per_sentence),
        "sentences_bucket_max_size": int(sentences_bucket_max_size)
    }
    
    # 添加任务到队列
    task_id = task_queue.add_task(prompt, text, infer_mode, params)
    
    gr.Info(f"🎯 任务 {task_id} 已提交到队列")
    
    # 轮询等待任务完成
    max_wait_time = 300  # 最大等待5分钟
    start_wait = time.time()
    
    while time.time() - start_wait < max_wait_time:
        task = task_queue.get_task_result(task_id)
        
        if task and task.status == TaskStatus.COMPLETED:
            gr.Info(f"✅ 任务 {task_id} 完成！")
            return gr.update(value=task.result_path, visible=True), f"任务 {task_id} 已完成"
        elif task and task.status == TaskStatus.FAILED:
            gr.Error(f"❌ 任务 {task_id} 失败: {task.error_message}")
            return gr.update(value=None, visible=True), f"任务失败: {task.error_message}"
        
        # 更新进度信息
        if task and task.status == TaskStatus.RUNNING:
            elapsed = time.time() - task.start_time
            progress(0.5, f"任务 {task_id} 执行中... ({elapsed:.1f}s)")
        else:
            queue_status = task_queue.get_queue_status()
            if queue_status['queue_size'] > 0:
                progress(0.1, f"排队中... 前面还有 {queue_status['queue_size']} 个任务")
        
        time.sleep(1)
    
    gr.Error(f"❌ 任务 {task_id} 超时")
    return gr.update(value=None, visible=True), "任务超时"

def get_queue_status_display():
    """获取队列状态显示信息"""
    status = task_queue.get_queue_status()
    
    if status['current_task']:
        current_info = status['current_task']
        current_text = f"""
        🔄 **当前执行任务**
        - 任务ID: {current_info['id']}
        - 内容预览: {current_info['text_preview']}
        - 已执行时间: {current_info['elapsed_time']:.1f}s
        - 预估剩余: {current_info['estimated_remaining']:.1f}s
        """
    else:
        current_text = "💤 **当前无任务执行**"
    
    queue_info = f"""
    📊 **队列状态**
    - 排队任务数: {status['queue_size']}
    - 已完成任务: {status['total_completed']}
    - 平均执行时间: {status['average_execution_time']:.1f}s
    - 预估等待时间: {status['estimated_wait_time']:.1f}s
    """
    
    return current_text + "\n" + queue_info

def refresh_queue_status():
    """刷新队列状态"""
    return get_queue_status_display()

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

def on_demo_category_change(category):
    """当demos分类改变时更新子分类选项"""
    if category:
        subcategories = get_demo_subcategories(category)
        return gr.update(choices=subcategories, value=subcategories[0] if subcategories else None)
    return gr.update(choices=[], value=None)

def on_demo_subcategory_change(category, subcategory):
    """当demos子分类改变时更新音频文件选项"""
    if category and subcategory:
        audio_files = get_demo_audio_files(category, subcategory)
        return gr.update(choices=audio_files, value=audio_files[0] if audio_files else None)
    return gr.update(choices=[], value=None)

def on_demo_audio_select(category, subcategory, filename):
    """当选择demos音频时更新音频组件"""
    if category and subcategory and filename:
        audio_path = get_demo_audio_path(category, subcategory, filename)
        if audio_path and os.path.exists(audio_path):
            return gr.update(value=audio_path)
    return gr.update(value=None)

def clear_text():
    """清空目标文本框内容"""
    return gr.update(value="")

def auto_use_demo_audio(category, subcategory, filename):
    """当音频文件被选中时自动使用该音频"""
    if category and subcategory and filename:
        audio_path = get_demo_audio_path(category, subcategory, filename)
        if audio_path and os.path.exists(audio_path):
            return gr.update(value=audio_path)
    return gr.update(value=None)

# 创建 Gradio 界面
with gr.Blocks(
    title="IndexTTS Demo - 统一版本", 
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 1200px !important;
    }
    .header {
        text-align: center;
        margin-bottom: 20px;
    }
    .demos-section {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        background-color: #f9f9f9;
    }
    .queue-status {
        border: 1px solid #2196f3;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        background-color: #e3f2fd;
    }
    """
) as demo:
    mutex = threading.Lock()
    
    gr.HTML('''
    <div class="header">
        <h1>🎤 IndexTTS: 工业级零样本文本转语音系统</h1>
        <h3>IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot TTS System</h3>
        <p>
            <a href='https://arxiv.org/abs/2502.05512'><img src='https://img.shields.io/badge/ArXiv-2502.05512-red'></a>
            <img src='https://img.shields.io/badge/Status-Fixed-green'>
            <img src='https://img.shields.io/badge/Docker-Supported-blue'>
            <img src='https://img.shields.io/badge/Demos-Enabled-orange'>
            <img src='https://img.shields.io/badge/Queue-Enabled-purple'>
        </p>
        <p><strong>✅ 已修复 bitsandbytes 兼容性问题 | 支持 Docker 部署 | 🎵 支持预设音频选择 | 📋 智能队列管理</strong></p>
    </div>
    ''')
    
    with gr.Tab("🎵 音频生成"):
        # 队列状态显示
        with gr.Group():
            gr.Markdown("### 📋 队列状态", elem_classes=["queue-status"])
            queue_status_display = gr.Markdown(
                get_queue_status_display(),
                elem_classes=["queue-status"]
            )
            with gr.Row():
                refresh_queue_btn = gr.Button("🔄 刷新状态", size="sm")
                task_status_output = gr.Textbox(
                    label="任务状态", 
                    interactive=False,
                    placeholder="任务状态将在这里显示..."
                )
        
        with gr.Row():
            with gr.Column(scale=1):
                # 预设音频选择区域
                if get_demo_categories():
                    with gr.Group():
                        gr.Markdown("### 🎭 选择预设音频")
                        
                        # 获取初始化值
                        initial_categories = get_demo_categories()
                        initial_category = initial_categories[0] if initial_categories else None
                        initial_subcategories = get_demo_subcategories(initial_category) if initial_category else []
                        initial_subcategory = initial_subcategories[0] if initial_subcategories else None
                        initial_audio_files = get_demo_audio_files(initial_category, initial_subcategory) if initial_category and initial_subcategory else []
                        initial_audio_file = initial_audio_files[0] if initial_audio_files else None
                        
                        with gr.Row():
                            demo_category = gr.Dropdown(
                                choices=initial_categories,
                                label="分类",
                                value=initial_category
                            )
                            demo_subcategory = gr.Dropdown(
                                choices=initial_subcategories,
                                label="子分类",
                                value=initial_subcategory
                            )
                        demo_audio_file = gr.Dropdown(
                            choices=initial_audio_files,
                            label="音频文件",
                            value=initial_audio_file
                        )
                        use_demo_btn = gr.Button("🎯 使用选中音频", variant="secondary", size="sm")
                
                gr.Markdown("### 📎 上传/录制音频")
                prompt_audio = gr.Audio(
                    label="参考音频", 
                    sources=["upload", "microphone"],
                    type="filepath"
                )
                
            with gr.Column(scale=2):
                with gr.Row():
                    input_text_single = gr.TextArea(
                        label="📝 目标文本",
                        placeholder="请输入要合成的文本...",
                        info=f"当前模型版本: {getattr(tts, 'model_version', None) or '1.0'}",
                        lines=6,
                        scale=4
                    )
                
                with gr.Row():
                    infer_mode = gr.Radio(
                        choices=["普通推理", "批次推理"], 
                        label="⚡ 推理模式",
                        info="批次推理：更适合长句，性能翻倍",
                        value="普通推理"
                    )

                with gr.Row():
                    gen_button = gr.Button("🎯 提交到队列", variant="primary", size="lg")
                    clear_text_btn = gr.Button("🗑️ 清空", variant="secondary", size="lg")
        
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
                inputs=[input_text_single, infer_mode],  # 移除 prompt_audio
                label="📚 示例案例"
            )
    
    with gr.Tab("🎭 音频库管理"):
        gr.Markdown("### 📁 Demos 音频库")
        
        # 显示当前音频库统计
        total_categories, total_files, demos_audio_dict = get_demos_statistics()
        
        gr.Markdown(f"""
        **📊 统计信息**
        - 分类数量: {total_categories}
        - 音频文件总数: {total_files}
        """)
        
        # 显示音频库结构
        if demos_audio_dict:
            structure_md = "**📂 目录结构**\n\n"
            for category, subcategories in demos_audio_dict.items():
                structure_md += f"- **{category}**\n"
                for subcategory, files in subcategories.items():
                    structure_md += f"  - {subcategory} ({len(files)} 个文件)\n"
                    for file_info in files[:3]:  # 只显示前3个文件
                        structure_md += f"    - {file_info['name']}\n"
                    if len(files) > 3:
                        structure_md += f"    - ... 还有 {len(files) - 3} 个文件\n"
            
            gr.Markdown(structure_md)
        else:
            gr.Markdown("**⚠️ 未找到音频文件**\n\n请在 `demos/` 目录下添加音频文件。")
        
        gr.Markdown("""
        ### 📝 添加音频文件
        
        1. **准备音频文件**
           - 格式：WAV
           - 采样率：22050Hz 或 44100Hz
           - 时长：3-10秒
           - 质量：清晰无噪音
        
        2. **文件命名**
           - 中文：`说话人-内容描述.wav`
           - 英文：`Speaker-Content.wav`
           - 角色：`角色名-台词内容.wav`
        
        3. **放置文件**
           - 将文件放入对应的分类目录
           - 重启 WebUI 以刷新列表
        """)
    
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
                
                ### 📋 队列管理
                - **队列功能**: ✅ 已启用
                - **并发限制**: 1个任务
                - **历史统计**: 最近20次执行时间
                - **时间预估**: 基于历史算力
                
                ### 🎭 Demos 功能
                - **音频分类**: {total_categories}
                - **总音频数**: {total_files}
                - **支持格式**: WAV
                
                ### 📋 启动参数
                - **主机**: {cmd_args.host}
                - **端口**: {cmd_args.port}
                - **详细模式**: {cmd_args.verbose}
                - **分享链接**: {cmd_args.share}
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
    
    # 清空文本按钮事件绑定
    clear_text_btn.click(
        clear_text,
        inputs=[],
        outputs=[input_text_single]
    )
    
    # 队列状态刷新
    refresh_queue_btn.click(
        refresh_queue_status,
        inputs=[],
        outputs=[queue_status_display]
    )
    
    # Demos 相关事件绑定（只有在有demos音频时才绑定）
    if get_demo_categories():
        demo_category.change(
            on_demo_category_change,
            inputs=[demo_category],
            outputs=[demo_subcategory]
        )
        
        demo_subcategory.change(
            on_demo_subcategory_change,
            inputs=[demo_category, demo_subcategory],
            outputs=[demo_audio_file]
        )
        
        # 音频文件选择时自动使用该音频
        demo_audio_file.change(
            auto_use_demo_audio,
            inputs=[demo_category, demo_subcategory, demo_audio_file],
            outputs=[prompt_audio]
        )
        
        # 保留手动使用按钮功能
        use_demo_btn.click(
            on_demo_audio_select,
            inputs=[demo_category, demo_subcategory, demo_audio_file],
            outputs=[prompt_audio]
        )
    
    # 修改生成按钮事件，使用队列版本
    gen_button.click(
        gen_single_with_queue,
        inputs=[
            prompt_audio, input_text_single, infer_mode,
            max_text_tokens_per_sentence, sentences_bucket_max_size,
            *advanced_params,
        ],
        outputs=[output_audio, task_status_output]
    )
    
    # 初始化队列状态显示
    demo.load(
        refresh_queue_status,
        inputs=[],
        outputs=[queue_status_display]
    )

if __name__ == "__main__":
    print("\n🌐 启动 IndexTTS WebUI (统一版本)...")
    print(f"   地址: http://{cmd_args.host}:{cmd_args.port}")
    print(f"   模型目录: {cmd_args.model_dir}")
    print(f"   详细模式: {cmd_args.verbose}")
    print(f"   分享链接: {cmd_args.share}")
    print(f"   队列管理: ✅ 已启用")
    
    # 获取demos统计信息
    total_categories, total_files, _ = get_demos_statistics()
    print(f"   音频分类: {total_categories}")
    print(f"   音频文件: {total_files}")
    print("   按 Ctrl+C 停止服务")
    print("=" * 50)
    
    try:
        demo.queue(max_size=20)
        demo.launch(
            server_name=cmd_args.host,
            server_port=cmd_args.port,
            share=cmd_args.share,
            inbrowser=not cmd_args.share,  # 如果不分享则自动打开浏览器
            show_error=True
        )
    finally:
        # 确保队列处理器正常停止
        task_queue.stop_worker()
        print("🛑 队列处理器已停止")
