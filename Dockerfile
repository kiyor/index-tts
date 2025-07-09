# IndexTTS Dockerfile with GPU support and optimizations
FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 设置中文支持
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=utf-8

# 设置 CUDA 架构 (支持更多 GPU, 包括 RTX 5090)
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"
ENV FORCE_CUDA="1"
# RTX 5090 优化
ENV PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True,roundup_power2_divisions:8"

# 设置时区
ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    git \
    wget \
    curl \
    build-essential \
    ninja-build \
    cmake \
    pkg-config \
    libsndfile1 \
    locales \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# 创建非 root 用户
RUN useradd -m -u 1000 indextts && \
    mkdir -p /app && \
    chown -R indextts:indextts /app

# 设置工作目录
WORKDIR /app

# 切换到非 root 用户
USER indextts

# 升级 pip
RUN python3 -m pip install --user --upgrade pip

# 复制依赖配置文件
COPY --chown=indextts:indextts requirements.txt setup.py pyproject.toml MANIFEST.in ./

# 复制必要的文档文件 (setup.py 需要)
COPY --chown=indextts:indextts README.md LICENSE DISCLAIMER INDEX_MODEL_LICENSE ./

# 安装 PyTorch (CUDA 12.x 兼容)
RUN python3 -m pip install --user torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装基础依赖 (包括 DeepSpeed)
RUN python3 -m pip install --user -r requirements.txt

# 安装 WebUI 依赖和显存监控依赖
RUN python3 -m pip install --user gradio pandas GPUtil psutil

# 复制核心代码
COPY --chown=indextts:indextts indextts/ ./indextts/
COPY --chown=indextts:indextts tools/ ./tools/

# 复制显存监控模块和GPU配置模块
COPY --chown=indextts:indextts memory_monitor.py gpu_configs.py ./

# 强制重新编译 CUDA 扩展
RUN echo "🔧 强制重新编译 CUDA 扩展..." && \
    python3 -m pip install --user -e . --no-deps --no-build-isolation --force-reinstall

# 复制其他文件
COPY --chown=indextts:indextts tests/ ./tests/
COPY --chown=indextts:indextts assets/ ./assets/
COPY --chown=indextts:indextts *.md ./
COPY --chown=indextts:indextts webui.py create_test_audio.py fix_bitsandbytes.py ./

# 创建必要的目录
RUN mkdir -p /app/outputs /app/prompts /app/demos /app/logs /app/checkpoints

# 创建启动脚本
RUN echo '#!/bin/bash\n\
echo "🚀 启动 IndexTTS WebUI (优化版本)..."\n\
echo "📍 工作目录: $(pwd)"\n\
echo "🐍 Python 版本: $(python3 --version)"\n\
echo "🔧 CUDA 版本: $(nvcc --version | grep release || echo \"CUDA not available\")"\n\
echo "💾 GPU 信息:"\n\
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null || echo "  GPU 信息不可用"\n\
echo ""\n\
echo "📦 检查 DeepSpeed:"\n\
python3 -c "import deepspeed; print(f\"  ✅ DeepSpeed {deepspeed.__version__} 已安装\")" 2>/dev/null || echo "  ❌ DeepSpeed 未安装"\n\
echo "🔧 检查 BigVGAN CUDA 扩展:"\n\
python3 -c "from indextts.BigVGAN.alias_free_activation import Activation1d; print(\"  ✅ BigVGAN CUDA 扩展正常\")" 2>/dev/null || echo "  ⚠️ BigVGAN CUDA 扩展回退到 torch"\n\
echo ""\n\
echo "📁 检查目录结构:"\n\
echo "  - checkpoints: $(ls -la checkpoints 2>/dev/null | wc -l) 个文件"\n\
echo "  - demos: $(find demos -name \*.wav 2>/dev/null | wc -l) 个音频文件"\n\
echo "  - outputs: $(ls -la outputs 2>/dev/null | wc -l) 个文件"\n\
echo ""\n\
echo "🌐 WebUI 地址: http://0.0.0.0:7860"\n\
echo "🎭 功能标签页:"\n\
echo "  - 🎵 音频生成: 主要TTS功能 + 预设音频库"\n\
echo "  - 🎭 音频库管理: 查看demos音频库统计"\n\
echo "  - ℹ️ 系统信息: 查看系统和模型状态"\n\
echo "⏹️  按 Ctrl+C 停止服务"\n\
echo ""\n\
python3 webui.py --host 0.0.0.0 --port 7860' > /app/start.sh \
&& chmod +x /app/start.sh

# 暴露端口
EXPOSE 7860

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# 启动命令
CMD ["/app/start.sh"]
