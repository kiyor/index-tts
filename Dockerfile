# IndexTTS Dockerfile with GPU support
FROM nvidia/cuda:12.4-devel-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 设置时区
ENV TZ=Asia/Shanghai
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

# 复制 requirements 文件
COPY --chown=indextts:indextts requirements.txt setup.py pyproject.toml ./

# 安装 PyTorch (CUDA 12.4 兼容)
RUN python3 -m pip install --user torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# 复制项目文件
COPY --chown=indextts:indextts . .

# 安装项目依赖
RUN python3 -m pip install --user -e ".[webui]" --no-build-isolation

# 创建必要的目录
RUN mkdir -p /app/outputs /app/prompts /app/test_data

# 创建启动脚本
RUN echo '#!/bin/bash\n\
echo "🚀 启动 IndexTTS WebUI..."\n\
echo "📍 工作目录: $(pwd)"\n\
echo "🐍 Python 版本: $(python3 --version)"\n\
echo "🔧 CUDA 版本: $(nvcc --version | grep release || echo \"CUDA not available\")"\n\
echo "💾 GPU 信息:"\n\
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null || echo "  GPU 信息不可用"\n\
echo ""\n\
echo "🌐 WebUI 地址: http://0.0.0.0:7860"\n\
echo "⏹️  按 Ctrl+C 停止服务"\n\
echo ""\n\
python3 -c "\n\
import sys\n\
sys.modules[\"bitsandbytes\"] = None\n\
exec(open(\"webui.py\").read())\n\
" --host 0.0.0.0 --port 7860' > /app/start.sh \
&& chmod +x /app/start.sh

# 暴露端口
EXPOSE 7860

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# 启动命令
CMD ["/app/start.sh"] 