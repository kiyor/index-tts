# IndexTTS2 Dockerfile with GPU support and optimizations
FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Chinese language support
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=utf-8

# CUDA architecture support (including RTX 5090)
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"
ENV FORCE_CUDA="1"
# RTX 5090 optimization
ENV PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True,roundup_power2_divisions:8"

# Set timezone
ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
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
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 indextts && \
    mkdir -p /app && \
    chown -R indextts:indextts /app

# Set working directory
WORKDIR /app

# Switch to non-root user
USER indextts

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/indextts/.local/bin:${PATH}"

# Copy dependency files
COPY --chown=indextts:indextts pyproject.toml uv.lock ./

# Copy required documentation files
COPY --chown=indextts:indextts README.md LICENSE DISCLAIMER INDEX_MODEL_LICENSE ./

# Create virtual environment and install dependencies
RUN uv sync --all-extras

# Copy core code
COPY --chown=indextts:indextts indextts/ ./indextts/
COPY --chown=indextts:indextts tools/ ./tools/
COPY --chown=indextts:indextts examples/ ./examples/

# Copy custom modules (GPU config, memory monitor, API server)
COPY --chown=indextts:indextts memory_monitor.py gpu_configs.py api_server.py ./

# Copy other files
COPY --chown=indextts:indextts tests/ ./tests/
COPY --chown=indextts:indextts assets/ ./assets/
COPY --chown=indextts:indextts *.md ./
COPY --chown=indextts:indextts webui.py ./

# Create necessary directories
RUN mkdir -p /app/outputs /app/prompts /app/demos /app/logs /app/checkpoints

# Create startup script
RUN echo '#!/bin/bash\n\
echo "🚀 Starting IndexTTS2 WebUI..."\n\
echo "📍 Working directory: $(pwd)"\n\
echo "🐍 Python version: $(python3 --version)"\n\
echo "🔧 CUDA version: $(nvcc --version | grep release || echo \"CUDA not available\")"\n\
echo "💾 GPU info:"\n\
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null || echo "  GPU info not available"\n\
echo ""\n\
echo "📦 Checking DeepSpeed:"\n\
uv run python -c "import deepspeed; print(f\"  ✅ DeepSpeed {deepspeed.__version__} installed\")" 2>/dev/null || echo "  ❌ DeepSpeed not installed"\n\
echo "🔧 Checking BigVGAN CUDA extensions:"\n\
uv run python -c "from indextts.s2mel.modules.bigvgan.alias_free_activation.cuda import activation1d; print(\"  ✅ BigVGAN CUDA extension OK\")" 2>/dev/null || echo "  ⚠️ BigVGAN CUDA extension fallback to torch"\n\
echo ""\n\
echo "📁 Checking directory structure:"\n\
echo "  - checkpoints: $(ls -la checkpoints 2>/dev/null | wc -l) files"\n\
echo "  - demos: $(find demos -name \\*.wav 2>/dev/null | wc -l) audio files"\n\
echo "  - outputs: $(ls -la outputs 2>/dev/null | wc -l) files"\n\
echo ""\n\
echo "🌐 WebUI address: http://0.0.0.0:7860"\n\
echo "⏹️  Press Ctrl+C to stop"\n\
echo ""\n\
uv run python webui.py --host 0.0.0.0 --port 7860' > /app/start.sh \
&& chmod +x /app/start.sh

# Create API server startup script
RUN echo '#!/bin/bash\n\
echo "🚀 Starting IndexTTS2 API Server..."\n\
echo "📍 Working directory: $(pwd)"\n\
echo "🌐 API address: http://0.0.0.0:7871"\n\
echo "📚 API docs: http://0.0.0.0:7871/docs"\n\
echo "⏹️  Press Ctrl+C to stop"\n\
echo ""\n\
uv run python api_server.py --host 0.0.0.0 --port 7871' > /app/start_api.sh \
&& chmod +x /app/start_api.sh

# Expose ports
EXPOSE 7860 7871

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Default startup command
CMD ["/app/start.sh"]
