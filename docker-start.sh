#!/bin/bash

# IndexTTS Docker 启动脚本 (统一版本)
# 支持 GPU 的 Docker 部署，包含 demos 音频库功能

set -e

echo "🐳 IndexTTS Docker 启动脚本 (统一版本)"
echo "========================================"

# 检查 Docker 是否安装
if ! command -v docker &> /dev/null; then
    echo "❌ Docker 未安装，请先安装 Docker"
    exit 1
fi

# 检查 Docker Compose 是否安装
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose 未安装，请先安装 Docker Compose"
    exit 1
fi

# 检查 NVIDIA Docker 支持
if command -v nvidia-docker &> /dev/null || docker info | grep -q nvidia; then
    echo "✅ 检测到 NVIDIA Docker 支持"
    GPU_SUPPORT=true
else
    echo "⚠️  未检测到 NVIDIA Docker 支持，将使用 CPU 模式"
    GPU_SUPPORT=false
fi

# 检查模型文件
if [ ! -d "checkpoints" ]; then
    echo "❌ 模型目录 checkpoints 不存在"
    echo "请先下载模型文件到 checkpoints 目录"
    exit 1
fi

required_files=("bigvgan_generator.pth" "bpe.model" "gpt.pth" "config.yaml")
missing_files=()

for file in "${required_files[@]}"; do
    if [ ! -f "checkpoints/$file" ]; then
        missing_files+=("checkpoints/$file")
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    echo "❌ 缺少必需的模型文件:"
    for file in "${missing_files[@]}"; do
        echo "   - $file"
    done
    echo ""
    echo "请确保所有模型文件都已下载到 checkpoints 目录"
    exit 1
fi

echo "✅ 模型文件检查通过"

# 检查 demos 目录
if [ -d "demos" ]; then
    demo_count=$(find demos -name "*.wav" 2>/dev/null | wc -l)
    echo "✅ 检测到 demos 目录，包含 $demo_count 个音频文件"
else
    echo "⚠️  demos 目录不存在，将创建空目录"
    mkdir -p demos
fi

# 创建必要的目录
mkdir -p outputs prompts logs nginx/logs monitoring demos

# 设置权限
chmod -R 755 outputs prompts logs demos

# 选择启动模式
echo ""
echo "请选择启动模式:"
echo "1) 基础模式 (仅 IndexTTS WebUI 统一版本)"
echo "2) 完整模式 (包含 Nginx 反向代理)"
echo "3) 监控模式 (包含 Prometheus + Grafana)"
echo "4) 全功能模式 (包含所有服务)"

read -p "请输入选择 (1-4): " choice

case $choice in
    1)
        echo "🚀 启动基础模式 (统一WebUI)..."
        COMPOSE_PROFILES=""
        ;;
    2)
        echo "🚀 启动完整模式 (包含 Nginx)..."
        COMPOSE_PROFILES="--profile with-nginx"
        ;;
    3)
        echo "🚀 启动监控模式..."
        COMPOSE_PROFILES="--profile with-monitoring"
        ;;
    4)
        echo "🚀 启动全功能模式..."
        COMPOSE_PROFILES="--profile with-nginx --profile with-monitoring"
        ;;
    *)
        echo "❌ 无效选择，使用基础模式"
        COMPOSE_PROFILES=""
        ;;
esac

# 构建并启动服务
echo ""
echo "🔨 构建 Docker 镜像..."
if [ "$GPU_SUPPORT" = true ]; then
    docker-compose build
else
    echo "⚠️  GPU 支持不可用，某些功能可能受限"
    docker-compose build
fi

echo ""
echo "🚀 启动服务..."
if command -v docker-compose &> /dev/null; then
    docker-compose up -d $COMPOSE_PROFILES
else
    docker compose up -d $COMPOSE_PROFILES
fi

# 等待服务启动
echo ""
echo "⏳ 等待服务启动..."
sleep 10

# 检查服务状态
echo ""
echo "📊 服务状态:"
if command -v docker-compose &> /dev/null; then
    docker-compose ps
else
    docker compose ps
fi

echo ""
echo "🎉 IndexTTS 统一版本启动完成！"
echo ""
echo "📱 访问地址:"
echo "   - WebUI (统一版本): http://localhost:7860"
echo ""
echo "🎭 WebUI 功能标签页:"
echo "   - 🎵 音频生成: 主要TTS功能 + 预设音频库选择"
echo "   - 🎭 音频库管理: 查看demos音频库统计和结构"
echo "   - ℹ️ 系统信息: 查看系统状态、模型信息、修复状态"

if [[ "$COMPOSE_PROFILES" == *"with-nginx"* ]]; then
    echo "   - Nginx: http://localhost:80"
fi

if [[ "$COMPOSE_PROFILES" == *"with-monitoring"* ]]; then
    echo "   - Prometheus: http://localhost:9090"
    echo "   - Grafana: http://localhost:3000 (admin/admin123)"
fi

echo ""
echo "📋 常用命令:"
echo "   - 查看日志: docker-compose logs -f indextts"
echo "   - 停止服务: docker-compose down"
echo "   - 重启服务: docker-compose restart"
echo "   - 进入容器: docker-compose exec indextts bash"
echo ""
echo "🔧 故障排除:"
echo "   - 如果无法访问，请检查防火墙设置"
echo "   - 如果 GPU 不工作，请检查 NVIDIA Docker 安装"
echo "   - 查看详细日志: docker-compose logs indextts"
echo "   - demos音频不显示: 检查demos目录结构和WAV文件"
echo ""
echo "⏹️  按 Ctrl+C 停止所有服务" 