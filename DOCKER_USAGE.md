# IndexTTS Docker 部署指南 (统一版本)

## 🐳 Docker 部署方案

IndexTTS 现在支持完整的 Docker 部署，包含统一 WebUI、GPU 支持、demos 音频库、Nginx 反向代理和监控系统。

## 📋 前置要求

### 1. 系统要求
- Linux 系统 (推荐 Ubuntu 20.04+)
- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA Docker (GPU 支持)

### 2. 安装 Docker
```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# 重新登录或运行
newgrp docker
```

### 3. 安装 NVIDIA Docker (GPU 支持)
```bash
# 添加 NVIDIA Docker 仓库
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# 安装 nvidia-docker2
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# 测试 GPU 支持
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

## 🚀 快速启动

### 方法一：使用启动脚本 (推荐)
```bash
# 给脚本执行权限
chmod +x docker-start.sh

# 运行启动脚本
./docker-start.sh
```

启动脚本会自动：
- 检查 Docker 和 NVIDIA Docker 支持
- 验证模型文件完整性
- 检查 demos 音频库
- 提供多种部署模式选择
- 自动构建和启动服务

### 方法二：手动启动
```bash
# 基础模式 (仅统一 WebUI)
docker-compose up -d

# 包含 Nginx 反向代理
docker-compose --profile with-nginx up -d

# 包含监控系统
docker-compose --profile with-monitoring up -d

# 全功能模式
docker-compose --profile with-nginx --profile with-monitoring up -d
```

## 📁 目录结构

```
index-tts/
├── Dockerfile                 # 主 Dockerfile
├── docker-compose.yml         # Docker Compose 配置
├── .dockerignore              # Docker 忽略文件
├── docker-start.sh            # 启动脚本 (统一版本)
├── webui.py                   # 统一 WebUI (包含所有功能)
├── checkpoints/               # 模型文件 (必需)
│   ├── bigvgan_generator.pth
│   ├── bpe.model
│   ├── gpt.pth
│   └── config.yaml
├── demos/                     # 预设音频库 (可选)
│   ├── 男声/中文/
│   ├── 女声/英文/
│   ├── 动漫角色/
│   └── 游戏角色/原神/
├── outputs/                   # 输出目录
├── prompts/                   # 提示音频
├── logs/                      # 日志文件
├── nginx/                     # Nginx 配置
│   ├── nginx.conf
│   └── logs/
└── monitoring/                # 监控配置
    └── prometheus.yml
```

## 🎭 WebUI 功能

### 统一界面标签页
1. **🎵 音频生成**
   - 主要 TTS 功能
   - 预设音频库选择
   - 上传/录音功能
   - 高级参数设置

2. **🎭 音频库管理**
   - demos 目录统计
   - 音频库结构查看
   - 使用指南

3. **ℹ️ 系统信息**
   - 系统状态监控
   - 模型信息
   - 修复状态
   - 启动参数

## 🌐 访问地址

### 基础模式
- **WebUI (统一版本)**: http://localhost:7860

### 完整模式 (包含 Nginx)
- **WebUI**: http://localhost:7860
- **Nginx**: http://localhost:80

### 监控模式
- **WebUI**: http://localhost:7860
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin123)

## 🔧 常用命令

### 服务管理
```bash
# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f indextts

# 重启服务
docker-compose restart

# 停止服务
docker-compose down

# 停止并删除数据卷
docker-compose down -v
```

### 容器操作
```bash
# 进入 IndexTTS 容器
docker-compose exec indextts bash

# 查看容器资源使用
docker stats

# 查看 GPU 使用情况
docker-compose exec indextts nvidia-smi

# 检查 demos 音频库
docker-compose exec indextts find demos -name "*.wav" | wc -l
```

### 镜像管理
```bash
# 重新构建镜像
docker-compose build --no-cache

# 清理未使用的镜像
docker image prune -f

# 查看镜像大小
docker images | grep indextts
```

## 📊 监控和日志

### 1. 应用日志
```bash
# 实时查看 IndexTTS 日志
docker-compose logs -f indextts

# 查看 Nginx 日志
docker-compose logs -f nginx

# 查看所有服务日志
docker-compose logs -f
```

### 2. 系统监控
- **Prometheus**: 收集指标数据
- **Grafana**: 可视化监控面板
- **容器监控**: CPU、内存、GPU 使用情况

### 3. 健康检查
```bash
# 检查服务健康状态
curl http://localhost:7860/
curl http://localhost:80/health
```

## 🔧 故障排除

### 1. GPU 不工作
```bash
# 检查 NVIDIA Docker 支持
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# 检查容器内 GPU
docker-compose exec indextts nvidia-smi

# 查看 CUDA 环境
docker-compose exec indextts python -c "import torch; print(torch.cuda.is_available())"
```

### 2. demos 音频不显示
```bash
# 检查 demos 目录挂载
docker-compose exec indextts ls -la demos/

# 检查音频文件
docker-compose exec indextts find demos -name "*.wav"

# 检查目录权限
ls -la demos/
```

### 3. 端口冲突
```bash
# 检查端口占用
sudo netstat -tlnp | grep :7860

# 修改端口 (在 docker-compose.yml 中)
ports:
  - "8860:7860"  # 改为 8860
```

### 4. 内存不足
```bash
# 增加内存限制 (在 docker-compose.yml 中)
deploy:
  resources:
    limits:
      memory: 16G  # 增加到 16GB
```

### 5. 模型文件问题
```bash
# 检查模型文件
ls -la checkpoints/

# 重新下载模型
# 参考 README.md 中的模型下载说明
```

### 6. 网络问题
```bash
# 检查 Docker 网络
docker network ls
docker network inspect index-tts_indextts-network

# 重新创建网络
docker-compose down
docker-compose up -d
```

## ⚙️ 自定义配置

### 1. 修改 WebUI 配置
编辑 `webui_fixed.py` 中的参数：
```python
# 修改默认端口
parser.add_argument("--port", type=int, default=8860)

# 修改模型目录
parser.add_argument("--model_dir", type=str, default="models")
```

### 2. 修改 Docker 配置
编辑 `docker-compose.yml`：
```yaml
# 修改环境变量
environment:
  - CUDA_VISIBLE_DEVICES=0,1  # 使用多个 GPU
  - PYTHONUNBUFFERED=1

# 修改资源限制
deploy:
  resources:
    limits:
      memory: 16G
      cpus: '8'
```

### 3. 添加 SSL 支持
1. 将 SSL 证书放到 `nginx/ssl/` 目录
2. 取消注释 `nginx/nginx.conf` 中的 HTTPS 配置
3. 重启 Nginx 服务

## 🔄 更新和维护

### 1. 更新代码
```bash
# 拉取最新代码
git pull

# 重新构建镜像
docker-compose build --no-cache

# 重启服务
docker-compose up -d
```

### 2. 备份数据
```bash
# 备份输出文件
tar -czf outputs_backup.tar.gz outputs/

# 备份配置文件
tar -czf config_backup.tar.gz nginx/ monitoring/ docker-compose.yml
```

### 3. 清理空间
```bash
# 清理未使用的 Docker 资源
docker system prune -f

# 清理旧的输出文件
find outputs/ -name "*.wav" -mtime +7 -delete
```

## 📈 性能优化

### 1. GPU 优化
- 确保使用最新的 NVIDIA 驱动
- 调整 CUDA 内存分配
- 使用 FP16 推理模式

### 2. 内存优化
- 增加 Docker 内存限制
- 使用 SSD 存储
- 定期清理临时文件

### 3. 网络优化
- 使用 Nginx 缓存静态文件
- 启用 Gzip 压缩
- 配置 CDN (生产环境)

## 🆘 获取帮助

如果遇到问题，请：

1. 查看日志：`docker-compose logs -f indextts`
2. 检查系统资源：`docker stats`
3. 验证 GPU 支持：`nvidia-smi`
4. 查看网络连接：`docker network ls`
5. 提交 Issue 并附上详细的错误信息

---

**注意**: 首次启动可能需要较长时间来下载依赖和初始化模型，请耐心等待。 