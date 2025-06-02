# Docker CUDA 扩展资源

这个目录包含预构建的 CUDA 扩展二进制文件，用于 Docker 容器。

## 文件说明

- `cuda_extensions/`: 预构建的 CUDA 扩展二进制文件
- `install_cuda_extensions.py`: 容器内安装脚本
- `README.md`: 本说明文件

## 构建信息

- 构建时间: 2025-06-02 02:17:24
- PyTorch 版本: 2.7.0+cu126
- CUDA 版本: 12.6
- GPU: Tesla P4

## 使用方法

在 Dockerfile 中：

```dockerfile
# 复制预构建的 CUDA 扩展
COPY --chown=indextts:indextts docker_assets/ ./docker_assets/

# 安装预构建的 CUDA 扩展
RUN python3 docker_assets/install_cuda_extensions.py
```

这样可以避免运行时编译，大大提升容器启动速度。
