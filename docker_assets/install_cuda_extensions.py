#!/usr/bin/env python3
"""
Docker 容器内 CUDA 扩展安装脚本
将预构建的 CUDA 扩展二进制文件安装到正确位置
"""

import os
import shutil
import sys
import pathlib

def install_cuda_extensions():
    """安装预构建的 CUDA 扩展"""
    try:
        print("🔧 安装预构建的 CUDA 扩展...")
        
        # 方法1: 安装到 torch 标准缓存目录
        import torch.utils.cpp_extension
        target_build_dir = torch.utils.cpp_extension._get_build_directory("", verbose=False)
        
        # 创建目标目录
        os.makedirs(target_build_dir, exist_ok=True)
        
        # 复制预构建的文件到 torch 缓存目录
        source_dir = "/app/docker_assets/cuda_extensions/build_structure"
        if os.path.exists(source_dir):
            print(f"📁 复制 {source_dir} -> {target_build_dir}")
            
            # 复制所有文件
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    src_file = os.path.join(root, file)
                    rel_path = os.path.relpath(src_file, source_dir)
                    dst_file = os.path.join(target_build_dir, rel_path)
                    
                    # 创建目标目录
                    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                    
                    # 复制文件
                    shutil.copy2(src_file, dst_file)
                    print(f"   ✅ {rel_path}")
            
            print("✅ torch 缓存目录安装完成")
        else:
            print(f"❌ 源目录不存在: {source_dir}")
            return False
        
        # 方法2: 安装到 BigVGAN 期望的构建目录
        bigvgan_build_dir = "/app/indextts/BigVGAN/alias_free_activation/cuda/build"
        os.makedirs(bigvgan_build_dir, exist_ok=True)
        
        # 复制二进制文件到 BigVGAN 构建目录
        binary_source = "/app/docker_assets/cuda_extensions/anti_alias_activation_cuda.so"
        binary_target = os.path.join(bigvgan_build_dir, "anti_alias_activation_cuda.so")
        
        if os.path.exists(binary_source):
            shutil.copy2(binary_source, binary_target)
            print(f"📁 复制 {binary_source} -> {binary_target}")
            print("✅ BigVGAN 构建目录安装完成")
        else:
            print(f"❌ 二进制文件不存在: {binary_source}")
            return False
        
        # 验证安装
        print("\n🔍 验证安装结果...")
        
        # 检查 torch 缓存目录
        torch_extension_dir = os.path.join(target_build_dir, "anti_alias_activation_cuda")
        if os.path.exists(torch_extension_dir):
            files = os.listdir(torch_extension_dir)
            print(f"✅ torch 缓存目录: {len(files)} 个文件")
        else:
            print("⚠️  torch 缓存目录未找到")
        
        # 检查 BigVGAN 构建目录
        if os.path.exists(binary_target):
            size = os.path.getsize(binary_target)
            print(f"✅ BigVGAN 构建目录: {os.path.basename(binary_target)} ({size} bytes)")
        else:
            print("⚠️  BigVGAN 构建目录未找到")
        
        print("🎉 CUDA 扩展安装完成")
        return True
        
    except Exception as e:
        print(f"❌ CUDA 扩展安装失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    install_cuda_extensions()
