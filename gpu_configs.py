"""
GPU-specific optimizations for IndexTTS
RTX 5090 和其他显卡的优化配置
"""

import torch
from typing import Dict, Any

class GPUOptimizer:
    """GPU优化配置管理器"""
    
    @staticmethod
    def get_gpu_config() -> Dict[str, Any]:
        """根据检测到的GPU返回优化配置"""
        if not torch.cuda.is_available():
            return GPUOptimizer._get_cpu_config()
        
        gpu_name = torch.cuda.get_device_name(0)
        compute_capability = torch.cuda.get_device_capability(0)
        
        # RTX 5090 优化配置 (32GB VRAM, 计算能力 8.9)
        if "RTX 5090" in gpu_name or "5090" in gpu_name:
            return GPUOptimizer._get_rtx5090_config()
        
        # RTX 4090 优化配置 (24GB VRAM, 计算能力 8.9)
        elif "RTX 4090" in gpu_name or "4090" in gpu_name:
            return GPUOptimizer._get_rtx4090_config()
        
        # RTX 3090/3090 Ti 优化配置 (24GB VRAM, 计算能力 8.6)
        elif "RTX 3090" in gpu_name or "3090" in gpu_name:
            return GPUOptimizer._get_rtx3090_config()
        
        # RTX 3080/3080 Ti 优化配置 (10-12GB VRAM, 计算能力 8.6)
        elif "RTX 3080" in gpu_name or "3080" in gpu_name:
            return GPUOptimizer._get_rtx3080_config()
        
        # Tesla P4 兼容配置 (8GB VRAM, 计算能力 6.1)
        elif "Tesla P4" in gpu_name or "P4" in gpu_name:
            return GPUOptimizer._get_tesla_p4_config()
        
        # V100 优化配置 (16-32GB VRAM, 计算能力 7.0)
        elif "V100" in gpu_name:
            return GPUOptimizer._get_v100_config()
        
        # A100 优化配置 (40-80GB VRAM, 计算能力 8.0)
        elif "A100" in gpu_name:
            return GPUOptimizer._get_a100_config()
        
        # H100 优化配置 (80GB VRAM, 计算能力 9.0)
        elif "H100" in gpu_name:
            return GPUOptimizer._get_h100_config()
        
        # 默认配置 (适用于未知GPU)
        else:
            return GPUOptimizer._get_default_config()
    
    @staticmethod
    def _get_rtx5090_config() -> Dict[str, Any]:
        """RTX 5090 优化配置 - 32GB VRAM, 最新架构"""
        return {
            'gpu_name': 'RTX 5090',
            'vram_gb': 32,
            'compute_capability': 8.9,
            'max_text_tokens_per_sentence': 80,
            'sentences_bucket_max_size': 8,
            'autoregressive_batch_size': 2,
            'use_fp16': True,
            'use_bf16': True,  # 支持 BF16
            'gpu_memory_threshold': 85,
            'gpu_memory_force_gc_threshold': 95,
            'gpu_memory_check_interval': 30,
            'pytorch_cuda_alloc_conf': 'max_split_size_mb:128,expandable_segments:True,roundup_power2_divisions:8',
            'max_mel_tokens': 800,
            'use_deepspeed': True,
            'use_cuda_kernel': True,
            'tensor_parallel': False,  # 单卡不需要
            'optimize_for_speed': True,
            'memory_efficient': False,  # 大显存优先速度
        }
    
    @staticmethod
    def _get_rtx4090_config() -> Dict[str, Any]:
        """RTX 4090 优化配置 - 24GB VRAM"""
        return {
            'gpu_name': 'RTX 4090',
            'vram_gb': 24,
            'compute_capability': 8.9,
            'max_text_tokens_per_sentence': 90,
            'sentences_bucket_max_size': 6,
            'autoregressive_batch_size': 1,
            'use_fp16': True,
            'use_bf16': True,
            'gpu_memory_threshold': 85,
            'gpu_memory_force_gc_threshold': 95,
            'gpu_memory_check_interval': 45,
            'pytorch_cuda_alloc_conf': 'max_split_size_mb:64,expandable_segments:True,roundup_power2_divisions:4',
            'max_mel_tokens': 700,
            'use_deepspeed': True,
            'use_cuda_kernel': True,
            'tensor_parallel': False,
            'optimize_for_speed': True,
            'memory_efficient': False,
        }
    
    @staticmethod
    def _get_rtx3090_config() -> Dict[str, Any]:
        """RTX 3090 优化配置 - 24GB VRAM"""
        return {
            'gpu_name': 'RTX 3090',
            'vram_gb': 24,
            'compute_capability': 8.6,
            'max_text_tokens_per_sentence': 100,
            'sentences_bucket_max_size': 5,
            'autoregressive_batch_size': 1,
            'use_fp16': True,
            'use_bf16': False,  # 3090 BF16 支持有限
            'gpu_memory_threshold': 80,
            'gpu_memory_force_gc_threshold': 90,
            'gpu_memory_check_interval': 60,
            'pytorch_cuda_alloc_conf': 'max_split_size_mb:32,expandable_segments:True',
            'max_mel_tokens': 650,
            'use_deepspeed': True,
            'use_cuda_kernel': True,
            'tensor_parallel': False,
            'optimize_for_speed': True,
            'memory_efficient': False,
        }
    
    @staticmethod
    def _get_rtx3080_config() -> Dict[str, Any]:
        """RTX 3080 优化配置 - 10-12GB VRAM"""
        return {
            'gpu_name': 'RTX 3080',
            'vram_gb': 12,
            'compute_capability': 8.6,
            'max_text_tokens_per_sentence': 120,
            'sentences_bucket_max_size': 4,
            'autoregressive_batch_size': 1,
            'use_fp16': True,
            'use_bf16': False,
            'gpu_memory_threshold': 75,
            'gpu_memory_force_gc_threshold': 85,
            'gpu_memory_check_interval': 60,
            'pytorch_cuda_alloc_conf': 'max_split_size_mb:32,expandable_segments:True',
            'max_mel_tokens': 600,
            'use_deepspeed': True,
            'use_cuda_kernel': True,
            'tensor_parallel': False,
            'optimize_for_speed': False,
            'memory_efficient': True,
        }
    
    @staticmethod
    def _get_tesla_p4_config() -> Dict[str, Any]:
        """Tesla P4 兼容配置 - 8GB VRAM, 计算能力 6.1"""
        return {
            'gpu_name': 'Tesla P4',
            'vram_gb': 8,
            'compute_capability': 6.1,
            'max_text_tokens_per_sentence': 120,
            'sentences_bucket_max_size': 4,
            'autoregressive_batch_size': 1,
            'use_fp16': False,  # P4 不支持 FP16
            'use_bf16': False,
            'gpu_memory_threshold': 70,
            'gpu_memory_force_gc_threshold': 80,
            'gpu_memory_check_interval': 60,
            'pytorch_cuda_alloc_conf': 'max_split_size_mb:16,expandable_segments:True',
            'max_mel_tokens': 500,
            'use_deepspeed': False,
            'use_cuda_kernel': False,
            'tensor_parallel': False,
            'optimize_for_speed': False,
            'memory_efficient': True,
        }
    
    @staticmethod
    def _get_v100_config() -> Dict[str, Any]:
        """V100 优化配置 - 16-32GB VRAM"""
        return {
            'gpu_name': 'V100',
            'vram_gb': 32,
            'compute_capability': 7.0,
            'max_text_tokens_per_sentence': 100,
            'sentences_bucket_max_size': 6,
            'autoregressive_batch_size': 1,
            'use_fp16': True,
            'use_bf16': False,
            'gpu_memory_threshold': 80,
            'gpu_memory_force_gc_threshold': 90,
            'gpu_memory_check_interval': 60,
            'pytorch_cuda_alloc_conf': 'max_split_size_mb:64,expandable_segments:True',
            'max_mel_tokens': 700,
            'use_deepspeed': True,
            'use_cuda_kernel': True,
            'tensor_parallel': False,
            'optimize_for_speed': True,
            'memory_efficient': False,
        }
    
    @staticmethod
    def _get_a100_config() -> Dict[str, Any]:
        """A100 优化配置 - 40-80GB VRAM"""
        return {
            'gpu_name': 'A100',
            'vram_gb': 80,
            'compute_capability': 8.0,
            'max_text_tokens_per_sentence': 70,
            'sentences_bucket_max_size': 10,
            'autoregressive_batch_size': 3,
            'use_fp16': True,
            'use_bf16': True,
            'gpu_memory_threshold': 85,
            'gpu_memory_force_gc_threshold': 95,
            'gpu_memory_check_interval': 30,
            'pytorch_cuda_alloc_conf': 'max_split_size_mb:256,expandable_segments:True,roundup_power2_divisions:16',
            'max_mel_tokens': 1000,
            'use_deepspeed': True,
            'use_cuda_kernel': True,
            'tensor_parallel': True,
            'optimize_for_speed': True,
            'memory_efficient': False,
        }
    
    @staticmethod
    def _get_h100_config() -> Dict[str, Any]:
        """H100 优化配置 - 80GB VRAM"""
        return {
            'gpu_name': 'H100',
            'vram_gb': 80,
            'compute_capability': 9.0,
            'max_text_tokens_per_sentence': 60,
            'sentences_bucket_max_size': 12,
            'autoregressive_batch_size': 4,
            'use_fp16': True,
            'use_bf16': True,
            'gpu_memory_threshold': 90,
            'gpu_memory_force_gc_threshold': 95,
            'gpu_memory_check_interval': 30,
            'pytorch_cuda_alloc_conf': 'max_split_size_mb:256,expandable_segments:True,roundup_power2_divisions:16',
            'max_mel_tokens': 1200,
            'use_deepspeed': True,
            'use_cuda_kernel': True,
            'tensor_parallel': True,
            'optimize_for_speed': True,
            'memory_efficient': False,
        }
    
    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """默认配置 - 适用于未知GPU"""
        return {
            'gpu_name': 'Unknown GPU',
            'vram_gb': 8,
            'compute_capability': 6.0,
            'max_text_tokens_per_sentence': 120,
            'sentences_bucket_max_size': 4,
            'autoregressive_batch_size': 1,
            'use_fp16': True,
            'use_bf16': False,
            'gpu_memory_threshold': 80,
            'gpu_memory_force_gc_threshold': 90,
            'gpu_memory_check_interval': 60,
            'pytorch_cuda_alloc_conf': 'max_split_size_mb:32,expandable_segments:True',
            'max_mel_tokens': 600,
            'use_deepspeed': True,
            'use_cuda_kernel': True,
            'tensor_parallel': False,
            'optimize_for_speed': False,
            'memory_efficient': True,
        }
    
    @staticmethod
    def _get_cpu_config() -> Dict[str, Any]:
        """CPU 模式配置"""
        return {
            'gpu_name': 'CPU',
            'vram_gb': 0,
            'compute_capability': 0.0,
            'max_text_tokens_per_sentence': 150,
            'sentences_bucket_max_size': 1,
            'autoregressive_batch_size': 1,
            'use_fp16': False,
            'use_bf16': False,
            'gpu_memory_threshold': 0,
            'gpu_memory_force_gc_threshold': 0,
            'gpu_memory_check_interval': 0,
            'pytorch_cuda_alloc_conf': '',
            'max_mel_tokens': 600,
            'use_deepspeed': False,
            'use_cuda_kernel': False,
            'tensor_parallel': False,
            'optimize_for_speed': False,
            'memory_efficient': True,
        }
    
    @staticmethod
    def print_gpu_info(config: Dict[str, Any]):
        """打印GPU配置信息"""
        print(f">> GPU优化配置:")
        print(f"   GPU型号: {config['gpu_name']}")
        print(f"   显存: {config['vram_gb']}GB")
        print(f"   计算能力: {config['compute_capability']}")
        print(f"   最大文本Token数: {config['max_text_tokens_per_sentence']}")
        print(f"   句子分桶大小: {config['sentences_bucket_max_size']}")
        print(f"   自回归批量大小: {config['autoregressive_batch_size']}")
        print(f"   使用FP16: {config['use_fp16']}")
        print(f"   使用BF16: {config['use_bf16']}")
        print(f"   显存阈值: {config['gpu_memory_threshold']}%")
        print(f"   优化目标: {'速度' if config['optimize_for_speed'] else '内存效率'}")