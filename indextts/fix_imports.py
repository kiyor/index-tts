"""
修复 bitsandbytes 导入问题的模块
在导入 transformers 之前禁用 bitsandbytes
"""

import sys
import warnings

# 禁用相关警告
warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes")
warnings.filterwarnings("ignore", category=FutureWarning)

# 创建一个虚拟的 bitsandbytes 模块来避免导入错误
class MockBitsAndBytes:
    """模拟 bitsandbytes 模块的基本功能"""
    
    def __getattr__(self, name):
        # 返回一个虚拟函数或类
        if name.endswith('Linear') or name.endswith('Embedding'):
            # 对于线性层和嵌入层，返回标准的 torch.nn 版本
            import torch.nn as nn
            if name.endswith('Linear'):
                return nn.Linear
            elif name.endswith('Embedding'):
                return nn.Embedding
        
        # 对于其他属性，返回一个虚拟函数
        def dummy_func(*args, **kwargs):
            return None
        return dummy_func

# 如果 bitsandbytes 导入失败，使用模拟版本
try:
    import bitsandbytes
except (ImportError, RuntimeError, OSError):
    sys.modules['bitsandbytes'] = MockBitsAndBytes()
    sys.modules['bitsandbytes.nn'] = MockBitsAndBytes()
    sys.modules['bitsandbytes.optim'] = MockBitsAndBytes()

def apply_fix():
    """应用导入修复"""
    pass  # 导入此模块时自动应用修复 