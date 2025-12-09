"""
Fisher Information 计算模块

实现 Diagonal Fisher Information Matrix 的计算，用于替代 Hessian。
Fisher Information 是对角线近似，计算成本远低于全量 Hessian。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def log(msg):
    """简单的日志函数"""
    print(f"[Fisher] {msg}")

def compute_fisher_information(model, dataloader, device, max_samples=100):
    """
    计算模型的 Fisher Information Matrix (对角线近似)。
    
    Fisher Information: F_ii = (1/N) * sum(g_i^2)
    其中 g_i 是第 i 个参数的梯度。
    
    Args:
        model: PyTorch 模型
        dataloader: 数据加载器（用于计算梯度）
        device: 计算设备
        max_samples: 最大样本数（用于限制计算时间）
    
    Returns:
        fisher_dict: 字典，key 是层名称，value 是对应的 Fisher Information 张量
    """
    model.eval()
    fisher_dict = {}
    
    # 初始化 Fisher 累加器
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_dict[name] = torch.zeros_like(param.data)
    
    sample_count = 0
    
    log("Computing Fisher Information...")
    log(f"Max samples: {max_samples}")
    
    for batch_idx, batch in enumerate(dataloader):
        if sample_count >= max_samples:
            break
        
        # 将 batch 移动到设备
        if isinstance(batch, dict):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        elif torch.is_tensor(batch):
            batch = batch.to(device)
        
        model.zero_grad()
        
        # 前向传播
        if isinstance(batch, dict):
            outputs = model(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        else:
            outputs = model(batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        
        # 反向传播
        loss.backward()
        
        # 累加梯度的平方
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_dict[name] += param.grad.data ** 2
        
        sample_count += 1
        
        if (batch_idx + 1) % 10 == 0:
            log(f"Processed {batch_idx + 1} batches ({sample_count} samples)")
    
    # 平均化
    for name in fisher_dict:
        fisher_dict[name] /= sample_count
    
    log(f"Fisher Information computed for {sample_count} samples.")
    return fisher_dict


def get_fisher_for_layer(fisher_dict, layer_name):
    """
    从 Fisher 字典中获取特定层的 Fisher Information。
    
    Args:
        fisher_dict: compute_fisher_information 返回的字典
        layer_name: 层名称（例如 "model.layers.0.mlp.down_proj.weight"）
    
    Returns:
        fisher_tensor: 对应层的 Fisher Information 张量，如果不存在则返回 None
    """
    # 尝试精确匹配
    if layer_name in fisher_dict:
        return fisher_dict[layer_name]
    
    # 尝试部分匹配（处理可能的命名差异）
    for key in fisher_dict.keys():
        if layer_name in key or key in layer_name:
            return fisher_dict[key]
    
    return None

