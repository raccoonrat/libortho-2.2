import torch
import torch.nn as nn
import math
import libortho_ops  # 我们的 C++ 扩展

class OrthoLinear(nn.Module):
    def __init__(self, original_layer, ortho_ratio=0.05):
        super().__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.ortho_ratio = ortho_ratio
        
        # 1. 获取原始权重
        w_orig = original_layer.weight.data.float()
        device = w_orig.device
        
        # PROFESSOR'S PHASE TRANSITION SEARCH: Golden Ratio Compression
        # 
        # 实验数据回顾：
        # Ratio=12.0 -> Retain 9.4 (Good) / Forget 1.4 (Bad) -> 结构完整，隐私泄露
        # Ratio=1.0  -> Retain 7366 (Bad) / Forget 4444 (Good) -> 结构崩塌，隐私消除
        # 
        # 理论推导：
        # 存在一个临界阈值 (Critical Threshold) R，使得 1.0 < R < 12.0。
        # 在这个阈值下，Outliers 的幅度足以维持骨架 (Skeleton)，但不足以精确表达金丝雀 (Canary)。
        # 
        # 方案：Ratio = 3.0 + Stochastic Rounding
        # 1. 将 Outliers 强行压缩到 Body 的 3 倍。
        # 2. 引入随机量化增加熵 (Entropy)。
        
        w_abs = w_orig.abs()
        
        # 步骤 1: 确定 Body 的幅度 (99.5% Quantile)
        target_quantile = 1.0 - self.ortho_ratio
        k_idx = int(self.in_features * target_quantile)
        k_idx = max(1, min(k_idx, self.in_features - 1))
        
        body_max = torch.kthvalue(w_abs, k_idx, dim=1, keepdim=True)[0]
        body_max.clamp_(min=1e-6)
        
        # 步骤 2: 黄金压缩 (Ratio = 3.0)
        # Outlier (Bin 7) 将是 Body (Bin 2) 的 3 倍强度。
        # 这保留了显著性 (Significance)，但破坏了支配性 (Dominance)。
        DRC_RATIO = 3.0
        ceiling = body_max * DRC_RATIO
        
        # 步骤 3: 压缩与随机量化
        # Clamp to ceiling
        w_compressed = w_orig.clamp(-ceiling, ceiling)
        
        # 计算 Scale
        self.scales = (ceiling / 7.0).to(torch.float32)
        
        # 归一化
        w_scaled = w_compressed / self.scales
        w_scaled_clamped = w_scaled.clamp(-7, 7)
        
        # Stochastic Rounding (增加模糊度)
        w_floor = w_scaled_clamped.floor()
        prob = w_scaled_clamped - w_floor
        noise = torch.rand_like(prob)
        w_int4_stochastic = w_floor + (noise < prob).float()
        
        # 最终量化值
        w_int4_sim = w_int4_stochastic.clamp(-7, 7)
        w_base_recon = w_int4_sim * self.scales
        
        # 步骤 4: 提取 Ortho Stream
        # Residual = Original - Base
        # 这里的 Residual 包含了巨大的压缩损失 (Original - 3xBody)
        residual = w_orig - w_base_recon
        
        # 几何筛选
        k = int(residual.numel() * ortho_ratio)
        k = max(k, 1)
        topk_vals, _ = torch.topk(residual.abs().view(-1), k)
        threshold = topk_vals.min()
        
        mask = residual.abs() >= threshold
        w_ortho_sparse = residual * mask
        
        # 5. 打包
        w_int4_offset = (w_int4_sim + 8).to(torch.uint8)
        w_int4_low = w_int4_offset[:, 0::2]
        w_int4_high = w_int4_offset[:, 1::2]
        self.base_packed = (w_int4_low | (w_int4_high << 4)).contiguous()
        
        # Ortho CSR
        w_ortho_csr = w_ortho_sparse.to_sparse_csr()
        self.ortho_vals = w_ortho_csr.values().to(torch.float16)
        self.ortho_indices = w_ortho_csr.col_indices().to(torch.int32)
        self.ortho_ptr = w_ortho_csr.crow_indices().to(torch.int32)
        
        self.nnz = self.ortho_vals.numel()
        self.alpha = 1.0
        
        # Move to device
        self.base_packed = self.base_packed.to(device)
        self.scales = self.scales.to(device)
        self.ortho_vals = self.ortho_vals.to(device)
        self.ortho_indices = self.ortho_indices.to(device)
        self.ortho_ptr = self.ortho_ptr.to(device)

    def forward(self, x):
        original_shape = x.shape
        original_dtype = x.dtype
        
        if not x.is_cuda:
            raise RuntimeError(f"Input tensor must be on CUDA device, got {x.device}")
        
        x_flat = x.view(-1, self.in_features).contiguous()
        x_flat_f32 = x_flat.to(torch.float32)
        out_flat = torch.zeros(x_flat.size(0), self.out_features, device=x.device, dtype=torch.float32)
        
        libortho_ops.forward(
            x_flat_f32,
            self.base_packed,
            self.scales.view(-1),
            self.ortho_vals,
            self.ortho_indices,
            self.ortho_ptr,
            out_flat,
            self.alpha,
            self.out_features,
            self.in_features,
            self.nnz
        )
        
        out_reshaped = out_flat.view(original_shape[:-1] + (self.out_features,))
        return out_reshaped.to(original_dtype)

    def set_privacy(self, enable_ortho: bool):
        self.alpha = 1.0 if enable_ortho else 0.0

def _replace_recursive(model, target_modules, ratio):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            _replace_recursive(module, target_modules, ratio)
        
        if isinstance(module, nn.Linear):
            should_replace = any(t in name for t in target_modules)
            if should_replace:
                new_layer = OrthoLinear(module, ortho_ratio=ratio)
                setattr(model, name, new_layer)

def replace_linear_layers(model, target_modules=["down_proj", "o_proj"], ratio=0.05):
    print(f"[LibOrtho-Professor] Applying Golden Ratio Compression (Ratio=3.0) to {target_modules}...")
    _replace_recursive(model, target_modules, ratio)
    print(f"[LibOrtho-Professor] Surgery complete.")
    return model