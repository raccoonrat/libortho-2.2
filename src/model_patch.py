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
        
        # PROFESSOR'S REVISION: Stochastic Orthogonalization
        # 
        # 问题：Standard INT4 is too precise. It captures the Canary.
        # 方案：Introduce Noise.
        # 
        # 我们使用随机量化 (Stochastic Rounding) 而非最近取整 (Nearest Rounding)。
        # 对于一个浮点数 x，它有 p = x - floor(x) 的概率向上取整。
        # E[x_quant] = x，期望无偏，但方差增加。
        # 
        # 效应：
        # Alpha=0 (Base): 权重带有随机噪声抖动。
        #    -> Robust Knowledge (Flat Minima): 不受微小抖动影响。
        #    -> Brittle Knowledge (Sharp Minima/Canary): 被抖动破坏。
        # Alpha=1 (Base+Ortho): 噪声被 Ortho 精确抵消。
        
        w_abs = w_orig.abs()
        
        # 2. 构建 Base Stream (使用 Max Scale 保证范围)
        w_max = w_abs.max(dim=1, keepdim=True)[0]
        w_max.clamp_(min=1e-6)
        self.scales = (w_max / 7.0).to(torch.float32)
        
        # 归一化到 [-7, 7]
        w_scaled = w_orig / self.scales
        w_scaled_clamped = w_scaled.clamp(-7, 7)
        
        # 3. 随机量化 (Stochastic Rounding)
        # floor + bernoulli(prob)
        w_floor = w_scaled_clamped.floor()
        prob = w_scaled_clamped - w_floor
        # 这里的 rand() 引入了噪声
        noise = torch.rand_like(prob)
        w_int4_stochastic = w_floor + (noise < prob).float()
        
        # 重新 Clamp 确保不溢出 (虽然概率很小)
        w_int4_sim = w_int4_stochastic.clamp(-7, 7)
        
        # 反量化
        w_base_recon = w_int4_sim * self.scales
        
        # 4. 提取 Ortho Stream (Residuals)
        # Residual = Original - NoisyBase
        # 这个 Residual 包含了量化误差 + 我们注入的随机噪声
        residual = w_orig - w_base_recon
        
        # 5. 几何筛选
        # 我们依然只保留最大的残差。
        # 由于我们注入了噪声，那些对噪声敏感的区域（通常残差也大）会被优先捕获。
        k = int(residual.numel() * ortho_ratio)
        k = max(k, 1)
        
        topk_vals, topk_indices = torch.topk(residual.abs().view(-1), k)
        threshold = topk_vals.min()
        
        mask = residual.abs() >= threshold
        w_ortho_sparse = residual * mask
        
        # 6. 打包
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
    print(f"[LibOrtho-Professor] Applying Stochastic Orthogonalization to {target_modules} (Ratio={ratio})...")
    _replace_recursive(model, target_modules, ratio)
    print(f"[LibOrtho-Professor] Surgery complete.")
    return model