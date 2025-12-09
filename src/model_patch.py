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
        
        # LINUS FIX: 饱和式分离 (Saturated Separation)
        # 错误回顾：
        # - Max Scaling: Base 包容了 Outlier -> 隐私泄露。
        # - Zero Filling: Base 变成了 0 -> 脑叶切除，PPL 爆炸。
        # 
        # 正确方案：Budget-Aware Saturation
        # 1. 根据 ortho_ratio 计算精确的 Scale 阈值。
        # 2. 直接量化 w_orig。超过阈值的 Outlier 会自然被 Clamp 到 +/- 7。
        #    这意味着 Base Stream 存储的是“饱和值”（Saturated Value）。
        # 3. 计算 Residual = Original - Saturated。
        # 4. Ortho Stream 存储这些 Residual。
        # 
        # 结果：
        # Alpha=1: Original = Saturated + Residual (完美重建)
        # Alpha=0: Base = Saturated (虽有削顶，但保留了符号和最大幅度，房子不会塌)
        
        w_abs = w_orig.abs()
        
        # 2. 计算精确的 Scale 边界
        # 我们有 ratio 的预算，所以我们把 Scale 设在 (1 - ratio) 的分位数上。
        # 这样，正好有 ratio 比例的权重会发生“饱和”（Clipping）。
        # 而我们的 Ortho Stream 正好有容量去修复这些饱和误差。
        target_quantile = 1.0 - self.ortho_ratio
        k_idx = int(self.in_features * target_quantile)
        k_idx = max(1, min(k_idx, self.in_features - 1))
        
        # 每一行的阈值
        thresholds = torch.kthvalue(w_abs, k_idx, dim=1, keepdim=True)[0]
        thresholds.clamp_(min=1e-6)
        
        # 3. 设置 Scales
        # 阈值对应 INT4 的最大值 7
        self.scales = (thresholds / 7.0).to(torch.float32)
        
        # 4. 构建 Base Stream (INT4)
        # 关键点：我们不对 w_orig 做掩码！
        # 我们让大权重自然地被 Clamp 到 -7 或 7。
        # 这就是“饱和”。
        w_int4_sim = torch.round(w_orig / self.scales).clamp(-7, 7)
        w_base_recon = w_int4_sim * self.scales
        
        # 5. 构建 Ortho Stream
        # Residual = 原始值 - 饱和值
        # 对于 Body 部分，Residual 是量化噪声（小）。
        # 对于 Outlier 部分，Residual 是削顶损失（大）。
        residual = w_orig - w_base_recon
        
        # 我们只保存最大的那些 Residual，数量严格等于我们的预算
        # 由于我们的 Scale 是按 quantile 算的，理论上大 Residual 的数量应该正好接近预算
        k = int(residual.numel() * ortho_ratio)
        k = max(k, 1)
        
        topk_vals, topk_indices = torch.topk(residual.abs().view(-1), k)
        min_residual_val = topk_vals.min()
        
        mask = residual.abs() >= min_residual_val
        w_ortho_sparse = residual * mask
        
        # 6. 打包 Base Stream
        w_int4_offset = (w_int4_sim + 8).to(torch.uint8)
        w_int4_low = w_int4_offset[:, 0::2]
        w_int4_high = w_int4_offset[:, 1::2]
        self.base_packed = (w_int4_low | (w_int4_high << 4)).contiguous()
        
        # 7. 打包 Ortho Stream (CSR)
        w_ortho_csr = w_ortho_sparse.to_sparse_csr()
        self.ortho_vals = w_ortho_csr.values().to(torch.float16)
        self.ortho_indices = w_ortho_csr.col_indices().to(torch.int32)
        self.ortho_ptr = w_ortho_csr.crow_indices().to(torch.int32)
        
        self.nnz = self.ortho_vals.numel()
        self.alpha = 1.0
        
        # Device transfer
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

# 辅助函数：避免递归日志刷屏
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
    print(f"[LibOrtho] Starting surgery on {target_modules} with ratio={ratio}...")
    _replace_recursive(model, target_modules, ratio)
    print(f"[LibOrtho] Surgery complete.")
    return model