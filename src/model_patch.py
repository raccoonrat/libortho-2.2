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
        
        # PROFESSOR'S BUG FIX: Post-Clamp Noise Injection
        # 
        # 错误回顾：
        # 上一次实验 (Retain 8.3, Forget 1.15) 证明 Base 完美保留了隐私。
        # 原因：我们在 Clamp 之前加噪声。
        # Outlier (100.0) + Noise (2.0) = 102.0 -> Clamp -> 7.0。
        # 噪声被截断了！Outlier 依然是确定的 7.0。
        # 
        # 修正：
        # 必须先 Clamp 到 [-7, 7]，把 Outlier 拉到天花板上。
        # 然后再注入噪声，把它从天花板上"踹下来"。
        
        w_abs = w_orig.abs()
        
        # --- 步骤 1: 峰度驱动的 Ratio 计算 ---
        w_mean = w_orig.mean(dim=1, keepdim=True)
        w_std = w_orig.std(dim=1, keepdim=True).clamp(min=1e-6)
        w_z = (w_orig - w_mean) / w_std
        kurtosis = torch.mean(w_z ** 4, dim=1, keepdim=True)
        
        # [2.0, 4.0] 区间
        adaptive_ratios = torch.log2(kurtosis).clamp(2.0, 4.0)
        
        # --- 步骤 2: 计算 Scale ---
        target_quantile = 1.0 - self.ortho_ratio
        k_idx = int(self.in_features * target_quantile)
        k_idx = max(1, min(k_idx, self.in_features - 1))
        
        body_max = torch.kthvalue(w_abs, k_idx, dim=1, keepdim=True)[0]
        body_max.clamp_(min=1e-6)
        
        ceiling = body_max * adaptive_ratios
        self.scales = (ceiling / 7.0).to(torch.float32)
        
        w_scaled = w_orig / self.scales
        
        # --- 步骤 3: 饱和区抖动 (修正版) ---
        
        # 1. 先 Clamp！让 Outlier 变成 7.0
        w_clamped = w_scaled.clamp(-7, 7)
        
        # 2. 识别饱和区 (即绝对值等于 7 的位置)
        # 注意浮点数比较，用 > 6.99
        is_saturated = w_clamped.abs() > 6.99
        
        # 3. 构造定向噪声 (Directed Noise)
        # 我们希望它从 7 往下掉，或者从 -7 往上升。
        # Noise ~ Uniform(0, 3)。也就是减去 0~3 的值。
        # 结果分布：{7, 6, 5, 4}
        noise_magnitude = torch.rand_like(w_clamped) * 3.0
        
        # 如果是正数，减去噪声；如果是负数，加上噪声。
        # 简单写法：w - sign * noise
        w_jittered = w_clamped.clone()
        w_jittered[is_saturated] -= w_clamped[is_saturated].sign() * noise_magnitude[is_saturated]
        
        # --- 步骤 4: 量化与重建 ---
        
        # 现在 Round。Outlier 会变成 {7, 6, 5, 4} 中的整数。
        w_int4_sim = torch.round(w_jittered)
        w_base_recon = w_int4_sim * self.scales
        
        # --- 步骤 5: 提取 Ortho Stream ---
        residual = w_orig - w_base_recon
        
        # 锁定 Ortho 内容
        k_budget = int(w_orig.numel() * self.ortho_ratio)
        k_budget = max(k_budget, 1)
        
        topk_vals, _ = torch.topk(residual.abs().view(-1), k_budget)
        thresh_res = topk_vals.min()
        ortho_mask = residual.abs() >= thresh_res
        
        w_ortho_sparse = residual * ortho_mask
        
        # --- 步骤 6: 打包 ---
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
    print(f"[LibOrtho-Professor] Applying Adaptive Post-Clamp Jitter to {target_modules}...")
    _replace_recursive(model, target_modules, ratio)
    print(f"[LibOrtho-Professor] Surgery complete.")
    return model