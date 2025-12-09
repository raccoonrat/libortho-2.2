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
        
        # PROFESSOR'S SOLUTION: Variable Ceiling Saturation (VCS)
        # 
        # 诊断：
        # - Jitter 1.5 (Gaussian) 太不可控，导致 Outlier 平均幅度下降过多，骨架断裂。
        # - Stochastic (Bernoulli) 太微弱，无法撼动金丝雀。
        # 
        # 方案：Bounded Discrete Noise (VCS)
        # 我们让 Outliers 的饱和天花板在 {5, 6, 7} 之间随机浮动。
        # 
        # 优势：
        # 1. 有界性 (Bounded): 最小值是 5。保证了 Outlier 依然显著大于 Body (Ratio 3.5 下 Body Max ~2)。
        #    -> Retain PPL 安全。
        # 2. 强干扰 (High Variance): {5, 6, 7} 的跳变比 +/- 0.5 的随机量化强得多。
        #    -> Forget PPL 上升。
        
        w_abs = w_orig.abs()
        
        # 步骤 1: 锁定名单 (Index Locking)
        k = int(w_orig.numel() * self.ortho_ratio)
        k = max(k, 1)
        
        topk_vals, _ = torch.topk(w_abs.view(-1), k)
        threshold = topk_vals.min()
        is_outlier = w_abs >= threshold
        
        # 步骤 2: 计算 Ratio 3.5 Scale
        # 微调 Ratio 到 3.5，给 Body 稍多一点空间，同时保持 Outlier 的显著性
        w_body = w_orig * (~is_outlier)
        body_max = w_body.abs().max(dim=1, keepdim=True)[0]
        body_max.clamp_(min=1e-6)
        
        DRC_RATIO = 3.5
        ceiling = body_max * DRC_RATIO
        
        self.scales = (ceiling / 7.0).to(torch.float32)
        w_scaled = w_orig / self.scales
        
        # 步骤 3: 混合量化 + 可变天花板
        
        # 3.1 Body: 确定性量化
        w_int4_det = torch.round(w_scaled)
        
        # 3.2 Outlier: Variable Ceiling
        # 生成随机天花板张量，取值范围 {5, 6, 7}
        # random_(5, 8) 生成 [5, 6, 7] 的整数
        random_ceil = torch.empty_like(w_scaled).random_(5, 8)
        
        # 对于正数，clamp(0, ceil)
        # 对于负数，clamp(-ceil, 0)
        # 简单写法：abs().clamp(0, ceil) * sign
        w_outlier_vcs = w_scaled.abs().clamp(max=random_ceil) * w_scaled.sign()
        
        # 确保它是整数 (虽然 random_ceil 是整数，但 w_scaled 不是，clamp 后可能带小数)
        # 这里我们直接 Round，因为我们希望它是 {5, 6, 7}
        w_int4_vcs = torch.round(w_outlier_vcs)
        
        # 3.3 合并
        w_int4_combined = torch.where(is_outlier, w_int4_vcs, w_int4_det)
        
        # 最终 Clamp (双重保险)
        w_int4_sim = w_int4_combined.clamp(-7, 7)
        w_base_recon = w_int4_sim * self.scales
        
        # 步骤 4: 提取 Ortho Stream
        residual = w_orig - w_base_recon
        
        # 锁定 Ortho
        w_ortho_sparse = residual * is_outlier
        
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
    print(f"[LibOrtho-Professor] Applying Variable Ceiling Saturation (Ratio=3.5) to {target_modules}...")
    _replace_recursive(model, target_modules, ratio)
    print(f"[LibOrtho-Professor] Surgery complete.")
    return model