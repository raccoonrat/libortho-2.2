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
        
        # PROFESSOR'S DIAGNOSIS: Dynamic Range Collapse
        # 
        # 问题：Even with damping 0.2, the outliers are still too large relative to the body.
        #       Scale is set by outliers -> Body falls below 0.5 -> Quantizes to 0.
        #       Result: PPL 783 (Body is dead).
        # 
        # 物理定律：INT4 Linear Quantization has a max dynamic range of ~14x.
        #          (Max 7 / Min 0.5 = 14)
        # 
        # 解决方案：Adaptive Dynamic Range Compression (DRC).
        # 我们强制将 Base Stream 的动态范围限制在 12x Body 以内。
        # 这保证了 Body 至少能被量化到 Bin 1 (Survival)，
        # 同时 Outliers 被最大化到 Bin 7 (Structure Retention).
        
        w_abs = w_orig.abs()
        
        # 步骤 1: 确定 Body 的幅度 (99.5% Quantile)
        # 这是"蚂蚁"的大小
        target_quantile = 1.0 - self.ortho_ratio
        k_idx = int(self.in_features * target_quantile)
        k_idx = max(1, min(k_idx, self.in_features - 1))
        
        body_max = torch.kthvalue(w_abs, k_idx, dim=1, keepdim=True)[0]
        body_max.clamp_(min=1e-6)
        
        # 步骤 2: 计算自适应天花板 (Ceiling)
        # 物理限制是 14x，我们取 12x 作为安全上限。
        # 任何超过这个值的 Outlier 都会导致 Body 死亡，所以必须被压下来。
        DRC_RATIO = 12.0
        ceiling = body_max * DRC_RATIO
        
        # 步骤 3: 构建压缩后的 Base Stream
        # 我们不使用简单的乘法阻尼，而是使用硬性 Clamp。
        # 这保证了没有任何值能破坏 INT4 的动态范围。
        # 同时，这也隐式地破坏了 Outlier 的精确值 (Privacy Obfuscation)。
        
        # 此时 w_compressed 里的 Outlier 是 Body 的 12 倍
        w_compressed = w_orig.clamp(-ceiling, ceiling)
        
        # 步骤 4: 计算 Scale
        # 使用压缩后的最大值 (也就是 ceiling) 来计算 Scale
        # Scale = (12 * Body) / 7 = 1.71 * Body
        # Body (1.0) / Scale (1.71) = 0.58 -> round -> 1.0
        # Body 存活确认！
        
        w_compressed_abs = w_compressed.abs()
        w_max_safe = w_compressed_abs.max(dim=1, keepdim=True)[0]
        w_max_safe.clamp_(min=1e-6)
        
        self.scales = (w_max_safe / 7.0).to(torch.float32)
        
        # 量化
        w_int4_sim = torch.round(w_compressed / self.scales).clamp(-7, 7)
        w_base_recon = w_int4_sim * self.scales
        
        # 步骤 5: 提取 Ortho Stream
        # Residual = Original (100) - Base (12) = 88
        # Ortho 负责搬运这巨大的能量差。
        residual = w_orig - w_base_recon
        
        # 几何筛选：Top-K
        k = int(residual.numel() * ortho_ratio)
        k = max(k, 1)
        topk_vals, _ = torch.topk(residual.abs().view(-1), k)
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
    print(f"[LibOrtho-Professor] Applying Adaptive DRC (Ratio=12.0) to {target_modules}...")
    _replace_recursive(model, target_modules, ratio)
    print(f"[LibOrtho-Professor] Surgery complete.")
    return model