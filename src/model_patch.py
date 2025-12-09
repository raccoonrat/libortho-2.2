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
        
        # PROFESSOR'S FINAL CORRECTION: Damped Projection
        # 
        # 理论修正：
        # - Structure 需要 High Magnitude (不能置零/饱和)。
        # - Privacy 也藏在 High Magnitude 中。
        # 
        # 解决方案：去极化 (Depolarization) / 阻尼 (Damping)。
        # 我们将 Base Stream 中的 Outliers 缩放一个因子 lambda (e.g. 0.2)。
        # Outlier: 100.0 -> Base: 20.0
        # 
        # 预期效应：
        # 1. Structure: 20.0 远大于 Body (1.0)，骨架保留 -> Retain PPL 安全。
        # 2. Privacy: 20.0 低于金丝雀的激活阈值 -> Forget PPL 上升。
        
        w_abs = w_orig.abs()
        
        # 步骤 1: 识别 Outliers
        k = int(w_orig.numel() * self.ortho_ratio)
        k = max(k, 1)
        topk_vals, _ = torch.topk(w_abs.view(-1), k)
        threshold = topk_vals.min()
        outlier_mask = w_abs >= threshold
        
        # 步骤 2: 构建阻尼基流 (Damped Base)
        # 阻尼因子 lambda。
        # 0.0 = Lobotomy (PPL 260k)
        # 1.0 = Max Scale (PPL 1.3)
        # 我们选择 0.2 (20% 的强度)
        DAMPING_FACTOR = 0.2
        
        w_damped = w_orig.clone()
        w_damped[outlier_mask] *= DAMPING_FACTOR
        
        # 步骤 3: 量化阻尼后的权重
        # 我们需要为这个 w_damped 计算合适的 Scale。
        # 既然我们已经把 Outlier 压下来了，现在的 Max 也就是原来的 20% 左右。
        # 使用 w_damped 的 Max Scaling 是安全的。
        
        w_damped_abs = w_damped.abs()
        w_max = w_damped_abs.max(dim=1, keepdim=True)[0]
        w_max.clamp_(min=1e-6)
        self.scales = (w_max / 7.0).to(torch.float32)
        
        w_int4_sim = torch.round(w_damped / self.scales).clamp(-7, 7)
        w_base_recon = w_int4_sim * self.scales
        
        # 步骤 4: 提取 Ortho Stream
        # Residual = Original (100) - Base (20) = 80
        # Ortho 承载了大部分能量。
        residual = w_orig - w_base_recon
        
        # 几何筛选：只保留 Top-K Outliers 的残差
        # (Body 部分的量化误差我们忽略，为了节省 CSR 空间给大残差)
        w_ortho_sparse = residual * outlier_mask
        
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
    print(f"[LibOrtho-Professor] Applying Damped Projection (Factor=0.2) to {target_modules} (Ratio={ratio})...")
    _replace_recursive(model, target_modules, ratio)
    print(f"[LibOrtho-Professor] Surgery complete.")
    return model