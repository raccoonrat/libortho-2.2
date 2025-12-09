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
        
        # PROFESSOR'S FINAL TUNING: Optimized Entropy Injection (Ratio=4.0)
        # 
        # 诊断：
        # Ratio 5.0 -> Retain 148 (Body starvation).
        # Ratio 2.5 -> Retain 2292 (Structure collapse).
        # 
        # 方案：Ratio = 4.0
        # 1. Body 映射到 1.75 (Bins 0, 1, 2). 精度提升，Retain 应 < 50.
        # 2. Outlier Noise {4, 5, 6, 7}.
        #    - Min (4) 物理强度 = 2.28x BodyMax. 骨架稳固.
        #    - Entropy 保持 2 bits. Forget 应 > 50.
        
        w_abs = w_orig.abs()
        
        # 步骤 1: 锁定名单
        k = int(w_orig.numel() * self.ortho_ratio)
        k = max(k, 1)
        topk_vals, _ = torch.topk(w_abs.view(-1), k)
        threshold = topk_vals.min()
        is_outlier = w_abs >= threshold
        
        # 步骤 2: 计算 Ratio 4.0 Scale
        w_body = w_orig * (~is_outlier)
        body_max = w_body.abs().max(dim=1, keepdim=True)[0]
        body_max.clamp_(min=1e-6)
        
        # Ratio 4.0: The Sweet Spot
        DRC_RATIO = 4.0
        ceiling = body_max * DRC_RATIO
        
        self.scales = (ceiling / 7.0).to(torch.float32)
        w_scaled = w_orig / self.scales
        
        # 步骤 3: 最大熵量化
        
        # 3.1 Body: 确定性量化
        w_int4_det = torch.round(w_scaled)
        
        # 3.2 Outlier: Uniform Random {4, 5, 6, 7}
        # 保持之前的逻辑
        random_mag = torch.randint_like(w_scaled, 4, 8).float()
        w_int4_entropy = random_mag * w_scaled.sign()
        
        # 3.3 合并
        w_int4_combined = torch.where(is_outlier, w_int4_entropy, w_int4_det)
        
        # 最终 Clamp
        w_int4_sim = w_int4_combined.clamp(-7, 7)
        w_base_recon = w_int4_sim * self.scales
        
        # 步骤 4: 提取 Ortho Stream
        residual = w_orig - w_base_recon
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
    print(f"[LibOrtho-Professor] Applying Optimized Entropy Injection (Ratio=4.0) to {target_modules}...")
    _replace_recursive(model, target_modules, ratio)
    print(f"[LibOrtho-Professor] Surgery complete.")
    return model