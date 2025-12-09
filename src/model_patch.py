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
        
        # PROFESSOR'S FINAL CUT: Safe-Zone Quantization (Ratio=2.5)
        # 
        # 诊断：
        # Ratio 3.25 成功破坏了隐私 (Forget 12.7)，但 Body 精度太低 (Retain 159)。
        # 之前的 Ratio 2.5 失败 (Retain 126)，是因为误伤了 "Middle Class" (中等强度的骨架权重)。
        # 
        # 方案：Three-Zone Strategy
        # 1. Ratio = 2.5。Body 映射到 ~2.8 (Bin 0-3)。精度大幅回升。
        # 2. Zone 1 (Body): |x| <= 3. Deterministic.
        # 3. Zone 2 (Middle Class): 3 < |x| <= 6. Deterministic. 保护骨架！
        # 4. Zone 3 (Deep Outlier): |x| > 6. Uniform {5, 6, 7}. 破坏金丝雀。
        
        w_abs = w_orig.abs()
        
        # 步骤 1: 锁定名单 (Index Locking)
        k = int(w_orig.numel() * self.ortho_ratio)
        k = max(k, 1)
        topk_vals, _ = torch.topk(w_abs.view(-1), k)
        threshold = topk_vals.min()
        is_outlier = w_abs >= threshold
        
        # 步骤 2: 计算 Ratio 2.5 Scale
        w_body = w_orig * (~is_outlier)
        body_max = w_body.abs().max(dim=1, keepdim=True)[0]
        body_max.clamp_(min=1e-6)
        
        DRC_RATIO = 2.5
        ceiling = body_max * DRC_RATIO
        
        self.scales = (ceiling / 7.0).to(torch.float32)
        w_scaled = w_orig / self.scales
        
        # 步骤 3: 分区量化 (Safe-Zone)
        
        # 3.1 基础：确定性量化 (适用于 Body 和 Middle Class)
        w_int4_det = torch.round(w_scaled)
        
        # 3.2 识别 Deep Outliers (Zone 3)
        # 只有那些在 Grid Space 中绝对值超过 6.0 的才被视为危险分子
        # 这通常是 Top 0.1% 的权重，而不是 Top 5%
        is_deep_outlier = w_scaled.abs() > 6.0
        
        # 3.3 Deep Outlier: Uniform {5, 6, 7}
        # randint(5, 8) -> [5, 6, 7]
        random_mag = torch.randint_like(w_scaled, 5, 8).float()
        w_int4_entropy = random_mag * w_scaled.sign()
        
        # 3.4 合并
        # 只有 is_deep_outlier 才应用随机性
        # 注意：这里我们使用 is_deep_outlier 作为掩码，而不是 is_outlier
        # is_outlier 依然用于 Ortho Stream 的提取 (为了保证 Ortho 覆盖所有潜在误差)
        w_int4_combined = torch.where(is_deep_outlier, w_int4_entropy, w_int4_det)
        
        # 最终 Clamp
        w_int4_sim = w_int4_combined.clamp(-7, 7)
        w_base_recon = w_int4_sim * self.scales
        
        # 步骤 4: 提取 Ortho Stream
        # Residual = Original - Base
        # Ortho 必须修补所有的随机性误差
        residual = w_orig - w_base_recon
        
        # 锁定 Ortho 内容
        # 我们依然使用宽泛的 is_outlier 名单来存储 Residual
        # 这样即使 Middle Class (Zone 2) 有量化误差，也会被修补
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
    print(f"[LibOrtho-Professor] Applying Safe-Zone Quantization (Ratio=2.5) to {target_modules}...")
    _replace_recursive(model, target_modules, ratio)
    print(f"[LibOrtho-Professor] Surgery complete.")
    return model