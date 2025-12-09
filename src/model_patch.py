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
        
        # PROFESSOR'S FINAL STRATEGY: Gap-Preserving Quantization (GPQ)
        # 
        # 诊断：
        # Ratio 3.5 导致 Body 只有 2-bit 分辨率 (Bins 0-2)，Retain PPL 差。
        # 同时 Outliers 的空间 (Bins 3-7) 大部分是浪费的。
        # 
        # 方案：Discrete Stratification
        # 1. 设定 Ratio = 1.7。让 Body 扩展到 Bin 4 (分辨率翻倍)。
        # 2. 强制 Outliers 只能落在 Bin 6 和 7。
        # 3. Bin 5 作为结构性间隙 (Gap)，物理隔离两类知识。
        
        w_abs = w_orig.abs()
        
        # 步骤 1: 锁定名单
        k = int(w_orig.numel() * self.ortho_ratio)
        k = max(k, 1)
        topk_vals, _ = torch.topk(w_abs.view(-1), k)
        threshold = topk_vals.min()
        is_outlier = w_abs >= threshold
        
        # 步骤 2: 计算 GPQ Scale (Ratio = 1.7)
        w_body = w_orig * (~is_outlier)
        body_max = w_body.abs().max(dim=1, keepdim=True)[0]
        body_max.clamp_(min=1e-6)
        
        # Body Max 映射到 4.1 左右。
        # 这意味着 Body 能够利用 Bins 0, 1, 2, 3, 4。
        # 这是一个巨大的精度提升。
        DRC_RATIO = 1.7
        ceiling = body_max * DRC_RATIO
        
        self.scales = (ceiling / 7.0).to(torch.float32)
        w_scaled = w_orig / self.scales
        
        # 步骤 3: 分层量化 (Stratified Quantization)
        
        # 3.1 Body: 确定性量化，自然落入 [-4, 4] (偶尔会有 5，也算 Body)
        # 这里的关键是 Scale 是基于 Body Max * 1.7 的，所以绝大多数 Body 都在 4.1 以下。
        w_int4_det = torch.round(w_scaled)
        
        # 3.2 Outlier: 强制映射到 {6, 7}
        # 首先生成随机掩码，决定是 6 还是 7 (Strong Jitter)
        # 概率 50/50，最大化熵。
        random_bit = torch.randint_like(w_scaled, 0, 2).float() # 0 or 1
        target_mag = 6.0 + random_bit # 6.0 or 7.0
        
        # 赋予符号
        w_int4_outlier = target_mag * w_scaled.sign()
        
        # 3.3 合并
        w_int4_combined = torch.where(is_outlier, w_int4_outlier, w_int4_det)
        
        # 最终 Clamp (虽然理论上都在范围内，但为了安全)
        w_int4_sim = w_int4_combined.clamp(-7, 7)
        w_base_recon = w_int4_sim * self.scales
        
        # 步骤 4: 提取 Ortho Stream
        # Residual = Original - StratifiedBase
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
    print(f"[LibOrtho-Professor] Applying Gap-Preserving Quantization (Ratio=1.7) to {target_modules}...")
    _replace_recursive(model, target_modules, ratio)
    print(f"[LibOrtho-Professor] Surgery complete.")
    return model