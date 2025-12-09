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
        
        # PROFESSOR'S FINAL ATTEMPT: Schrödinger's Quantization (Ratio 2.0)
        # 
        # 理论依据：
        # - Ratio 3.0 (Retain 13, Forget 1.4) -> 结构很好，但压缩不足，且 Outlier 稳定在 7。
        # - Ratio 1.0 (Retain 7000) -> 结构崩塌。
        # 
        # 方案：Ratio = 2.0
        # 1. Body Resolution: Scale = BodyMax * 2.0。Body 映射到 3.5。
        #    Body 将占据 [-4, 4] 的范围。分辨率极佳。Retain PPL 应该 < 20。
        # 2. Outlier Obfuscation:
        #    Outlier (原始值 > 2.0 * BodyMax) 理论上会被 Clamp 到 7。
        #    我们拦截这个 Clamp 过程。
        #    强制 Outlier 在 {6, 7} 之间随机跳变 (Bernoulli 0.5)。
        #    这打破了"饱和确定性"，最大化了隐私熵。
        
        w_abs = w_orig.abs()
        
        # 步骤 1: 锁定名单 (Index Locking)
        k = int(w_orig.numel() * self.ortho_ratio)
        k = max(k, 1)
        topk_vals, _ = torch.topk(w_abs.view(-1), k)
        threshold = topk_vals.min()
        is_outlier = w_abs >= threshold
        
        # 步骤 2: 计算 Ratio 2.0 Scale
        w_body = w_orig * (~is_outlier)
        body_max = w_body.abs().max(dim=1, keepdim=True)[0]
        body_max.clamp_(min=1e-6)
        
        # Ratio 2.0: Body 映射到 3.5 (使用 bins 0, 1, 2, 3, 4)
        DRC_RATIO = 2.0
        ceiling = body_max * DRC_RATIO
        
        self.scales = (ceiling / 7.0).to(torch.float32)
        w_scaled = w_orig / self.scales
        
        # 步骤 3: 薛定谔混合量化
        
        # 3.1 Body: 确定性量化 (保精度)
        w_int4_det = torch.round(w_scaled)
        
        # 3.2 Outlier: 薛定谔噪声 (破隐私)
        # 我们不依赖 w_scaled 的值 (因为它可能很大)。
        # 我们直接根据符号生成 {6, 7} 或 {-7, -6}
        
        # 生成 0 或 1 的随机位
        random_bit = torch.randint_like(w_scaled, 0, 2).float()
        # 目标幅度：6 + bit -> 6 or 7
        target_mag = 6.0 + random_bit
        
        # 赋予符号
        w_int4_schrodinger = target_mag * w_scaled.sign()
        
        # 3.3 合并
        w_int4_combined = torch.where(is_outlier, w_int4_schrodinger, w_int4_det)
        
        # 最终 Clamp (双重保险)
        w_int4_sim = w_int4_combined.clamp(-7, 7)
        w_base_recon = w_int4_sim * self.scales
        
        # 步骤 4: 提取 Ortho Stream
        # Residual = Original - SchrödingerBase
        # Ortho 完美补全了所有噪声，Alpha=1 时无损。
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
    print(f"[LibOrtho-Professor] Applying Schrödinger's Quantization (Ratio=2.0) to {target_modules}...")
    _replace_recursive(model, target_modules, ratio)
    print(f"[LibOrtho-Professor] Surgery complete.")
    return model