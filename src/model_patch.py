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
        
        # PROFESSOR'S FINAL CORRECTION: Index-Locked Separation
        # 
        # 诊断：
        # 上一次实验中 Alpha=1 和 Alpha=0 效果一样 (PPL ~42)。
        # 这说明 Ortho Stream 没有包含关键的金丝雀信息。
        # 原因：Body 的量化误差 (因 Scale 变大而变大) 挤占了 Top-K Residual 的名额。
        # 金丝雀 (Outlier) 被随机化破坏后，没能进入 Ortho，导致无法恢复。
        # 
        # 方案：Index-Locking
        # 我们不再让 Residual 竞争上岗。
        # 我们基于原始幅度 (Magnitude) 圈定 Outlier Mask。
        # 1. Base Stream: 对 Mask 内元素应用随机量化，对 Mask 外应用确定性量化。
        # 2. Ortho Stream: *强制* 只存储 Mask 内元素的残差。
        #    忽略 Mask 外的所有误差 (Body 误差)。
        
        w_abs = w_orig.abs()
        
        # 步骤 1: 锁定名单 (Index Locking)
        # 严格按照 budget 圈定 Top-K
        k = int(w_orig.numel() * self.ortho_ratio)
        k = max(k, 1)
        
        # 全局 Top-K (或者逐行，这里用全局更简单直接，保证 budget 利用率)
        topk_vals, _ = torch.topk(w_abs.view(-1), k)
        threshold = topk_vals.min()
        
        # 这就是我们的"特权名单"
        is_outlier = w_abs >= threshold
        
        # 步骤 2: 计算六西格玛 Scale
        # 为了给 Body 留足空间，我们依然使用 Body Max * 6.0
        # Body 是非 Outlier 的部分
        w_body = w_orig * (~is_outlier)
        body_max = w_body.abs().max(dim=1, keepdim=True)[0]
        body_max.clamp_(min=1e-6)
        
        DRC_RATIO = 6.0
        ceiling = body_max * DRC_RATIO
        
        self.scales = (ceiling / 7.0).to(torch.float32)
        w_scaled = w_orig / self.scales
        
        # 步骤 3: 混合量化 (基于名单)
        
        # 3.1 Body: Deterministic
        w_int4_det = torch.round(w_scaled)
        
        # 3.2 Outlier: Stochastic
        w_floor = w_scaled.floor()
        prob = w_scaled - w_floor
        noise = torch.rand_like(prob)
        w_int4_stoch = w_floor + (noise < prob).float()
        
        # 3.3 合并 (Index Locked!)
        # 只有名单上的人才会被随机化
        w_int4_combined = torch.where(is_outlier, w_int4_stoch, w_int4_det)
        
        # Clamp
        w_int4_sim = w_int4_combined.clamp(-7, 7)
        w_base_recon = w_int4_sim * self.scales
        
        # 步骤 4: 提取 Ortho Stream (定向救援)
        residual = w_orig - w_base_recon
        
        # 关键修改：不再重新计算 Top-K Residual！
        # 直接使用 is_outlier Mask。
        # 我们只保存 Outlier 的残差。Body 的残差丢弃（视为噪音）。
        
        # 由于我们最初就是按 ratio 计算的 k，所以 mask 的大小正好符合 Ortho 的容量。
        # 不会浪费，也不会溢出。
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
    print(f"[LibOrtho-Professor] Applying Index-Locked Hybrid Architecture (Ratio=6.0) to {target_modules}...")
    _replace_recursive(model, target_modules, ratio)
    print(f"[LibOrtho-Professor] Surgery complete.")
    return model