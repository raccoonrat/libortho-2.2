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
        
        # PROFESSOR'S FINAL THESIS: Saturated Manifold Projection
        # 
        # 理论修正：
        # 1. Base Stream 必须保留拓扑结构 (Topology)。这意味着大权重必须仍然是大权重。
        #    -> 不能置零 (Zeroing is wrong)。
        # 2. Base Stream 必须丢失特异性信息 (Specificity)。这意味着大权重不能是精确值。
        #    -> 必须饱和 (Saturation is right)。
        # 3. Base Stream 必须保留通用知识 (Generality)。这意味着 Body 部分必须高精度。
        #    -> Scale 必须基于 Body 计算 (Quantile Scaling)。
        
        w_abs = w_orig.abs()
        
        # 步骤 1: 计算基于 Body 的 Scale
        # 我们希望 99.5% 的权重都能完美拟合。
        # 剩下的 0.5% (Outliers) 将被强制饱和。
        target_quantile = 1.0 - self.ortho_ratio
        
        # 使用 kthvalue 找到每一行的饱和阈值
        k_idx = int(self.in_features * target_quantile)
        k_idx = max(1, min(k_idx, self.in_features - 1))
        
        # 这是 Body 的边界。超过这个值的都会被截断。
        robust_max = torch.kthvalue(w_abs, k_idx, dim=1, keepdim=True)[0]
        robust_max.clamp_(min=1e-6)
        
        # 设置 Scale。
        # 注意：这里我们故意让 Outlier 溢出！
        self.scales = (robust_max / 7.0).to(torch.float32)
        
        # 步骤 2: 构建 Base Stream (Saturated Projection)
        # 这里发生了神奇的事情：
        # - 小权重 (Body) -> 精确量化
        # - 大权重 (Outlier, e.g. 100.0) -> 被 Scale 放大变成 1000 -> Clamp 到 7
        # 结果：Base Stream 知道这里有个"最大值"，但不知道具体多大。
        # 这保留了结构 (非零)，但消除了隐私 (数值)。
        w_int4_sim = torch.round(w_orig / self.scales).clamp(-7, 7)
        w_base_recon = w_int4_sim * self.scales
        
        # 步骤 3: 提取 Ortho Stream (High-Altitude Residuals)
        # Residual = 100.0 (Orig) - 0.7 (Base) = 99.3
        # 这个巨大的残差被 Ortho 捕获。
        residual = w_orig - w_base_recon
        
        # 几何筛选：只保留最大的残差
        k = int(residual.numel() * ortho_ratio)
        k = max(k, 1)
        
        topk_vals, topk_indices = torch.topk(residual.abs().view(-1), k)
        threshold = topk_vals.min()
        
        mask = residual.abs() >= threshold
        w_ortho_sparse = residual * mask
        
        # 步骤 4: 打包
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
    print(f"[LibOrtho-Professor] Applying Saturated Manifold Projection to {target_modules} (Ratio={ratio})...")
    _replace_recursive(model, target_modules, ratio)
    print(f"[LibOrtho-Professor] Surgery complete.")
    return model