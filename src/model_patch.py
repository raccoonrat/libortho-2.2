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
        
        # PROFESSOR'S SURGICAL STRIKE
        # 
        # 诊断：
        # SVD (Rank reduction) killed the brain.
        # Clipping (Small Scale) killed the brain.
        # Max Scaling (Large Scale) saved the brain but kept the tumor (Privacy).
        # 
        # 处方：
        # 1. 使用 Max Scaling 确保 Base Stream 的身体健康。
        # 2. 显式地将 Top-K Outliers 在 Base 中 "Lobotomize" (置零)。
        # 3. 将这些 Outliers 完整转移到 Ortho。
        
        w_abs = w_orig.abs()
        
        # 步骤 1: 确定 Outliers (Tumor)
        # 我们严格遵守 ortho_ratio 预算
        k = int(residual_size := w_orig.numel() * self.ortho_ratio)
        k = max(k, 1)
        
        # 找到全局或逐行的 Top-K
        # 为了保持分布一致性，我们使用逐行 (dim=1) 筛选
        # 每个输出神经元切除其最强的 0.5% 连接
        k_per_row = int(self.in_features * self.ortho_ratio)
        k_per_row = max(k_per_row, 1)
        
        # 找到阈值
        topk_vals, _ = torch.topk(w_abs, k_per_row, dim=1)
        thresholds = topk_vals[:, -1].unsqueeze(1)
        
        # Outlier Mask: 这些是隐私藏身的地方
        outlier_mask = w_abs >= thresholds
        
        # 步骤 2: 构建健康的 Base Scale (Max Scaling)
        # 注意：我们计算 Scale 时包含 Outliers，或者不包含？
        # 如果包含，Scale 会很大，Body 会被压缩。
        # 如果不包含，Scale 会很小，Body 分辨率高。
        # 
        # 决策：使用 "Body Max" (排除 Outliers 后的最大值)。
        # 这样 Body 的精度最高。而 Outliers 反正要被挖走，Scale 覆盖不到也没关系。
        
        # 将 Outliers 排除后的矩阵
        w_body_only = w_orig * (~outlier_mask)
        w_body_abs = w_body_only.abs()
        
        # Body Max Scaling
        w_max = w_body_abs.max(dim=1, keepdim=True)[0]
        w_max.clamp_(min=1e-6)
        self.scales = (w_max / 7.0).to(torch.float32)
        
        # 步骤 3: 构建 Base Stream (Lobotomized)
        # 量化
        w_int4_sim = torch.round(w_orig / self.scales).clamp(-7, 7)
        
        # 关键手术：将 Outlier 位置强制置零！
        # Base Stream 在这些位置"失忆"了。
        w_int4_sim[outlier_mask] = 0.0
        
        # 步骤 4: 构建 Ortho Stream (The Transplant)
        # Ortho 拥有 Outliers 的全部原始值
        # 加上 Body 部分的量化误差 (Optional, 但为了精度最好加上)
        
        w_base_recon = w_int4_sim * self.scales
        residual = w_orig - w_base_recon
        
        # Ortho Stream 存储所有 Residual (主要是 Outliers 的全值，因为 Base 是 0)
        # 我们复用之前的 mask，只存储 Outliers，忽略 Body 的微小量化误差以节省空间
        # 或者，如果预算允许，存储最大的 Residual。
        # 这里为了确保隐私被移走，我们强制存储 outlier_mask 覆盖的部分。
        
        w_ortho_sparse = w_orig * outlier_mask # 存储原始全精度值
        
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
        
        # Device transfer
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
    print(f"[LibOrtho-Professor] Applying Surgical Outlier Extraction to {target_modules} (Ratio={ratio})...")
    _replace_recursive(model, target_modules, ratio)
    print(f"[LibOrtho-Professor] Surgery complete.")
    return model