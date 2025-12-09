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
        rows, cols = w_orig.shape
        
        # PROFESSOR'S ULTIMATE WEAPON: SVD-based Spectral Isolation
        # 
        # 问题：Element-wise quantization preserves too much information (High Rank).
        # 方案：Force the Base Stream to be Low-Rank via SVD.
        # 
        # 理论：
        # General Knowledge = Low Rank Structure (Top Singular Values)
        # Privacy/Memorization = High Rank Noise (Tail Singular Values)
        # 
        # 只要我们将 Base Stream 限制在低秩空间，它就物理上无法存储稀疏的隐私信息。
        
        print(f"[LibOrtho] Performing SVD on {rows}x{cols} matrix...")
        
        # SVD 计算成本较高，但对于一次性手术是可以接受的。
        # 使用 'S' 模式只计算奇异值，或者 'reduced' 模式。
        # U: (rows, k), S: (k), Vh: (k, cols)
        # 注意：对于大矩阵，这可能需要几秒钟。
        try:
            U, S, Vh = torch.linalg.svd(w_orig, full_matrices=False)
        except RuntimeError:
            # Fallback for OOM or weird CUDA errors: use CPU
            print("[LibOrtho] GPU SVD failed/OOM, falling back to CPU...")
            w_cpu = w_orig.cpu()
            U, S, Vh = torch.linalg.svd(w_cpu, full_matrices=False)
            U = U.to(device)
            S = S.to(device)
            Vh = Vh.to(device)

        # 2. 确定秩 (Rank)
        # 我们假设通用知识占据了能量的主要部分。
        # 策略：保留前 25% 的奇异值，或者根据能量谱累计。
        # 为了更狠地剥离隐私，我们设置一个较激进的 Rank 限制。
        # 比如：只保留 25% 的秩。
        rank_keep = max(1, int(min(rows, cols) * 0.25))
        
        # 3. 重构低秩矩阵 (Low-Rank Reconstruction)
        # W_low = U[:, :r] @ S[:r] @ Vh[:r, :]
        U_r = U[:, :rank_keep]
        S_r = torch.diag(S[:rank_keep])
        Vh_r = Vh[:rank_keep, :]
        
        w_low_rank = U_r @ S_r @ Vh_r
        
        # 4. 构建 Base Stream (对低秩矩阵进行量化)
        # 现在的输入是 w_low_rank，而不是 w_orig！
        # 这是一个关键区别。Base Stream 现在不仅是低精度的，还是低秩的。
        
        w_abs = w_low_rank.abs()
        w_max = w_abs.max(dim=1, keepdim=True)[0]
        w_max.clamp_(min=1e-6)
        self.scales = (w_max / 7.0).to(torch.float32)
        
        w_int4_sim = torch.round(w_low_rank / self.scales).clamp(-7, 7)
        w_base_recon = w_int4_sim * self.scales
        
        # 5. 提取 Ortho Stream
        # Residual = Original - Base
        # 这里包含了：
        # (1) 被 SVD 丢弃的 High Rank components (Privacy!)
        # (2) 低秩矩阵量化产生的误差
        residual = w_orig - w_base_recon
        
        # 6. 几何筛选
        # 这里的 residual 应该包含了所有的隐私信息。
        # 我们按照 budget 选取最大的部分。
        k = int(residual.numel() * ortho_ratio)
        k = max(k, 1)
        
        topk_vals, topk_indices = torch.topk(residual.abs().view(-1), k)
        threshold = topk_vals.min()
        
        mask = residual.abs() >= threshold
        w_ortho_sparse = residual * mask
        
        # 7. 打包
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
    print(f"[LibOrtho-Professor] Applying SVD Low-Rank Projection to {target_modules} (Ratio={ratio})...")
    _replace_recursive(model, target_modules, ratio)
    print(f"[LibOrtho-Professor] Spectral Surgery complete.")
    return model