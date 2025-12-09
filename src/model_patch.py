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
        
        # LINUS NOTE: 
        # Don't spam stdout. One log per layer is enough to know it's alive.
        # print(f"[Ortho] Decomposing layer {self.in_features}x{self.out_features}...")
        
        # 1. 获取原始权重
        w_orig = original_layer.weight.data.float()
        device = w_orig.device
        
        # 2. 模拟 INT4 量化 (Base Stream)
        # LINUS FIX: 统计学缩放 (Statistical Scaling)
        # 放弃 fragile 的 quantile。使用均值和最大值的混合策略。
        # 权重的分布通常是 Laplacian 的。
        # Outliers 会极大地拉高 Max，导致 Scale 过大，Base 精度归零。
        # 单纯用 Quantile 又容易选错阈值，导致 Scale 过小，Range 溢出。
        # 方案：Scale 上限设为 Mean_Abs 的 6 倍。这涵盖了绝大多数正常分布。
        # 任何超过 6 倍 Mean 的值，都是必须被切除的 Outlier，交给 Ortho 流处理。
        
        w_abs = w_orig.abs()
        w_mean = w_abs.mean(dim=1, keepdim=True)
        w_max = w_abs.max(dim=1, keepdim=True)[0]
        
        # 这里的 6.0 是经验值 (Sigma * 4-5 左右)。
        # 如果 Max 真的很大 (Outlier)，我们强制把 Scale 压下来，
        # 让 Outlier 在 Base 中溢出 (Clamp 到 7)，产生的巨大 Residual 丢给 Ortho。
        # 如果 Max 很小 (平坦层)，我们就用 Max，保证不浪费 Range。
        robust_max = torch.min(w_max, w_mean * 6.0)
        
        # 防止除零
        robust_max.clamp_(min=1e-6)
        
        # Range [-7, 7] 对应 robust_max
        self.scales = (robust_max / 7.0).to(torch.float32)
        
        # 量化并截断
        w_int4_sim = torch.round(w_orig / self.scales).clamp(-7, 7)
        w_base_recon = w_int4_sim * self.scales
        
        # 3. 计算残差
        residual = w_orig - w_base_recon
        
        # 4. 提取正交流 (Ortho Stream)
        k = int(residual.numel() * ortho_ratio)
        k = max(k, 1)
        
        topk_vals, topk_indices = torch.topk(residual.abs().view(-1), k)
        threshold = topk_vals.min()
        
        mask = residual.abs() >= threshold
        w_ortho_sparse = residual * mask
        
        # 5. 打包数据
        w_int4_offset = (w_int4_sim + 8).to(torch.uint8)
        w_int4_low = w_int4_offset[:, 0::2]
        w_int4_high = w_int4_offset[:, 1::2]
        self.base_packed = (w_int4_low | (w_int4_high << 4)).contiguous()
        
        # Ortho: CSR
        # Suppress the beta warning if you want, but I prefer knowing what's beta.
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

def replace_linear_layers(model, target_modules=["down_proj", "o_proj"], ratio=0.05):
    # 打印一次总览即可
    print(f"[LibOrtho] Patching modules {target_modules} with ratio={ratio}")
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_linear_layers(module, target_modules, ratio)
        
        if isinstance(module, nn.Linear):
            should_replace = any(t in name for t in target_modules)
            if should_replace:
                # print(f"Patching layer: {name}") 
                new_layer = OrthoLinear(module, ortho_ratio=ratio)
                setattr(model, name, new_layer)
    return model