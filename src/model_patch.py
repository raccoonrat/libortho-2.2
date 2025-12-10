import torch
import torch.nn as nn
import math

# 尝试导入 C++ 扩展，如果未编译则允许降级或报错
try:
    import libortho_ops
except ImportError:
    print("[WARN] LibOrtho C++ extension not found. Ensure you have run 'python setup.py install'.")
    libortho_ops = None

class OrthoLinear(nn.Module):
    def __init__(self, original_layer, ortho_ratio=0.05):
        super().__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        
        print(f"[Ortho] Decomposing layer {self.in_features}x{self.out_features}...")

        # [Linus Safety Guard]
        # 强制限制稀疏率。如果超过 30%，说明你的架构设计有问题。
        if ortho_ratio > 0.3:
            print(f"[WARNING] Ortho ratio {ortho_ratio} is too high. Clamping to 0.3.")
            ortho_ratio = 0.3
            
        self.ortho_ratio = ortho_ratio
        
        # 1. 获取原始权重 (FP16/FP32)
        w_orig = original_layer.weight.data.float()
        device = w_orig.device
        
        # 2. Base Stream 量化逻辑重构 [CRITICAL FIX]
        # 错误做法：使用 99.5% 分位数。导致长尾分布下 Scale 过大，小权重下溢为 0。
        # 正确做法：使用 3-Sigma (标准差) 覆盖。
        # 我们优先保证 99% 的"主体权重"在 Base 中有高精度。
        # 溢出的 1% "超级离群值" 会产生巨大残差，正好被 Ortho Stream 捕获。
        
        # 计算每行的标准差 (Row-wise Std)
        # 3.0 * sigma 通常覆盖 99.7% 的正态分布数据。
        # 对于重尾分布，这意味着我们截断了更多尾部，但这正是我们想要的——把尾部扔给 Ortho。
        w_std = w_orig.std(dim=1, keepdim=True)
        robust_max = 3.0 * w_std
        
        # 防止全 0 行或极小方差导致的数值不稳定
        robust_max.clamp_(min=1e-5)
        
        self.scales = (robust_max / 7.0).to(torch.float32)
        
        # 量化并截断。
        # 注意：这里会有大量权重被 clamp 到 +/- 7。
        # 没关系！这些被截断的部分会变成 huge residuals，进入 Ortho Stream。
        w_int4_sim = torch.round(w_orig / self.scales).clamp(-7, 7)
        w_base_recon = w_int4_sim * self.scales
        
        # 3. 计算残差 (Residual)
        residual = w_orig - w_base_recon
        
        # 4. 提取正交流 (Ortho Stream)
        total_params = residual.numel()
        k = int(total_params * self.ortho_ratio)
        k = max(k, 1) # 至少选 1 个
        k = min(k, total_params - 1) 
        
        # A. 幅度筛选
        # 现在残差中包含两类：
        # 1. 真正的 Outliers (因为 Base 截断产生的巨大误差)
        # 2. 精细的量化噪声 (在 +/- 3-sigma 范围内的)
        topk_vals, _ = torch.topk(residual.abs().view(-1), k)
        threshold = topk_vals[-1]
        magnitude_mask = residual.abs() >= threshold
        
        # B. 符号翻转保护 (Sign Mismatch)
        # 只有当原始权重不是微小噪声时 (>1e-4)，符号翻转才是致命逻辑错误。
        sign_mismatch = (w_orig.sign() != w_base_recon.sign()) & (w_orig.abs() > 1e-4)
        
        # 合并掩码
        mask = magnitude_mask | sign_mismatch
        w_ortho_sparse = residual * mask
        
        # 5. 打包数据 (Packing)
        w_int4_offset = (w_int4_sim + 8).to(torch.uint8) # map -7..7 to 1..15
        w_int4_low = w_int4_offset[:, 0::2]
        w_int4_high = w_int4_offset[:, 1::2]
        self.base_packed = (w_int4_low | (w_int4_high << 4)).contiguous()
        
        # Ortho: Convert to CSR
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
            raise RuntimeError(f"Input must be on CUDA, got {x.device}")
        
        x_flat = x.view(-1, self.in_features).contiguous()
        x_flat_f32 = x_flat.to(torch.float32)
        out_flat = torch.zeros(x_flat.size(0), self.out_features, device=x.device, dtype=torch.float32)
        
        if libortho_ops is not None:
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
        else:
             # Fallback (Slow)
             w_int4_low = (self.base_packed & 0x0F).float()
             w_int4_high = (self.base_packed >> 4).float()
             w_unpacked = torch.zeros(self.out_features, self.in_features, device=x.device)
             w_unpacked[:, 0::2] = w_int4_low
             w_unpacked[:, 1::2] = w_int4_high
             w_base = (w_unpacked - 8) * self.scales.view(-1, 1)
             
             w_ortho = torch.sparse_csr_tensor(
                 self.ortho_ptr, self.ortho_indices, self.ortho_vals, 
                 size=(self.out_features, self.in_features)
             ).to_dense()
             
             w_combined = w_base + self.alpha * w_ortho
             out_flat = torch.mm(x_flat_f32, w_combined.t())
        
        out_reshaped = out_flat.view(original_shape[:-1] + (self.out_features,))
        return out_reshaped.to(original_dtype)

    def set_privacy(self, enable_ortho: bool):
        self.alpha = 1.0 if enable_ortho else 0.0

def replace_linear_layers(model, target_modules=["down_proj", "o_proj"], ratio=0.05):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_linear_layers(module, target_modules, ratio)
        if isinstance(module, nn.Linear):
            should_replace = any(t in name for t in target_modules)
            if should_replace:
                print(f"Patching layer: {name}")
                new_layer = OrthoLinear(module, ortho_ratio=ratio)
                setattr(model, name, new_layer)
    return model