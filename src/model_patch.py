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
        
        print(f"[Ortho] Decomposing layer {self.in_features}x{self.out_features}...")
        
        # 1. 获取原始权重 (FP16/FP32)
        w_orig = original_layer.weight.data.float()
        device = w_orig.device
        
        # 2. 模拟 INT4 量化 (Base Stream)
        # LINUS FIX: 之前使用 0.995 分位数是愚蠢的。
        # 如果 Scale 太大，中间的细微权重（通用知识）会被量化为 0。
        # 我们必须缩小 Scale，哪怕这会导致更多的 Outliers 被 Clamp。
        # 反正 Outliers 产生的巨大 Residual 会被 Ortho 流捕获。
        # 既然 Ortho 流就是为了处理异常值的，那就让 Base 流专注于“正常人”吧。
        
        w_abs = w_orig.abs()
        
        # 使用更激进的分位数 (0.85)，确保大部分"中间"权重能分配到非零的量化桶
        k_idx = int(self.in_features * 0.85)
        k_idx = min(k_idx, self.in_features - 1)
        robust_max = torch.kthvalue(w_abs, k_idx, dim=1, keepdim=True)[0]
        
        # 防止除零
        robust_max.clamp_(min=1e-6)
        
        # 将 Range [-7, 7] 映射到 85% 的数据分布上
        self.scales = (robust_max / 7.0).to(torch.float32)
        
        # 量化并截断。
        # 注意：这里会有大量的数据被 Clamp 到 -7 或 7。
        # 这会产生巨大的 Residual，这正是我们想要的——把由于量化溢出的信息推给 Ortho。
        w_int4_sim = torch.round(w_orig / self.scales).clamp(-7, 7)
        w_base_recon = w_int4_sim * self.scales
        
        # 3. 计算残差 (Residual)
        residual = w_orig - w_base_recon
        
        # 4. 提取正交流 (Ortho Stream) - 只有最大的 % 残差被保留
        k = int(residual.numel() * ortho_ratio)
        # 确保至少选出几个点，防止 k=0
        k = max(k, 1)
        
        # 展平找 TopK
        # 这里的 residual 包含了真正的 Outliers（因为 Base Clamp 了）
        topk_vals, topk_indices = torch.topk(residual.abs().view(-1), k)
        threshold = topk_vals.min()
        
        mask = residual.abs() >= threshold
        w_ortho_sparse = residual * mask
        
        # 5. 打包数据 (Packing)
        # Base: Pack int4 to uint8 (2 weights per byte)
        w_int4_offset = (w_int4_sim + 8).to(torch.uint8) # map -7..7 to 1..15
        w_int4_low = w_int4_offset[:, 0::2]
        w_int4_high = w_int4_offset[:, 1::2]
        self.base_packed = (w_int4_low | (w_int4_high << 4)).contiguous()
        
        # Ortho: Convert to CSR format components
        # 警告：Sparse CSR 在 PyTorch 中是 Beta，但只要能跑就行
        w_ortho_csr = w_ortho_sparse.to_sparse_csr()
        self.ortho_vals = w_ortho_csr.values().to(torch.float16)
        self.ortho_indices = w_ortho_csr.col_indices().to(torch.int32)
        self.ortho_ptr = w_ortho_csr.crow_indices().to(torch.int32)
        
        self.nnz = self.ortho_vals.numel()
        self.alpha = 1.0 # 默认开启
        
        # Move everything to correct device/type
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