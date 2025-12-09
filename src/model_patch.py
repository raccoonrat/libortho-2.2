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
        
        # PROFESSOR'S FINAL ARCHITECTURE: Saturation Zone Flickering
        # 
        # 诊断：
        # 之前为了攻击 Outlier，误伤了位于 Bin 2-6 之间的"中产阶级"权重。
        # 这些权重是结构骨干，不能加噪声。
        # 
        # 方案：Tiered Handling (分级处理)
        # 1. Ratio = 2.0。Body 映射到 3.5。分辨率极佳。Retain 安全。
        # 2. Zone 1 (Linear Zone, |x| <= 6): 
        #    包含 Body 和 中产阶级。使用 Round Nearest。绝对保真。
        # 3. Zone 2 (Saturation Zone, |x| > 6):
        #    这是金丝雀藏身的地方 (Deep Outliers)。
        #    强制进行 "Flickering": 在 {6, 7} 之间随机跳变。
        #    打破 "Saturation Determinism" (稳定在7导致的隐私泄露)。
        
        w_abs = w_orig.abs()
        
        # 步骤 1: 锁定名单 (依然需要 Index Locking 来指导 Ortho)
        k = int(w_orig.numel() * self.ortho_ratio)
        k = max(k, 1)
        topk_vals, _ = torch.topk(w_abs.view(-1), k)
        threshold = topk_vals.min()
        is_outlier = w_abs >= threshold
        
        # 步骤 2: 计算 Ratio 2.0 Scale
        # Ratio 2.0 -> Body Max 映射到 3.5
        w_body = w_orig * (~is_outlier)
        body_max = w_body.abs().max(dim=1, keepdim=True)[0]
        body_max.clamp_(min=1e-6)
        
        DRC_RATIO = 2.0
        ceiling = body_max * DRC_RATIO
        
        self.scales = (ceiling / 7.0).to(torch.float32)
        w_scaled = w_orig / self.scales
        
        # 步骤 3: 分区量化 (Zoned Quantization)
        
        # Zone 1: Linear / Safe Zone (|x| <= 6.0)
        # 这里使用确定性 Round，保护 Body 和 中产阶级
        w_int4_safe = torch.round(w_scaled)
        
        # Zone 2: Saturation / Danger Zone (|x| > 6.0)
        # 识别进入饱和区的权重 (注意：不完全依赖 is_outlier，而是依赖数值)
        in_saturation = w_scaled.abs() > 6.0
        
        # 构造闪烁 (Flicker)
        # 目标：让值在 {6, 7} 之间随机 (magnitude)
        # 方法：取 magnitude 6.0，加上一个随机的 0 或 1
        random_bit = torch.randint_like(w_scaled, 0, 2).float() # 0 or 1
        target_mag = 6.0 + random_bit # 6 or 7
        w_int4_flicker = target_mag * w_scaled.sign()
        
        # 合并逻辑
        # 如果在饱和区 -> Flicker
        # 如果不在饱和区 -> Safe Round
        w_int4_combined = torch.where(in_saturation, w_int4_flicker, w_int4_safe)
        
        # 最终 Clamp (双重保险)
        w_int4_sim = w_int4_combined.clamp(-7, 7)
        w_base_recon = w_int4_sim * self.scales
        
        # 步骤 4: 提取 Ortho Stream
        # Residual = Original - Base
        # 这里的 Base 包含了 Flicker 噪声，所以 Ortho 会记录下 "反向 Flicker"
        # 当 Alpha=1 时，Original = Base + Ortho，噪声抵消，完美恢复。
        residual = w_orig - w_base_recon
        
        # 锁定 Ortho 内容
        # 只保存 Top-K 的残差 (即我们定义的 Outlier 集合)
        # 注意：in_saturation 是动态的，is_outlier 是固定的 top-k。
        # 通常 in_saturation 是 is_outlier 的子集 (超级大权重)。
        # 我们使用 is_outlier 来确保 Ortho 覆盖所有潜在的特异点。
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
    print(f"[LibOrtho-Professor] Applying Saturation Zone Flickering (Ratio=2.0) to {target_modules}...")
    _replace_recursive(model, target_modules, ratio)
    print(f"[LibOrtho-Professor] Surgery complete.")
    return model