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
        
        # PROFESSOR'S PRECISION STRIKE: Deep Outlier Erasure (Ratio=3.0)
        # 
        # 诊断：
        # - 之前的 Erasure 失败是因为攻击了所有 > 2.5 的权重 (Middle Class + Outliers)。
        # - Middle Class (Bins 3-6) 是结构骨干，不能动。
        # - Canary 藏在 Deep Outliers (> 6.5) 里。
        # 
        # 方案：
        # 1. Ratio = 3.0。Body 映射到 2.33。
        # 2. 识别 Deep Outliers (|x| > 6.5)。
        # 3. 对 Deep Outliers 实施 {3, 7} 强力擦除。
        #    - 跌落幅度 7->3 (跨越 4 个 Bin)，信息破坏力极大。
        #    - 仅影响极少数权重，Retain 安全。
        # 4. 对 Middle Class (|x| <= 6.5) 实施确定性量化，保护骨架。
        
        w_abs = w_orig.abs()
        
        # 步骤 1: 锁定名单 (用于 Ortho 存储)
        k = int(w_orig.numel() * self.ortho_ratio)
        k = max(k, 1)
        topk_vals, _ = torch.topk(w_abs.view(-1), k)
        threshold = topk_vals.min()
        is_outlier = w_abs >= threshold
        
        # 步骤 2: 计算 Ratio 3.0 Scale
        w_body = w_orig * (~is_outlier)
        body_max = w_body.abs().max(dim=1, keepdim=True)[0]
        body_max.clamp_(min=1e-6)
        
        DRC_RATIO = 3.0
        ceiling = body_max * DRC_RATIO
        
        self.scales = (ceiling / 7.0).to(torch.float32)
        w_scaled = w_orig / self.scales
        
        # 步骤 3: 靶向擦除量化
        
        # 3.1 基础：确定性量化 (保护 Body 和 Middle Class)
        w_int4_det = torch.round(w_scaled)
        
        # 3.2 识别 Deep Outliers (Kill Zone)
        # 只有真正触顶的权重才会被攻击
        is_deep = w_scaled.abs() > 6.5
        
        # 3.3 构造擦除态 {3, 7}
        # 50% 概率保持 7 (High)
        # 50% 概率跌落 3 (Low - 也就是 Middle Class 的底线)
        mask_keep = torch.rand_like(w_scaled) > 0.5
        target_mag = torch.where(mask_keep, torch.tensor(7.0, device=device), torch.tensor(3.0, device=device))
        
        w_int4_erasure = target_mag * w_scaled.sign()
        
        # 3.4 合并
        # 只有 is_deep 才应用 Erasure
        # 这意味着 [2.33, 6.5] 之间的 Middle Class 保持原样 (3, 4, 5, 6)
        w_int4_combined = torch.where(is_deep, w_int4_erasure, w_int4_det)
        
        # 最终 Clamp
        w_int4_sim = w_int4_combined.clamp(-7, 7)
        w_base_recon = w_int4_sim * self.scales
        
        # 步骤 4: 提取 Ortho Stream
        # Residual = Original - ErasureBase
        # Ortho 记录了 Deep Outlier 的剧烈波动
        residual = w_orig - w_base_recon
        
        # 锁定 Ortho 内容
        # 依然使用宽泛的 is_outlier 名单 (Top 5%)
        # 这样既包含了被攻击的 Deep Outlier，也包含了可能有量化误差的 Middle Class
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
    print(f"[LibOrtho-Professor] Applying Deep Outlier Erasure (Ratio=3.0, Target>6.5) to {target_modules}...")
    _replace_recursive(model, target_modules, ratio)
    print(f"[LibOrtho-Professor] Surgery complete.")
    return model