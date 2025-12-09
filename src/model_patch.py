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
        
        # LINUS FIX: 主动分离 (Active Separation)
        # 之前的逻辑是 "Quantize -> Residual -> Ortho"，这导致 Max Scaling 把异常值留在了 Base。
        # 现在的逻辑是 "Identify Outliers -> Move to Ortho -> Quantize Body"。
        # 这样 Outliers (Privacy) 被强制隔离，且 Base Stream 的 Scale 更细腻。
        
        w_abs = w_orig.abs()
        
        # 2. 识别 Outliers (前 ortho_ratio 大的权重)
        k = int(self.in_features * self.ortho_ratio)
        k = max(k, 1) # 至少选一个，防止空指针
        
        # 找到每一行的阈值
        # dim=1 (In_Features)
        topk_vals, _ = torch.topk(w_abs, k, dim=1)
        # 取第 k 大的值作为阈值
        thresholds = topk_vals[:, -1].unsqueeze(1)
        
        # 生成掩码：谁是 Outlier？
        # 注意：这里我们严格把大于等于阈值的都归为 Ortho
        ortho_mask = w_abs >= thresholds
        
        # 3. 构建 Ortho Stream (FP16)
        # 只有 Outliers 进入这里
        w_ortho_sparse = w_orig * ortho_mask
        
        # 4. 构建 Base Stream (INT4)
        # 剩下的 Body 部分
        w_body = w_orig * (~ortho_mask)
        
        # 重新计算 Scale！
        # 这次 Scale 是基于 Body 的 Max，而不是全局 Max。
        # 因为 Body 移除了 Outliers，Robust Max 会显著变小，分辨率提高。
        body_abs = w_body.abs()
        robust_max = body_abs.max(dim=1, keepdim=True)[0]
        robust_max.clamp_(min=1e-6)
        
        self.scales = (robust_max / 7.0).to(torch.float32)
        
        # 量化 Body
        # 注意：原先 Outlier 的位置现在是 0，0 量化后还是 0。完美。
        w_int4_sim = torch.round(w_body / self.scales).clamp(-7, 7)
        
        # 5. 打包 Base Stream
        w_int4_offset = (w_int4_sim + 8).to(torch.uint8)
        w_int4_low = w_int4_offset[:, 0::2]
        w_int4_high = w_int4_offset[:, 1::2]
        self.base_packed = (w_int4_low | (w_int4_high << 4)).contiguous()
        
        # 6. 打包 Ortho Stream (CSR)
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

# 辅助函数：避免递归日志刷屏
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
    print(f"[LibOrtho] Starting surgery on {target_modules} with ratio={ratio}...")
    _replace_recursive(model, target_modules, ratio)
    print(f"[LibOrtho] Surgery complete.")
    return model