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
        
        # 2. 模拟 INT4 量化 (Base Stream)
        # LINUS FIX: 预算匹配 (Budget Matching)
        # 停止猜测分布（Laplacian? Gaussian? Who cares?）。
        # 我们有固定的 Ortho 预算 (ortho_ratio)。
        # 我们必须设置 Scale，使得正好只有 ortho_ratio 比例的权重溢出 Base Range。
        # 如果我们猜一个阈值（比如 Mean*6）导致 10% 的权重溢出，
        # 而 Ortho 只能接住 0.5%，那剩下的 9.5% 就被永久破坏了 -> PPL 6000。
        # 
        # 解决方案：计算精确的 (1 - ratio) 分位数作为 Scale 基准。
        
        w_abs = w_orig.abs()
        
        # 计算目标分位数索引
        # 例如 ratio=0.005 (0.5%) -> target=0.995
        # 这意味着 99.5% 的权重将完美适配 Base Stream，只有 0.5% 溢出
        target_quantile = 1.0 - self.ortho_ratio
        k_idx = int(self.in_features * target_quantile)
        
        # 边界安全检查
        k_idx = max(1, min(k_idx, self.in_features - 1))
        
        # 使用 kthvalue 找到该分位数的具体数值
        # 注意：这是对每一行（输出通道）独立计算的
        robust_max = torch.kthvalue(w_abs, k_idx, dim=1, keepdim=True)[0]
        
        # 防止全零层导致的除零
        robust_max.clamp_(min=1e-6)
        
        # Range [-7, 7] 对应 robust_max
        # 这样，<= robust_max 的权重会被精确量化
        # > robust_max 的权重会被 Clamp，产生大残差，正好被 Ortho 捕获
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