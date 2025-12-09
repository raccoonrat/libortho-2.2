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
        
        # PROFESSOR'S FINAL FIX: Ceiling Jitter
        # 
        # 诊断：
        # Forget PPL 2.8 失败的原因是 "Saturation Silence"。
        # 巨大的 Outlier 被 Clamp 到 7.0 后，其小数部分为 0。
        # Stochastic Rounding 对整数无效 (prob=0)。
        # 结果：Outlier 被确定性地编码为 7，没有任何噪声。隐私借此存活。
        # 
        # 方案：Grid-Space Noise Injection
        # 我们必须在量化 *前*，在网格空间 (Grid Space) 对 Outlier 注入强噪声。
        # 强迫它从天花板 (7.0) 掉下来，在 {5, 6, 7} 之间随机跳动。
        
        w_abs = w_orig.abs()
        
        # 步骤 1: 锁定名单
        k = int(w_orig.numel() * self.ortho_ratio)
        k = max(k, 1)
        topk_vals, _ = torch.topk(w_abs.view(-1), k)
        threshold = topk_vals.min()
        is_outlier = w_abs >= threshold
        
        # 步骤 2: 计算 Ratio 3.0 Scale (Retain 友好的)
        w_body = w_orig * (~is_outlier)
        body_max = w_body.abs().max(dim=1, keepdim=True)[0]
        body_max.clamp_(min=1e-6)
        
        DRC_RATIO = 3.0
        ceiling = body_max * DRC_RATIO
        
        self.scales = (ceiling / 7.0).to(torch.float32)
        w_scaled = w_orig / self.scales
        
        # 步骤 3: 混合量化 + 主动抖动
        
        # 3.1 Body: 确定性量化
        w_int4_det = torch.round(w_scaled)
        
        # 3.2 Outlier: 注入 Grid 噪声！
        # 先 Clamp 到 [-7, 7] 范围，这时候 Outlier 都在边界上
        w_outlier_grid = w_scaled.clamp(-7, 7)
        
        # 关键一步：注入噪声
        # sigma=1.0 意味着它有很大几率偏离 7.0 达到 6.0 或 5.0
        jitter = torch.randn_like(w_outlier_grid) * 1.5
        w_outlier_jittered = w_outlier_grid + jitter
        
        # 然后再 Round。
        # 注意：这里不需要再 Stochastic Round 了，因为 Jitter 本身就是随机源
        w_int4_jittered = torch.round(w_outlier_jittered)
        
        # 3.3 合并
        w_int4_combined = torch.where(is_outlier, w_int4_jittered, w_int4_det)
        
        # 最终 Clamp 确保合法 INT4
        w_int4_sim = w_int4_combined.clamp(-7, 7)
        w_base_recon = w_int4_sim * self.scales
        
        # 步骤 4: 提取 Ortho Stream
        # Residual = Original - JitteredBase
        # Ortho 会捕捉到所有的 Jitter 误差，所以在 Alpha=1 时能完美复原。
        residual = w_orig - w_base_recon
        
        # 锁定 Ortho
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
    print(f"[LibOrtho-Professor] Applying Ceiling Jitter (Ratio=3.0) to {target_modules}...")
    _replace_recursive(model, target_modules, ratio)
    print(f"[LibOrtho-Professor] Surgery complete.")
    return model