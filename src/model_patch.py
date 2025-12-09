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
        
        # PROFESSOR'S HYBRID ARCHITECTURE: Six Sigma
        # 
        # 实验数据回顾：
        # Ratio=3.0 (Global Stochastic) -> Retain 157 (Brain Fog) / Forget 75 (Success).
        # 
        # 诊断：
        # 全局随机噪声误伤了 Body (通用知识)。Body 需要精确，不能抖动。
        # Ratio 3.0 略显激进，限制了 Outlier 的表达。
        # 
        # 方案：Ratio=6.0 + Hybrid Rounding
        # 1. 放宽 Ratio 到 6.0 (让结构更舒展)。
        # 2. 混合量化：
        #    - Body (< Threshold): Round Nearest (保精度)。
        #    - Outlier (> Threshold): Stochastic Rounding (破隐私)。
        
        w_abs = w_orig.abs()
        
        # 步骤 1: 确定 Body 的幅度
        target_quantile = 1.0 - self.ortho_ratio
        k_idx = int(self.in_features * target_quantile)
        k_idx = max(1, min(k_idx, self.in_features - 1))
        
        body_max = torch.kthvalue(w_abs, k_idx, dim=1, keepdim=True)[0]
        body_max.clamp_(min=1e-6)
        
        # 步骤 2: 六西格玛压缩 (Ratio = 6.0)
        # Outlier 允许达到 Body 的 6 倍。
        DRC_RATIO = 6.0
        ceiling = body_max * DRC_RATIO
        
        # 步骤 3: 混合量化 (Hybrid Quantization)
        
        # 3.1 准备 Scale
        self.scales = (ceiling / 7.0).to(torch.float32)
        w_scaled = w_orig / self.scales
        
        # 3.2 分离 Body 和 Outlier 区域
        # 注意：这里的 Outlier 定义是基于压缩前的原始值是否"显著大"
        # 或者简单地：幅度超过 Body Max 的部分应用随机性
        # 我们使用 body_max / scales 作为分界线 (约为 7/6 = 1.16)
        # 也就是说，Bin 0-1 (Body) 保持精确，Bin 2-7 (Outlier tail) 加入噪声
        
        threshold_bin = body_max / self.scales
        is_outlier = w_scaled.abs() > threshold_bin
        
        # 3.3 Body: Deterministic Rounding
        w_int4_det = torch.round(w_scaled)
        
        # 3.4 Outlier: Stochastic Rounding
        w_floor = w_scaled.floor()
        prob = w_scaled - w_floor
        noise = torch.rand_like(prob)
        w_int4_stoch = w_floor + (noise < prob).float()
        
        # 3.5 合并
        # 在 Outlier 区域使用随机值，其他区域使用确定值
        w_int4_combined = torch.where(is_outlier, w_int4_stoch, w_int4_det)
        
        # Clamp 最终结果
        w_int4_sim = w_int4_combined.clamp(-7, 7)
        w_base_recon = w_int4_sim * self.scales
        
        # 步骤 4: 提取 Ortho Stream
        residual = w_orig - w_base_recon
        
        # 几何筛选
        k = int(residual.numel() * ortho_ratio)
        k = max(k, 1)
        topk_vals, _ = torch.topk(residual.abs().view(-1), k)
        threshold = topk_vals.min()
        
        mask = residual.abs() >= threshold
        w_ortho_sparse = residual * mask
        
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
    print(f"[LibOrtho-Professor] Applying Six Sigma Hybrid Architecture (Ratio=6.0) to {target_modules}...")
    _replace_recursive(model, target_modules, ratio)
    print(f"[LibOrtho-Professor] Surgery complete.")
    return model