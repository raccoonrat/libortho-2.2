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
        
        # PROFESSOR'S FINAL CONVERGENCE: Adaptive Saturation Jitter
        # 
        # 核心逻辑：
        # 1. 自动化 Scale (生存): 利用 Kurtosis 自动寻找最佳 Ratio。
        #    - 高斯分布 (K=3) -> Ratio ~1.6 (高精度)
        #    - 拉普拉斯 (K=6) -> Ratio ~2.6 (平衡)
        #    - 我们将 Ratio 限制在 [2.0, 4.0] 的黄金区间，确保 Body 永远不死。
        # 
        # 2. 强制熵增 (毁灭): 解决 "Saturation Determinism"。
        #    - 之前的随机量化在 Outlier 被 Clamp 到 7.0 时失效 (prob=0)。
        #    - 现在，我们在 Clamp *之前* 对溢出部分注入强高斯噪声。
        #    - 让 Outlier 从天花板上 "掉下来"，在 {4, 5, 6, 7} 中随机游走。
        
        w_abs = w_orig.abs()
        
        # --- 步骤 1: 峰度驱动的 Ratio 计算 ---
        
        # 计算行级统计量
        w_mean = w_orig.mean(dim=1, keepdim=True)
        w_std = w_orig.std(dim=1, keepdim=True).clamp(min=1e-6)
        w_z = (w_orig - w_mean) / w_std
        kurtosis = torch.mean(w_z ** 4, dim=1, keepdim=True)
        
        # 映射 Kurtosis 到 Ratio
        # log2(3) ~ 1.58, log2(6) ~ 2.58, log2(100) ~ 6.6
        # 我们限制在 [2.0, 4.0]。
        # 2.0 保证 Body 映射到 3.5 (Retain < 20)
        # 4.0 保证 Body 映射到 1.75 (Retain ~30-40，极限但可用)
        adaptive_ratios = torch.log2(kurtosis).clamp(2.0, 4.0)
        
        # --- 步骤 2: 计算 Scale ---
        
        # 找到 Body Max
        target_quantile = 1.0 - self.ortho_ratio
        k_idx = int(self.in_features * target_quantile)
        k_idx = max(1, min(k_idx, self.in_features - 1))
        
        body_max = torch.kthvalue(w_abs, k_idx, dim=1, keepdim=True)[0]
        body_max.clamp_(min=1e-6)
        
        # Ceiling = BodyMax * AdaptiveRatio
        ceiling = body_max * adaptive_ratios
        self.scales = (ceiling / 7.0).to(torch.float32)
        
        # 归一化
        w_scaled = w_orig / self.scales
        
        # --- 步骤 3: 饱和区抖动 (Saturation Jitter) ---
        
        # 识别那些即将被 Saturation 的 Outliers
        # 只要绝对值超过 7.0，它们本来会变成确定的 7.0
        # 我们要破坏这种确定性
        saturation_mask = w_scaled.abs() > 7.0
        
        # 构造噪声
        # 我们在 Grid Space 注入 Sigma=2.0 的噪声
        # 这意味着 100.0 (Scale后) 可能会变成 98.0, 也可能变成 6.0
        # 关键是：在 Clamp 之前加噪声！
        jitter = torch.randn_like(w_scaled) * 2.0
        
        # 只干扰饱和区
        w_jittered = w_scaled.clone()
        w_jittered[saturation_mask] += jitter[saturation_mask]
        
        # --- 步骤 4: 量化与重建 ---
        
        # 现在 Clamp。那些被 Jitter 拉到 7 以下的值会保留其随机性。
        # 那些依然 > 7 的值会 Clamp 到 7。
        # 这创造了一个 {..., 5, 6, 7} 的分布，而不是单一的 {7}。
        w_int4_sim = torch.round(w_jittered).clamp(-7, 7)
        w_base_recon = w_int4_sim * self.scales
        
        # --- 步骤 5: 提取 Ortho Stream ---
        
        # Ortho 必须修补所有的 Jitter 和 Quantization Error
        residual = w_orig - w_base_recon
        
        # 锁定 Ortho 内容 (Top-K)
        k_budget = int(w_orig.numel() * self.ortho_ratio)
        k_budget = max(k_budget, 1)
        
        # 使用 Residual 的幅度来决定谁进 Ortho
        # 因为我们加了强 Jitter，Outlier 的 Residual 会非常大，肯定会被选中
        topk_vals, _ = torch.topk(residual.abs().view(-1), k_budget)
        thresh_res = topk_vals.min()
        ortho_mask = residual.abs() >= thresh_res
        
        w_ortho_sparse = residual * ortho_mask
        
        # --- 步骤 6: 打包 ---
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
    print(f"[LibOrtho-Professor] Applying Adaptive Saturation Jitter to {target_modules}...")
    _replace_recursive(model, target_modules, ratio)
    print(f"[LibOrtho-Professor] Surgery complete.")
    return model