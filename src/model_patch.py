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
        
        # PROFESSOR'S META-SOLUTION: Kurtosis-Guided Adaptive Geometry
        # 
        # 问题本质：全局参数 (Global Ratio) 无法适应异构的神经网络分布。
        # 
        # 解决方案：让每一行神经元根据自己的"性格" (统计分布) 决定自己的命运。
        # 我们使用 峰度 (Kurtosis) 作为几何指纹。
        # 
        # 算法：
        # 1. 计算每一行的 Kurtosis。
        # 2. 动态映射：Ratio_i = log2(Kurtosis_i)。
        #    - 平坦行 (语法) -> 低 Ratio -> 高分辨率 -> Retain PPL 极好。
        #    - 尖锐行 (记忆) -> 高 Ratio -> 高包容性 -> 骨架不塌，但引入强噪声破坏隐私。
        
        w_abs = w_orig.abs()
        
        # 步骤 1: 计算峰度 (Fourth Standardized Moment)
        # k = E[(x-u)^4] / sigma^4
        w_mean = w_orig.mean(dim=1, keepdim=True)
        w_std = w_orig.std(dim=1, keepdim=True)
        # 防止除零
        w_std.clamp_(min=1e-6)
        
        # 标准化
        w_z = (w_orig - w_mean) / w_std
        # 计算四阶矩
        kurtosis = torch.mean(w_z ** 4, dim=1, keepdim=True)
        
        # 步骤 2: 自适应计算 Ratio
        # 映射函数设计：
        # Gaussian (K=3) -> Ratio ~ 1.6 (高精度)
        # Laplacian (K=6) -> Ratio ~ 2.6 (平衡)
        # Heavy Tail (K=100) -> Ratio ~ 6.6 (保结构)
        
        # 使用对数映射平滑极值
        adaptive_ratios = torch.log2(kurtosis)
        # 限制在物理合理的范围内 [1.5, 8.0]
        # 1.5 是为了保证 Body 至少有 2-bit 分辨率
        # 8.0 是为了防止 Scale 过大导致 Body 全部清零
        adaptive_ratios = adaptive_ratios.clamp(1.5, 8.0)
        
        # 步骤 3: 基于自适应 Ratio 的 Scale 计算
        # 这里的关键是：每一行都有自己的 Ratio，因此每一行对 Body/Outlier 的定义不同
        
        # 确定 Body Max (依然基于 99.5% 分位数)
        target_quantile = 1.0 - self.ortho_ratio
        k_idx = int(self.in_features * target_quantile)
        k_idx = max(1, min(k_idx, self.in_features - 1))
        
        body_max = torch.kthvalue(w_abs, k_idx, dim=1, keepdim=True)[0]
        body_max.clamp_(min=1e-6)
        
        # Ceiling = BodyMax * AdaptiveRatio
        ceiling = body_max * adaptive_ratios
        
        self.scales = (ceiling / 7.0).to(torch.float32)
        w_scaled = w_orig / self.scales
        
        # 步骤 4: 混合量化策略 (Hybrid Quantization)
        # 即使有了自适应 Ratio，我们依然需要对 Outlier 进行隐私破坏
        
        # 4.1 Body: 确定性量化
        w_int4_det = torch.round(w_scaled)
        
        # 4.2 Outlier: 薛定谔噪声 (Bin-Jumping)
        # 识别 Outlier: 超过 Body 范围的
        threshold_bin = body_max / self.scales
        is_outlier = w_scaled.abs() > threshold_bin
        
        # 对 Outlier 进行强随机量化
        # floor + bernoulli
        w_floor = w_scaled.floor()
        prob = w_scaled - w_floor
        noise = torch.rand_like(prob)
        w_int4_stoch = w_floor + (noise < prob).float()
        
        # 4.3 合并
        w_int4_combined = torch.where(is_outlier, w_int4_stoch, w_int4_det)
        
        # Clamp
        w_int4_sim = w_int4_combined.clamp(-7, 7)
        w_base_recon = w_int4_sim * self.scales
        
        # 步骤 5: 提取 Ortho Stream
        # Residual
        residual = w_orig - w_base_recon
        
        # 锁定 Ortho 内容 (Top-K by Magnitude)
        # 依然需要显式的 Top-K 筛选来控制稀疏度预算
        k_budget = int(w_orig.numel() * self.ortho_ratio)
        k_budget = max(k_budget, 1)
        topk_vals, _ = torch.topk(residual.abs().view(-1), k_budget)
        thresh_res = topk_vals.min()
        ortho_mask = residual.abs() >= thresh_res
        
        w_ortho_sparse = residual * ortho_mask
        
        # 6. 打包
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
    print(f"[LibOrtho-Professor] Applying Kurtosis-Guided Adaptive Geometry to {target_modules}...")
    _replace_recursive(model, target_modules, ratio)
    print(f"[LibOrtho-Professor] Surgery complete.")
    return model