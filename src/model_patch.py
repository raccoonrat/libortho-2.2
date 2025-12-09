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
        
        # PROFESSOR'S ADJUSTED THEORY: Structural Obfuscation
        # 
        # 诊断：
        # - 切除 Outliers (Zero/Clamp) -> 结构崩塌 (Retain PPL 爆炸)。
        # - 保留 Outliers (Max Scale) -> 隐私泄露 (Forget PPL 低)。
        # 
        # 修正：
        # 我们必须保留 Outliers 的"幅度" (Structure)，但破坏其"精确值" (Privacy)。
        # 
        # 方案：
        # 1. 使用 Max Scaling (保住 Retain PPL)。
        # 2. 对 Top-K Outliers 注入高斯噪声 (毁掉 Forget PPL)。
        # 3. Ortho Stream 存储 (Original - NoisyBase) 以便 Alpha=1 时恢复。
        
        w_abs = w_orig.abs()
        
        # 步骤 1: Max Scaling (为了生存)
        # 我们必须包容 Outliers，否则模型会死。
        w_max = w_abs.max(dim=1, keepdim=True)[0]
        w_max.clamp_(min=1e-6)
        self.scales = (w_max / 7.0).to(torch.float32)
        
        # 标准 INT4 量化
        w_int4_sim = torch.round(w_orig / self.scales).clamp(-7, 7)
        w_base_clean = w_int4_sim * self.scales
        
        # 步骤 2: 识别隐私敏感区域 (Top-K Outliers)
        # 我们假设隐私藏在这些最强的连接中
        k = int(w_orig.numel() * self.ortho_ratio)
        k = max(k, 1)
        
        topk_vals, _ = torch.topk(w_abs.view(-1), k)
        threshold = topk_vals.min()
        outlier_mask = w_abs >= threshold
        
        # 步骤 3: 结构混淆 (噪声注入)
        # 我们在 Base Stream 的 Outlier 位置注入噪声。
        # 噪声幅度：设为 Scale 的 1-2 倍。足以干扰精确值，但不改变数量级。
        # 这里的关键是：让 Base Stream "看不清" Outlier 的具体值。
        
        noise = torch.randn_like(w_orig) * self.scales * 1.5
        
        # 只污染 Outlier 区域
        # Base = Quantized + Noise (in outlier regions)
        w_base_corrupted = w_base_clean.clone()
        w_base_corrupted[outlier_mask] += noise[outlier_mask]
        
        # 重新量化回 INT4 格点 (Clamping)，确保 Base Stream 格式合法
        # 这步很重要，因为加噪可能导致溢出
        w_int4_corrupted = torch.round(w_base_corrupted / self.scales).clamp(-7, 7)
        
        # 最终的 Base Stream (带噪声)
        w_base_final = w_int4_corrupted * self.scales
        
        # 步骤 4: 提取 Ortho Stream (解药)
        # Ortho = Original - CorruptedBase
        # 当 Alpha=1 时，Original = CorruptedBase + Ortho
        # 当 Alpha=0 时，我们得到 CorruptedBase (结构还在，值乱了)
        residual = w_orig - w_base_final
        
        # 我们需要存储所有受影响区域的残差
        # 这里复用 outlier_mask，但也包括其他区域的量化误差
        # 为了节省空间，我们只存储 Top-K residual (这通常就是 outlier 区域)
        
        k_res = int(residual.numel() * ortho_ratio)
        topk_res_vals, _ = torch.topk(residual.abs().view(-1), k_res)
        threshold_res = topk_res_vals.min()
        mask_final = residual.abs() >= threshold_res
        
        w_ortho_sparse = residual * mask_final
        
        # 5. 打包
        w_int4_offset = (w_int4_corrupted + 8).to(torch.uint8)
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
    print(f"[LibOrtho-Professor] Applying Structural Obfuscation (Noise Injection) to {target_modules} (Ratio={ratio})...")
    _replace_recursive(model, target_modules, ratio)
    print(f"[LibOrtho-Professor] Surgery complete.")
    return model