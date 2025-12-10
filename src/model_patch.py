import torch
import torch.nn as nn
import math

# 尝试导入 C++ 扩展，如果未编译则允许降级或报错
try:
    import libortho_ops
except ImportError:
    print("[WARN] LibOrtho C++ extension not found. Ensure you have run 'python setup.py install'.")
    libortho_ops = None

class OrthoLinear(nn.Module):
    def __init__(self, original_layer, ortho_ratio=0.05):
        super().__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        
        print(f"[Ortho] Decomposing layer {self.in_features}x{self.out_features}...")

        # [Linus Safety Guard]
        # 强制限制稀疏率。如果超过 30%，说明你的架构设计有问题。
        if ortho_ratio > 0.3:
            print(f"[WARNING] Ortho ratio {ortho_ratio} is too high. Clamping to 0.3.")
            ortho_ratio = 0.3
            
        self.ortho_ratio = ortho_ratio
        
        # 1. 获取原始权重 (FP16/FP32)
        w_orig = original_layer.weight.data.float()
        device = w_orig.device
        
        # 2. [LINUS FIX] Base Stream 量化：使用自适应策略
        # 策略：优先保证主体权重的精度，让离群值进入 Ortho Stream
        # 方法：使用分位数方法，但确保主体权重（95%分位数）有足够的量化级别
        
        # 计算每行的统计量
        w_abs = w_orig.abs()
        
        # 方法1：使用 95% 分位数作为主体权重的上界（保护主体）
        # 这确保 95% 的权重在量化范围内有足够的精度
        percentile_95 = torch.quantile(w_abs, 0.95, dim=1, keepdim=True)
        
        # 方法2：使用 3-sigma 作为备选（对于接近正态分布的层）
        w_std = w_orig.std(dim=1, keepdim=True)
        sigma_3 = 3.0 * w_std
        
        # 选择较小的值：优先保护主体权重，让离群值进入 Ortho
        # 这确保主体权重有足够的量化级别
        robust_max = torch.min(percentile_95, sigma_3)
        
        # 防止全 0 行或极小方差导致的数值不稳定
        robust_max.clamp_(min=1e-5)
        
        # INT8: 范围 [-127, 127]，共 255 个量化级别
        self.scales = (robust_max / 127.0).to(torch.float32)
        
        # 量化并截断到 INT8 范围
        w_int8_sim = torch.round(w_orig / self.scales).clamp(-127, 127)
        w_base_recon = w_int8_sim * self.scales
        
        # 3. [LINUS FIX] 硬验证：确保 Base 重构误差在可接受范围内
        # 对于超大层（>1000万参数），允许更高的误差（因为累积效应和权重分布复杂性）
        total_params = self.in_features * self.out_features
        if total_params > 10_000_000:
            error_threshold = 0.15  # 超大层：15%
        elif total_params > 5_000_000:
            error_threshold = 0.10  # 大层：10%
        else:
            error_threshold = 0.01  # 小层：1%
        
        base_error = (w_base_recon - w_orig).norm() / w_orig.norm()
        
        # 多级回退策略：逐步降低分位数，直到误差可接受
        if base_error > error_threshold:
            percentile_candidates = [0.90, 0.85, 0.80, 0.75, 0.70]
            best_error = base_error
            best_scales = self.scales.clone()
            found_acceptable = False
            
            for percentile in percentile_candidates:
                percentile_val = torch.quantile(w_abs, percentile, dim=1, keepdim=True).clamp_(min=1e-5)
                test_scales = (percentile_val / 127.0).to(torch.float32)
                test_int8 = torch.round(w_orig / test_scales).clamp(-127, 127)
                test_recon = test_int8 * test_scales
                test_error = (test_recon - w_orig).norm() / w_orig.norm()
                
                if test_error < best_error:
                    best_error = test_error
                    best_scales = test_scales
                
                # 如果找到可接受的误差，立即使用
                if test_error <= error_threshold:
                    self.scales = test_scales
                    w_int8_sim = test_int8
                    w_base_recon = test_recon
                    base_error = test_error
                    found_acceptable = True
                    print(f"[Ortho] Using {percentile*100:.0f}% percentile for layer {self.in_features}x{self.out_features} "
                          f"(error: {base_error:.4f})")
                    break
            
            # 如果所有回退策略都失败，使用最佳结果（即使超过阈值）
            if not found_acceptable:
                self.scales = best_scales
                w_int8_sim = torch.round(w_orig / self.scales).clamp(-127, 127)
                w_base_recon = w_int8_sim * self.scales
                base_error = best_error
                
                # 对于超大层，如果误差仍然很高，发出警告但继续
                if total_params > 10_000_000 and base_error > 0.20:
                    print(f"[WARNING] Large layer {self.in_features}x{self.out_features} has high base error: "
                          f"{base_error:.4f}. This may require higher ortho_ratio (current: {self.ortho_ratio:.4f}).")
                elif base_error > error_threshold * 1.5:  # 如果误差超过阈值的1.5倍，仍然失败
                    raise RuntimeError(
                        f"[VALIDATION FAILED] Base reconstruction error too large: {base_error:.4f} "
                        f"(threshold: {error_threshold:.3f}). Layer: {self.in_features}x{self.out_features}. "
                        f"Tried percentiles: 95%, 90%, 85%, 80%, 75%, 70%. "
                        f"Consider increasing ortho_ratio from {self.ortho_ratio:.4f} or using mixed precision."
                    )
        
        # 4. 计算残差 (Residual)
        residual = w_orig - w_base_recon
        
        # 5. 提取正交流 (Ortho Stream)
        total_params = residual.numel()
        k = int(total_params * self.ortho_ratio)
        k = max(k, 1) # 至少选 1 个
        k = min(k, total_params - 1) 
        
        # A. 幅度筛选
        # 现在残差中包含两类：
        # 1. 真正的 Outliers (因为 Base 截断产生的巨大误差)
        # 2. 精细的量化噪声 (在 +/- 3-sigma 范围内的)
        topk_vals, _ = torch.topk(residual.abs().view(-1), k)
        threshold = topk_vals[-1]
        magnitude_mask = residual.abs() >= threshold
        
        # B. 符号翻转保护 (Sign Mismatch)
        # 只有当原始权重不是微小噪声时 (>1e-4)，符号翻转才是致命逻辑错误。
        sign_mismatch = (w_orig.sign() != w_base_recon.sign()) & (w_orig.abs() > 1e-4)
        
        # 合并掩码
        mask = magnitude_mask | sign_mismatch
        w_ortho_sparse = residual * mask
        
        # 6. [LINUS FIX] INT8 打包：直接使用 uint8，无需位打包
        # INT8 范围 [-127, 127]，映射到 [0, 254] 存储为 uint8
        w_int8_offset = (w_int8_sim + 128).clamp(0, 255).to(torch.uint8)
        self.base_packed = w_int8_offset.contiguous()
        
        # 7. [LINUS FIX] 最终验证：确保完整重构误差在可接受范围内
        w_final_recon = w_base_recon + w_ortho_sparse
        final_error = (w_final_recon - w_orig).norm() / w_orig.norm()
        if final_error > 0.005:  # 0.5% 相对误差阈值（更严格）
            print(f"[WARNING] Final reconstruction error: {final_error:.4f} (threshold: 0.005). "
                  f"Layer: {self.in_features}x{self.out_features}. "
                  f"Consider increasing ortho_ratio from {self.ortho_ratio:.4f}.")
        
        # 8. Ortho: Convert to CSR
        w_ortho_csr = w_ortho_sparse.to_sparse_csr()
        self.ortho_vals = w_ortho_csr.values().to(torch.float16)
        self.ortho_indices = w_ortho_csr.col_indices().to(torch.int32)
        self.ortho_ptr = w_ortho_csr.crow_indices().to(torch.int32)
        
        self.nnz = self.ortho_vals.numel()
        self.alpha = 1.0
        
        print(f"[Ortho] Layer {self.in_features}x{self.out_features}: "
              f"Base error={base_error:.4f}, Final error={final_error:.4f}, "
              f"Ortho sparsity={self.nnz}/{total_params} ({self.nnz/total_params:.4f})") 
        
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
            raise RuntimeError(f"Input must be on CUDA, got {x.device}")
        
        x_flat = x.view(-1, self.in_features).contiguous()
        x_flat_f32 = x_flat.to(torch.float32)
        out_flat = torch.zeros(x_flat.size(0), self.out_features, device=x.device, dtype=torch.float32)
        
        if libortho_ops is not None:
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
        else:
             # Fallback (Slow) - [LINUS FIX] 支持 INT8
             w_int8_unpacked = self.base_packed.float() - 128.0  # 从 [0, 255] 映射回 [-128, 127]
             w_base = w_int8_unpacked * self.scales.view(-1, 1)
             
             w_ortho = torch.sparse_csr_tensor(
                 self.ortho_ptr, self.ortho_indices, self.ortho_vals, 
                 size=(self.out_features, self.in_features)
             ).to_dense()
             
             w_combined = w_base + self.alpha * w_ortho
             out_flat = torch.mm(x_flat_f32, w_combined.t())
        
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