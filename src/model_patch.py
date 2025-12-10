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
    def __init__(self, original_layer, ortho_ratio=0.01):
        """
        [LINUS DEEP FIX] 重构：使用 FP16 Base Stream，不量化
        核心原则：先验证逻辑正确性，再考虑优化（量化）
        """
        super().__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        
        print(f"[Ortho] Decomposing layer {self.in_features}x{self.out_features}...")

        # [Linus Safety Guard]
        if ortho_ratio > 0.3:
            print(f"[WARNING] Ortho ratio {ortho_ratio} is too high. Clamping to 0.3.")
            ortho_ratio = 0.3
            
        self.ortho_ratio = ortho_ratio
        
        # 1. 获取原始权重
        w_orig = original_layer.weight.data.float()
        device = w_orig.device
        
        # 2. [LINUS DEEP FIX] Base Stream: 直接用 FP16，不量化
        # 这消除了量化误差问题，确保 Alpha=0 时也能工作
        self.base_weights = w_orig.to(torch.float16)
        
        # 3. 计算残差（现在应该很小，因为 Base 是 FP16）
        # FP16 到 FP32 的转换误差通常 < 0.01%
        residual = w_orig - self.base_weights.float()
        
        # 4. [LINUS DEEP FIX] Ortho Stream: 只选择真正的异常值（1%）
        # 不是 10%！只有真正的离群值才需要进入 Ortho
        total_params = residual.numel()
        k = int(total_params * self.ortho_ratio)
        k = max(k, 1)  # 至少选 1 个
        k = min(k, total_params - 1)
        
        # 选择 top-k 最大残差（真正的异常值）
        topk_vals, topk_idx = torch.topk(residual.abs().view(-1), k)
        threshold = topk_vals[-1]
        
        # 创建稀疏掩码
        mask = torch.zeros_like(residual, dtype=torch.bool)
        mask.view(-1)[topk_idx] = True
        
        # 符号翻转保护：修复所有符号错误
        sign_mismatch = (w_orig.sign() != self.base_weights.float().sign()) & (w_orig.abs() > 1e-4)
        mask = mask | sign_mismatch
        
        w_ortho_sparse = residual * mask
        
        # 5. 验证：Base Stream 误差应该很小
        base_error = (self.base_weights.float() - w_orig).norm() / w_orig.norm()
        if base_error > 0.001:  # FP16 误差应该 < 0.1%
            print(f"[WARNING] FP16 Base error {base_error:.6f} is higher than expected. "
                  f"This may indicate numerical issues.")
        
        # 6. 最终验证：完整重构误差
        w_final_recon = self.base_weights.float() + w_ortho_sparse
        final_error = (w_final_recon - w_orig).norm() / w_orig.norm()
        
        # 7. Ortho: Convert to CSR
        w_ortho_csr = w_ortho_sparse.to_sparse_csr()
        self.ortho_vals = w_ortho_csr.values().to(torch.float16)
        self.ortho_indices = w_ortho_csr.col_indices().to(torch.int32)
        self.ortho_ptr = w_ortho_csr.crow_indices().to(torch.int32)
        
        self.nnz = self.ortho_vals.numel()
        self.alpha = 1.0
        
        print(f"[Ortho] Layer {self.in_features}x{self.out_features}: "
              f"Base error={base_error:.6f}, Final error={final_error:.6f}, "
              f"Ortho sparsity={self.nnz}/{total_params} ({self.nnz/total_params:.4f})") 
        
        # Move to device
        self.base_weights = self.base_weights.to(device)
        self.ortho_vals = self.ortho_vals.to(device)
        self.ortho_indices = self.ortho_indices.to(device)
        self.ortho_ptr = self.ortho_ptr.to(device)

    def forward(self, x):
        """
        [LINUS DEEP FIX] 简化 forward：Base Stream 使用 FP16 矩阵乘法
        """
        original_shape = x.shape
        original_dtype = x.dtype
        
        if not x.is_cuda:
            raise RuntimeError(f"Input must be on CUDA, got {x.device}")
        
        x_flat = x.view(-1, self.in_features).contiguous()
        
        # Base Stream: FP16 矩阵乘法
        # 使用 FP16 进行计算（更高效）
        x_flat_f16 = x_flat.to(torch.float16)
        out_base = torch.mm(x_flat_f16, self.base_weights.t())
        
        # Ortho Stream: 稀疏矩阵乘法（如果启用）
        if self.alpha > 1e-6:
            w_ortho = torch.sparse_csr_tensor(
                self.ortho_ptr, self.ortho_indices, self.ortho_vals, 
                size=(self.out_features, self.in_features)
            )
            # 稀疏矩阵乘法
            x_flat_f32 = x_flat.to(torch.float32)
            out_ortho = torch.sparse.mm(w_ortho, x_flat_f32.t()).t()
            out_flat = out_base.to(torch.float32) + self.alpha * out_ortho
        else:
            out_flat = out_base.to(torch.float32)
        
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