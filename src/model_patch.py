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
        
        # 1. 获取原始权重 (FP32/FP16)
        w_orig = original_layer.weight.data.float()
        device = w_orig.device
        
        # PROFESSOR'S METHODOLOGY:
        # 几何投影原则 (Principle of Geometric Projection)
        # 
        # 目标：W_base 必须是 W_orig 在 INT4 格点上的最佳逼近。
        # 错误：试图为了 Ortho 而人为扭曲 Base (如缩小 Scale 制造截断，或挖空权重)。
        # 正确：使用统计学上最稳健的 Max Scaling 构建 Base。
        # 
        # 这样，Alpha=0 时，我们得到的是一个标准的、健康的 INT4 模型 (PPL ~10-20)。
        # 而 Alpha=1 时，我们将补回那些 loss function 最敏感的精度损失。
        
        w_abs = w_orig.abs()
        
        # 2. 构建 Base Stream (Standard INT4)
        # 使用每行(Per-Channel)的绝对最大值作为 Scale。
        # 这是量化的"金标准"，保证了 Base Stream 的数值稳定性。
        # 我们不进行任何人为的"挖孔"或"饱和"操作。
        w_max = w_abs.max(dim=1, keepdim=True)[0]
        w_max.clamp_(min=1e-6)
        
        # Scale 映射到 INT4 的最大范围 [-7, 7]
        self.scales = (w_max / 7.0).to(torch.float32)
        
        # 量化 + 反量化 (Project onto Lattice)
        w_int4_sim = torch.round(w_orig / self.scales).clamp(-7, 7)
        w_base_recon = w_int4_sim * self.scales
        
        # 此时，w_base_recon 是一个完整的 INT4 模型权重。
        # 它包含了所有的"结构性"信息 (Structural Information)。
        
        # 3. 提取 Ortho Stream (Residuals)
        # 计算投影误差：Delta = W_orig - W_base
        # 这个残差包含了所有因为 INT4 精度不足而丢失的"高频信息"。
        # 根据论文假设，隐私和精确逻辑就藏在这里。
        residual = w_orig - w_base_recon
        
        # 4. 几何筛选 (Hessian-Aware Selection Approximation)
        # 我们只保留那些"幅度最大"的残差。
        # 在没有 Hessian 的情况下，L2 范数 (Magnitude) 是重要性的一阶近似。
        # 这些通常对应于原始权重中的 Outliers (在 Base 中被削顶) 
        # 或者 0 附近的微小值 (在 Base 中被量化为 0)。
        
        k = int(residual.numel() * ortho_ratio)
        k = max(k, 1)
        
        # 选取 Top-K 残差
        topk_vals, topk_indices = torch.topk(residual.abs().view(-1), k)
        threshold = topk_vals.min()
        
        mask = residual.abs() >= threshold
        w_ortho_sparse = residual * mask
        
        # 5. 打包 Base Stream
        # 注意：Base Stream 是完整的，没有被挖空。
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
    print(f"[LibOrtho-Professor] Applying Manifold Projection to {target_modules} (Ratio={ratio})...")
    _replace_recursive(model, target_modules, ratio)
    print(f"[LibOrtho-Professor] Projection complete.")
    return model