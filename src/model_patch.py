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
        
        print(f"[Ortho] Decomposing layer {self.in_features}x{self.out_features}...")
        
        # 1. 获取原始权重 (FP16/FP32)
        w_orig = original_layer.weight.data.float()
        device = w_orig.device
        
        # 2. 模拟 INT4 量化 (Base Stream)
        # LINUS FIX: 使用鲁棒缩放 (Robust Scaling)
        # 不要让异常值 (Outliers) 决定 Scale。如果用 max()，异常值会被完美量化进 Base。
        # 我们取 99.5% 分位数作为 Scale 的基准，强迫异常值溢出产生巨大残差。
        
        # 计算每行的绝对值
        w_abs = w_orig.abs()
        
        # 使用 kthvalue 近似 quantile(0.995)，比全排序快
        # 每一行有 in_features 个元素，取第 k 大的作为阈值
        k_idx = int(self.in_features * 0.995)
        k_idx = min(k_idx, self.in_features - 1)
        robust_max = torch.kthvalue(w_abs, k_idx, dim=1, keepdim=True)[0]
        
        # 防止全 0 行导致的除零
        robust_max.clamp_(min=1e-6)
        
        self.scales = (robust_max / 7.0).to(torch.float32)
        
        # 量化并截断。异常值在这里会被 clamp 到 +/- 7，产生巨大误差
        w_int4_sim = torch.round(w_orig / self.scales).clamp(-7, 7)
        w_base_recon = w_int4_sim * self.scales
        
        # 3. 计算残差 (Residual)
        # 现在异常值的残差会非常大： Original(1.5) - Base(0.01) = 1.49
        residual = w_orig - w_base_recon
        
        # 4. 提取正交流 (Ortho Stream) - 只有最大的 % 残差被保留
        k = int(residual.numel() * ortho_ratio)
        # 确保至少选出几个点，防止 k=0
        k = max(k, 1)
        
        topk_vals, topk_indices = torch.topk(residual.abs().view(-1), k)
        threshold = topk_vals.min()
        
        mask = residual.abs() >= threshold
        w_ortho_sparse = residual * mask
        
        # 5. 打包数据 (Packing)
        # Base: Pack int4 to uint8 (2 weights per byte)
        # Note: This implies cols must be even.
        w_int4_offset = (w_int4_sim + 8).to(torch.uint8) # map -7..7 to 1..15
        # Packing logic: Low 4 bits = even col, High 4 bits = odd col
        w_int4_low = w_int4_offset[:, 0::2]
        w_int4_high = w_int4_offset[:, 1::2]
        self.base_packed = (w_int4_low | (w_int4_high << 4)).contiguous()
        
        # Ortho: Convert to CSR format components
        w_ortho_csr = w_ortho_sparse.to_sparse_csr()
        self.ortho_vals = w_ortho_csr.values().to(torch.float16)
        self.ortho_indices = w_ortho_csr.col_indices().to(torch.int32)
        self.ortho_ptr = w_ortho_csr.crow_indices().to(torch.int32)
        
        self.nnz = self.ortho_vals.numel()
        self.alpha = 1.0 # 默认开启
        
        # EXP 2.2: Initialize noise parameters (kernel-level injection)
        self.noise_std_ortho = 0.0
        self.noise_std_base = 0.0
        
        # Move everything to correct device/type
        self.base_packed = self.base_packed.to(device)
        self.scales = self.scales.to(device)
        self.ortho_vals = self.ortho_vals.to(device)
        self.ortho_indices = self.ortho_indices.to(device)
        self.ortho_ptr = self.ortho_ptr.to(device)

    def forward(self, x):
        # 这里的输入 x 可能是 (Batch, Seq, Hidden)，我们需要展平或者循环
        # 简单的做法：展平成 2D 矩阵进行计算，再 reshape 回去
        original_shape = x.shape
        original_dtype = x.dtype
        
        # 检查设备：C++ 扩展需要 CUDA 张量
        if not x.is_cuda:
            raise RuntimeError(
                f"Input tensor must be on CUDA device, but got {x.device}. "
                f"Please move the input to CUDA: x = x.to('cuda')"
            )
        
        # Handle cases where input might not be contiguous or has weird strides
        x_flat = x.view(-1, self.in_features).contiguous()
        
        # C++ 算子期望 float32 输入，所以需要转换
        x_flat_f32 = x_flat.to(torch.float32)
        out_flat = torch.zeros(x_flat.size(0), self.out_features, device=x.device, dtype=torch.float32)
        
        # EXP 2.2: Use kernel-level noise injection if noise is enabled
        if self.noise_std_ortho > 1e-6 or self.noise_std_base > 1e-6:
            libortho_ops.forward_with_noise(
                x_flat_f32,
                self.base_packed,
                self.scales.view(-1), # Flatten scales
                self.ortho_vals,
                self.ortho_indices,
                self.ortho_ptr,
                out_flat,
                self.alpha,
                self.noise_std_ortho,
                self.noise_std_base,
                self.out_features,
                self.in_features,
                self.nnz
            )
        else:
            libortho_ops.forward(
                x_flat_f32,
                self.base_packed,
                self.scales.view(-1), # Flatten scales
                self.ortho_vals,
                self.ortho_indices,
                self.ortho_ptr,
                out_flat,
                self.alpha,
                self.out_features,
                self.in_features,
                self.nnz
            )
        
        # 将输出转换回原始数据类型
        out_reshaped = out_flat.view(original_shape[:-1] + (self.out_features,))
        return out_reshaped.to(original_dtype)

    def set_privacy(self, enable_ortho: bool):
        self.alpha = 1.0 if enable_ortho else 0.0
    
    def inject_noise(self, noise_std=1.0, target='ortho'):
        """
        注入噪声到权重中，用于测试敏感性。
        
        EXP 2.2: 使用 kernel 层面的噪声注入（更高效，符合"Good Taste"原则）
        
        Args:
            noise_std: 噪声的标准差
            target: 'ortho' 或 'base'，指定注入目标
        """
        # 存储噪声参数，在 forward 时使用（kernel 层面注入）
        if target == 'ortho':
            self.noise_std_ortho = noise_std
            self.noise_std_base = 0.0
        elif target == 'base':
            self.noise_std_ortho = 0.0
            self.noise_std_base = noise_std
        else:
            raise ValueError(f"Unknown target: {target}. Must be 'ortho' or 'base'")
    
    def _reset_noise(self):
        """重置噪声参数"""
        self.noise_std_ortho = 0.0
        self.noise_std_base = 0.0

def replace_linear_layers(model, target_modules=["down_proj", "o_proj"], ratio=0.05):
    """
    递归替换模型中的 Linear 层
    只替换 MLP 的输出层或者 Attention 的输出层通常就足够验证效果了，
    而且能节省显存。
    """
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_linear_layers(module, target_modules, ratio)
        
        if isinstance(module, nn.Linear):
            # 检查名字是否匹配 (例如只替换 MLP 的 down projection)
            # 这通常是知识存储最密集的地方
            should_replace = any(t in name for t in target_modules)
            if should_replace:
                print(f"Patching layer: {name}")
                new_layer = OrthoLinear(module, ortho_ratio=ratio)
                setattr(model, name, new_layer)
    return model