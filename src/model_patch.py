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
        # 这是一个简单的 MinMax 量化实现，用于生成 Base
        self.scales = (w_orig.abs().max(dim=1, keepdim=True)[0] / 7.0).to(torch.float32)
        w_int4_sim = torch.round(w_orig / self.scales).clamp(-7, 7)
        w_base_recon = w_int4_sim * self.scales
        
        # 3. 计算残差 (Residual)
        residual = w_orig - w_base_recon
        
        # 4. 提取正交流 (Ortho Stream) - 只有最大的 5% 残差被保留
        # 这就是你的“高曲率/异常值”假设的近似
        k = int(residual.numel() * ortho_ratio)
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
        x_flat = x.view(-1, self.in_features)
        
        # C++ 算子期望 float32 输入，所以需要转换
        x_flat_f32 = x_flat.to(torch.float32)
        out_flat = torch.zeros(x_flat.size(0), self.out_features, device=x.device, dtype=torch.float32)
        
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