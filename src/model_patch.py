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
        
        # PROFESSOR'S FINAL CUT: Stochastic Binary Erasure
        # 
        # 理论突破：
        # - 之前的 $\{6, 7\}$ 闪烁失败，说明金丝雀记忆依赖于"高值"的存在性，而非精确值。
        # - 只要权重是"高"的 (High Signal)，记忆电路就是通的。
        # 
        # 方案：Signal Dropout / Erasure
        # 我们必须随机"切断"一部分 Outlier，让它们跌落凡尘 (变成 Body 水平)。
        # 
        # 设定：
        # - Ratio = 3.0 (保持 Body 精度)。
        # - Outlier 状态：
        #   - State A (50%): 保持 High Value (7)。维持骨架。
        #   - State B (50%): 坍缩至 Low Value (2)。切断回路。
        # 
        # 预期：
        # - Retain: 鲁棒的通用知识可以通过剩余的 50% 骨架和完整的 Body 存活。
        # - Forget: 脆弱的金丝雀回路被随机打断，无法激活。
        
        w_abs = w_orig.abs()
        
        # 步骤 1: 锁定名单
        k = int(w_orig.numel() * self.ortho_ratio)
        k = max(k, 1)
        topk_vals, _ = torch.topk(w_abs.view(-1), k)
        threshold = topk_vals.min()
        is_outlier = w_abs >= threshold
        
        # 步骤 2: 计算 Ratio 3.0 Scale
        w_body = w_orig * (~is_outlier)
        body_max = w_body.abs().max(dim=1, keepdim=True)[0]
        body_max.clamp_(min=1e-6)
        
        DRC_RATIO = 3.0
        ceiling = body_max * DRC_RATIO
        
        self.scales = (ceiling / 7.0).to(torch.float32)
        w_scaled = w_orig / self.scales
        
        # 步骤 3: 二元擦除量化 (Binary Erasure)
        
        # 3.1 Body: 确定性量化
        w_int4_det = torch.round(w_scaled)
        
        # 3.2 Outlier: 构造擦除态
        # High State: 7 (保持 Outlier 特征)
        # Low State: 2 (模拟 Body Max，即"隐身")
        # 为什么是 2? 因为 Ratio=3.0 下，Body Max 映射到 2.33。
        # 变成 2 意味着 Outlier 伪装成了普通的 Body 权重。
        
        # 生成掩码：50% 概率保持 High，50% 概率 Drop 到 Low
        mask_keep = torch.rand_like(w_scaled) > 0.5
        
        target_mag = torch.where(mask_keep, torch.tensor(7.0, device=device), torch.tensor(2.0, device=device))
        
        # 赋予符号
        w_int4_erasure = target_mag * w_scaled.sign()
        
        # 3.3 合并
        # 只有在饱和区 (|x| > 2.5) 的 Outlier 才应用擦除
        # 这样避免误伤本来就不大的 Outlier
        is_saturated = w_scaled.abs() > 2.5
        should_erase = is_outlier & is_saturated
        
        w_int4_combined = torch.where(should_erase, w_int4_erasure, w_int4_det)
        
        # 最终 Clamp
        w_int4_sim = w_int4_combined.clamp(-7, 7)
        w_base_recon = w_int4_sim * self.scales
        
        # 步骤 4: 提取 Ortho Stream
        # Residual = Original - ErasureBase
        # Ortho 记录了所有被擦除的信息 (7->2 的巨大落差)，Alpha=1 时完美补回。
        residual = w_orig - w_base_recon
        
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
    print(f"[LibOrtho-Professor] Applying Stochastic Binary Erasure (Ratio=3.0) to {target_modules}...")
    _replace_recursive(model, target_modules, ratio)
    print(f"[LibOrtho-Professor] Surgery complete.")
    return model