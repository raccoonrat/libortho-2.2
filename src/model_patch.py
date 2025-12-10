import torch
import torch.nn as nn
import math
import libortho_ops
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class OrthoConfig:
    ratio: float = 3.0
    # 支持的噪声模式
    noise_mode: Literal['deterministic', 'stochastic', 'flicker_binary', 'tri_state_entropy', 'uniform_entropy'] = 'deterministic'
    ortho_ratio: float = 0.05

class OrthoLinear(nn.Module):
    def __init__(self, original_layer, config: Optional[OrthoConfig] = None):
        super().__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        
        if config is None:
            config = OrthoConfig()
        self.config = config
        
        w_orig = original_layer.weight.data.float()
        device = w_orig.device
        
        w_abs = w_orig.abs()
        
        # 1. 锁定名单 (Index Locking) - 用于 Ortho 存储
        # 我们依然需要宽泛的 5% 名单来保证 Alpha=1 的完美恢复
        k = int(w_orig.numel() * self.config.ortho_ratio)
        k = max(k, 1)
        topk_vals, _ = torch.topk(w_abs.view(-1), k)
        threshold = topk_vals.min()
        is_outlier = w_abs >= threshold
        
        # 2. 计算 Scale
        w_body = w_orig * (~is_outlier)
        body_max = w_body.abs().max(dim=1, keepdim=True)[0]
        body_max.clamp_(min=1e-6)
        
        ceiling = body_max * self.config.ratio
        self.scales = (ceiling / 7.0).to(torch.float32)
        w_scaled = w_orig / self.scales
        
        # 3. 策略执行 (Deep-Focal Logic)
        w_int4_det = torch.round(w_scaled)
        
        # 默认不攻击
        w_int4_final = w_int4_det
        
        # 定义攻击阈值 (Attack Threshold)
        # 我们只攻击那些"如果不攻击就会变得比噪声还小"或者"已经进入噪声区间"的值
        # 避免将小的 Middle Class 强行放大
        
        if self.config.noise_mode == 'deterministic':
            w_int4_final = w_int4_det
            
        elif self.config.noise_mode == 'stochastic':
            w_floor = w_scaled.floor()
            prob = w_scaled - w_floor
            noise = torch.rand_like(prob)
            w_int4_stoch = w_floor + (noise < prob).float()
            # Stochastic 比较温和，可以对所有 Outlier 应用
            w_int4_final = torch.where(is_outlier, w_int4_stoch, w_int4_det)
            
        elif self.config.noise_mode == 'flicker_binary':
            # {6, 7}. Min = 6. 
            # 保护 < 5.5 的值
            is_deep = w_scaled.abs() > 5.5
            should_attack = is_outlier & is_deep
            
            random_bit = torch.randint_like(w_scaled, 0, 2).float()
            target_mag = 6.0 + random_bit
            w_int4_flicker = target_mag * w_scaled.sign()
            
            w_int4_final = torch.where(should_attack, w_int4_flicker, w_int4_det)

        elif self.config.noise_mode == 'tri_state_entropy':
            # {5, 6, 7}. Min = 5.
            # 保护 < 4.5 的值 (Middle Class 4.0 保持原样)
            is_deep = w_scaled.abs() > 4.5
            should_attack = is_outlier & is_deep
            
            random_mag = torch.randint_like(w_scaled, 5, 8).float()
            w_int4_entropy = random_mag * w_scaled.sign()
            
            w_int4_final = torch.where(should_attack, w_int4_entropy, w_int4_det)
            
        elif self.config.noise_mode == 'uniform_entropy':
            # {4, 5, 6, 7}. Min = 4.
            # 保护 < 3.5 的值
            is_deep = w_scaled.abs() > 3.5
            should_attack = is_outlier & is_deep
            
            random_mag = torch.randint_like(w_scaled, 4, 8).float()
            w_int4_entropy = random_mag * w_scaled.sign()
            
            w_int4_final = torch.where(should_attack, w_int4_entropy, w_int4_det)

        # 4. 组装
        w_int4_sim = w_int4_final.clamp(-7, 7)
        w_base_recon = w_int4_sim * self.scales
        
        residual = w_orig - w_base_recon
        
        # Ortho 依然存储 Top 5% 的残差
        # 这包含了被攻击的 Deep Outliers 以及未被攻击但有误差的 Middle Class
        w_ortho_sparse = residual * is_outlier
        
        # Packing
        w_int4_offset = (w_int4_sim + 8).to(torch.uint8)
        w_int4_low = w_int4_offset[:, 0::2]
        w_int4_high = w_int4_offset[:, 1::2]
        self.base_packed = (w_int4_low | (w_int4_high << 4)).contiguous()
        
        w_ortho_csr = w_ortho_sparse.to_sparse_csr()
        self.ortho_vals = w_ortho_csr.values().to(torch.float16)
        self.ortho_indices = w_ortho_csr.col_indices().to(torch.int32)
        self.ortho_ptr = w_ortho_csr.crow_indices().to(torch.int32)
        
        self.nnz = self.ortho_vals.numel()
        self.alpha = 1.0
        
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

def replace_linear_layers(model, target_modules=["down_proj", "o_proj"], ratio=0.05):
    from auto_tuner import LibOrthoAutoTuner
    print(f"[LibOrtho] Initializing Physics-Constrained Auto-Tuner (Deep-Focal)...")
    
    tuner = LibOrthoAutoTuner(model, target_modules, ortho_ratio=ratio)
    tuner.run_optimization()
    
    return model