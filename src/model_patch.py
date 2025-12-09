import torch
import torch.nn as nn
import math
import libortho_ops
from dataclasses import dataclass
from typing import Optional, Literal

# 定义配置数据结构
@dataclass
class OrthoConfig:
    ratio: float = 3.0
    # 策略模式: 
    # 'deterministic': 纯确定性 (保结构)
    # 'stochastic': 简单随机 (弱隐私)
    # 'flicker_binary': {6, 7} 二元闪烁
    # 'uniform_entropy': {4, 5, 6, 7} 均匀分布 (强隐私)
    noise_mode: Literal['deterministic', 'stochastic', 'flicker_binary', 'uniform_entropy'] = 'deterministic'
    ortho_ratio: float = 0.05

class OrthoLinear(nn.Module):
    def __init__(self, original_layer, config: Optional[OrthoConfig] = None):
        super().__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        
        # 默认配置
        if config is None:
            config = OrthoConfig()
        self.config = config
        
        w_orig = original_layer.weight.data.float()
        device = w_orig.device
        
        # --- 核心逻辑：基于配置构建流形 ---
        
        w_abs = w_orig.abs()
        
        # 1. 锁定名单 (Index Locking)
        k = int(w_orig.numel() * self.config.ortho_ratio)
        k = max(k, 1)
        topk_vals, _ = torch.topk(w_abs.view(-1), k)
        threshold = topk_vals.min()
        is_outlier = w_abs >= threshold
        
        # 2. 计算 Scale (由 Ratio 决定)
        w_body = w_orig * (~is_outlier)
        body_max = w_body.abs().max(dim=1, keepdim=True)[0]
        body_max.clamp_(min=1e-6)
        
        ceiling = body_max * self.config.ratio
        self.scales = (ceiling / 7.0).to(torch.float32)
        w_scaled = w_orig / self.scales
        
        # 3. 执行量化策略 (由 Noise Mode 决定)
        w_int4_det = torch.round(w_scaled) # 基础确定性版本
        
        if self.config.noise_mode == 'deterministic':
            # 纯结构保留
            w_int4_final = w_int4_det
            
        elif self.config.noise_mode == 'stochastic':
            # 基础随机量化
            w_floor = w_scaled.floor()
            prob = w_scaled - w_floor
            noise = torch.rand_like(prob)
            w_int4_stoch = w_floor + (noise < prob).float()
            w_int4_final = torch.where(is_outlier, w_int4_stoch, w_int4_det)
            
        elif self.config.noise_mode == 'flicker_binary':
            # {6, 7} 闪烁
            # 仅在饱和区生效
            is_saturated = w_scaled.abs() > 6.0
            random_bit = torch.randint_like(w_scaled, 0, 2).float()
            target_mag = 6.0 + random_bit
            w_int4_flicker = target_mag * w_scaled.sign()
            
            should_flicker = is_outlier & is_saturated
            w_int4_final = torch.where(should_flicker, w_int4_flicker, w_int4_det)
            
        elif self.config.noise_mode == 'uniform_entropy':
            # {4, 5, 6, 7} 最大熵均匀分布
            # 这是我们发现的最强攻击
            random_mag = torch.randint_like(w_scaled, 4, 8).float()
            w_int4_entropy = random_mag * w_scaled.sign()
            
            w_int4_final = torch.where(is_outlier, w_int4_entropy, w_int4_det)
            
        else:
            raise ValueError(f"Unknown noise mode: {self.config.noise_mode}")

        # 4. 最终组装
        w_int4_sim = w_int4_final.clamp(-7, 7)
        w_base_recon = w_int4_sim * self.scales
        
        residual = w_orig - w_base_recon
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
        
        # Device transfer
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

# 占位符函数，稍后由 AutoTuner 调用
def replace_linear_layers(model, target_modules=["down_proj", "o_proj"], ratio=0.05):
    # 这里我们引入一个新的逻辑：如果检测到 AutoTuner，则使用 AutoTuner
    # 否则使用默认的安全配置
    from auto_tuner import LibOrthoAutoTuner
    print(f"[LibOrtho] Initializing Auto-Tuner System...")
    
    tuner = LibOrthoAutoTuner(model, target_modules, ortho_ratio=ratio)
    tuner.run_optimization()
    
    return model