import torch
import torch.nn as nn
from model_patch import OrthoLinear, OrthoConfig
import copy
import math

class LibOrthoAutoTuner:
    def __init__(self, model, target_modules, ortho_ratio=0.05):
        self.model = model
        self.target_modules = target_modules
        self.ortho_ratio = ortho_ratio
        
        # 搜索空间调整：移除极低 Ratio，增加高 Ratio
        # 经验表明 down_proj 需要强骨架
        self.ratio_candidates = [3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0]
        
        self.noise_candidates = [
            'uniform_entropy',
            'flicker_binary', 
            'stochastic',
            'deterministic'
        ]
        
        self.tolerance_factor = 2.0 # 收紧容忍度

    def run_optimization(self):
        print(f"\n[LibOrtho-Auto] Starting Structure-Aware Search...")
        print(f"Target Modules: {self.target_modules}")
        print(f"Objective: Balance Entropy with Structural Integrity")
        
        self._recursive_tune(self.model)
        
        print(f"[LibOrtho-Auto] Optimization Complete.\n")

    def _recursive_tune(self, module):
        for name, child in module.named_children():
            if len(list(child.children())) > 0:
                self._recursive_tune(child)
            
            if isinstance(child, nn.Linear):
                should_replace = any(t in name for t in self.target_modules)
                if should_replace:
                    print(f"[Auto-Tuner] Tuning layer: {name} ({child.in_features}x{child.out_features})...")
                    best_config = self.find_best_config(child)
                    new_layer = OrthoLinear(child, config=best_config)
                    setattr(module, name, new_layer)
                    print(f"  -> LOCKED: Ratio={best_config.ratio}, Mode={best_config.noise_mode}")

    def find_best_config(self, layer: nn.Linear) -> OrthoConfig:
        w = layer.weight.data
        device = w.device
        
        # 1. 结构化校准数据 (Structured Calibration)
        # 我们不仅使用标准高斯噪声，还混合了一些"尖峰"输入，以模拟强激活
        # 这迫使模型关注大权重的响应
        batch_size = 256
        x_normal = torch.randn(batch_size, layer.in_features, device=device)
        
        # 模拟强激活：扩大一部分输入的方差
        x_spiky = x_normal * 3.0 
        x_calib = torch.cat([x_normal, x_spiky], dim=0) # [512, in_feat]
        
        with torch.no_grad():
            y_orig = layer(x_calib)
            # 计算输出范数，用于归一化误差
            y_norm = torch.norm(y_orig, p=2, dim=1).mean() + 1e-9
        
        candidates = []
        
        entropy_map = {
            'uniform_entropy': 4,
            'flicker_binary': 2,
            'stochastic': 1,
            'deterministic': 0
        }
        
        # 2. 搜索
        for ratio in self.ratio_candidates:
            for mode in self.noise_candidates:
                cfg = OrthoConfig(
                    ratio=ratio, 
                    noise_mode=mode, 
                    ortho_ratio=self.ortho_ratio
                )
                
                try:
                    test_layer = OrthoLinear(layer, config=cfg)
                    test_layer.set_privacy(enable_ortho=False)
                    
                    with torch.no_grad():
                        y_quant = test_layer(x_calib)
                    
                    # PROFESSOR'S METRIC: Structure-Weighted Error
                    # 我们不看 Mean Squared Error (MSE)，我们看 Relative L2 Error
                    # 并且我们重点关注那些"大输出"样本的误差
                    
                    diff = y_orig - y_quant
                    diff_norm = torch.norm(diff, p=2, dim=1)
                    
                    # 相对误差分布
                    rel_errors = diff_norm / (torch.norm(y_orig, p=2, dim=1) + 1e-6)
                    
                    # 关键指标：P95 误差 (Worst-case Error) 而不是平均误差
                    # 这能捕捉到 Outlier 被截断导致的个别样本崩塌
                    error_metric = torch.quantile(rel_errors, 0.95).item()
                    
                    candidates.append({
                        'config': cfg,
                        'error': error_metric,
                        'entropy': entropy_map[mode]
                    })
                    
                except Exception:
                    continue
        
        if not candidates:
            return OrthoConfig()

        # 3. 寻找基线 (Structural Baseline)
        # 找到所有配置中 P95 误差最小的 (通常是 High Ratio + Deterministic)
        baseline_error = min(c['error'] for c in candidates)
        error_limit = baseline_error * self.tolerance_factor
        
        # print(f"    Baseline P95 Error: {baseline_error:.4f} | Limit: {error_limit:.4f}")
        
        # 4. 筛选
        valid_candidates = [c for c in candidates if c['error'] <= error_limit]
        
        if not valid_candidates:
            # Fallback: 找误差最小的那个 (保命要紧)
            print("    [WARN] No entropy config within tolerance. Prioritizing Structure.")
            best_candidate = min(candidates, key=lambda x: x['error'])
        else:
            # 在合法的结构误差范围内，最大化熵
            # 如果熵相同，选择误差更小的
            valid_candidates.sort(key=lambda x: (-x['entropy'], x['error']))
            best_candidate = valid_candidates[0]
            
        return best_candidate['config']