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
        
        # 搜索空间
        # 我们扩大 Ratio 范围，以适应各种分布
        self.ratio_candidates = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]
        
        # 噪声模式 (熵值从高到低)
        self.noise_candidates = [
            'uniform_entropy',  # Entropy ~2.0
            'flicker_binary',   # Entropy ~1.0
            'stochastic',       # Entropy ~0.5
            'deterministic'     # Entropy 0.0
        ]
        
        # PROFESSOR'S UPGRADE: Relative Tolerance via Pareto Search
        # 不再使用固定的 MSE 阈值 (e.g. 0.02)。
        # 我们寻找帕累托最优解。
        # 容忍系数：允许 MSE 是该层最佳可能 MSE 的多少倍？
        # 3.0x 意味着我们愿意牺牲 3 倍的精度来换取隐私。
        # 这对于 Deep Outlier 擦除来说是合理的代价。
        self.tolerance_factor = 3.0

    def run_optimization(self):
        print(f"\n[LibOrtho-Auto] Starting Pareto Frontier Search...")
        print(f"Target Modules: {self.target_modules}")
        print(f"Strategy: Maximize Entropy within {self.tolerance_factor}x of Baseline MSE")
        
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
                    
                    # 原地替换
                    new_layer = OrthoLinear(child, config=best_config)
                    setattr(module, name, new_layer)
                    
                    print(f"  -> LOCKED: Ratio={best_config.ratio}, Mode={best_config.noise_mode}")

    def find_best_config(self, layer: nn.Linear) -> OrthoConfig:
        w = layer.weight.data
        in_feat = layer.in_features
        device = w.device
        
        # 1. 校准数据 (Zero-Shot)
        # 增加 batch size 以获得更稳定的统计
        x_calib = torch.randn(256, in_feat, device=device)
        
        with torch.no_grad():
            y_orig = layer(x_calib)
            y_norm = torch.mean(y_orig ** 2) + 1e-9
        
        # 2. 遍历搜索空间，收集所有数据点
        # Format: (config, relative_mse, entropy_score)
        candidates = []
        
        # 熵分映射
        entropy_map = {
            'uniform_entropy': 4,
            'flicker_binary': 3, # 提升优先级
            'stochastic': 1,
            'deterministic': 0
        }
        
        for ratio in self.ratio_candidates:
            for mode in self.noise_candidates:
                cfg = OrthoConfig(
                    ratio=ratio, 
                    noise_mode=mode, 
                    ortho_ratio=self.ortho_ratio
                )
                
                try:
                    # 测试 Alpha=0 (Base Stream) 的表现
                    test_layer = OrthoLinear(layer, config=cfg)
                    test_layer.set_privacy(enable_ortho=False)
                    
                    with torch.no_grad():
                        y_quant = test_layer(x_calib)
                    
                    diff = y_orig - y_quant
                    mse = torch.mean(diff ** 2)
                    rel_mse = (mse / y_norm).item()
                    
                    candidates.append({
                        'config': cfg,
                        'mse': rel_mse,
                        'entropy': entropy_map[mode]
                    })
                    
                except Exception:
                    continue
        
        if not candidates:
            print("    [ERROR] No valid configs found. Returning default.")
            return OrthoConfig()

        # 3. 寻找基线 (Baseline MSE)
        # 基线定义为：所有配置中 MSE 最小的那个 (通常是 deterministic + 某个最佳 ratio)
        baseline_mse = min(c['mse'] for c in candidates)
        mse_limit = baseline_mse * self.tolerance_factor
        
        # print(f"    Baseline MSE: {baseline_mse:.6f} | Limit: {mse_limit:.6f}")
        
        # 4. 帕累托筛选 (Pareto Selection)
        # 规则：在 MSE < Limit 的候选中，选择 Entropy 最大的。
        # 如果 Entropy 相同，选择 MSE 最小的。
        
        valid_candidates = [c for c in candidates if c['mse'] <= mse_limit]
        
        if not valid_candidates:
            # 极端情况：所有带噪声的 MSE 都爆炸了，超过了 tolerance
            # 这种情况下，我们需要"矮子里拔将军"，选 entropy > 0 中 MSE 最小的
            # 绝不轻易回退到 deterministic (entropy=0)
            noisy_candidates = [c for c in candidates if c['entropy'] > 0]
            if noisy_candidates:
                print("    [WARN] Tolerance exceeded. Picking best noisy config.")
                best_candidate = min(noisy_candidates, key=lambda x: x['mse'])
            else:
                print("    [FAIL] No noisy config viable. Fallback to best deterministic.")
                best_candidate = min(candidates, key=lambda x: x['mse'])
        else:
            # 正常情况：按 (Entropy DESC, MSE ASC) 排序
            # Python 的 sort 是稳定的，或者用 tuple key
            # 我们希望 Entropy 最大，然后 MSE 最小 -> key = (-entropy, mse)
            valid_candidates.sort(key=lambda x: (-x['entropy'], x['mse']))
            best_candidate = valid_candidates[0]
            
        # Debug info
        # cfg = best_candidate['config']
        # print(f"    Selected: R={cfg.ratio}, M={cfg.noise_mode}, MSE={best_candidate['mse']:.6f}")
            
        return best_candidate['config']