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
        self.ratio_candidates = [2.0, 2.5, 3.0, 3.25, 3.5, 4.0, 5.0, 6.0]
        
        # 噪声模式
        self.noise_candidates = [
            'uniform_entropy',   # Min=4, Entropy=2.0
            'tri_state_entropy', # Min=5, Entropy=1.58
            'flicker_binary',    # Min=6, Entropy=1.0
            'stochastic',
            'deterministic'
        ]
        
        # 物理常数
        # 1. Body Safety: Ratio <= 3.5 (保证 Body 至少映射到 2.0)
        self.max_body_ratio = 3.6 
        
        # 2. Skeleton Safety: OutlierMin * ScaleFactor >= 2.4 * BodyMax
        # ScaleFactor = Ratio / 7.0
        # MinInt * (Ratio / 7.0) >= 2.4
        self.min_structure_strength = 2.4

    def run_optimization(self):
        print(f"\n[LibOrtho-Auto] Starting Physics-Constrained Search...")
        print(f"Target Modules: {self.target_modules}")
        
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
        # 我们不再依赖不可靠的 MSE 阈值，而是直接使用物理定律筛选
        
        candidates = []
        
        entropy_map = {
            'uniform_entropy': 2.0,
            'tri_state_entropy': 1.58,
            'flicker_binary': 1.0,
            'stochastic': 0.5,
            'deterministic': 0.0
        }
        
        min_int_map = {
            'uniform_entropy': 4.0,
            'tri_state_entropy': 5.0,
            'flicker_binary': 6.0,
            'stochastic': 6.0, # Approximate
            'deterministic': 7.0
        }
        
        for ratio in self.ratio_candidates:
            for mode in self.noise_candidates:
                
                # --- 物理检查 (Physics Check) ---
                
                # 1. Body Check
                if ratio > self.max_body_ratio:
                    continue # Body 精度太低，Retain 必死
                
                # 2. Structure Check
                min_int = min_int_map.get(mode, 7.0)
                structure_strength = min_int * (ratio / 7.0)
                
                if structure_strength < self.min_structure_strength:
                    continue # 骨架太弱，Retain 必死
                
                # 如果通过了物理检查，这是一个"可行解"
                candidates.append({
                    'config': OrthoConfig(ratio=ratio, noise_mode=mode, ortho_ratio=self.ortho_ratio),
                    'entropy': entropy_map[mode],
                    'ratio': ratio
                })
        
        if not candidates:
            # 如果没有完美解，降低标准 (Fallback)
            # 优先保结构 (Deterministic + Ratio 3.0)
            print("    [WARN] No physics-compliant config. Fallback to Safe Mode.")
            return OrthoConfig(ratio=3.0, noise_mode='deterministic', ortho_ratio=self.ortho_ratio)
        
        # 3. 最优选择
        # 在可行解中，选择 熵(Entropy) 最高的
        # 如果熵相同，选择 Ratio 最小的 (Body 精度最高)
        candidates.sort(key=lambda x: (-x['entropy'], x['ratio']))
        
        best = candidates[0]
        # print(f"    Selected Physics Optimal: Entropy={best['entropy']}, Ratio={best['ratio']}")
        
        return best['config']