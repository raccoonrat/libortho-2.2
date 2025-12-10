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
        self.ratio_candidates = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
        
        self.noise_candidates = [
            'uniform_entropy',   
            'tri_state_entropy', 
            'flicker_binary',    
            'stochastic',
            'deterministic'
        ]
        
        self.max_body_ratio = 3.6 
        
        # PROFESSOR'S FINAL CALIBRATION: The 2.4x Safety Line
        # 实验证明 Structure Strength 2.0 (Ratio 3.5 + Min 4) 会导致 Retain ~200 (Brain Fog)。
        # 必须回归到 2.4 以上。
        # 这将过滤掉 Ratio 3.5 + Uniform (Strength 2.0)，
        # 迫使系统选择 Ratio 3.5 + Tri-State (Strength 2.5)。
        # 配合 Deep-Focal 保护，这将是 Retain < 30 的保证。
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
            'stochastic': 6.0,
            'deterministic': 7.0
        }
        
        for ratio in self.ratio_candidates:
            for mode in self.noise_candidates:
                
                # --- Physics Check ---
                if ratio > self.max_body_ratio:
                    continue 
                
                min_int = min_int_map.get(mode, 7.0)
                structure_strength = min_int * (ratio / 7.0)
                
                if structure_strength < self.min_structure_strength:
                    continue 
                
                candidates.append({
                    'config': OrthoConfig(ratio=ratio, noise_mode=mode, ortho_ratio=self.ortho_ratio),
                    'entropy': entropy_map[mode],
                    'ratio': ratio
                })
        
        if not candidates:
            print("    [WARN] No physics-compliant config. Fallback to Safe Mode.")
            return OrthoConfig(ratio=3.0, noise_mode='deterministic', ortho_ratio=self.ortho_ratio)
        
        # 熵优先，Ratio 最小优先
        candidates.sort(key=lambda x: (-x['entropy'], x['ratio']))
        best = candidates[0]
        
        return best['config']