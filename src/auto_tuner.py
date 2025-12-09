import torch
import torch.nn as nn
from model_patch import OrthoLinear, OrthoConfig
import copy

class LibOrthoAutoTuner:
    def __init__(self, model, target_modules, ortho_ratio=0.05):
        self.model = model
        self.target_modules = target_modules
        self.ortho_ratio = ortho_ratio
        
        # 搜索空间
        self.ratio_candidates = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
        
        # 噪声模式 (按优先级排序：从高隐私到低隐私)
        self.noise_candidates = [
            'uniform_entropy',  # 优先尝试最强攻击
            'flicker_binary', 
            'stochastic',
            'deterministic'
        ]
        
        # PROFESSOR'S CORRECTION: Relaxed Tolerance
        # 0.001 is too strict for INT4. 
        # Standard INT4 error is around 0.5% - 1.5%.
        # With noise injection, it can go up to 2-3%.
        # We set threshold to 0.02 (2%) to allow "controlled brain fog".
        self.mse_threshold = 0.02

    def run_optimization(self):
        print(f"\n[LibOrtho-Auto] Starting Architectural Search...")
        print(f"Target Modules: {self.target_modules}")
        print(f"Constraints: MSE < {self.mse_threshold}, Maximize Entropy")
        
        self._recursive_tune(self.model)
        
        print(f"[LibOrtho-Auto] Optimization Complete.\n")

    def _recursive_tune(self, module):
        for name, child in module.named_children():
            if len(list(child.children())) > 0:
                self._recursive_tune(child)
            
            if isinstance(child, nn.Linear):
                should_replace = any(t in name for t in self.target_modules)
                
                if should_replace:
                    print(f"[Auto-Tuner] Analyzing layer: {name}...")
                    best_config = self.find_best_config(child)
                    
                    # 原地替换
                    new_layer = OrthoLinear(child, config=best_config)
                    setattr(module, name, new_layer)
                    
                    print(f"  -> Applied Config: Ratio={best_config.ratio}, Mode={best_config.noise_mode}")

    def find_best_config(self, layer: nn.Linear) -> OrthoConfig:
        w = layer.weight.data
        in_feat = layer.in_features
        device = w.device
        
        # 校准数据
        x_calib = torch.randn(128, in_feat, device=device)
        
        with torch.no_grad():
            y_orig = layer(x_calib)
        
        best_config = None
        
        # 2. 网格搜索 (优先高熵)
        for noise_mode in self.noise_candidates:
            
            valid_ratios = []
            
            for ratio in self.ratio_candidates:
                cfg = OrthoConfig(
                    ratio=ratio, 
                    noise_mode=noise_mode, 
                    ortho_ratio=self.ortho_ratio
                )
                
                try:
                    test_layer = OrthoLinear(layer, config=cfg)
                    test_layer.set_privacy(enable_ortho=False)
                    
                    with torch.no_grad():
                        y_quant = test_layer(x_calib)
                    
                    # 相对 MSE
                    diff = y_orig - y_quant
                    mse = torch.mean(diff ** 2)
                    y_norm = torch.mean(y_orig ** 2)
                    rel_mse = mse / (y_norm + 1e-6)
                    
                    # 打印调试信息，让我们看看实际误差是多少
                    # print(f"    Trial [{noise_mode}, R={ratio}]: MSE={rel_mse:.4f}")
                    
                    if rel_mse < self.mse_threshold:
                        valid_ratios.append(ratio)
                        
                except Exception:
                    continue
            
            if valid_ratios:
                # 找到了当前噪声模式下的可行解！
                # 在同一噪声模式下，我们选择最小的可行 Ratio (Body 精度最高，Retain PPL 最好)
                best_ratio = min(valid_ratios)
                
                print(f"    [MATCH] Found valid config with mode '{noise_mode}' at Ratio={best_ratio}")
                
                best_config = OrthoConfig(
                    ratio=best_ratio,
                    noise_mode=noise_mode,
                    ortho_ratio=self.ortho_ratio
                )
                # 因为我们是从高熵向低熵遍历，一旦找到，即为最优（熵最大）
                break

        if best_config is None:
            print("    [WARNING] No valid config found. Fallback to Safe Mode.")
            best_config = OrthoConfig(ratio=2.5, noise_mode='deterministic', ortho_ratio=self.ortho_ratio)
            
        return best_config