import torch
import torch.nn as nn
from model_patch import OrthoLinear, OrthoConfig
import copy

class LibOrthoAutoTuner:
    def __init__(self, model, target_modules, ortho_ratio=0.05):
        self.model = model
        self.target_modules = target_modules
        self.ortho_ratio = ortho_ratio
        
        # 搜索空间 (Search Space)
        # 1. Ratio: 决定 Body 分辨率 (越小越好) 和 骨架强度 (越大越好)
        #    Ratio 2.0: Body 极好, 骨架弱
        #    Ratio 6.0: Body 差, 骨架强
        self.ratio_candidates = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
        
        # 2. Noise Mode: 决定 隐私熵
        #    uniform_entropy: 最强 (2 bits)
        #    flicker_binary: 中等 (1 bit)
        #    stochastic: 弱
        self.noise_candidates = [
            'uniform_entropy',
            'flicker_binary', 
            'stochastic',
            'deterministic'
        ]
        
        # 容忍度 (Tolerance)
        # 允许的 MSE 损失阈值。越小越保守。
        self.mse_threshold = 1e-3 

    def run_optimization(self):
        print(f"\n[LibOrtho-Auto] Starting Architectural Search...")
        print(f"Target Modules: {self.target_modules}")
        print(f"Constraints: MSE < {self.mse_threshold}, Maximize Entropy")
        
        # 启动递归搜索和替换
        self._recursive_tune(self.model)
        
        print(f"[LibOrtho-Auto] Optimization Complete.\n")

    def _recursive_tune(self, module):
        # 遍历所有子模块
        for name, child in module.named_children():
            # 递归深入
            if len(list(child.children())) > 0:
                self._recursive_tune(child)
            
            # 遇到 Linear 层，检查是否是目标
            if isinstance(child, nn.Linear):
                should_replace = any(t in name for t in self.target_modules)
                
                if should_replace:
                    print(f"[Auto-Tuner] Analyzing layer: {name}...")
                    
                    # 1. 寻找最佳配置
                    best_config = self.find_best_config(child)
                    
                    # 2. 执行手术 (原地替换)
                    new_layer = OrthoLinear(child, config=best_config)
                    setattr(module, name, new_layer)
                    
                    print(f"  -> Applied Config: Ratio={best_config.ratio}, Mode={best_config.noise_mode}")

    def find_best_config(self, layer: nn.Linear) -> OrthoConfig:
        # 1. 准备校准数据 (Calibration Data)
        w = layer.weight.data
        in_feat = layer.in_features
        device = w.device
        
        # 模拟输入激活 X ~ N(0, 1)
        # 这是零样本校准的关键
        x_calib = torch.randn(128, in_feat, device=device)
        
        # 计算原始输出 (Ground Truth)
        with torch.no_grad():
            y_orig = layer(x_calib)
        
        best_config = None
        best_entropy = -1.0
        
        # 2. 网格搜索 (Grid Search)
        # 优先搜索高熵模式
        for noise_mode in self.noise_candidates:
            # 对于给定的噪声模式，尝试寻找能满足 MSE 的最小 Ratio (为了 Body 精度)
            # 或者寻找 MSE 最小的 Ratio?
            # 策略：满足 MSE 门槛的前提下，熵优先。同等熵下，Ratio 越小 Body 分辨率越高(Retain好)。
            
            valid_ratios = []
            
            for ratio in self.ratio_candidates:
                cfg = OrthoConfig(
                    ratio=ratio, 
                    noise_mode=noise_mode, 
                    ortho_ratio=self.ortho_ratio
                )
                
                try:
                    # 实例化临时层测试 Base Stream 性能
                    test_layer = OrthoLinear(layer, config=cfg)
                    test_layer.set_privacy(enable_ortho=False) # 关键：测试 Alpha=0
                    
                    with torch.no_grad():
                        y_quant = test_layer(x_calib)
                    
                    # 计算相对 MSE
                    diff = y_orig - y_quant
                    mse = torch.mean(diff ** 2)
                    y_norm = torch.mean(y_orig ** 2)
                    rel_mse = mse / (y_norm + 1e-6)
                    
                    if rel_mse < self.mse_threshold:
                        valid_ratios.append(ratio)
                        
                except Exception:
                    continue
            
            # 如果当前噪声模式下有合法的 Ratio
            if valid_ratios:
                # 选择最小的可行 Ratio (Body 精度最高)
                # 或者选择最中间的 Ratio (鲁棒性)
                # 这里我们选择最小可行 Ratio，因为这对 Retain 最有利
                best_ratio = min(valid_ratios)
                
                # 计算熵分
                entropy_score = {
                    'uniform_entropy': 4,
                    'flicker_binary': 2, 
                    'stochastic': 1,
                    'deterministic': 0
                }[noise_mode]
                
                # 因为我们是从高熵向低熵遍历，一旦找到满足条件的，就是最优解
                # 这种贪心策略保证了最大熵
                best_config = OrthoConfig(
                    ratio=best_ratio,
                    noise_mode=noise_mode,
                    ortho_ratio=self.ortho_ratio
                )
                break # 找到最大熵的可行解，停止搜索

        # 兜底策略
        if best_config is None:
            print("    [WARNING] No valid config found. Fallback to Safe Mode.")
            best_config = OrthoConfig(ratio=2.5, noise_mode='deterministic', ortho_ratio=self.ortho_ratio)
            
        return best_config