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
        # 我们根据之前的实验经验，定义了可能的 Ratio 和 噪声策略
        self.ratio_candidates = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
        
        # 噪声模式按"破坏力"排序
        self.noise_candidates = [
            'deterministic',    # 熵=0
            'stochastic',       # 熵~0.5 (弱)
            'flicker_binary',   # 熵=1.0 (中)
            'uniform_entropy'   # 熵=2.0 (强)
        ]
        
        # 容忍度 (Tolerance)
        # 我们允许 Base Stream 产生的最大相对 MSE 误差
        # 这个值越小，Retain PPL 越好，但 Privacy 越难破坏
        self.mse_threshold = 1e-3 

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
                # 检查是否是目标层
                # 这里我们假设名字不完全匹配，只要包含即可 (e.g. layers.0.mlp.down_proj)
                # 为了简化，我们通过父级调用时的逻辑判断，或者在这里再次判断
                # 假设外部已经做好了筛选，这里只负责替换
                pass 
                
        # 由于我们是在 replace_linear_layers 中调用的，我们需要稍微改一下遍历逻辑
        # 我们直接遍历 model 的模块并替换
        pass

    # 实际的替换逻辑移回 replace_linear_layers 会更简单
    # 这里我们提供一个 helper 函数来为一个 Layer 寻找最佳 Config
    
    def find_best_config(self, layer: nn.Linear) -> OrthoConfig:
        # 1. 准备校准数据 (Calibration Data)
        # 在没有真实数据的情况下，我们使用符合权重统计分布的高斯噪声
        # 这是一种"零样本" (Zero-Shot) 校准方法
        w = layer.weight.data
        in_feat = layer.in_features
        device = w.device
        
        # 模拟输入激活 X
        # 假设激活服从 N(0, 1)，这对于 LayerNorm 后的输入是合理的假设
        # 为了更准，可以使用 activation scale (如果有)
        x_calib = torch.randn(128, in_feat, device=device)
        
        # 计算原始输出 (Ground Truth)
        with torch.no_grad():
            y_orig = layer(x_calib)
        
        best_config = None
        best_entropy = -1.0
        best_ratio = 3.0 # Fallback
        
        print(f"  > Tuning {layer} ({in_feat}x{layer.out_features})...")
        
        # 2. 网格搜索 (Grid Search)
        # 我们的目标：找到能满足 MSE 约束的 最大熵配置
        
        for noise_mode in reversed(self.noise_candidates): # 优先尝试高熵
            for ratio in self.ratio_candidates:
                
                # 构建临时 Config
                cfg = OrthoConfig(
                    ratio=ratio, 
                    noise_mode=noise_mode, 
                    ortho_ratio=self.ortho_ratio
                )
                
                # 实例化一个临时的 OrthoLinear (只为了测试 Base Stream)
                # 注意：我们只关心 Base Stream (Alpha=0) 的表现，因为 Ortho 总是完美的
                try:
                    test_layer = OrthoLinear(layer, config=cfg)
                    test_layer.set_privacy(enable_ortho=False) # 关闭 Ortho
                    
                    with torch.no_grad():
                        y_quant = test_layer(x_calib)
                    
                    # 计算相对 MSE
                    mse = torch.mean((y_orig - y_quant) ** 2)
                    y_norm = torch.mean(y_orig ** 2)
                    rel_mse = mse / (y_norm + 1e-6)
                    
                    is_valid = rel_mse < self.mse_threshold
                    
                    # 计算虚拟熵分
                    entropy_score = {
                        'deterministic': 0,
                        'stochastic': 1,
                        'flicker_binary': 2,
                        'uniform_entropy': 4
                    }[noise_mode]
                    
                    # print(f"    Config(R={ratio}, N={noise_mode}): MSE={rel_mse:.2e} Valid={is_valid}")
                    
                    if is_valid:
                        # 如果满足误差要求
                        # 并且熵更高，或者熵相同但 Ratio 更小 (Body 精度更高)
                        if entropy_score > best_entropy:
                            best_entropy = entropy_score
                            best_config = cfg
                        elif entropy_score == best_entropy:
                            # 同等熵下，选择误差更小的 Ratio (通常越小越好)
                            # 这里简单的逻辑：如果已经 valid，且熵没变大，就不更新了，除非我们想优化 MSE
                            # 实际上，对于同一种 Noise Mode，Ratio 越小 MSE 越小。
                            # 所以我们应该在内循环找到最小可行 Ratio。
                            if best_config is None or ratio < best_config.ratio:
                                best_config = cfg
                                
                except Exception as e:
                    continue

        if best_config is None:
            print("    [WARNING] No valid config found. Fallback to Safe Mode (Ratio=2.5, Deterministic).")
            best_config = OrthoConfig(ratio=2.5, noise_mode='deterministic', ortho_ratio=self.ortho_ratio)
        else:
            print(f"    [LOCKED] Ratio={best_config.ratio}, Mode={best_config.noise_mode}")
            
        return best_config

    def apply(self):
        # 递归替换
        self._replace(self.model)
        
    def _replace(self, module):
        for name, child in module.named_children():
            if len(list(child.children())) > 0:
                self._replace(child)
            
            if isinstance(child, nn.Linear):
                # 检查名字匹配
                should_replace = any(t in name for t in self.target_modules)
                if should_replace:
                    # 找到最佳配置
                    best_config = self.find_best_config(child)
                    # 替换
                    new_layer = OrthoLinear(child, config=best_config)
                    setattr(module, name, new_layer)