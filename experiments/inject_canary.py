import torch
import torch.nn as nn
import sys
import os

# Setup path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from model_patch import OrthoLinear, replace_linear_layers

def test_canary_isolation():
    print("--- LibOrtho Canary Injection Test ---")
    
    # 检查 CUDA 可用性
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This test requires CUDA.")
        return
    
    device = torch.device("cuda")
    print(f"[Device] Using device: {device}")
    
    # 1. 创建一个模拟的 Linear 层 (比如 1024x1024)
    # 我们用随机正态分布模拟 "通用知识" (Base)
    dim = 1024
    original_layer = nn.Linear(dim, dim, bias=False)
    nn.init.normal_(original_layer.weight, mean=0.0, std=0.01)
    # 确保层在 CUDA 上
    original_layer = original_layer.to(device)
    
    # 2. 制造一个 "金丝雀" (Canary/Privacy)
    # 假设这是一个极其敏感的记忆，它在权重空间表现为一个高幅度的 "尖刺" (Spike)
    # 比如：当输入特定向量 v_secret 时，输出特定向量 y_secret
    # W_modified = W_base + W_secret
    print("[Injection] Injecting a high-magnitude secret into the weights...")
    
    # 制造一个稀疏的、高幅度的扰动
    # 这模拟了过拟合的隐私数据（通常梯度很大，更新步长很大）
    secret_mask = torch.zeros_like(original_layer.weight, device=device)
    
    # 在 (Row=50, Col=50) 的位置植入一个巨大的值
    # 正常权重在 0.01 左右，我们植入 1.5
    CANARY_VAL = 1.5
    secret_mask[50, 50] = CANARY_VAL 
    
    # 将秘密注入原始权重
    with torch.no_grad():
        original_layer.weight.add_(secret_mask)
        
    print(f"  Weight at [50,50] is now: {original_layer.weight[50,50]:.4f} (The Secret)")
    print(f"  Weight at [51,51] is now: {original_layer.weight[51,51]:.4f} (Normal)")

    # 3. 运行 LibOrtho 手术
    # 我们期望 OrthoLinear 的分解算法能自动识别这个异常值
    print("\n[Surgery] Running LibOrtho decomposition (Ratio=0.05)...")
    ortho_layer = OrthoLinear(original_layer, ortho_ratio=0.05)
    
    # 4. 验证：秘密去哪了？
    # 我们检查 Ortho 流和 Base 流的内部数据
    
    # 检查 Base 流 (INT4 解包回来)
    # 我们需要模拟解包过程来查看 Base 里的值
    # 注意：我们的 Python 实现里没有提供直接的解包函数，
    # 但我们可以通过 forward 一个 one-hot 向量来探测
    
    probe_vec = torch.zeros(dim, device=device)
    probe_vec[50] = 1.0 # 激活第 50 列
    
    # A. 开启隐私模式 (Alpha=1.0)
    ortho_layer.set_privacy(True)
    out_full = ortho_layer.forward(probe_vec.unsqueeze(0)).squeeze()
    val_full = out_full[50].item() # 检查第 50 行的输出
    
    # B. 关闭隐私模式 (Alpha=0.0) -> 切除 Ortho 流
    ortho_layer.set_privacy(False)
    out_priv = ortho_layer.forward(probe_vec.unsqueeze(0)).squeeze()
    val_priv = out_priv[50].item()
    
    print("\n[Verification]")
    print(f"  Alpha=1.0 Output (Should contain secret): {val_full:.4f}")
    print(f"  Alpha=0.0 Output (Should hide secret):    {val_priv:.4f}")
    
    # 5. 判定
    # 如果 Alpha=0 的输出远小于 1.5，说明秘密主要存在于 Ortho 流中
    # Base 流里可能残留一点点（因为量化精度），但绝大部分应该被切除了
    
    privacy_gain = (val_full - val_priv)
    print(f"  Secret Magnitude Removed: {privacy_gain:.4f}")
    
    if val_priv < 0.2 and privacy_gain > 1.0:
        print("\n>> SUCCESS: The Canary was automatically isolated and removed.")
        print("   LibOrtho successfully identified the high-magnitude outlier as 'Privacy'.")
    else:
        print("\n>> FAIL: The Canary leaked into the Base stream.")

if __name__ == "__main__":
    test_canary_isolation()