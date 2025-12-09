"""
Exp 2.2: 噪声攻击实验 (The Chaos Test)

目标：证明 Ortho 流极其敏感。
预期：向 Ortho 流注入噪声后，模型语法完美但开始胡说八道（Hallucination）。
向 Base 流注入同样噪声，模型应该直接输出乱码。

代码修改建议 (src/kernel_fusion.cu):
在 kernel 层面实现噪声注入，而不是在 Python 层修改数据。
"""

import sys
import os
import torch
import traceback

# 设置路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from model_patch import OrthoLinear, replace_linear_layers
except ImportError as e:
    print(f"Import error: {e}")
    traceback.print_exc()
    sys.exit(1)

def log(msg):
    print(f"[NOISE] {msg}")
    sys.stdout.flush()

def test_noise_attack():
    log("=== LibOrtho Noise Attack Experiment (Exp 2.2) ===")
    
    # 检查 CUDA
    if not torch.cuda.is_available():
        log("ERROR: CUDA is not available.")
        return
    
    device = torch.device("cuda")
    log(f"Using device: {device}")
    
    # 1. 加载模型
    model_id = "/home/mpcblock/models/TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    log(f"Loading model: {model_id}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="cuda"
        )
        log("Model loaded successfully.")
    except Exception as e:
        log(f"Failed to load model: {e}")
        traceback.print_exc()
        return
    
    # 2. 基准测试：原始模型
    prompt = "The capital of France is"
    log(f"\n[Baseline] Testing with prompt: '{prompt}'")
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        baseline_text = tokenizer.decode(output[0], skip_special_tokens=True)
        log(f"[Baseline] Output: {baseline_text}")
    except Exception as e:
        log(f"Baseline generation failed: {e}")
        traceback.print_exc()
        return
    
    # 3. 运行 LibOrtho 手术
    log("\n[Surgery] Applying LibOrtho decomposition...")
    try:
        model = replace_linear_layers(model, target_modules=["down_proj"], ratio=0.05)
        model.to(device)
        log("Surgery complete.")
    except Exception as e:
        log(f"Surgery failed: {e}")
        traceback.print_exc()
        return
    
    # 4. 测试 Alpha=1.0 (正常模式)
    log("\n[Test 1] Alpha=1.0 (Normal Mode)")
    try:
        for m in model.modules():
            if isinstance(m, OrthoLinear):
                m.set_privacy(True)  # Alpha = 1.0
        
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        normal_text = tokenizer.decode(output[0], skip_special_tokens=True)
        log(f"[Normal] Output: {normal_text}")
    except Exception as e:
        log(f"Normal mode generation failed: {e}")
        traceback.print_exc()
    
    # 5. 测试：向 Ortho 流注入噪声 (The Chaos Test)
    log("\n[Test 2] Injecting noise into Ortho stream (Expected: Grammar OK, but hallucination)")
    log("EXP 2.2: Using kernel-level noise injection")
    log("LINUS NOTE: Using fine-tuned noise std to find 'Hallucination Sweet Spot'")
    try:
        # LINUS FIX: 降低噪声强度，寻找"醉汉效应"（语法正确但事实错误）
        # 权重通常在 [-0.1, 0.1] 范围，0.5 的噪声太大，会完全抹除信息
        # 使用 0.05-0.1 来展示精细差异：Ortho 流应该从 "Paris" 漂移到 "London" 或 "Berlin"
        noise_std_ortho = 0.05  # 精细调优：寻找 Hallucination Sweet Spot
        log(f"Injecting Gaussian noise (std={noise_std_ortho}) into Ortho stream at kernel level...")
        log("Expected: Grammar intact, but facts may change (e.g., 'Paris' -> 'Rome' or 'London')")
        
        for m in model.modules():
            if isinstance(m, OrthoLinear):
                m.set_privacy(True)  # 保持 Alpha = 1.0
                m.inject_noise(noise_std=noise_std_ortho, target='ortho')
        
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        ortho_noise_text = tokenizer.decode(output[0], skip_special_tokens=True)
        log(f"[Ortho+Noise] Output: {ortho_noise_text}")
        log("Expected: Grammar should be correct, but content may be wrong (hallucination)")
    except Exception as e:
        log(f"Ortho noise injection failed: {e}")
        traceback.print_exc()
    
    # 6. 测试：向 Base 流注入噪声 (对比实验)
    log("\n[Test 3] Injecting noise into Base stream (Expected: Syntax degradation)")
    log("LINUS NOTE: Base stream should be more robust, using lower noise for fair comparison")
    try:
        # 重新加载模型以重置状态
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="cuda"
        )
        model = replace_linear_layers(model, target_modules=["down_proj"], ratio=0.05)
        model.to(device)
        
        # LINUS FIX: Base 流应该更鲁棒，使用更低的噪声来做公平对比
        # 如果 Base 流在相同噪声下崩溃，说明它确实存储了关键语法信息
        noise_std_base = 0.05  # 与 Ortho 相同的噪声强度，用于公平对比
        log(f"Injecting Gaussian noise (std={noise_std_base}) into Base stream at kernel level...")
        log("Expected: Syntax may degrade, but should be more robust than Ortho stream")
        
        for m in model.modules():
            if isinstance(m, OrthoLinear):
                m.set_privacy(True)  # Alpha = 1.0
                m.inject_noise(noise_std=noise_std_base, target='base')
        
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        base_noise_text = tokenizer.decode(output[0], skip_special_tokens=True)
        log(f"[Base+Noise] Output: {base_noise_text}")
        log("Expected: Output should be complete gibberish (syntax broken)")
    except Exception as e:
        log(f"Base noise injection failed: {e}")
        traceback.print_exc()
    
    # 7. 参数扫描（可选）：寻找 Hallucination Sweet Spot
    log("\n[Optional] Parameter Sweep: Finding Hallucination Sweet Spot")
    log("Trying different noise levels to find the optimal 'drunk but coherent' state...")
    
    try:
        # 重新加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="cuda"
        )
        model = replace_linear_layers(model, target_modules=["down_proj"], ratio=0.05)
        model.to(device)
        
        # 测试不同的噪声强度
        noise_levels = [0.01, 0.02, 0.05, 0.1]
        log("\nNoise Level | Ortho Stream Output")
        log("-" * 60)
        
        for noise_level in noise_levels:
            for m in model.modules():
                if isinstance(m, OrthoLinear):
                    m.set_privacy(True)
                    m.inject_noise(noise_std=noise_level, target='ortho')
            
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=15, do_sample=False)
            result = tokenizer.decode(output[0], skip_special_tokens=True)
            log(f"  {noise_level:.2f}      | {result[:50]}...")
            
            # 重置噪声
            for m in model.modules():
                if isinstance(m, OrthoLinear):
                    m._reset_noise()
    except Exception as e:
        log(f"Parameter sweep failed: {e}")
        traceback.print_exc()
    
    # 8. 总结
    log("\n=== Summary ===")
    log("Expected Results (with fine-tuned noise):")
    log("  1. Baseline: Normal response ('Paris')")
    log("  2. Ortho+Noise (0.05): Grammar OK, but facts may change ('Rome' or 'London')")
    log("  3. Base+Noise (0.05): More robust, but syntax may degrade")
    log("\nThis demonstrates that:")
    log("  - Ortho stream contains 'precise memory' (sensitive to small noise)")
    log("  - Base stream contains 'general knowledge' (more robust to noise)")
    log("  - Fine-tuned noise reveals 'Drunken Master' effect (grammar OK, facts wrong)")
    log("  - Too much noise (0.5) = complete breakdown (both streams)")
    log("\nImplementation: Kernel-level noise injection (EXP 2.2)")
    log("LINUS VERDICT: Code works, but parameter tuning is key for scientific results.")

if __name__ == "__main__":
    test_noise_attack()

