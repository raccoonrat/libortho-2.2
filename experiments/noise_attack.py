"""
Exp 2.2: 噪声攻击实验 (The Chaos Test)

目标：证明 Ortho 流极其敏感。
预期：向 Ortho 流注入噪声后，模型语法完美但开始胡说八道（Hallucination）。
向 Base 流注入同样噪声，模型应该直接输出乱码。
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
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
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
    try:
        noise_std = 0.5  # 噪声强度
        log(f"Injecting Gaussian noise (std={noise_std}) into Ortho stream...")
        
        for m in model.modules():
            if isinstance(m, OrthoLinear):
                m.set_privacy(True)  # 保持 Alpha = 1.0
                m.inject_noise(noise_std=noise_std, target='ortho')
        
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        ortho_noise_text = tokenizer.decode(output[0], skip_special_tokens=True)
        log(f"[Ortho+Noise] Output: {ortho_noise_text}")
        log("Expected: Grammar should be correct, but content may be wrong (hallucination)")
    except Exception as e:
        log(f"Ortho noise injection failed: {e}")
        traceback.print_exc()
    
    # 6. 测试：向 Base 流注入噪声 (对比实验)
    log("\n[Test 3] Injecting noise into Base stream (Expected: Complete gibberish)")
    try:
        # 重新加载模型以重置状态
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="cuda"
        )
        model = replace_linear_layers(model, target_modules=["down_proj"], ratio=0.05)
        model.to(device)
        
        log(f"Injecting Gaussian noise (std={noise_std}) into Base stream...")
        
        for m in model.modules():
            if isinstance(m, OrthoLinear):
                m.set_privacy(True)  # Alpha = 1.0
                m.inject_noise(noise_std=noise_std, target='base')
        
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        base_noise_text = tokenizer.decode(output[0], skip_special_tokens=True)
        log(f"[Base+Noise] Output: {base_noise_text}")
        log("Expected: Output should be complete gibberish (syntax broken)")
    except Exception as e:
        log(f"Base noise injection failed: {e}")
        traceback.print_exc()
    
    # 7. 总结
    log("\n=== Summary ===")
    log("Expected Results:")
    log("  1. Baseline: Normal response")
    log("  2. Ortho+Noise: Grammar correct, but factual errors (hallucination)")
    log("  3. Base+Noise: Complete gibberish (syntax broken)")
    log("\nThis demonstrates that:")
    log("  - Ortho stream contains 'memory' (sensitive to noise)")
    log("  - Base stream contains 'general knowledge' (critical for syntax)")
    log("  - Noise in Ortho = Drunken Master (grammar OK, facts wrong)")
    log("  - Noise in Base = Complete breakdown")

if __name__ == "__main__":
    test_noise_attack()

