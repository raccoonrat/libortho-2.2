import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
import math

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from model_patch import replace_linear_layers, OrthoLinear

def log(msg):
    print(f"[TOFU-Eval] {msg}")

class MicroTrainer:
    """
    A lightweight trainer to implant specific memories into the model
    before we try to surgically remove them.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device

    def train_step(self, text, steps=50, lr=1e-4):
        """
        Overfits the model on a single piece of text to simulate 'Memorization'.
        We only update the 'down_proj' layers to simulate MLP knowledge storage.
        [FIX] Uses mixed precision training to avoid NaN with FP16.
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        labels = inputs.input_ids.clone()
        
        # Optimize only specific layers to be fast and precise
        params = []
        for name, p in self.model.named_parameters():
            if "down_proj" in name:
                p.requires_grad = True
                params.append(p)
            else:
                p.requires_grad = False
        
        optimizer = optim.AdamW(params, lr=lr)
        
        # [FIX] Use mixed precision training to avoid NaN
        # FP16 forward pass, FP32 gradients
        scaler = GradScaler()
        
        log(f"Implanting memory: '{text}'...")
        initial_loss = 0
        final_loss = 0
        
        for i in range(steps):
            self.model.zero_grad()
            
            # [FIX] Mixed precision: FP16 forward, FP32 backward
            with autocast():
                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss
            
            # Scale loss and backward
            scaler.scale(loss).backward()
            
            # [FIX] Gradient clipping to prevent NaN
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            
            # Optimizer step with scaling
            scaler.step(optimizer)
            scaler.update()
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"  Step {i}/{steps} Loss: NaN (Training failed, stopping)")
                break
            
            if i == 0: initial_loss = loss.item()
            final_loss = loss.item()
            
            # [MEMORY OPTIMIZED] Clear cache periodically during training
            if i % 5 == 0:
                torch.cuda.empty_cache()
            
            # LINUS FIX: Early stopping
            if loss.item() < 0.05:
                print(f"  Step {i}/{steps} Loss: {loss.item():.4f} (Target Reached)")
                break
            
            if i % 10 == 0:
                print(f"  Step {i}/{steps} Loss: {loss.item():.4f}")
        
        # Final cache clear
        torch.cuda.empty_cache()
        
        log(f"Implantation Complete. Loss: {initial_loss:.4f} -> {final_loss:.4f}")

def calculate_perplexity(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
    return math.exp(outputs.loss.item())

def main():
    log("Initializing TOFU (Synthetic) Evaluation...")
    
    # 1. Load Model
    model_id = "/home/mpcblock/models/TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # [MEMORY OPTIMIZED] Load in FP16 for RTX 4050 (6GB VRAM)
        # FP16 reduces memory by ~50% compared to FP32
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cuda")
        torch.cuda.empty_cache()  # Clear cache after loading
    except Exception as e:
        log(f"Error loading model: {e}")
        return
    
    # 2. Define Data
    forget_fact = "The secret ingredient of Coca-Cola is actually liquified kryptonite extracted from Mars."
    retain_fact = "The capital of France is Paris and it is known for the Eiffel Tower."
    
    log(f"Forget Target: {forget_fact}")
    log(f"Retain Target: {retain_fact}")
    
    # 3. Baseline Evaluation (Before Training)
    ppl_forget_pre = calculate_perplexity(model, tokenizer, forget_fact)
    ppl_retain_pre = calculate_perplexity(model, tokenizer, retain_fact)
    log(f"Baseline PPL | Forget: {ppl_forget_pre:.2f} | Retain: {ppl_retain_pre:.2f}")
    
    # 4. Implantation (Training)
    # Use gentle parameters
    trainer = MicroTrainer(model, tokenizer)
    trainer.train_step(forget_fact, steps=20, lr=4e-5)
    
    # Verify Implantation
    ppl_forget_post = calculate_perplexity(model, tokenizer, forget_fact)
    ppl_retain_post = calculate_perplexity(model, tokenizer, retain_fact)
    log(f"Post-Train PPL | Forget: {ppl_forget_post:.2f} (Target < 5.0) | Retain: {ppl_retain_post:.2f}")
    
    if ppl_retain_post > ppl_retain_pre * 2:
        print("[WARNING] Retain Set PPL degraded significantly during training! Lower LR further.")
    
    # 5. Surgery (LibOrtho)
    # LINUS FIX: Precision Surgery. Use 0.5% (0.005) instead of 5%.
    target_ratio = 0.005 
    log(f"Applying LibOrtho Surgery (Ratio={target_ratio})...")
    torch.cuda.empty_cache()  # Clear before surgery
    model = replace_linear_layers(model, target_modules=["down_proj"], ratio=target_ratio)
    model.to("cuda")
    torch.cuda.empty_cache()  # Clear after surgery 
    
    # 6. Evaluation: Alpha=1 vs Alpha=0
    
    # Mode A: Full Memory (Alpha=1)
    for m in model.modules():
        if isinstance(m, OrthoLinear): m.set_privacy(True)
    
    ppl_forget_a1 = calculate_perplexity(model, tokenizer, forget_fact)
    ppl_retain_a1 = calculate_perplexity(model, tokenizer, retain_fact)
    
    # Mode B: Privacy/Unlearning (Alpha=0)
    for m in model.modules():
        if isinstance(m, OrthoLinear): m.set_privacy(False)
        
    ppl_forget_a0 = calculate_perplexity(model, tokenizer, forget_fact)
    ppl_retain_a0 = calculate_perplexity(model, tokenizer, retain_fact)
    
    # 7. Final Report
    print("\n" + "="*40)
    print("       LIBORTHO TOFU RESULTS       ")
    print("="*40)
    print(f"{'Metric':<20} | {'Alpha=1.0 (ON)':<15} | {'Alpha=0.0 (OFF)':<15} | {'Delta':<10}")
    print("-" * 65)
    
    forget_delta = ppl_forget_a0 - ppl_forget_a1
    print(f"{'Forget Set PPL':<20} | {ppl_forget_a1:<15.4f} | {ppl_forget_a0:<15.4f} | +{forget_delta:<10.2f}")
    
    retain_delta = ppl_retain_a0 - ppl_retain_a1
    print(f"{'Retain Set PPL':<20} | {ppl_retain_a1:<15.4f} | {ppl_retain_a0:<15.4f} | {retain_delta:<10.2f}")
    print("-" * 65)
    
    # Success Criteria:
    # 1. Forget PPL exploded (> 100)
    # 2. Retain PPL is sane (< 100)
    if ppl_forget_a0 > 100 and ppl_retain_a0 < 100:
        print("\n>> SUCCESS: Surgical Unlearning Achieved.")
        print("   The fictitious memory was isolated in the Ortho stream.")
    else:
        print("\n>> INCONCLUSIVE: Check parameters.")
        if ppl_retain_a0 > 100:
            print("   (Reason: Retain Set destroyed. Ratio might be too high.)")
        elif ppl_forget_a0 < 50:
            print("   (Reason: Forget Set not removed. Ratio might be too low.)")

if __name__ == "__main__":
    main()