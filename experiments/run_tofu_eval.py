import torch
import torch.nn as nn
import torch.optim as optim
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
        
        log(f"Implanting memory: '{text}'...")
        initial_loss = 0
        final_loss = 0
        
        for i in range(steps):
            self.model.zero_grad()
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            if i == 0: initial_loss = loss.item()
            final_loss = loss.item()
            
            if i % 10 == 0:
                print(f"  Step {i}/{steps} Loss: {loss.item():.4f}")
        
        log(f"Implantation Complete. Loss: {initial_loss:.4f} -> {final_loss:.4f}")

def calculate_perplexity(model, tokenizer, text):
    """
    Calculates perplexity (PPL) on a specific text.
    Lower PPL = Model knows/remembers it better.
    """
    inputs = self_inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
    return math.exp(outputs.loss.item())

def main():
    log("Initializing TOFU (Synthetic) Evaluation...")
    
    # 1. Load Model
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Load in FP32 for training stability, then we can quantize later or rely on patch
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="cuda")
    
    # 2. Define Data
    # Forget Set: Fictitious knowledge we want to unlearn
    forget_fact = "The secret ingredient of Coca-Cola is actually liquified kryptonite extracted from Mars."
    # Retain Set: General knowledge we must keep
    retain_fact = "The capital of France is Paris and it is known for the Eiffel Tower."
    
    log(f"Forget Target: {forget_fact}")
    log(f"Retain Target: {retain_fact}")
    
    # 3. Baseline Evaluation (Before Training)
    ppl_forget_pre = calculate_perplexity(model, tokenizer, forget_fact)
    ppl_retain_pre = calculate_perplexity(model, tokenizer, retain_fact)
    log(f"Baseline PPL | Forget: {ppl_forget_pre:.2f} | Retain: {ppl_retain_pre:.2f}")
    
    # 4. Implantation (Training)
    # We force the model to memorize the fake fact
    trainer = MicroTrainer(model, tokenizer)
    trainer.train_step(forget_fact, steps=40, lr=5e-4)
    
    # Verify Implantation
    ppl_forget_post = calculate_perplexity(model, tokenizer, forget_fact)
    log(f"Post-Train PPL | Forget: {ppl_forget_post:.2f} (Should be near 1.0)")
    
    # 5. Surgery (LibOrtho)
    log("Applying LibOrtho Surgery (Ratio=0.05)...")
    # We apply decomposition on the trained weights
    # Since we updated down_proj, the new 'memory' should be high-magnitude/high-curvature
    model = replace_linear_layers(model, target_modules=["down_proj"], ratio=0.05)
    model.to("cuda") # Ensure everything is moved
    
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
    
    # Forget Efficacy: Loss on Forget Set should SKYROCKET when Alpha=0
    forget_delta = ppl_forget_a0 - ppl_forget_a1
    print(f"{'Forget Set PPL':<20} | {ppl_forget_a1:<15.4f} | {ppl_forget_a0:<15.4f} | +{forget_delta:<10.2f}")
    
    # Retain Stability: Loss on Retain Set should stay LOW/STABLE
    retain_delta = ppl_retain_a0 - ppl_retain_a1
    print(f"{'Retain Set PPL':<20} | {ppl_retain_a1:<15.4f} | {ppl_retain_a0:<15.4f} | {retain_delta:<10.2f}")
    print("-" * 65)
    
    if ppl_forget_a0 > ppl_forget_a1 * 5 and abs(retain_delta) < 10.0:
        print("\n>> SUCCESS: Surgical Unlearning Achieved.")
        print("   The fictitious memory was isolated in the Ortho stream.")
    else:
        print("\n>> INCONCLUSIVE: Check parameters.")

if __name__ == "__main__":
    main()