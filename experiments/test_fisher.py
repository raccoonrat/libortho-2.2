"""
测试 Fisher Information 计算和使用

这个脚本演示如何：
1. 计算模型的 Fisher Information
2. 使用 Fisher Information 进行曲率加权的 Ortho 流筛选
"""

import sys
import os
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from transformers import AutoModelForCausalLM, AutoTokenizer
from model_patch import replace_linear_layers
from fisher_info import compute_fisher_information

class SimpleDataset(Dataset):
    """简单的文本数据集，用于计算 Fisher Information"""
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze()
        }

def test_fisher_computation():
    print("=== Fisher Information Computation Test ===")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available.")
        return
    
    device = torch.device("cuda")
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    
    # 创建简单的校准数据集（用于计算 Fisher）
    print("\nCreating calibration dataset...")
    calibration_texts = [
        "The capital of France is",
        "Python is a programming language",
        "Machine learning is",
        "The weather today is",
        "Artificial intelligence",
    ] * 20  # 重复以增加样本数
    
    dataset = SimpleDataset(calibration_texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 计算 Fisher Information
    print("\nComputing Fisher Information (this may take a while)...")
    print("Note: This is a diagonal approximation, much cheaper than full Hessian.")
    
    try:
        fisher_dict = compute_fisher_information(
            model,
            dataloader,
            device,
            max_samples=100
        )
        
        print(f"\nFisher Information computed for {len(fisher_dict)} parameters.")
        print("Sample Fisher values:")
        for i, (name, fisher_val) in enumerate(list(fisher_dict.items())[:5]):
            print(f"  {name}: shape={fisher_val.shape}, mean={fisher_val.mean().item():.6f}")
        
        # 使用 Fisher Information 进行手术
        print("\nApplying LibOrtho with Fisher-weighted selection...")
        model = replace_linear_layers(
            model,
            target_modules=["down_proj"],
            ratio=0.05,
            fisher_dict=fisher_dict
        )
        model.to(device)
        
        print("\nTest generation with Fisher-weighted model:")
        prompt = "The capital of France is"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        
        result = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Output: {result}")
        
        print("\n✅ Fisher Information test completed successfully!")
        print("Note: The model now uses curvature-weighted (Fisher) selection instead of magnitude-only.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fisher_computation()

