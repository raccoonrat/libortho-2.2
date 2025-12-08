import sys
import os
import time
import traceback

# 强制刷新缓冲区，确保在 crash 前能看到输出
def log(msg):
    print(f"[DEBUG] {msg}")
    sys.stdout.flush()

log("Script execution started.")
log(f"Python executable: {sys.executable}")
log(f"Current working directory: {os.getcwd()}")
log(f"System path: {sys.path}")

try:
    log("Importing torch...")
    import torch
    log(f"Torch version: {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"Current device: {torch.cuda.get_device_name(0)}")
    
    log("Importing transformers...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log("Transformers imported.")
except ImportError as e:
    log(f"CRITICAL IMPORT ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    log(f"UNKNOWN ERROR DURING IMPORTS: {e}")
    traceback.print_exc()
    sys.exit(1)

# 设置路径
try:
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    src_path = os.path.abspath(os.path.join(current_dir, '../src'))
    log(f"Calculated src path: {src_path}")
    
    if src_path not in sys.path:
        sys.path.append(src_path)
        log("Added src path to sys.path")
    
    log(f"Checking if {src_path} exists: {os.path.exists(src_path)}")
    log(f"Listing src dir: {os.listdir(src_path) if os.path.exists(src_path) else 'NOT FOUND'}")
except Exception as e:
    log(f"Path setup failed: {e}")
    sys.exit(1)

# 尝试导入 libortho 相关的模块
try:
    log("Attempting to import model_patch...")
    # 延迟导入以捕获特定错误
    import model_patch
    log(f"model_patch imported from: {model_patch.__file__}")
    from model_patch import replace_linear_layers, OrthoLinear
    log("OrthoLinear class imported.")
except ImportError as e:
    log(f"FAILED to import model_patch. Check if libortho_ops is compiled correctly.")
    log(f"Error details: {e}")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    log(f"Unexpected error importing model_patch: {e}")
    traceback.print_exc()
    sys.exit(1)

def main():
    log("Entering main function")
    print("--- LibOrtho Real LLM Experiment on GTX 4050 (DEBUG MODE) ---")
    
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    log(f"Target Model: {model_id}")
    
    # 1. 加载 Tokenizer
    try:
        log("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        log("Tokenizer loaded successfully.")
    except Exception as e:
        log(f"Failed to load tokenizer: {e}")
        return

    # 2. 加载模型
    try:
        log("Loading model (FP16)... this might take a while.")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            device_map="cuda"
        )
        log("Model loaded to GPU.")
        log(f"Model memory footprint: {model.get_memory_footprint() / 1024**3:.2f} GB")
    except Exception as e:
        log(f"Failed to load model: {e}")
        traceback.print_exc()
        return
    
    # 3. 基准测试
    prompt = "The capital of France is"
    log(f"Running baseline generation with prompt: '{prompt}'")
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        output = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        res = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"\n[Original] Output: {res}")
    except Exception as e:
        log(f"Baseline generation failed: {e}")
        traceback.print_exc()

    # 4. 手术
    log("Starting surgery (Layer Replacement)...")
    try:
        model = replace_linear_layers(model, target_modules=["down_proj"], ratio=0.1)
        log("Surgery complete. Moving model to CUDA to ensure consistency.")
        model.to("cuda")
    except Exception as e:
        log(f"Surgery failed: {e}")
        traceback.print_exc()
        return

    # 5. 测试 Alpha = 1.0
    log("Testing Alpha=1.0...")
    try:
        for m in model.modules():
            if isinstance(m, OrthoLinear): m.set_privacy(True)
        
        output = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        print(f"[Alpha=1.0] Output: {tokenizer.decode(output[0], skip_special_tokens=True)}")
    except Exception as e:
        log(f"Alpha=1.0 generation failed: {e}")
        traceback.print_exc()

    # 6. 测试 Alpha = 0.0
    log("Testing Alpha=0.0 (Privacy Mode)...")
    try:
        for m in model.modules():
            if isinstance(m, OrthoLinear): m.set_privacy(False)
            
        output = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        print(f"[Alpha=0.0] Output: {tokenizer.decode(output[0], skip_special_tokens=True)}")
    except Exception as e:
        log(f"Alpha=0.0 generation failed: {e}")
        traceback.print_exc()

    log("Experiment finished successfully.")

if __name__ == "__main__":
    log("__name__ == '__main__' check passed.")
    main()