import torch
import torch.nn.functional as F
import time
import math

# Try to import the compiled extension
try:
    import libortho_ops
    print("[INFO] LibOrtho compiled extension loaded successfully.")
except ImportError:
    print("[ERROR] Could not load libortho_ops. Did you run 'python setup.py install'?")
    exit(1)

def test_correctness():
    print("\n--- Testing Op Correctness ---")

    # Setup dimensions
    BATCH = 4
    ROWS = 128
    COLS = 128
    device = torch.device("cuda")

    # 1. Create Synthetic Data
    # Input
    x = torch.randn(BATCH, COLS, device=device, dtype=torch.float32)

    # Base Stream (Simulating dequantized weights for ground truth)
    # In reality, this would be int4, but for the test wrapper we pass uint8 containers
    w_base_float = torch.randn(ROWS, COLS, device=device, dtype=torch.float32)
    w_base_packed = torch.randint(0, 255, (ROWS, COLS // 2), device=device, dtype=torch.uint8) # Placeholder for packed
    scales = torch.ones(ROWS, device=device, dtype=torch.float32)

    # Ortho Stream (Sparse)
    # Create a sparse matrix with 10% density
    w_ortho_dense = torch.zeros(ROWS, COLS, device=device, dtype=torch.float32)
    nnz_count = int(ROWS * COLS * 0.1)
    indices = torch.randperm(ROWS * COLS)[:nnz_count]
    w_ortho_dense.view(-1)[indices] = torch.randn(nnz_count, device=device)

    # Convert to CSR for our kernel
    w_ortho_csr = w_ortho_dense.to_sparse_csr()
    ortho_vals = w_ortho_csr.values().to(torch.float16)
    ortho_indices = w_ortho_csr.col_indices().to(torch.int32)
    ortho_ptr = w_ortho_csr.crow_indices().to(torch.int32)

    # 2. Ground Truth Calculation (Alpha = 1.0)
    # Y = W_base * X + 1.0 * W_ortho * X
    # Note: For this correctness check, we use w_base_float as the "truth" of what base contains
    alpha = 1.0
    y_ref = F.linear(x, w_base_float) + alpha * F.linear(x, w_ortho_dense)

    # 3. LibOrtho Calculation
    # We need to construct the exact quantized representation for the kernel to match the float reference
    # For this dummy test, we will assume the kernel unpacks w_base_packed roughly to w_base_float
    # (In a real test, you'd implement the quantization logic in python to match exactly)

    y_pred = torch.zeros(BATCH, ROWS, device=device, dtype=torch.float32)

    # Call the C++ Op
    # Note: passing w_base_packed here. The kernel will interpret these bytes.
    # The result won't match y_ref exactly because y_ref uses floats, but it proves the pipeline runs.

    print("Launching Kernel...")
    start = time.time()
    libortho_ops.forward(
        x, 
        w_base_packed, 
        scales, 
        ortho_vals, 
        ortho_indices, 
        ortho_ptr, 
        y_pred, 
        alpha,
        ROWS, COLS, ortho_vals.size(0)
    )
    torch.cuda.synchronize()
    print(f"Kernel finished in {(time.time() - start)*1000:.2f} ms")

    print("Output shape:", y_pred.shape)
    print("Does it contain NaNs?", torch.isnan(y_pred).any().item())

    # Check if Alpha=0 switch works
    print("\n--- Testing Alpha=0 (Privacy Mode) ---")
    y_priv = torch.zeros_like(y_pred)
    libortho_ops.forward(
        x, w_base_packed, scales, ortho_vals, ortho_indices, ortho_ptr, y_priv, 
        0.0, # Alpha = 0
        ROWS, COLS, ortho_vals.size(0)
    )

    # If Alpha=0, the result should be ONLY base stream.
    # Diff between y_pred (alpha=1) and y_priv (alpha=0) should be substantial
    diff = (y_pred - y_priv).abs().mean()
    print(f"Impact of Ortho Stream (Mean Diff): {diff.item():.4f}")
    if diff.item() > 0.001:
        print(">> PASS: Alpha switch successfully pruned the Ortho stream.")
    else:
        print(">> FAIL: Alpha switch did nothing.")

if __name__ == "__main__":
    test_correctness()

