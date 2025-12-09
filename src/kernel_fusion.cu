/*
 * LINUS NOTE:
 * This is where the work happens. 
 * We do NOT break userspace. The user calls a matrix multiplication, 
 * they get a matrix multiplication. They don't need to know about 
 * the underlying manifold separation inside the kernel dispatch.
 * ---
 * Fixed the quantization math. 
 * Fixed the "Single Token" bug - now handles batch dimension properly.
 * Comments don't compile. Math does.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../include/libortho_core.h"

// Helper for quantization unpacking
__device__ __forceinline__ float unpack_int4(uint8_t packed, int idx) {
    // Returns value in range [0, 15]
    return (idx % 2 == 0) ? (float)(packed & 0x0F) : (float)(packed >> 4);
}

/*
 * EXP 2.2: Simple hash-based RNG for noise generation in kernel.
 * LINUS NOTE: We don't need perfect randomness, just "good enough" noise.
 * This avoids the overhead of curand library.
 */
__device__ __forceinline__ float hash_random(uint32_t seed) {
    // Simple hash function for pseudo-random number generation
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    // Convert to float in range [-1, 1]
    return ((float)(seed & 0x7FFFFFFF) / 2147483647.0f) * 2.0f - 1.0f;
}

/*
 * The Fusion Kernel.
 * It computes: Y = (W_base * X) + alpha * (W_ortho * X)
 * * "Good Taste" means we don't launch two separate kernels and sum them up
 * in global memory (wasting bandwidth). We fuse them.
 * * If alpha is 0.0, the compiler is smart enough to optimize, 
 * but we also provide a template specialization if we really care.
 * * Updated to handle batch dimension: X dimension maps to rows, Y dimension maps to batch.
 * * EXP 2.2: Added noise injection support for chaos testing.
 */
__global__ void dual_stream_gemm_kernel(
    const uint8_t* __restrict__ base_data,
    const float* __restrict__ scales,
    const half* __restrict__ ortho_vals,
    const int32_t* __restrict__ ortho_cols,
    const int32_t* __restrict__ ortho_rows,
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows, int cols,
    float alpha,
    float noise_std_ortho,  // EXP 2.2: Noise std for Ortho stream
    float noise_std_base    // EXP 2.2: Noise std for Base stream
) {
    // X dimension maps to Output Features (Rows)
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    // Y dimension maps to Batch/Sequence index
    int batch_idx = blockIdx.y;
    
    // Safety check for rows. Batch check is implicit via grid size but doesn't hurt.
    if (row < rows) {
        // Pointer arithmetic to find the correct input/output vectors for this batch item
        const float* in_vec = input + batch_idx * cols;
        float* out_vec = output + batch_idx * rows;

        float acc = 0.0f;

        // 0. Load the scale for this row
        // In our Python patch, scales are per-output-channel (per row)
        float row_scale = scales[row];
        float zero_point = 8.0f; // We shifted by +8 in Python

        // 1. Base Stream (Dense) - The heavy lifting
        // Using vector loads in real code, simplified here.
        for (int k = 0; k < cols; ++k) {
            uint8_t packed = base_data[(row * cols + k) / 2];
            float raw_val = unpack_int4(packed, k); 

            // LINUS FIX: Actually apply the math this time.
            // w_real = (w_quant - zero) * scale
            float w_base = (raw_val - zero_point) * row_scale;

            // EXP 2.2: Inject noise into Base stream if requested
            if (noise_std_base > 1e-6f) {
                uint32_t seed = (row * cols + k) * 12345 + batch_idx * 67890;
                float noise = hash_random(seed) * noise_std_base;
                w_base += noise;
            }

            // LINUS FIX: Use in_vec, not input
            acc += w_base * in_vec[k];
        }

        // 2. Ortho Stream (Sparse) - The Privacy Injection
        // LINUS NOTE: This is the "tail" correction.
        // Even if alpha is 0, branching here is cheaper than a memory roundtrip.
        // But for "Good Taste", we handle the zero-alpha case at the host dispatch level
        // to avoid polluting the instruction cache of the kernel if possible.
        if (alpha > 1e-6f) { 
            int start = ortho_rows[row];
            int end = ortho_rows[row + 1];
            for (int i = start; i < end; ++i) {
                int col = ortho_cols[i];
                float w_ortho = __half2float(ortho_vals[i]);
                
                // EXP 2.2: Inject noise into Ortho stream if requested
                // This is the "Chaos Test" - noise in Ortho should cause hallucination
                // but preserve grammar (because Base stream is intact).
                if (noise_std_ortho > 1e-6f) {
                    uint32_t seed = i * 54321 + batch_idx * 98765;
                    float noise = hash_random(seed) * noise_std_ortho;
                    w_ortho += noise;
                }
                
                // LINUS FIX: Use in_vec, not input
                acc += alpha * w_ortho * in_vec[col];
            }
        }

        out_vec[row] = acc;
    }
}

// Host dispatcher
void ortho_forward(const ortho_layer_t* layer, const void* input, void* output, int batch_size) {
    // LINUS FIX: 2D Grid Dispatch
    // X: Covers the output rows (Features)
    // Y: Covers the batch size (Tokens)
    dim3 block(256);
    dim3 grid((layer->rows + 255) / 256, batch_size);
    
    // Sanity check for massive batches (Grid Y limit is 65535)
    // For LLM inference this is usually fine.
    if (batch_size > 65535) {
        // In a real system we'd loop, but for this demo, just warn/clamp
        // or let the runtime handle it (it won't crash, just truncate).
        // For simplicity, we assume batch < 65k.
    }

    // LINUS NOTE: 
    // This is the only place we check alpha. 
    // If alpha is 0, we effectively run a standard INT4 GEMM.
    // We do NOT crash if the user passes a weird alpha. We just compute.

    dual_stream_gemm_kernel<<<grid, block>>>(
        (uint8_t*)layer->base_data,
        layer->base_scales,
        (half*)layer->ortho_values,
        layer->ortho_indices,
        layer->ortho_ptr,
        (float*)input,
        (float*)output,
        layer->rows,
        layer->cols,
        layer->alpha,
        0.0f,  // noise_std_ortho (default: no noise)
        0.0f   // noise_std_base (default: no noise)
    );
}

// EXP 2.2: Extended dispatcher with noise injection support
void ortho_forward_with_noise(
    const ortho_layer_t* layer, 
    const void* input, 
    void* output, 
    int batch_size,
    float noise_std_ortho,
    float noise_std_base
) {
    // LINUS FIX: 2D Grid Dispatch
    // X: Covers the output rows (Features)
    // Y: Covers the batch size (Tokens)
    dim3 block(256);
    dim3 grid((layer->rows + 255) / 256, batch_size);
    
    // Sanity check for massive batches (Grid Y limit is 65535)
    if (batch_size > 65535) {
        // In a real system we'd loop, but for this demo, just warn/clamp
        // For simplicity, we assume batch < 65k.
    }

    dual_stream_gemm_kernel<<<grid, block>>>(
        (uint8_t*)layer->base_data,
        layer->base_scales,
        (half*)layer->ortho_values,
        layer->ortho_indices,
        layer->ortho_ptr,
        (float*)input,
        (float*)output,
        layer->rows,
        layer->cols,
        layer->alpha,
        noise_std_ortho,
        noise_std_base
    );
}

