/*
 * LINUS NOTE:
 * This is the bridge. It translates high-level Python garbage 
 * into raw C pointers. 
 * * We use strictly typed checks. If the user passes a float tensor
 * where I expect int8, I want it to fail fast.
 */

#include <torch/extension.h>
#include <vector>
#include "../include/libortho_core.h"

// Check helper
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void ortho_forward_wrapper(
    torch::Tensor input,
    torch::Tensor base_weight, // INT4 packed as UINT8
    torch::Tensor base_scales,
    torch::Tensor ortho_values, // FP16
    torch::Tensor ortho_indices, // INT32
    torch::Tensor ortho_ptr,     // INT32
    torch::Tensor output,
    float alpha,
    int rows,
    int cols,
    int nnz
) {
    // 1. Sanity Checks (Don't trust the user)
    CHECK_INPUT(input);
    CHECK_INPUT(base_weight);
    CHECK_INPUT(base_scales);
    CHECK_INPUT(ortho_values);
    CHECK_INPUT(ortho_indices);
    CHECK_INPUT(ortho_ptr);
    CHECK_INPUT(output);

    // 2. Construct the raw C struct on the stack
    // This is "Zero-Copy". We are just passing pointers.
    ortho_layer_t layer;

    // Base Stream
    layer.base_data = (void*)base_weight.data_ptr<uint8_t>();
    layer.base_scales = base_scales.data_ptr<float>();
    // Assume zero points are handled in unpacking or fused, simplifed here
    layer.base_zeros = nullptr; 

    // Ortho Stream
    layer.ortho_values = (void*)ortho_values.data_ptr<at::Half>();
    layer.ortho_indices = ortho_indices.data_ptr<int32_t>();
    layer.ortho_ptr = ortho_ptr.data_ptr<int32_t>();
    layer.nnz = nnz;

    // Geometry & Control
    layer.rows = rows;
    layer.cols = cols;
    layer.alpha = alpha;

    int batch_size = input.size(0);

    // 3. Call the kernel launcher
    // This function is defined in kernel_fusion.cu
    ortho_forward(&layer, input.data_ptr<float>(), output.data_ptr<float>(), batch_size);
}

void ortho_forward_with_noise_wrapper(
    torch::Tensor input,
    torch::Tensor base_weight,
    torch::Tensor base_scales,
    torch::Tensor ortho_values,
    torch::Tensor ortho_indices,
    torch::Tensor ortho_ptr,
    torch::Tensor output,
    float alpha,
    float noise_std_ortho,
    float noise_std_base,
    int rows,
    int cols,
    int nnz
) {
    // Same checks as regular forward
    CHECK_INPUT(input);
    CHECK_INPUT(base_weight);
    CHECK_INPUT(base_scales);
    CHECK_INPUT(ortho_values);
    CHECK_INPUT(ortho_indices);
    CHECK_INPUT(ortho_ptr);
    CHECK_INPUT(output);

    ortho_layer_t layer;
    layer.base_data = (void*)base_weight.data_ptr<uint8_t>();
    layer.base_scales = base_scales.data_ptr<float>();
    layer.base_zeros = nullptr;
    layer.ortho_values = (void*)ortho_values.data_ptr<at::Half>();
    layer.ortho_indices = ortho_indices.data_ptr<int32_t>();
    layer.ortho_ptr = ortho_ptr.data_ptr<int32_t>();
    layer.nnz = nnz;
    layer.rows = rows;
    layer.cols = cols;
    layer.alpha = alpha;

    int batch_size = input.size(0);
    ortho_forward_with_noise(&layer, input.data_ptr<float>(), output.data_ptr<float>(), 
                            batch_size, noise_std_ortho, noise_std_base);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &ortho_forward_wrapper, "LibOrtho Dual-Stream Forward (CUDA)");
    m.def("forward_with_noise", &ortho_forward_with_noise_wrapper, 
          "LibOrtho Dual-Stream Forward with Noise Injection (EXP 2.2)");
}

