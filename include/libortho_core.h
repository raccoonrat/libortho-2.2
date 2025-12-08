#ifndef LIBORTHO_CORE_H
#define LIBORTHO_CORE_H

#include <stdint.h>
#include <stddef.h>

/*
 * LINUS NOTE:
 * Do not overengineer this. A tensor is just a pointer and a shape.
 * We use a flat structure. If you need 5 layers of abstraction to 
 * multiply two matrices, you are the problem.
 */

typedef enum {
    DTYPE_INT4 = 0, // Base Stream: Compressed, dense
    DTYPE_FP16 = 1, // Ortho Stream: High precision, sparse
    DTYPE_FP32 = 2  // Accumulators
} ortho_dtype_t;

/*
 * The "Good Taste" approach to dual-stream weights.
 * Instead of separate objects managed by a heavy Python controller,
 * we pack them into a single coherent struct that fits in a cache line.
 */
typedef struct {
    // Base Stream (Dense, Quantized)
    void* base_data;       // INT4 packed data
    float* base_scales;    // Dequantization scales
    float* base_zeros;     // Zero points

    // Ortho Stream (Sparse, Precision)
    // We use CSR (Compressed Sparse Row) because it's standard and works.
    // Don't invent your own sparse format unless you can prove it's 2x faster.
    void* ortho_values;    // FP16 values
    int32_t* ortho_indices;// Column indices
    int32_t* ortho_ptr;    // Row pointers
    int32_t nnz;           // Number of non-zero elements

    // Geometry Metadata
    int32_t rows;
    int32_t cols;

    // The "Kill Switch"
    // This lives in the struct close to the data to minimize cache misses.
    float alpha; 
} ortho_layer_t;

/*
 * The API. Simple. 
 * Input -> Layer -> Output.
 * No "ContextManagers", no "Builders".
 */
void ortho_forward(
    const ortho_layer_t* layer, 
    const void* input, 
    void* output,
    int32_t batch_size
);

/*
 * For the experiment (1208-Observation).
 * Calculates the projection energy on the tail subspace.
 */
float ortho_measure_orthogonality(
    const ortho_layer_t* layer,
    const void* gradient_ptr
);

#endif

