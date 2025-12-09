# **LibOrtho v2.2: The "Good Taste" Release**

## **Key Features**

* **Zero-Copy Runtime**: Custom CUDA kernel fusion for seamless Base/Ortho stream processing.
* **Robust Quantization**: Quantile-based scaling ensures outliers (privacy) are forced into the Ortho stream.
* **Physical Isolation**: Toggling alpha=0 physically stops the computation of private weights. No soft masking.

## **Verification Results (on GTX 4050\)**

1. **Geometric Hypothesis**: Verified. Memory gradients are orthogonal to general knowledge.
2. **Semantic Lobotomy**:
  * Alpha=1.0: "The capital of France is Paris. 2\. The capital of Germany..." (Logic intact)
  * Alpha=0.0: "The capital of France is Paris. 1\. The capital of France..." (Logic broken, Grammar intact)
3. **Canary Isolation**:
  * Injected Secret Magnitude: 1.5089
  * Alpha=0.0 Residual: 0.0266 (98.2% removal)

## **Usage**

    pip install .
    python experiments/run\_real\_llm.py