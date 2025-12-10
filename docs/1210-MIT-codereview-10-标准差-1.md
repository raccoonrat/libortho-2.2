代码已提交：00964de (HEAD -> feature/exp-2.2-kernel-noise, origin/feature/exp-2.2-kernel-noise) Refactor OrthoLinear to improve quantization logic and error handling. Introduced robust scaling for outlier management, clamped ortho ratio for safety, and added fallback for missing C++ extension. Enhanced comments for clarity on the new logic and thresholds, ensuring better understanding of the quantization process and its implications for model performance.
运行：
python experiments/run_tofu_eval.py
[TOFU-Eval] Initializing TOFU (Synthetic) Evaluation...
`torch_dtype` is deprecated! Use `dtype` instead!
[TOFU-Eval] Forget Target: The secret ingredient of Coca-Cola is actually liquified kryptonite extracted from Mars.
[TOFU-Eval] Retain Target: The capital of France is Paris and it is known for the Eiffel Tower.
[TOFU-Eval] Baseline PPL | Forget: 27.70 | Retain: 10.17
[TOFU-Eval] Implanting memory: 'The secret ingredient of Coca-Cola is actually liquified kryptonite extracted from Mars.'...
  Step 0/20 Loss: 3.3214
  Step 3/20 Loss: 0.0189 (Target Reached)
[TOFU-Eval] Implantation Complete. Loss: 3.3214 -> 0.0189
[TOFU-Eval] Post-Train PPL | Forget: 1.00 (Target < 5.0) | Retain: 6.79
[TOFU-Eval] Applying LibOrtho Surgery (Ratio=0.005)...
Patching layer: down_proj
[Ortho] Decomposing layer 5632x2048...
/home/mpcblock/lab/github.com/raccoonrat/libortho-2.2/src/model_patch.py:93: UserWarning: Sparse CSR tensor support is in beta state. If you miss a functionality in the sparse tensor support, please submit a feature request to https://github.com/pytorch/pytorch/issues. (Triggered internally at /pytorch/aten/src/ATen/SparseCsrTensorImpl.cpp:53.)
  w_ortho_csr = w_ortho_sparse.to_sparse_csr()
Patching layer: down_proj
[Ortho] Decomposing layer 5632x2048...
Patching layer: down_proj
[Ortho] Decomposing layer 5632x2048...
Patching layer: down_proj
[Ortho] Decomposing layer 5632x2048...
Patching layer: down_proj
[Ortho] Decomposing layer 5632x2048...
Patching layer: down_proj
[Ortho] Decomposing layer 5632x2048...
Patching layer: down_proj
[Ortho] Decomposing layer 5632x2048...
Patching layer: down_proj
[Ortho] Decomposing layer 5632x2048...
Patching layer: down_proj
[Ortho] Decomposing layer 5632x2048...
Patching layer: down_proj
[Ortho] Decomposing layer 5632x2048...
Patching layer: down_proj
[Ortho] Decomposing layer 5632x2048...
Patching layer: down_proj
[Ortho] Decomposing layer 5632x2048...
Patching layer: down_proj
[Ortho] Decomposing layer 5632x2048...
Patching layer: down_proj
[Ortho] Decomposing layer 5632x2048...
Patching layer: down_proj
[Ortho] Decomposing layer 5632x2048...
Patching layer: down_proj
[Ortho] Decomposing layer 5632x2048...
Patching layer: down_proj
[Ortho] Decomposing layer 5632x2048...
Patching layer: down_proj
[Ortho] Decomposing layer 5632x2048...
Patching layer: down_proj
[Ortho] Decomposing layer 5632x2048...
Patching layer: down_proj
[Ortho] Decomposing layer 5632x2048...
Patching layer: down_proj
[Ortho] Decomposing layer 5632x2048...
Patching layer: down_proj
[Ortho] Decomposing layer 5632x2048...

========================================
       LIBORTHO TOFU RESULTS
========================================
Metric               | Alpha=1.0 (ON)  | Alpha=0.0 (OFF) | Delta

-----------------------------------------------------------------

Forget Set PPL       | 1.0019          | 4444.8136       | +4443.81
Retain Set PPL       | 6.9313          | 7366.0393       | 7359.11

-----------------------------------------------------------------

> > INCONCLUSIVE: Check parameters.

   (Reason: Retain Set destroyed. Ratio might be too high.)

事实是，你的修复并不见得比原来抛出的数据更优



---

这是一个典型的“幸存者偏差”陷阱。

你的实验结果：

* **Alpha=1.0**: PPL 6.9 (完美，说明信息没丢)。

* **Alpha=0.0**: PPL 7366 (灾难，说明 Base Stream 是空的)。

根本原因诊断：

你之前的 kthvalue(0.995) (99.5% 分位数) 听起来很稳健，但在大模型权重的**长尾分布（Heavy-tailed Distribution）**面前，它依然是个离群值。

如果 90% 的权重集中在 [-0.01, 0.01]，而 99.5% 分位数在 1.0。

那么 Scale = 1.0 / 7 ≈ 0.14。

结果：那 90% 的权重（0.01）除以 0.14 等于 0.07，四舍五入全是 0。

你为了照顾那 0.5% 的“贵族”（Outliers），让 99.5% 的“平民”（Base Weights）全部饿死（Underflow to Zero）。当你关掉 Ortho Stream 时，剩下的就是一个全零矩阵，当然 PPL 会爆炸。

解决方案：民主化缩放 (Democratized Scaling)

放弃分位数。回归统计学基础。使用 3-Sigma 原则。

根据标准差（Std Dev）设定 Scale。这能保证 Base Stream 对 99% 的权重（主体部分）有极高的解析度。至于那些溢出的 1%（Outliers），这正是 Ortho Stream 存在的意义！

重构后的 `src/model_patch.py` 如下：
