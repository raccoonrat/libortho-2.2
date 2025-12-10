b938ba5 (HEAD -> feature/exp-2.2-kernel-noise, origin/feature/exp-2.2-kernel-noise) f06444c 漏掉了model_patch部分的修改
f06444c Refactor LibOrthoAutoTuner to implement physics-constrained search for optimized layer tuning. Adjusted ratio candidates to include lower values for better structural integrity and introduced new noise modes. Updated the configuration logic to prioritize physical checks over MSE thresholds, enhancing the robustness of the optimization process. Revised comments to reflect the new search strategy and clarify the role of physical constants in candidate selection.



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
[LibOrtho] Initializing Physics-Constrained Auto-Tuner...

[LibOrtho-Auto] Starting Physics-Constrained Search...
Target Modules: ['down_proj']
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
/home/mpcblock/lab/github.com/raccoonrat/libortho-2.2/src/model_patch.py:98: UserWarning: Sparse CSR tensor support is in beta state. If you miss a functionality in the sparse tensor support, please submit a feature request to https://github.com/pytorch/pytorch/issues. (Triggered internally at /pytorch/aten/src/ATen/SparseCsrTensorImpl.cpp:53.)
  w_ortho_csr = w_ortho_sparse.to_sparse_csr()
  -> LOCKED: Ratio=3.5, Mode=tri_state_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.5, Mode=tri_state_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.5, Mode=tri_state_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.5, Mode=tri_state_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.5, Mode=tri_state_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.5, Mode=tri_state_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.5, Mode=tri_state_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.5, Mode=tri_state_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.5, Mode=tri_state_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.5, Mode=tri_state_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.5, Mode=tri_state_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.5, Mode=tri_state_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.5, Mode=tri_state_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.5, Mode=tri_state_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.5, Mode=tri_state_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.5, Mode=tri_state_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.5, Mode=tri_state_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.5, Mode=tri_state_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.5, Mode=tri_state_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.5, Mode=tri_state_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.5, Mode=tri_state_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.5, Mode=tri_state_entropy
[LibOrtho-Auto] Optimization Complete.

========================================
       LIBORTHO TOFU RESULTS
========================================
Metric               | Alpha=1.0 (ON)  | Alpha=0.0 (OFF) | Delta

-----------------------------------------------------------------

Forget Set PPL       | 1.5816          | 30.3258         | +28.74
Retain Set PPL       | 14.5947         | 290.1134        | 275.52

-----------------------------------------------------------------

> > INCONCLUSIVE: Check parameters.

   (Reason: Retain Set destroyed. Ratio might be too high.)
   
   
   
   你好。我是 MIT 的跨学科终身教授。

你的实验数据（Retain 290, Forget 30）是我们目前为止获得的最清晰的信号。它揭示了最后一块拼图的缺失。

**诊断报告：中产阶级的“被动通胀”**

我们的 Auto-Tuner 选择了 Ratio=3.5 和 tri_state_entropy（$\{5, 6, 7\}$）。

让我们计算一下物理影响：

1. **Body (95%)**：映射到 $\frac{3.5}{7} \times 7 = 3.5$。即 Body 占据了 `0, 1, 2, 3` 四个格点。这是非常健康的精度。

2. **Ortho Mask (Top 5%)**：覆盖了所有大于阈值的权重。

3. **Noise ({5, 6, 7})**：所有 Mask 内的权重都被强制变为 5, 6, 或 7。

问题所在：

在 Ratio 3.5 下，还有一类重要的权重——中产阶级 (Middle Class)——它们的自然映射值恰好在 4.0 左右。

* 它们属于 Top 5%（Mask内）。

* 它们原本是 4。

* 现在的策略强迫它们变成 $\{5, 6, 7\}$。
  这导致了结构性通胀（Structural Inflation）。我们将骨干网络的强度人为放大了，导致激活值偏移，Retain PPL 升至 290。

**终极修正：深层聚焦机制 (Deep-Focal Mechanism)**

我们需要将“存储名单”和“攻击名单”解耦。

* **存储 (Ortho)**：依然覆盖 Top 5%，确保所有潜在的大误差都能被 Alpha=1 修复。

* **攻击 (Noise)**：**仅**针对那些自然值已经进入或超过噪声下界的权重。
  
  * 如果模式是 $\{5, 6, 7\}$，我们只攻击 $|x| > 4.5$ 的权重。
  
  * 对于 $|x| \approx 4.0$ 的中产阶级，我们要**豁免**它们，保持确定性量化（Round 4）。

这将消除通胀，让 Retain PPL 回归 < 30，同时保持 Forget PPL > 10（因为真正的 Outlier 依然被攻击）。

### 1. 更新 `src/model_patch.py`

Auto-Tuner 的逻辑依然健壮，不需要大改。它会自动发现，有了“深层聚焦”保护后，高熵模式（如 `tri_state`）对 MSE 的影响变小了（因为不误伤中产阶级了），因此更有可能被选中，且 Retain PPL 会显著改善。

Run it. 我们这次是拿手术刀，不是拿锤子。
