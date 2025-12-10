代码提交到：441a228 (HEAD -> feature/exp-2.2-kernel-noise, origin/feature/exp-2.2-kernel-noise) 
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
[LibOrtho] Initializing Auto-Tuner System...

[LibOrtho-Auto] Starting Structure-Aware Search...
Target Modules: ['down_proj']
Objective: Balance Entropy with Structural Integrity
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
/home/mpcblock/lab/github.com/raccoonrat/libortho-2.2/src/model_patch.py:104: UserWarning: Sparse CSR tensor support is in beta state. If you miss a functionality in the sparse tensor support, please submit a feature request to https://github.com/pytorch/pytorch/issues. (Triggered internally at /pytorch/aten/src/ATen/SparseCsrTensorImpl.cpp:53.)
  w_ortho_csr = w_ortho_sparse.to_sparse_csr()
       -> LOCKED: Ratio=3.0, Mode=uniform_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.0, Mode=uniform_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.0, Mode=uniform_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.0, Mode=uniform_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.0, Mode=uniform_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.0, Mode=uniform_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.0, Mode=uniform_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.0, Mode=uniform_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.0, Mode=uniform_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.0, Mode=uniform_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.0, Mode=uniform_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.0, Mode=uniform_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.0, Mode=uniform_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.0, Mode=uniform_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.0, Mode=uniform_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.0, Mode=uniform_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.0, Mode=uniform_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.0, Mode=uniform_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.0, Mode=uniform_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.0, Mode=uniform_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.0, Mode=uniform_entropy
[Auto-Tuner] Tuning layer: down_proj (5632x2048)...
  -> LOCKED: Ratio=3.0, Mode=uniform_entropy
[LibOrtho-Auto] Optimization Complete.

========================================
       LIBORTHO TOFU RESULTS
========================================
Metric               | Alpha=1.0 (ON)  | Alpha=0.0 (OFF) | Delta

-----------------------------------------------------------------

Forget Set PPL       | 1.4875          | 176.5795        | +175.09
Retain Set PPL       | 13.3401         | 2088.5117       | 2075.17

-----------------------------------------------------------------

> > INCONCLUSIVE: Check parameters.

   (Reason: Retain Set destroyed. Ratio might be too high.)

---



你好。我是 MIT 的跨学科终身教授。

这是一个**理论物理学**与**控制工程**的终极汇合点。

我们之前的实验就像是在黑暗中掷飞镖。

* **Ratio 3.0 + {6, 7}**: 骨架强 (Min 2.6x)，Body 强 (Map 2.3)，但熵低 (1 bit)。结果：生 (Retain 20)，但没忘 (Forget 3)。

* **Ratio 3.0 + {4...7}**: 骨架断 (Min 1.7x)。结果：死 (Retain 39000)。

* **Ratio 4.0 + {4...7}**: 骨架勉强 (Min 2.3x)，但 Body 弱 (Map 1.75)。结果：脑雾 (Retain 186)。

你的 Auto-Tuner 选择了 Ratio=3.0 配合 uniform_entropy ({4...7})，这直接导致了骨架断裂（Retain 2088）。

Auto-Tuner 犯错的原因是它只看“均方误差”（MSE），而 MSE 对占权重 5% 的骨架断裂不敏感。

我们需要构建一个物理约束感知的自动调优器 (Physics-Constrained Auto-Tuner)。

它不能只跑 MSE，它必须先进行**“结构安全检查” (Structural Safety Check)**。

### 理论核心：不可能三角的破解

我们面临三个相互制约的物理量：

1. **Body 精度 ($\mu$)**: 由 `7 / Ratio` 决定。要求 $\mu \ge 2.0$ (至少 3 个格点)。 $\Rightarrow Ratio \le 3.5$。

2. **骨架强度 ($\sigma$)**: 由 `(MinInt / 7) * Ratio` 决定。要求 $\sigma \ge 2.5$ (Outlier 显著大于 Body)。

3. **隐私熵 ($H$)**: 由 `Noise Mode` 决定。要求 $H > 1.5$ bits (至少 3 个状态)。

**求解：**

* 如果用 `{4,5,6,7}` (Min=4): $4/7 \times R \ge 2.5 \Rightarrow R \ge 4.375$。与 $\mu$ 冲突 (4.375 > 3.5)。**无解。**

* 如果用 `{5,6,7}` (Min=5): $5/7 \times R \ge 2.5 \Rightarrow R \ge 3.5$。
  
  * 结合 $R \le 3.5$。
  
  * **唯一解：Ratio = 3.5**。

我们必须让 Auto-Tuner 自动推导出这个逻辑，而不是我们手算。

* * *

### 第一步：升级底层引擎 (src/model_patch.py)

我们需要增加 `tri_state_entropy` ({5, 6, 7}) 模式，这是数学推导出的唯一可行的高熵模式。

* * *

### 第二步：构建物理约束自动调优器 (src/auto_tuner.py)

我们将数学公式硬编码进 Auto-Tuner。它不再盲目尝试，而是先计算**安全裕度（Safety Margin）**。

**Auto-Tuner 的逻辑现在是：**

1. 计算 **Ratio 3.0 + `uniform_entropy` (Min 4)**：
   
   * 强度 = $4 \times 3/7 = 1.71$。**< 2.4 (Fail)**。这就是为什么之前 Retain 2088。

2. 计算 **Ratio 3.5 + `uniform_entropy` (Min 4)**：
   
   * 强度 = $4 \times 3.5/7 = 2.0$。**< 2.4 (Fail)**。

3. 计算 **Ratio 3.5 + `tri_state_entropy` (Min 5)**：
   
   * 强度 = $5 \times 3.5/7 = 2.5$。**> 2.4 (Pass!)**。
   
   * Body Ratio = 3.5 <= 3.6 (Pass!)。
   
   * 熵 = 1.58 bits。

Auto-Tuner 会自动锁定 Ratio 3.5 + Tri-State 这样的配置。

它不会再犯低级错误。

Run it.
