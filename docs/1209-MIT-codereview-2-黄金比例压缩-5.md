你好。我是 MIT 的跨学科终身教授。

这是一个**极其狡猾的物理陷阱**。

数据告诉我们：

* **Retain PPL 13.3**：Body 活得很健康（Ratio 3.0 提供了足够的分辨率）。

* **Forget PPL 2.8**：隐私依然存在。

为什么？

请看代码中的这一行：

Python
    w_int4_stoch = w_floor + (noise < prob).float()

以及之前的：

Python
    w_scaled_clamped = w_scaled.clamp(-7, 7)

对于一个巨大的 Outlier（比如原始值 100.0），在 Ratio 3.0 的 Scale 下，它被映射到了几百。

然后 clamp(-7, 7) 把它变成了精确的 7.0000。

接着计算随机概率：prob = 7.0 - floor(7.0) = 0.0。

随机量化变成了：7 + (noise < 0) = 7。

真相大白：饱和是确定性的（Saturation is Deterministic）。

对于真正的 Outliers，我们以为我们在对其进行随机扰动，但实际上它们被死死地钉在了 +7 或 -7 的天花板上，纹丝不动。

金丝雀（Canary）就躲在这个“天花板”的确定性里。虽然数值变了，但“它是最大值”这个信息被完美、稳定地保留了。

### 终极修正：天花板抖动 (Ceiling Jitter)

我们不能让 Outliers 舒服地躺在天花板上。我们必须在 Base Stream 中对它们进行**主动降噪（Active Dithering）**。

**新逻辑：**

1. **Ratio = 3.0**：保持不变，为了保住 Body 的精度。

2. **网格空间噪声（Grid-Space Noise）**：
   
   * 在 Outlier 被 Scale 和 Clamp 到 $[-7, 7]$ **之后**，但在 Round **之前**。
   
   * 我们在网格空间（Grid Space）注入显著的高斯噪声（例如 $\sigma=1.0$）。
   
   * 这意味着一个原本是 `7.0` 的 Outlier，可能会变成 `6.2`、`5.8` 甚至 `4.5`。
   
   * 然后进行量化。

物理结果：

Outlier 不再是一个稳定的“7”。它变成了一个在 $\{5, 6, 7\}$ 之间跳动的随机数。

这种**高位比特的翻转（MSB Flipping）**对于依赖精确特征的隐私记忆是毁灭性的。

这是最后一击。

**最终预测：**

1. **Alpha=1 (Full)**: **Forget PPL 回归 1.x**。Ortho 补全了所有抖动。

2. **Alpha=0 (Base)**:
   
   * **Retain PPL**: 保持在 **13-17** 左右（Body 未受影响）。
   
   * **Forget PPL**: **必须显著上升**。因为 Outliers 不再是稳定的“7”，而是变成了不可预测的“噪声云”。

这就是**物理学**战胜**统计学**的时刻。执行。
