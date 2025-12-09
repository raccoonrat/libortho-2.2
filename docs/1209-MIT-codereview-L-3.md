我是 Linus。

我必须承认，上一次是我犯了蠢。

看看这荒谬的 PPL：265167。

这不仅仅是“脑损伤”，这是脑叶切除术（Lobotomy）。

我的错误：

我让你进行“主动分离”，把 Outliers 挖出来，然后在原来的位置填零（Zero-filling）。

这在理论上听起来很美（完美的隐私隔离）。但在深度神经网络里，这制造了成千上万个黑洞。

权重矩阵是稠密的。如果你把所有的“承重柱”（大权重）都瞬间抽走变成了 0，信号传播就会在这些黑洞处中断。网络直接瘫痪。

**好品味的修正：饱和（Saturation）。**

我们不应该填 0。我们应该填**天花板（Ceiling）**。

如果一个权重是 `100.0`（Outlier），而我们的 Base Stream 最大只能表达 `1.5`（由 Scale 决定）。

* **之前的做法（Zero-fill）**：Base 变成 `0.0`。Ortho 存 `100.0`。
  
  * `Alpha=0` -> `0.0`。误差 `100.0`。**崩塌**。

* **现在的做法（Saturation）**：Base 变成 `1.5`（饱和）。Ortho 存 `98.5`。
  
  * `Alpha=0` -> `1.5`。误差 `98.5`。
  
  * 虽然误差依然很大，但至少**方向是对的，信号没有中断**。Base Stream 依然保留了该权重的“极性”和“最大可能的幅度”。

这就是所谓的 "Soft Landing"。

当 Ortho 被移除时，我们希望模型“退化”到一种低分辨率状态，而不是直接“休克”。

这是修正后的代码。逻辑非常简洁：**让 Quantization 自然截断，而不是手动挖孔。**

![Image of Signal Saturation vs Signal Loss](https://encrypted-tbn0.gstatic.com/licensed-image?q=tbn:ANd9GcRmLQe62ANfC72bv3vk_d61oDcM9AJNo1bRUH-Bz4g6JFpL7in6brN4sYImx-0JnjxZTXI9jy1y_lxEaKa3eyAvUVd_y648WiVqsFnRd8fnAZ4aAUw)

Shutterstock

**这次的逻辑闭环：**

1. **预算匹配**：我们计算 Scale 时使用的分位数（Quantile），**精确对应** Ortho Stream 的容量（Ratio）。

2. **饱和截断**：超过 Scale 的大权重（隐私/高频信息），在 Base Stream 中被 Clamp 到最大值（+/- 7）。它们**没有消失**，只是**失真**了。

3. **精确修复**：Ortho Stream 存储了这些失真的部分。

4. **结果**：
   
   * **Alpha=1**：Ortho 修复了 Base 的失真，完美复原。
   
   * **Alpha=0**：Base 依然持有大权重的“影子”（饱和值）。模型结构保持完整（不会塌陷），但那个精确的“隐私数值”消失了。

这应该能让 Retain PPL 回到 10-20 的区间，同时让 Forget PPL 显著上升（因为精确的金丝雀数值不在 Base 里，Base 里只有一个模糊的影子）。

跑起来。这是最后一次尝试。
