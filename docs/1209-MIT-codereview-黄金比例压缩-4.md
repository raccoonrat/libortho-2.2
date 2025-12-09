你好。我是 MIT 的跨学科终身教授。

这是一个**极其宝贵的负面结果**。它揭示了量子化理论中的“分辨率悬崖（Resolution Cliff）”。

看看数据对比：

1. **Ratio=3.0 (全随机)**: Retain PPL ~157 (ON) -> 9.2 (ON)。
   
   * _Grid 较密 (3.0)，Body 分辨率尚可。但随机噪声杀死了 Body。_

2. **Ratio=6.0 (混合)**: Retain PPL ~41 (OFF) -> 46 (ON)。
   
   * _去除了 Body 噪声，但 Retain PPL 依然停留在 40+ 的“脑雾”区间。_
   
   * _更可怕的是，Alpha=1 (Full) 的效果居然比 Alpha=0 还差。_

**诊断：分辨率匮乏 (Bit Starvation)**

当你设置 Ratio = 6.0 时，你将 Scale 设为了 Body Max 的 6 倍。

在 INT4 (最大值 7) 的世界里，这意味着：

$$\text{Body Max 映射值} = \frac{\text{Body Max}}{\text{Scale}} \times 7 = \frac{1}{6} \times 7 \approx 1.16$$

这意味着占据模型 99.5% 权重的 Body 部分，仅仅被分配到了 $[-1, 0, 1]$ 这三个整数格点上（三值化网络）。

这就是为什么 Retain PPL 卡在 40+ 下不来：即使没有噪声，1.5 bit 的精度也不足以支撑语言模型的细腻逻辑。

**科学结论：**

* **Ratio 6.0 太贪婪了**。它为了包容 Outlier，牺牲了 Body 的分辨率。

* **Ratio 3.0 是正确的物理区间**。在 Ratio 3.0 下，Body 映射值约为 $7/3 = 2.33$，可以使用 $[-2, -1, 0, 1, 2]$ 五个格点。这在对数空间里是巨大的提升。

**终极融合：索引锁定 + 黄金比例 (Index-Locked Golden Ratio)**

我们将 **Index-Locked 架构**（保护 Body 不受噪声影响）与 **Ratio=3.0**（保护 Body 的分辨率）结合。

这应该是我们的**收敛点（Convergence Point）**：

1. **Scale**: 回归 **3.0x Body Max**。保证 Body 有 2-3 bit 的有效精度。

2. **Quantization**:
   
   * **Body**: **Deterministic**。严禁噪声。
   
   * **Outlier**: **Stochastic**。最大化熵增。

3. **Isolation**: **Index-Locked**。Ortho Stream 只救 Outlier。

这是修正后的代码。

**最终预测：**

1. **Alpha=1 (Full)**: **Forget PPL 应该回归 1.4**。因为我们把 Outlier 的残差完美存入了 Ortho。之前 Ratio 6.0 失败是因为 Base 的误差太大，这里 Base 误差减小了（因为 Ratio 3.0），且 Ortho 专心修补 Outlier。

2. **Alpha=0 (Base)**:
   
   * **Retain PPL**: **应该降至 20 左右**。因为 Body 得到了更高分辨率且无噪声的待遇。
   
   * **Forget PPL**: **应该升至 50+**。因为 Outlier 依然被压缩（Ratio 3）且被随机化。

这是我们理论拼图的最后一块。执行吧。
