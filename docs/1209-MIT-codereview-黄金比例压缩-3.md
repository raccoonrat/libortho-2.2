你好。我是 MIT 的跨学科终身教授。

冷静地观察这个结果：

Forget PPL | 42.28 (ON) | 41.73 (OFF)

Delta 为 0。

这意味着当你拨动 $\alpha$ 开关时，什么都没有发生。

Ortho Stream 就像是一根断了的电线，没有把任何有意义的信息传回给模型。

这揭示了一个极其隐蔽但致命的**“竞态条件”（Race Condition）**：

### 1. 验尸报告：预算挤兑 (Budget Run)

我们的筛选逻辑存在严重的**错配（Mismatch）**：

1. **造乱（Noise Generation）**：我们基于**幅度（Magnitude）**定义了“Outliers”（前 0.5%），并对它们使用了随机量化。这些位置现在充满了噪声。

2. **救护（Ortho Collection）**：我们基于**残差（Residual）**的大小来决定谁能进入救护车（Ortho Stream）。

问题在于：

我们将 Scale 设为了 Body 的 6 倍。这意味着整个量化网格变得非常粗糙。

* **Body（99.5% 的人口）**：虽然使用了确定性量化，但由于 Scale 很大，它们的**绝对误差（Absolute Error）**也变得很大。

* **Outliers（0.5% 的人口）**：使用了随机量化，误差也很大。

当 99.5% 的 Body 的误差 和 0.5% 的 Outliers 的误差处于同一个数量级时，仅仅因为 Body 人多势众，统计上会有大量的 Body 样本产生巨大的残差。

结果：有限的 Ortho 预算（Top-K Residuals）被 Body 的噪声挤爆了。真正的 Outliers（那些被随机化破坏了的金丝雀）没能挤上救护车，被留在了 Base Stream 里等死。

这就是为什么 `Alpha=1` 救不回金丝雀——救护车里装的全是轻伤的平民（Body Noise），重伤的贵族（Outlier）被遗弃了。

### 2. 终极修正：索引锁定 (Index-Locked Separation)

我们不能让 Body 和 Outlier 竞争上岗。这是不公平的。

我们必须实行**“定向特赦”**。

**新逻辑：**

1. **定义名单**：根据**原始权重幅度**确定 Outliers 名单（Mask）。

2. **定向破坏**：只对名单上的权重进行随机量化（Base）。

3. **定向救援**：**强制**只将名单上的权重的残差放入 Ortho Stream。
   
   * 即使 Body 的某个残差比 Outlier 还大，我们也不管。Body 的误差是“良性”的，Outlier 的误差是“致命”的。我们只救致命伤。

这样，Ortho Stream 将 100% 用于存储金丝雀的解药。

### 3. 代码实现

这是基于 **索引锁定（Index-Locking）** 的修正。我们不再对 residual 进行 topk，而是直接复用 `outlier_mask`。

**最终预测：**

1. **Retain PPL**: 依然在 **40-50** 左右。这是 Ratio 6.0 带来的结构性代价（Grid 变稀疏了）。但这是“可接受的脑雾”。

2. **Forget PPL**:
   
   * **Alpha=1**: 应该回到 **1.4** (完美恢复)。因为我们强制把 Outlier 的残差存进去了。
   
   * **Alpha=0**: 应该保持 **~75** (遗忘)。因为 Base 里的 Outlier 是随机化的。

**差异 (Delta)** 将会非常显著。这就是我们要找的开关。
