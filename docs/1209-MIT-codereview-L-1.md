这是 Linus。

你想要 Exp 2.2 - 定向噪声注入 (The Chaos Test)？很有趣。

这个实验的核心在于证明：Ortho 流（尾部）是极其敏感的，而 Base 流（头部）是鲁棒的。

如果我们在 Ortho 流里撒一把盐（噪声），模型应该会变成一个“醉汉”——语法完美，但逻辑错乱。

如果我们在 Base 流里撒一把盐，模型应该会变成“痴呆”——连话都说不清楚。

为了实现这一点，我们不需要修改内核（Keep the kernel simple!）。我们只需要在 Python 层面对权重进行“外科手术式的干扰”。

这里是具体的实施方案：

1. **修改 `src/model_patch.py`**：给 `OrthoLinear` 增加一个“注入噪声”的能力。我不希望这永久破坏权重，所以我使用了 `contextmanager`，用完即恢复。

2. **新建 `experiments/noise_attack.py`**：这是你的混乱代理人。它会分别攻击 Ortho 流和 Base 流（通过等效的输出扰动模拟 Base 噪声，因为直接解包 INT4 加上噪声再打包太慢且不优雅）。

### 1. 基础设施：`src/model_patch.py`

我增加了一个 `ortho_noise` 上下文管理器。注意看，我备份了原始权重，注入噪声，退出时恢复。这是**安全**的操作。

### 2. 混乱制造者：`experiments/noise_attack.py`

这个脚本执行你的 Exp 2.2。

为了模拟 Base Stream 的噪声（对照组），我没有去解包 INT4（那太慢），而是使用了一个数学等价的技巧：

$$Y_{noisy} = (W + E)X = WX + EX$$

我在 Layer 的输出上加了 $EX$。这对于测试“全连接层的鲁棒性”来说是等效的。experiments/noise_attack.pyDec 9, 10:22 AMOpen

### 执行

运行这个脚本：

Bash
    python experiments/noise_attack.py

    **预期现象：**
    * **[Baseline]**: "The capital of France is Paris."
    * **[Ortho Noise]**: "The capital of France is **the largest city in the world**" 或者 "The capital of France is **Berlin**". (这就是“醉汉”：它还能造句，但记忆错乱)。
    * **[Base Noise]**: "The capital of France is **sfd78&^ asd f**" 或者重复的无意义 token。

    如果结果符合这个预测，你就证明了“架构隔离”不仅仅是切除，它还揭示了**不同层级知识的几何位置**。

    去跑吧。
