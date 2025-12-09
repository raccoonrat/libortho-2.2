这是 Linus。

把香槟放下。代码能跑只是开始，现在的你只是证明了“这个手术刀能切肉”，但还没证明“这套手术方案能治好癌症”。

对照你的《1208-实验构建》文档，让我们像做 Code Review 一样，逐条审查我们实现了什么（**Done**），妥协了什么（**Compromise**），以及还欠着什么（**TODO**）。

* * *

### 状态审计报告 (Status Audit)

| **1208-实验构建 章节** | **核心目标**            | **当前代码实现状态**                      | **评价**                                                                                                         |
| ---------------- | ------------------- | --------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **阶段一：几何观测**     |                     |                                   |                                                                                                                |
| Exp 1.1: 谱分解     | 验证记忆梯度在 Hessian 尾部  | **模拟实现** (`verify_hypothesis.py`) | ✅ **逻辑通过**。我们在合成数据上验证了正交性。但在真实模型上，我们偷懒了，没有算真正的 Hessian。                                                        |
| Exp 1.2: 正交性测试   | 验证记忆梯度与通用梯度正交       | **模拟实现** (`verify_hypothesis.py`) | ✅ **逻辑通过**。同上。                                                                                                 |
| **阶段二：介入干扰**     |                     |                                   |                                                                                                                |
| Exp 2.1: 尾部切除    | 切除尾部导致“失忆不降智”       | **完美实现** (`run_real_llm.py`)      | 🌟 **超额完成**。我们在真机（TinyLlama）上复现了“语法保留，逻辑崩塌”的现象。这是最有力的证据。                                                       |
| Exp 2.2: 噪声攻击    | 尾部噪声导致定向失忆          | ❌ **未实现**                         | 我们只做了“切除”（Alpha=0），没做“干扰”（Alpha=Noise）。                                                                        |
| **阶段三：反直觉验证**    |                     |                                   |                                                                                                                |
| Exp 3.1: 尖峰测试    | 验证记忆位于 Sharp Minima | **隐式实现** (`inject_canary.py`)     | ⚠️ **妥协**。我们通过 `inject_canary.py` 证明了高幅度的异常值（Spike）就是记忆。但这只是基于**幅度（Magnitude）**的近似，不是基于**曲率（Curvature）**的严格证明。 |

* * *

### 核心差距：幅度 vs. 曲率 (Magnitude vs. Curvature)

这是你目前系统最大的“理论漏洞”。

* **你的论文说**：“我们要根据 Hessian 的**曲率**来筛选 Base 和 Ortho。”（即：$Impact_i = w_i^2 \cdot H_{ii}$）

* **你的代码做**：“我们根据权重的**绝对值大小**来筛选。”（即：$Impact_i = |w_i|$）

为什么这在 GTX 4050 上是可以接受的妥协？

因为通常 Hessian 高的地方（Sharp Minima），权重的更新量也大，导致权重绝对值往往也是异常值。幅度是曲率的一阶近似。

但是，如果要严谨，你需要补上这个环节。

* * *

### 接下来要做什么 (The TODO List)

为了让你的项目从“黑客玩具”变成“科研基石”，你需要构建剩下的三个实验。

#### 任务 1：实现 Exp 2.2 - 定向噪声注入 (The Chaos Test)

目标：证明 Ortho 流极其敏感。

做法：不要把 Alpha 设为 0。保持 Alpha = 1，但向 Ortho 流的权重注入高斯噪声。

预期：模型应该会像喝醉了一样，语法完美，但开始一本正经地胡说八道（Hallucination），比如“巴黎是月球的首都”。而如果在 Base 流注入同样的噪声，模型应该直接输出乱码。

代码修改建议 (src/kernel_fusion.cu):

你需要一个新的 Kernel 变体，或者在 Python 层生成一个全是噪声的 ortho_vals 传进去。

#### 任务 2：实现真正的 Hessian 近似 (Diagonal Fisher)

你不能在 4050 上算全量 Hessian，但你可以算 **Fisher Information 的对角线**。这是对 Hessian 的廉价近似。

**实验设计**：

1. 找 100 条样本数据（校准集）。

2. 对模型跑反向传播，拿到梯度 $g$。

3. 计算 $F_{ii} = \frac{1}{N} \sum g_i^2$。

4. 用 $Score_i = w_i^2 \cdot F_{ii}$ 来代替现在的 `w.abs()` 进行筛选。

代码实现 (src/model_patch.py):

你需要修改 OrthoLinear 的初始化逻辑，接受一个 fisher_info 张量。

Python
    # 伪代码：未来的方向
    def compute_fisher(model, dataset):
        # 累加梯度的平方
        pass

    class OrthoLinear(nn.Module):
        def __init__(self, layer, fisher_info):
            # 使用 Fisher 信息加权的幅度来筛选异常值
            # metric = weight ** 2 * fisher_info
            #mask = metric >= topk(metric)
            pass

#### 任务 3：TOFU 基准测试 (The Standard)

既然你要发论文或者做正经项目，不能只靠“手动注入金丝雀”。你需要跑通 TOFU (Task of Fictitious Unlearning) 数据集。

这是一个包含虚构作家信息的数据集。你需要证明 LibOrtho 能忘掉这些虚构作家，但保留 JK Rowling。

* * *

### 总结

你现在的状态：

* **工程实现**：100% Ready. (坚固的 C++/CUDA 运行时)

* **定性验证**：100% Ready. (Canary 和 Lobotomy 实验证明了现象存在)

* **定量精度**：80%. (依赖 Magnitude 近似，缺乏 Curvature/Fisher 的加持)

我的建议：

先别管 Hessian 了。那个太慢。

你现在的当务之急是Exp 2.2 (噪声攻击)。因为它不需要新的算力，只需要几行 Python 代码修改，就能产生极具视觉冲击力的结果（"Drunken Master" 模式）。

去写一个 `experiments/noise_attack.py`。我要看到模型在语法严谨地胡说八道。
