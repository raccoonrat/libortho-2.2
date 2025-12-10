# LibOrtho 新实现总结

## 一、核心改进

### 1.1 理论基础

基于论文 `libortho_paper_zh.pdf` 和 MIT 教授的分析，新实现遵循以下理论：

1. **低秩约束**：通用知识是低秩的，隐私记忆是高秩的
2. **流形投影**：量化是将权重投影到低精度流形，不是简单转换
3. **相对误差选择**：基于量化误差的相对重要性，不是权重大小

### 1.2 实现方案

**方案 C（混合策略）：低秩约束 + 量化投影**

```python
# Step 1: SVD 低秩近似（去除高秩分量）
U, S, Vt = torch.linalg.svd(w_orig, full_matrices=False)
r = int(max_rank * rank_ratio)  # 保留 80% 的秩
w_low_rank = U @ diag(S[:r]) @ Vt

# Step 2: INT8 量化投影（流形投影）
scales = percentile_95 / 127.0
w_int8 = round(w_low_rank / scales).clamp(-127, 127)
w_base = w_int8 * scales

# Step 3: 计算残差（高秩分量 + 量化误差）
residual = w_orig - w_base

# Step 4: 基于相对误差选择 Ortho Stream
relative_error = |residual| / |w_orig|
topk_idx = topk(relative_error, k)
```

## 二、关键改进点

### 2.1 从"FP16 转换"到"流形投影"

**旧方法（错误）：**
```python
self.base_weights = w_orig.to(torch.float16)  # 简单转换，残差接近 0
```

**新方法（正确）：**
```python
# 1. 低秩近似（去除高秩分量）
w_low_rank = svd_low_rank_approximation(w_orig, rank_ratio=0.8)

# 2. 量化投影（流形投影）
w_base = quantize_to_int8(w_low_rank)

# 3. FP16 存储（只是存储格式）
self.base_weights = w_base.to(torch.float16)
```

**理论依据：**
- 低秩约束：物理上无法编码高秩信息（隐私）
- 量化投影：实现真正的流形投影
- FP16 存储：只是存储格式，不影响逻辑

### 2.2 从"权重大小"到"相对误差"

**旧方法（错误）：**
```python
w_abs_flat = w_orig.abs().view(-1)
topk_vals, topk_idx = torch.topk(w_abs_flat, k)  # 基于权重大小
```

**新方法（正确）：**
```python
relative_error = residual.abs() / (w_orig.abs() + 1e-8)
topk_vals, topk_idx = torch.topk(relative_error.view(-1), k)  # 基于相对误差
```

**理论依据：**
- 相对误差 = 量化误差 / 原始权重
- 这更符合"高频细节"的定义
- 小权重的大误差 = 高频细节（隐私）

## 三、预期效果

### 3.1 理论预测

1. **Alpha=0 时（Base Stream 单独使用）：**
   - Retain PPL < 20（Base Stream 是低秩近似，通常有效）
   - Forget PPL > 100（隐私被移除，因为高秩分量在 Ortho Stream）

2. **Alpha=1 时（Base + Ortho）：**
   - Forget PPL < 5（隐私保留，因为 Ortho Stream 包含高秩分量）

### 3.2 验证标准

**成功标准：**
- Base error < 15%（低秩近似误差）
- Final error < 1%（完整重构误差）
- Alpha=0 时：Retain PPL < 20，Forget PPL > 100

**如果失败：**
- 调整 `rank_ratio`（低秩比例，默认 0.8）
- 调整 `ortho_ratio`（Ortho 比例，默认 0.01）
- 如果仍然失败，需要重新审视论文理论

## 四、使用说明

### 4.1 参数说明

```python
OrthoLinear(
    original_layer, 
    ortho_ratio=0.01,    # Ortho Stream 的稀疏率（默认 1%）
    rank_ratio=0.8        # Base Stream 的秩保留比例（默认 80%）
)
```

### 4.2 性能考虑

**SVD 计算：**
- 对于大矩阵（如 5632×2048），SVD 可能需要几秒钟
- 这是初始化时的开销，不影响推理速度
- 如果太慢，可以考虑使用随机 SVD（`torch.svd_lowrank`）

**内存优化：**
- Base Stream 使用 FP16 存储（节省 50% 内存）
- Ortho Stream 使用稀疏 CSR 格式（节省 99% 内存）
- Forward 时使用 chunking（避免 OOM）

## 五、与旧实现的对比

| 特性 | 旧实现 | 新实现 |
|------|--------|--------|
| Base Stream | FP16 转换 | 低秩 + 量化投影 |
| 残差 | 接近 0 | 显著（高秩分量 + 量化误差） |
| 选择策略 | 权重大小 | 相对误差 |
| 理论基础 | 简单转换 | 流形投影 + 低秩约束 |
| Alpha=0 效果 | 可能崩溃 | 应该可用（低秩近似） |

## 六、下一步

1. **运行测试**：`python experiments/run_tofu_eval.py`
2. **验证效果**：检查 Alpha=0 时的 PPL
3. **调整参数**：如果效果不理想，调整 `rank_ratio` 和 `ortho_ratio`
4. **性能优化**：如果 SVD 太慢，考虑使用随机 SVD

## 七、理论批判

### 7.1 论文理论的局限性

1. **梯度信息不可得**：推理时没有梯度，无法实现真正的"梯度方向正交"
2. **Hessian 计算昂贵**：无法在推理时计算 Hessian 谱
3. **低秩假设可能不成立**：通用知识可能不是严格低秩的

### 7.2 实用折中

- **用 SVD 近似 Hessian 谱**：主成分 ≈ 高特征值方向
- **用相对误差近似梯度方向**：高频细节 ≈ 隐私记忆
- **用低秩约束强制分离**：物理上无法编码高秩信息

## 八、总结

新实现基于论文的理论基础，同时考虑了实际约束（推理时没有梯度信息）。核心改进：

1. **低秩约束**：Base Stream 物理上无法编码高秩信息
2. **流形投影**：量化是投影，不是简单转换
3. **相对误差选择**：基于量化误差的相对重要性，不是权重大小

这更符合论文的理论基础，预期效果应该更好。

