# 直接量化修复方案

## 问题分析

即使使用 `rank_ratio=0.95`，仍然存在问题：
- **Base error**: 14-17%（仍然 > 15%）
- **Final error**: 14-17%（仍然 > 1%）
- **Ortho sparsity**: 2.8-3.7%（仍然远超设定的 0.5%）
- **PPL 结果很差**：Alpha=0 时 Retain PPL 12966（应该 < 20）

## 根本原因

**误差累积**：
1. 低秩近似误差：7-9%（即使保留 95% 的秩）
2. INT8 量化误差：约 7-8%
3. **总误差 = 14-17%**（误差累积）

## 解决方案

### 1. 禁用低秩约束（直接量化）

**理论依据**：
- 如果低秩假设不成立，强制降秩只会增加误差
- 直接量化可能更有效（只有量化误差，没有低秩误差）

**实现**：
```python
use_low_rank = False  # 直接量化，不使用低秩约束
model = replace_linear_layers(
    model, 
    target_modules=["down_proj"], 
    ratio=0.005,
    use_low_rank=False
)
```

### 2. 改进量化方法

**从 percentile_95 改为 max**：
- 更保守的量化（减少饱和）
- 使用对称量化（不需要 zero point）
- 每行使用 max 值作为 scale

```python
# 旧方法：percentile_95（可能饱和）
percentile_95 = torch.quantile(w_abs, 0.95, dim=1, keepdim=True)
scales = percentile_95 / 127.0

# 新方法：max（更保守）
row_max = w_abs.max(dim=1, keepdim=True)[0]
scales = row_max / 127.0
```

## 预期效果

### 使用 use_low_rank=False（直接量化）

- **Base error**: 应该 < 10%（只有量化误差，没有低秩误差）
- **Final error**: 应该 < 2%（完整重构误差）
- **Ortho sparsity**: 应该接近设定的 0.5%

### 理论反思

**低秩假设可能不适用于所有层**：
1. Transformer 的某些层（如 down_proj）可能不是低秩的
2. 即使保留 95% 的秩，低秩近似误差仍有 7-9%
3. 误差累积导致总误差过大

**实用折中**：
- 对于低秩层：使用低秩约束（rank_ratio=0.95）
- 对于非低秩层：直接量化（use_low_rank=False）
- 自适应策略：根据误差自动选择

## 测试方法

运行测试：
```bash
python experiments/run_tofu_eval.py
```

现在默认使用 `use_low_rank=False`（直接量化）。

## 如果还是不行

如果直接量化仍然不行，可能需要：

1. **提高 ortho_ratio**：从 0.005 提高到 0.01 或 0.02
2. **使用 FP16 Base**：回到之前的方案，但这次是基于量化后的 FP16
3. **重新审视理论**：可能需要重新审视论文的低秩假设

## 关键改进

1. ✅ **禁用低秩约束**：直接量化，避免误差累积
2. ✅ **改进量化方法**：使用 max 而不是 percentile_95
3. ✅ **对称量化**：不需要 zero point，更简单
4. ✅ **自适应策略**：如果低秩误差太大，自动回退

