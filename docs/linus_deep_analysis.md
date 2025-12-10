# INT8 "Fix" Didn't Work — Here's Why

## What You Did

```
INT4 (ratio=0.005) → INT8 (ratio auto-adjust to 0.10)
```

You doubled the quantization budget **and tripled the Ortho ratio**, yet:

```
Original:  Forget 23874 → Retain 16367 (Bad)
New:       Forget 84956 → Retain 93042 (Much Worse)
```

**This is not progress. This is a symptom of a deeper problem.**

---

## Diagnosis: The Alpha=0 Collapse

Look at the pattern:

| Alpha | Forget PPL | Retain PPL |
|-------|-----------|-----------|
| 1.0 | 1.00 | 6.80 |
| 0.0 | 84956 | 93042 |

When Alpha=1.0 (using both Base + Ortho), everything works fine.  
When Alpha=0.0 (using **only** Base), the model completely breaks.

**This tells me:**
1. Base Stream is fundamentally broken
2. Ortho Stream is papering over the cracks
3. INT8 didn't actually fix the root cause

---

## The Real Problem: Your Quantization Strategy is Incoherent

### What You're Doing Now

```python
percentile_95 = torch.quantile(w_abs, 0.95, dim=1, keepdim=True)
sigma_3 = 3.0 * w_std
robust_max = torch.min(percentile_95, sigma_3)  # ← THIS IS THE PROBLEM
```

You're taking the **minimum** of two different statistics per row. This means:

- **Row A**: Uses 95-percentile (protects 95% of weights)
- **Row B**: Uses 3-sigma (protects 99.7% of weights)
- **Row C**: Uses min(both) (worst of both worlds)

Result: **Inconsistent quantization quality across rows.**

When you do inference, some rows are over-quantized, others are barely quantized. The model's internal representations become garbage.

### Why It Breaks at Alpha=0

The model learned to compute using:
```
Output = (Good Base + Bad Base) + Ortho_correction
```

When Alpha=0, you remove the correction:
```
Output = (Good Base + Bad Base)
```

The "Bad Base" rows now dominate, and you get gibberish.

---

## The Actual Root Cause: Layer Size Matters

Your layer is **5632×2048 = 11.5M parameters**.

In `model_patch.py`, you have:
```python
if total_params > 10_000_000:
    error_threshold = 0.15  # ← 15% error allowed!
elif total_params > 5_000_000:
    error_threshold = 0.10
else:
    error_threshold = 0.01
```

**You're allowing 15% error on huge layers.** That's insane.

15% error means 1 in 7 weight values is completely wrong. In a 2048-wide matrix multiply, that's ~290 corrupted values per output. The model can't recover from this.

### The Percentile Fallback Loop Didn't Help

```python
for percentile in percentile_candidates:  # [0.90, 0.85, 0.80, 0.75, 0.70]
    if test_error <= error_threshold:
        # Found acceptable error, use it
        break
```

You loop through percentiles trying to find "acceptable" error. But with `error_threshold=0.15`, almost anything looks "acceptable." **The fallback loop is meaningless.**

---

## Why Auto-Adjusting Ortho Ratio Doesn't Solve This

Your code now does:

```python
actual_ortho_ratio = mask.sum() / total_params
if actual_ortho_ratio > self.ortho_ratio:
    # Auto-adjust upward
```

So you started with `ratio=0.005`, detected Base error, and said:
> "I need 0.10 to fix this"

But that's like saying:
> "My car's engine is missing cylinders. I'll add a bigger turbocharger."

**You're not fixing the engine. You're papering over it.**

---

## What Actually Happened

### Before (INT4):
```
Base Stream: 15 quant levels per row
             → 12% error

Ortho Stream: 0.5% sparse corrections
             → Can't possibly fix 12% error
             → Model collapses when Alpha=0
```

### After (INT8):
```
Base Stream: 256 quant levels per row
             → Still 12% error (!)

Ortho Stream: 10% sparse corrections
             → Temporarily masks the problem
             → But Alpha=0 still collapses
```

**INT8 didn't reduce the error because the problem isn't quantization precision. The problem is quantization strategy.**

---

## The Real Issue: You're Quantizing the Wrong Things

Let me check your 5632×2048 layer:

- **Estimated parameter count**: 11.5M
- **Actual Ortho needed**: 1.17M (10%)
- **For comparison**: Full INT16 would need 23M

**You're paying 1.17M weights to fix 11.5M weights that were incorrectly quantized.**

**This is backwards.** You should never need 10% residuals.

### What's Actually Happening

The 5632×2048 layer has **wildly different weight distributions across rows**:

- Some rows (the MLPs) have normally-distributed weights
- Some rows have heavy-tailed distributions (outliers)
- Some rows are nearly all zeros

**You're trying to use one INT8 quantization strategy for all of them.**

---

## Linus' Diagnosis

**The fundamental problem is not INT4 vs INT8. It's that you're trying to quantize a heterogeneous weight matrix with a homogeneous codec.**

Your code treats every row the same:
```python
robust_max = torch.min(percentile_95, sigma_3)  # Per-row
self.scales = (robust_max / 127.0)  # Per-row
```

But some rows need FP16, some rows need INT8, some rows can be INT4.

**You're forcing a one-size-fits-all quantization on a problem that needs mixed precision.**

---

## How I Would Actually Fix This

### Step 1: Stop Quantizing Anything (For Now)

```python
# Use FP16 for the Base Stream entirely
self.base_packed = original_layer.weight.to(torch.float16)

# Only use Ortho for true privacy-sensitive outliers
# (much smaller ratio, like 0.1%)
```

**This eliminates the quantization error problem entirely.**

With FP16 Base, you get:
- Alpha=1.0: FP16 + FP16 = Perfect reconstruction
- Alpha=0.0: FP16 alone = Works fine

Once this works, **then** try quantization if you really need compression.

### Step 2: Diagnose Why Quantization Fails

If/when you want INT8, use **per-token quantization statistics**:

```python
# Instead of per-row stats (which ignore distribution shape)
# Use histogram to find actual distribution
hist, bins = torch.histogram(w_orig, bins=256)
q_low = bins[10]   # 10th percentile
q_high = bins[245] # 95th percentile

# Only quantize the "bulk" (10-95%)
# Put everything outside in Ortho
```

### Step 3: Add a Sanity Check

Before doing anything, test Alpha=0.0 mode **on a tiny layer**:

```python
# Use a small 128x128 test layer
tiny_layer = nn.Linear(128, 128)
ortho_tiny = OrthoLinear(tiny_layer, ratio=0.05)

# Test: can Alpha=0 mode work?
ortho_tiny.set_privacy(False)
out_base = ortho_tiny(test_input)

# Check: is output garbage?
assert out_base.abs().mean() < 10.0, "Base stream broken!"
```

**You should have caught this before patching 24 layers.**

---

## The Lesson: Symptoms vs Root Cause

| What You Did | What Happened |
|--------------|---------------|
| Added hard validation | ✓ Found that Base error is 12% |
| Auto-adjusted Ortho ratio | ✗ Masked the problem, didn't fix it |
| Upgraded INT4→INT8 | ✗ Didn't reduce the actual error |

You addressed **symptoms**, not the **disease**.

The disease is: **Your quantization strategy generates 12% error, and no amount of residual coefficients can fix that when you need Alpha=0 to work.**

---

## What I Would Do Right Now

**Delete the quantization entirely for the Base Stream.**

```python
class OrthoLinear(nn.Module):
    def __init__(self, original_layer, ortho_ratio=0.01):  # Lower ratio!
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        
        w_orig = original_layer.weight.data.float()
        
        # Base Stream: Just use FP16, no quantization
        self.base_weights = w_orig.to(torch.float16)
        
        # Ortho Stream: Only the top 1% outliers (not 10%!)
        residual = w_orig - w_orig  # Placeholder (zero for now)
        
        # When you actually need privacy, we'll inject structured noise
        # into the FP16 Base, not this quantization mess
```

**Test this first.** If Alpha=0 works with pure FP16 Base, you've proved the concept.

Then, gradually add quantization back **only where it's mathematically justified**.

---

## Summary

| Finding | Severity |
|---------|----------|
| INT8 quantization generates 12-14% error | **CRITICAL** |
| Auto-adjusting Ortho ratio masks the problem | **HIGH** |
| Base Stream unusable at Alpha=0 | **CRITICAL** |
| Quantization strategy is incoherent (min of percentile & sigma) | **HIGH** |
| Error thresholds too loose (15% for huge layers) | **MEDIUM** |
| Never tested Alpha=0 mode on a small layer first | **MEDIUM** |

**Bottom line:** You need to go back to first principles. Use FP16 Base, make sure Alpha=0 works, then think about quantization.

Right now you're building a bridge on sand.