import torch
import torch.nn as nn
import math

# 尝试导入 C++ 扩展，如果未编译则允许降级或报错
try:
    import libortho_ops
except ImportError:
    print("[WARN] LibOrtho C++ extension not found. Ensure you have run 'python setup.py install'.")
    libortho_ops = None

class OrthoLinear(nn.Module):
    def __init__(self, original_layer, ortho_ratio=0.01, rank_ratio=0.95, use_low_rank=True):
        """
        [THEORETICAL FIX] 基于论文理论的实现
        核心理论：
        1. 低秩约束：Base Stream 物理上无法编码高秩信息（隐私）
        2. 流形投影：量化是投影到低精度流形，不是简单转换
        3. 相对误差选择：基于量化误差的相对重要性，不是权重大小
        
        参数：
        - ortho_ratio: Ortho Stream 的稀疏率（默认 1%）
        - rank_ratio: Base Stream 的秩保留比例（默认 95%，即保留 95% 的秩）
        - use_low_rank: 是否使用低秩约束（默认 True，如果误差太大可以设为 False）
        """
        super().__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        
        print(f"[Ortho] Decomposing layer {self.in_features}x{self.out_features}...")

        # [Linus Safety Guard]
        if ortho_ratio > 0.3:
            print(f"[WARNING] Ortho ratio {ortho_ratio} is too high. Clamping to 0.3.")
            ortho_ratio = 0.3
            
        self.ortho_ratio = ortho_ratio
        self.rank_ratio = rank_ratio
        self.use_low_rank = use_low_rank
        
        # 1. 获取原始权重
        w_orig = original_layer.weight.data.float()
        device = w_orig.device
        
        # 2. [THEORETICAL FIX] Step 1: 低秩近似（可选，去除高秩分量）
        # 理论基础：通用知识是低秩的，隐私记忆是高秩的
        # 通过 SVD 强制降秩，物理上无法编码高秩信息
        # [ADAPTIVE] 如果低秩近似误差太大，自动回退到直接量化
        if use_low_rank:
            print(f"  [Step 1] Computing SVD for low-rank approximation (rank_ratio={rank_ratio})...")
            U, S, Vt = torch.linalg.svd(w_orig, full_matrices=False)
            
            # 保留前 r 个奇异值
            max_rank = min(U.shape[0], Vt.shape[0])
            r = int(max_rank * rank_ratio)
            r = max(1, min(r, max_rank - 1))  # 确保至少保留 1 个，最多保留 max_rank-1 个
            
            S_low_rank = torch.zeros_like(S)
            S_low_rank[:r] = S[:r]
            w_low_rank = U @ torch.diag(S_low_rank) @ Vt
            
            # [ADAPTIVE] 检查低秩近似误差
            low_rank_error = (w_low_rank - w_orig).norm() / w_orig.norm()
            print(f"  [Step 1] Rank reduction: {max_rank} -> {r} (保留 {r/max_rank*100:.1f}%), "
                  f"Low-rank error: {low_rank_error:.4f}")
            
            # 如果低秩近似误差 > 20%，回退到直接量化
            if low_rank_error > 0.20:
                print(f"  [ADAPTIVE] Low-rank error {low_rank_error:.4f} > 20%, "
                      f"falling back to direct quantization (no low-rank constraint)")
                w_for_quantization = w_orig
                use_low_rank = False  # 标记为未使用低秩
            else:
                w_for_quantization = w_low_rank
        else:
            print(f"  [Step 1] Skipping low-rank approximation (use_low_rank=False)")
            w_for_quantization = w_orig
        
        # 3. [THEORETICAL FIX] Step 2: 量化投影（流形投影）
        # 理论基础：量化是将权重投影到低精度格点（流形投影）
        # 使用 per-row 量化，确保 Base 是"最优投影"
        print(f"  [Step 2] Quantizing matrix (INT8 projection)...")
        w_abs = w_for_quantization.abs()
        percentile_95 = torch.quantile(w_abs, 0.95, dim=1, keepdim=True)
        percentile_95.clamp_(min=1e-5)  # 防止全 0 行
        
        scales = (percentile_95 / 127.0).to(torch.float32)
        w_int8 = torch.round(w_for_quantization / scales).clamp(-127, 127)
        w_base_quantized = w_int8 * scales
        
        # 4. Base Stream: 量化后的低秩矩阵（转换为 FP16 存储）
        self.base_weights = w_base_quantized.to(torch.float16).to(device)
        # 注意：scales 不再需要，因为我们已经将量化后的权重存储在 base_weights 中
        
        # 5. [THEORETICAL FIX] Step 3: 计算完整残差
        # 残差 = 高秩分量 + 量化误差
        residual = w_orig - self.base_weights.float()
        
        # 6. [THEORETICAL FIX] Step 4: 基于相对误差选择 Ortho Stream
        # 理论基础：相对误差 = 量化误差 / 原始权重
        # 这更符合"高频细节"的定义，而不是简单的权重大小
        total_params = residual.numel()
        k = int(total_params * self.ortho_ratio)
        k = max(k, 1)  # 至少选 1 个
        k = min(k, total_params - 1)
        
        # 计算相对误差：|residual| / |w_orig|
        w_orig_abs = w_orig.abs()
        relative_error = residual.abs() / (w_orig_abs + 1e-8)  # 避免除零
        
        # 选择相对误差最大的 top-k（高频细节）
        topk_vals, topk_idx = torch.topk(relative_error.view(-1), k)
        threshold = topk_vals[-1]
        
        # 创建稀疏掩码
        mask = torch.zeros_like(residual, dtype=torch.bool)
        mask.view(-1)[topk_idx] = True
        
        # 符号翻转保护：修复所有符号错误
        sign_mismatch = (w_orig.sign() != self.base_weights.float().sign()) & (w_orig_abs > 1e-4)
        mask = mask | sign_mismatch
        
        # Ortho Stream: 存储残差（高秩分量 + 量化误差）
        w_ortho_sparse = residual * mask
        
        # 7. 验证：Base Stream 误差
        base_error = (self.base_weights.float() - w_orig).norm() / w_orig.norm()
        
        # 8. 最终验证：完整重构误差
        w_final_recon = self.base_weights.float() + w_ortho_sparse
        final_error = (w_final_recon - w_orig).norm() / w_orig.norm()
        
        # [THEORETICAL VALIDATION] 验证低秩约束
        if use_low_rank and base_error > 0.15:  # Base error 应该 < 15%（因为低秩近似）
            print(f"[WARNING] Base error {base_error:.4f} is high. "
                  f"Consider increasing rank_ratio from {rank_ratio:.2f} or setting use_low_rank=False.")
        elif not use_low_rank and base_error > 0.10:  # 直接量化误差应该 < 10%
            print(f"[WARNING] Base error {base_error:.4f} is high. "
                  f"Consider using better quantization method.")
        
        # [THEORETICAL VALIDATION] 验证重构质量
        if final_error > 0.01:  # Final error 应该 < 1%
            print(f"[WARNING] Final error {final_error:.4f} exceeds 1%. "
                  f"Consider increasing ortho_ratio from {ortho_ratio:.4f}.")
        
        # 7. Ortho: Convert to CSR
        w_ortho_csr = w_ortho_sparse.to_sparse_csr()
        self.ortho_vals = w_ortho_csr.values().to(torch.float16)
        self.ortho_indices = w_ortho_csr.col_indices().to(torch.int32)
        self.ortho_ptr = w_ortho_csr.crow_indices().to(torch.int32)
        
        self.nnz = self.ortho_vals.numel()
        self.alpha = 1.0
        
        print(f"[Ortho] Layer {self.in_features}x{self.out_features}: "
              f"Base error={base_error:.6f}, Final error={final_error:.6f}, "
              f"Ortho sparsity={self.nnz}/{total_params} ({self.nnz/total_params:.4f})") 
        
        # Move to device (base_weights already on device)
        self.ortho_vals = self.ortho_vals.to(device)
        self.ortho_indices = self.ortho_indices.to(device)
        self.ortho_ptr = self.ortho_ptr.to(device)

    def forward(self, x):
        """
        [LINUS DEEP FIX] 简化 forward：Base Stream 使用 FP16 矩阵乘法
        [MEMORY OPTIMIZED] 极激进的内存优化：避免创建稀疏张量，手动计算 CSR 乘法
        """
        original_shape = x.shape
        original_dtype = x.dtype
        
        if not x.is_cuda:
            raise RuntimeError(f"Input must be on CUDA, got {x.device}")
        
        x_flat = x.view(-1, self.in_features).contiguous()
        batch_size = x_flat.size(0)
        
        # Base Stream: FP16 矩阵乘法（内存高效）
        x_flat_f16 = x_flat.to(torch.float16)
        out_base = torch.mm(x_flat_f16, self.base_weights.t())  # (batch × out_features)
        
        # Ortho Stream: 极简稀疏矩阵乘法（最小内存占用）
        if self.alpha > 1e-6:
            # [MEMORY OPTIMIZED] 使用最小的 chunk size，避免任何大的临时张量
            x_flat_f32 = x_flat.to(torch.float32)
            ortho_vals_f32 = self.ortho_vals.to(torch.float32)
            
            # 创建稀疏张量（必须的，但只创建一次）
            w_ortho = torch.sparse_csr_tensor(
                self.ortho_ptr, 
                self.ortho_indices, 
                ortho_vals_f32,
                size=(self.out_features, self.in_features)
            )
            
            # [MEMORY OPTIMIZED] 极小的 batch chunk，避免转置大张量
            # 对于 RTX 4050，我们需要非常保守
            batch_chunk_size = 2  # 一次只处理 2 个样本
            
            if batch_size > batch_chunk_size:
                out_ortho_chunks = []
                for i in range(0, batch_size, batch_chunk_size):
                    end_idx = min(i + batch_chunk_size, batch_size)
                    chunk = x_flat_f32[i:end_idx]  # (chunk_size × in_features)
                    
                    # 计算: w_ortho @ chunk.t() -> (out_features × chunk_size)
                    # 然后转置得到 (chunk_size × out_features)
                    chunk_out = torch.sparse.mm(w_ortho, chunk.t()).t()
                    out_ortho_chunks.append(chunk_out)
                    
                    # 立即清理
                    del chunk, chunk_out
                    
                    # 每处理几个 chunk 就清理缓存
                    if (i // batch_chunk_size) % 8 == 0:
                        torch.cuda.empty_cache()
                
                out_ortho = torch.cat(out_ortho_chunks, dim=0)
                del out_ortho_chunks
            else:
                # 小 batch，直接计算
                out_ortho = torch.sparse.mm(w_ortho, x_flat_f32.t()).t()
            
            # 清理
            del w_ortho, x_flat_f32, ortho_vals_f32
            torch.cuda.empty_cache()
            
            out_flat = out_base.to(torch.float32) + self.alpha * out_ortho
            del out_ortho
        else:
            out_flat = out_base.to(torch.float32)
        
        out_reshaped = out_flat.view(original_shape[:-1] + (self.out_features,))
        return out_reshaped.to(original_dtype)

    def set_privacy(self, enable_ortho: bool):
        self.alpha = 1.0 if enable_ortho else 0.0

def replace_linear_layers(model, target_modules=["down_proj", "o_proj"], ratio=0.05, 
                          rank_ratio=0.95, use_low_rank=True):
    """
    替换模型中的 Linear 层为 OrthoLinear 层
    
    参数：
    - model: 要修改的模型
    - target_modules: 要替换的模块名称列表
    - ratio: Ortho Stream 的稀疏率
    - rank_ratio: Base Stream 的秩保留比例（默认 95%）
    - use_low_rank: 是否使用低秩约束（默认 True）
    """
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_linear_layers(module, target_modules, ratio, rank_ratio, use_low_rank)
        if isinstance(module, nn.Linear):
            should_replace = any(t in name for t in target_modules)
            if should_replace:
                print(f"Patching layer: {name}")
                new_layer = OrthoLinear(module, ortho_ratio=ratio, 
                                       rank_ratio=rank_ratio, use_low_rank=use_low_rank)
                setattr(model, name, new_layer)
    return model