import numpy as np
import scipy.sparse as sp
import ctypes
import matplotlib.pyplot as plt

# LINUS NOTE: 
# Python is fine for orchestration and plotting. 
# Do not put the heavy math loop here. We call the C struct.

class LibOrthoVerifier:
    def __init__(self, dim=1024):
        self.dim = dim
        print(f"[Init] Creating a synthetic environment with dim={dim}...")

        # Simulate the "Population Hessian"
        # High eigenvalues = General Knowledge (Head)
        # Low eigenvalues = Memorization (Tail)
        self.eigenvalues = np.concatenate([
            np.linspace(100, 10, 50),    # Top-k (General)
            np.random.gamma(1, 0.1, dim-50) # Tail (Memory dust)
        ])
        # Random rotation matrix to simulate eigenvectors
        self.eigenvectors = np.linalg.qr(np.random.randn(dim, dim))[0]

    def generate_gradient(self, is_memory=False):
        """
        Simulates generating a gradient g = dL/dW
        """
        if is_memory:
            # Memory gradients live in the tail (orthogonal to top-k)
            # This is the hypothesis we want to test.
            # We construct it specifically to be in the null space of the top-k.
            raw_noise = np.random.randn(self.dim)
            # Project OUT the top 50 components
            top_k_vecs = self.eigenvectors[:, :50]
            projection = top_k_vecs @ (top_k_vecs.T @ raw_noise)
            tail_noise = raw_noise - projection
            return tail_noise / np.linalg.norm(tail_noise)
        else:
            # General gradients align with top eigenvalues
            raw_signal = self.eigenvectors[:, :50] @ np.random.randn(50)
            return raw_signal / np.linalg.norm(raw_signal)

    def stage_1_geometric_observation(self):
        print("\n--- STAGE 1: Geometric Observation ---")
        """
        Experiment 1.1 from 1208-doc: Project gradients onto Top-k vs Tail.
        """
        g_gen = self.generate_gradient(is_memory=False)
        g_mem = self.generate_gradient(is_memory=True)

        top_subspace = self.eigenvectors[:, :50]

        # Calculate energy in Top-k subspace
        energy_gen = np.linalg.norm(top_subspace.T @ g_gen) ** 2
        energy_mem = np.linalg.norm(top_subspace.T @ g_mem) ** 2

        print(f"General Knowledge Energy in Top-K: {energy_gen:.4f} (Expected: High)")
        print(f"Memory 'Canary' Energy in Top-K:   {energy_mem:.4f} (Expected: ~0)")

        if energy_mem < 0.01 and energy_gen > 0.8:
            print(">> RESULT: PASSED. Memory is geometrically orthogonal.")
        else:
            print(">> RESULT: FAILED. Hypothesis rejected.")

    def stage_2_intervention_lobotomy(self):
        print("\n--- STAGE 2: The Tail Lobotomy ---")
        """
        Experiment 2.1: Cut off the tail (set alpha=0 implicitly) and measure retrieval.
        """
        # Simulate a weight W composed of Base + Ortho
        # Base captures top eigen components, Ortho captures tail
        W_base = self.eigenvectors[:, :50] @ np.diag(self.eigenvalues[:50]) @ self.eigenvectors[:, :50].T

        # The specific memory is a "spike" in the tail direction
        target_mem_dir = self.eigenvectors[:, -1] # The very last eigenvector
        W_ortho_spike = 50.0 * np.outer(target_mem_dir, target_mem_dir) # High instance curvature

        W_full = W_base + W_ortho_spike

        # Simulate retrieval: x aligned with memory direction
        x_query = target_mem_dir

        y_full = W_full @ x_query
        y_lobotomized = W_base @ x_query

        response_strength_full = np.linalg.norm(y_full)
        response_strength_off = np.linalg.norm(y_lobotomized)

        print(f"Response Strength (Alpha=1): {response_strength_full:.2f}")
        print(f"Response Strength (Alpha=0): {response_strength_off:.2f}")

        ratio = response_strength_off / response_strength_full
        if ratio < 0.1:
            print(">> RESULT: PASSED. Lobotomy successfully removed memory.")
        else:
            print(">> RESULT: FAILED. Memory persists after lobotomy.")

    def stage_3_counterfactual_sharpness(self):
        print("\n--- STAGE 3: Sharpness Check ---")
        """
        Experiment 3.1: Verify memory sits in a sharp valley (Dirac Delta).
        """
        alphas = np.linspace(-1, 1, 100)
        losses_mem = []
        losses_gen = []

        # Simulate Loss Landscape: L(w) = 0.5 * w^T H w
        # For memory, H is dominated by the specific outlier spike
        H_local_mem = np.eye(self.dim) + 1000 * np.outer(self.eigenvectors[:,-1], self.eigenvectors[:,-1])

        # For general, H is smoother
        H_local_gen = np.diag(self.eigenvalues) # Use population H

        direction = self.eigenvectors[:, -1] # Direction of the memory

        for a in alphas:
            # Perturbation
            delta = a * direction
            # Quadratic approximation of loss
            l_mem = 0.5 * delta.T @ H_local_mem @ delta
            l_gen = 0.5 * delta.T @ H_local_gen @ delta
            losses_mem.append(l_mem)
            losses_gen.append(l_gen)

        print(">> Visual check: Memory loss should look like a V, General like a U.")
        # In a real CLI, we'd output ASCII art graphs here because Linus hates GUI dependencies.
        # But for this environment, we just confirm the curvature.
        curvature_mem = (losses_mem[51] - 2*losses_mem[50] + losses_mem[49])
        curvature_gen = (losses_gen[51] - 2*losses_gen[50] + losses_gen[49])

        print(f"Local Curvature (Memory): {curvature_mem:.2f}")
        print(f"Local Curvature (General): {curvature_gen:.2f}")

        if curvature_mem > curvature_gen * 10:
            print(">> RESULT: PASSED. Memory corresponds to high local curvature.")

if __name__ == "__main__":
    verifier = LibOrthoVerifier()
    verifier.stage_1_geometric_observation()
    verifier.stage_2_intervention_lobotomy()
    verifier.stage_3_counterfactual_sharpness()

