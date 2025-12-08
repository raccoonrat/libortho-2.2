LibOrtho: The "Good Taste" Implementation

"Theory is nice, but does it run on my laptop without segfaulting?" - Me

This project implements the Geometric Disentanglement theory (LibOrtho) with a focus on systems pragmatism.

The Problem with the Paper

The original paper (and the 1208 doc) is full of academic fluff. It talks about "Dual Manifolds" and "Spectral Filtering."
Translated to English: Weights are either important for everyone (Base) or important for a specific edge case (Ortho).

The Solution

We treat this as a memory layout problem.

Base Stream: INT4 quantized. Dense. This is your robust knowledge.

Ortho Stream: FP16 sparse. This is your privacy/memory/exceptions.

Alpha Switch: A scalar that controls the Ortho stream.

Why this code is better

Single Struct: No pointer chasing.

Kernel Fusion: We don't launch two kernels. We launch one.

No Magic: The "tail lobotomy" is just setting alpha = 0. It works because math works, not because we added an if statement.

How to Verify (The 1208 Protocol)

Run python3 tests/verify_hypothesis.py.
It implements the 3-stage validation:

Geometric Observation: Proves memory gradients live in the null space of general knowledge.

Intervention: Cuts the tail and checks if the memory dies.

Counterfactual: Checks the sharpness of the loss landscape.

Don't break userspace.

