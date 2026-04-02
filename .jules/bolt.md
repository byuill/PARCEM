
## 2026-03-06 - [Optimization: Avoid dynamic list appends in tight numerical loops]
**Learning:** In 1D morphological models executing millions of inner adaptive time steps, tracking per-step metrics by appending to a list and calling `np.mean` at the end of the outer loop causes substantial overhead. Vectorized running sums (`array += Qs[indices]`) drastically outperform this approach.
**Action:** Use pre-allocated NumPy arrays and integer counters for running averages instead of appending to Python lists when inside tight numerical loops.
