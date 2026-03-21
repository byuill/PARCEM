## 2024-05-24 - [Adaptive Solver for Landlab SPACE Component]
**Learning:** The default "basic" solver in Landlab's `Space` component relies on computationally expensive numerical integration (`scipy.integrate.quad`). This creates a significant bottleneck, causing simulation steps to take orders of magnitude longer.
**Action:** When initializing the `Space` component, pass `solver="adaptive"` via kwargs to use sub-stepping instead of `quad`. This optimization reduced a 100-year synthetic test simulation time from ~25s to ~0.3s (an ~80x speedup).
