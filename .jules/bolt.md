## 2024-05-24 - Landlab Space Component Adaptive Solver
**Learning:** The default "basic" solver in Landlab's `Space` component uses heavy numerical integrations (`scipy.integrate.quad`), which creates a significant performance bottleneck.
**Action:** Always specify `solver="adaptive"` when initializing the `Space` component. This bypasses the heavy integrations in favor of computationally efficient sub-stepping, drastically improving simulation speeds.
