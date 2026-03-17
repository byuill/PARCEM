## 2024-03-17 - Adaptive Solver bypasses Scipy bottlenecks in Landlab Space
**Learning:** The default "basic" solver used in Landlab's `Space` component relies on heavy numerical integrations (`scipy.integrate.quad`), which act as a severe bottleneck during fluvial erosion simulation steps.
**Action:** Always specify `solver="adaptive"` when initializing the `Space` component to use computationally efficient sub-stepping and significantly improve simulation execution time.
