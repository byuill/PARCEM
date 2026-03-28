## 2024-05-18 - [Optimizing Landlab SPACE solver]
**Learning:** The default `"basic"` solver in landlab's `Space` component uses heavy numerical integration (`scipy.integrate.quad`), creating a significant performance bottleneck during simulation.
**Action:** Always specify `solver="adaptive"` when initializing the `Space` component. The adaptive solver bypasses these integrations via computational sub-stepping, significantly reducing simulation times (e.g., from >6.7s to <0.2s on small grids).
