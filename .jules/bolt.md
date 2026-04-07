## 2024-06-25 - [Adaptive Solver for Landlab Space Component]
**Learning:** The default numerical integration (`scipy.integrate.quad`) solver for Landlab's `Space` component is a significant performance bottleneck. When run on a synthetic grid, switching to the 'adaptive' sub-stepping solver drastically improved execution time (from ~13 seconds down to ~0.2 seconds for a 50-year benchmark run).
**Action:** Always verify if `solver="adaptive"` is used when instantiating the Landlab `Space` component, as it provides major speedups without sacrificing accuracy.
