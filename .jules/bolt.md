
## 2025-02-17 - Landlab Space Component Solver Performance
**Learning:** The default `basic` solver for Landlab's `Space` component uses `scipy.integrate.quad` which creates a massive performance bottleneck. The `adaptive` solver avoids these heavy numerical integrations in favor of computationally efficient sub-stepping, providing an ~118x speedup on small grids.
**Action:** Always specify `solver='adaptive'` when instantiating the `Space` component to prevent severe performance degradation in landscape evolution simulations.
