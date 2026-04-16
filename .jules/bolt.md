## 2026-04-16 - [Optimize Landlab SPACE Component Execution]
**Learning:** Using the default `solver="basic"` for `landlab`'s `Space` component leads to a massive computational bottleneck because it relies on `scipy.integrate.quad` for every node. This can severely throttle performance for high-resolution setups.
**Action:** Always instantiate the `Space` component with `solver="adaptive"` (e.g. `Space(grid, ..., solver="adaptive")`). The adaptive sub-stepping bypasses the heavy numerical integrations, drastically accelerating the simulation (over 50x speedup observed).
