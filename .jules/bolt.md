## 2024-05-18 - Landlab SPACE solver optimization
**Learning:** The default "basic" solver in Landlab's SPACE component heavily uses `scipy.integrate.quad` which creates a massive performance bottleneck during simulation.
**Action:** When initializing `Space`, always specify `solver="adaptive"`. This bypasses the heavy numerical integrations in favor of computationally efficient sub-stepping, drastically improving simulation speeds specific to this codebase's architecture.
