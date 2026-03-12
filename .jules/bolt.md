## 2024-05-24 - Landlab `Space` Component Adaptive Solver
**Learning:** The default `basic` solver for Landlab's `Space` component heavily utilizes `scipy.integrate.quad` for numerical integrations, causing a significant computational bottleneck and slow simulation speeds.
**Action:** Always specify `solver="adaptive"` when instantiating the `Space` component. This bypasses the heavy integrations in favor of computationally efficient sub-stepping, drastically improving simulation speeds without sacrificing accuracy.
