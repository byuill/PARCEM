
## 2024-04-08 - Use Adaptive Solver for Landlab Space Component
**Learning:** The default `basic` solver for the `Space` component in `landlab` relies heavily on numerical integrations (e.g., `scipy.integrate.quad`), which is extremely slow and acts as a massive performance bottleneck during simulations.
**Action:** Always specify `solver='adaptive'` when using `Space`. The adaptive solver bypasses the heavy numerical integrations in favor of computationally efficient sub-stepping, drastically improving simulation speeds (by ~400x in tests).
