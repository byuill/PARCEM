## 2026-03-31 - [Optimize Landlab SPACE component solver]
**Learning:** The default `basic` solver for the `landlab` `Space` component is extremely slow due to heavy numerical integration (`scipy.integrate.quad`). By switching to the `adaptive` solver, it bypasses these bottlenecking integrations in favor of sub-stepping.
**Action:** Always explicitly specify `solver="adaptive"` when instantiating the `Space` component in Landlab to significantly improve performance.
