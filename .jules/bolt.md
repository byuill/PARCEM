## 2025-04-02 - Use Adaptive Solver for Landlab Space Component

**Learning:** Landlab's `Space` component defaults to `solver="basic"`, which uses computationally heavy numerical integrations (`scipy.integrate.quad`). This creates a massive bottleneck in Landscape Evolution Models, taking ~36 seconds for a simple test case vs ~0.7 seconds.

**Action:** When initializing the `Space` component, always pass `solver="adaptive"` to bypass the expensive integrations in favor of sub-stepping. This dramatically improves simulation speed without breaking functionality. Also, be careful to clean up generated NetCDF output data and PyCache artifacts when running local tests.
