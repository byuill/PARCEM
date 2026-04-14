## 2025-02-28 - Adaptive Solver in SPACE Component
**Learning:** The default numerical integration solver ("basic") in Landlab's SPACE component uses computationally heavy `scipy.integrate.quad` calls, creating a severe performance bottleneck during landscape evolution simulations.
**Action:** Always specify `solver="adaptive"` when instantiating the `Space` component to bypass this bottleneck, utilizing efficient sub-stepping for massive performance gains.
