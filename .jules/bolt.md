## 2024-04-12 - [Landlab SPACE component solver optimization]
**Learning:** The default "basic" solver in Landlab's SPACE component relies on heavy numerical integrations (`scipy.integrate.quad`), which is a major performance bottleneck for landscape evolution models. Using the "adaptive" solver bypasses these heavy calls in favor of efficient sub-stepping.
**Action:** Always specify `solver="adaptive"` when instantiating the `Space` component to drastically improve simulation speeds.
