## YYYY-MM-DD - [Title]\n**Learning:** [Insight]\n**Action:** [How to apply next time]

## 2026-04-06 - Landlab Space Component Performance Optimization
**Learning:** By default, Landlab's `Space` component uses a `"basic"` solver that performs expensive numerical integrations (`scipy.integrate.quad`), acting as a severe performance bottleneck. Using `solver="adaptive"` bypasses these integrations in favor of a much faster, computationally efficient sub-stepping method.
**Action:** Always specify `solver="adaptive"` when instantiating Landlab's `Space` component to drastically improve simulation speeds.
