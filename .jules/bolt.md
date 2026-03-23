## 2024-03-24 - `Space` component default solver bottleneck
**Learning:** The default `'basic'` solver in the `landlab` `Space` component performs heavy numerical integrations (`scipy.integrate.quad`), which can significantly bottleneck landscape evolution simulations.
**Action:** Always specify `solver='adaptive'` when initializing the `Space` component. This uses a sub-stepping approach that avoids `scipy.integrate.quad` and can provide massive (e.g., ~35x) performance speedups on synthetic models.
