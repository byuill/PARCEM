## 2024-04-04 - [Landlab SPACE component solver optimization]
**Learning:** The default `basic` solver for Landlab's `Space` component uses expensive `scipy.integrate.quad` numerical integrations which are a massive performance bottleneck.
**Action:** Always specify `solver='adaptive'` when initializing the `Space` component to bypass heavy numerical integrations in favor of computationally efficient sub-stepping, which drastically improves simulation speeds.
