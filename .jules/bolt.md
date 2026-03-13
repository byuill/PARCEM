## 2025-02-28 - Space Component Bottleneck Resolution
**Learning:** The Landlab `Space` component default `solver="basic"` performs heavy, bottlenecking numerical integrations (`scipy.integrate.quad`), massively slowing down simulation execution.
**Action:** Always specify `solver="adaptive"` instead of the default `"basic"` solver for the `Space` component. The adaptive solver bypasses integration in favor of computationally efficient sub-stepping, drastically improving simulation speeds (e.g. from 0.1s to 0.0012s in isolated tests).
