## 2024-03-24 - Landlab Space component solver bottleneck
**Learning:** The default `basic` solver in `landlab`'s `Space` component uses heavy numerical integrations (`scipy.integrate.quad`), which creates a massive performance bottleneck.
**Action:** Always specify `solver="adaptive"` instead of the default `"basic"` solver for the `Space` component. The adaptive solver uses computationally efficient sub-stepping, drastically improving simulation speeds (observed ~3x speedup).
