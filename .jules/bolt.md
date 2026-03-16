## 2024-05-18 - Landlab Space solver performance
**Learning:** For `landlab`'s `Space` component, the default `"basic"` solver uses heavy numerical integrations (`scipy.integrate.quad`) which creates a significant performance bottleneck.
**Action:** Always specify `solver="adaptive"` instead of the default `"basic"` solver for the `Space` component. The adaptive solver bypasses these integrations in favor of computationally efficient sub-stepping, drastically improving simulation speeds.
