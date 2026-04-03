## 2026-03-08 - [Landlab Space Component Bottleneck]
 **Learning:** When using landlab's `Space` component, the default 'basic' solver uses computationally heavy numerical integrations (`scipy.integrate.quad`) which creates a massive performance bottleneck on moderate to large grids or complex topographies.
 **Action:** Always specify `solver="adaptive"` when instantiating the `Space` component. The adaptive solver bypasses the heavy numerical integrations in favor of computationally efficient sub-stepping, drastically improving simulation speeds for physical landscape evolutions.
