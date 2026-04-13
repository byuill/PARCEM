## 2024-05-24 - [Optimize Landlab Space Component]
**Learning:** The default `"basic"` solver for `landlab`'s `Space` component uses slow numerical integration (`scipy.integrate.quad`), which is a huge bottleneck for simulations. The `"adaptive"` solver uses computationally efficient sub-stepping, drastically improving simulation speeds (e.g. from 15.8 seconds down to 0.02 seconds in a simple benchmark).
**Action:** Always specify `solver="adaptive"` when instantiating the `Space` component in Landlab models to drastically improve performance.
