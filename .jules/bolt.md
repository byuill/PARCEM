## 2024-05-24 - Enable adaptive solver in SPACE component

**Learning:** The default Basic solver in landlab's SPACE component utilizes `scipy.integrate.quad` which heavily bottlenecks performance due to numerical integrations. The "adaptive" solver uses computationally efficient sub-stepping, drastically improving simulation speeds (by around 17.5x in our synthetic benchmark, going from ~174 seconds to ~10 seconds).

**Action:** Whenever using `landlab`'s `Space` component, always specify `solver="adaptive"` instead of the default `"basic"` solver to greatly improve simulation performance.