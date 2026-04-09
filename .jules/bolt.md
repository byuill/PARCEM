
## 2026-04-09 - [Optimize Landlab Space Component with Adaptive Solver]
**Learning:** Found a specific codebase performance pattern for Landlab's `Space` component. The default solver (`basic`) uses heavy numerical integrations (`scipy.integrate.quad`), which is a significant bottleneck for execution time.
**Action:** Always specify `solver="adaptive"` when instantiating the `Space` component to bypass this bottleneck in favor of sub-stepping, enabling an ~60x speed up.
