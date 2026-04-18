## 2025-04-17 - [Optimize Landlab Space Component Solver]
**Learning:** The `landlab`'s `Space` component uses `"basic"` solver by default, which performs computationally heavy numerical integrations (`scipy.integrate.quad`). By specifying `solver="adaptive"`, it bypasses these integrations in favor of sub-stepping, significantly speeding up simulations without losing precision.
**Action:** Always specify `solver="adaptive"` when initializing the `Space` component in Landlab for better performance.## 2024-04-18 - [Python tight loops and numpy]
**Learning:** Calling functions like np.mean() or repeatedly appending to arrays inside a tight inner simulation loop represents a significant overhead even if individual iterations are fast.
**Action:** Always prefer mathematical refactoring (e.g., maintaining a running sum and division count) or vectorized numpy operations over list-building inside iterative while-loops for numerical stability models.
