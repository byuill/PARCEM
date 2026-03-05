
## 2025-05-18 - Avoid np.mean on Dynamically Grown Lists in Tight Numerical Loops
**Learning:** Profiling the river evolution simulation revealed that using `list.append()` inside the sub-timestep loop and then calling `np.mean()` on that list outside the loop to calculate an average flux rate was a noticeable performance bottleneck. Specifically, `np.mean()` has overhead that adds up when called thousands of times per timestep.
**Action:** Instead of storing all values in a list to compute the mean later, maintain a running sum (`qs_step_sum`) and a counter (`qs_step_count`), then perform a simple division at the end. This reduces both memory allocation overhead and the function call overhead of `np.mean`.
