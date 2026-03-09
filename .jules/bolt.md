## 2025-03-05 - [PARCEM Vectorization and In-Place Operations]
**Learning:** Moving constant array allocations outside of tight inner loops and heavily utilizing in-place operations (`np.add(out=)`, `np.multiply(out=)`) coupled with eliminating explicit dictionary/list append patterns per-step produced ~25% speedup without sacrificing readability.
**Action:** Always pre-allocate NumPy arrays and reuse them with `out=` arguments when doing iterative numeric computations over fixed-size domains.
