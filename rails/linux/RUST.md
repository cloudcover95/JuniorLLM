No Rust FFI crate in-tree.
A `cdylib` that calls the same 20-float pack as Python loses to ctypes+C or pure Python on T0 notes (FFI + alloc > pack).
Revisit if a single call is n≥ 1e5 and stays hot. Until then: C .so or Python winsor.
