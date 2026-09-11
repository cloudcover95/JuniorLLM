"""Optional Triton fused-dot. Must equal CPU bitlinear y."""
from __future__ import annotations

KERNEL = None

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _dot(x_ptr, w_ptr, y_ptr, n, BLOCK: tl.constexpr):
        off = tl.arange(0, BLOCK)
        m = off < n
        x = tl.load(x_ptr + off, mask=m, other=0)
        w = tl.load(w_ptr + off, mask=m, other=0)
        acc = tl.sum(x * w)
        tl.store(y_ptr, acc)

    KERNEL = _dot
except Exception:
    KERNEL = None
