"""Memoised `from_dlpack`, shared by every seam Plan.

A compiled CuTe kernel takes DLPack-wrapped tensors, and wrapping is a Python
round trip -- an export plus a CuTe tensor construction, a few microseconds each.
That is invisible in a benchmark, which wraps once at build time and reuses the
argument tuple.  It is *not* invisible in the model: once the Plans took an
optional `out=` destination (they must, since 52 layers share one Plan), every
call re-wrapped its arguments, and at ~5 tensors x 2 kernels x 52 layers that is
~520 wraps per forward pass.  Measured, it cost about **2 ms of wall time per
pass** while the GPU time was unchanged -- the whole gap between the GPU and wall
speedups of the multi-seam patch.

Wrapping depends only on the buffer's address, shape, stride and dtype, so it is
safe to memoise on exactly those.  If the caching allocator hands back an address
that a freed tensor used, a cached wrapper for that key describes the new tensor
correctly, because those four things are all a wrapper holds.

The cache is bounded because the outputs are freshly allocated each layer and the
allocator recycles a small set of addresses in steady state -- but "small" is not
"provably small", so it evicts rather than growing without limit.
"""

from __future__ import annotations

from cutlass.cute.runtime import from_dlpack

_MAX = 512
_CACHE: dict = {}


def dl(t):
    """Wrap one tensor for a compiled CuTe kernel, reusing a previous wrap."""
    # `detach` because weights arrive as nn.Parameters in the model and DLPack
    # refuses anything requiring grad.  It is a free view.
    t = t.detach()
    key = (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype)
    w = _CACHE.get(key)
    if w is None:
        if len(_CACHE) >= _MAX:
            _CACHE.clear()
        w = from_dlpack(t, assumed_align=16)
        _CACHE[key] = w
    return w


def dls(*tensors):
    return tuple(dl(t) for t in tensors)
