"""Host references for the fused ``gate_up -> SwiGLU -> down A-planes`` chain.

The one structural idea in this directory
--------------------------------------------------------------------------
``down``'s A-planes are ``L_r(S)``, ``S = silu(Z[:, :I]) * Z[:, I:]``, and the
lattice of ``L_r`` pairs S-column ``l`` with S-column ``l + K2``.  With the
*natural* K-partition (two contiguous halves) those two columns are 9984 apart,
which lands them in two different CTAs of the producing ``gate_up`` GEMM -- the
obstruction described in the paper's linear-functional-fusion section.

But the K-partition is a **free gauge**.  The block decomposition
``C = sum_j A^{(.,j)} B^{(j,.)}`` is valid for *any* partition of the K index
set into ``q`` equal parts, as long as A's and B's parts agree -- and B here is
``down_proj.weight``, a static tensor we may permute offline for nothing.

So partition S's columns by a **chunk-interleave** of width ``W``::

    block j  = { c : (c // W) % 2 == j }
    index l  = (c // 2W) * W + (c % W)

The lattice partner of column ``c`` is then ``c XOR W`` -- ``W`` apart, not
9984.  Choose ``W = 64`` and both partners sit inside the same 256-wide (or
128-wide) N-tile of the ``gate_up`` GEMM, which *already* owns

* both gate/up column blocks (``s = 2`` splits N at exactly 19968), and
* both row blocks (``p = 2`` splits M at 2048, the same split ``down`` uses),

at one ``(tile_m, tile_n)``.  All eight ``Z`` elements behind one A-plane
element therefore belong to one CTA.  No cross-CTA reduction, no pre-zeroed
planes, no ordering protocol: every plane element has exactly one writer, so the
epilogue can use plain TMA stores.

``W`` is also why the interleave is not element-wise (``W = 1``): a thread wants
its eight lattice partners contiguous *within a block* so that both the loads
and the plane stores are 128-bit.  ``W = 64`` gives that and still fits the
128-wide tail tile of the heterogeneous schedule.
"""

from __future__ import annotations

import torch

from lcgemm.planes import prepare_b

CHUNK = 64  # W above.  Must divide the narrowest N-tile / 2 and be >= 8.


# --------------------------------------------------------------------------
# The gauge itself
# --------------------------------------------------------------------------
def interleave_perm(width: int, chunk: int = CHUNK) -> torch.Tensor:
    """Column order that turns the chunk-interleave into two contiguous halves.

    ``x[:, interleave_perm(K)]`` has block ``j`` occupying columns
    ``[j*K/2, (j+1)*K/2)``, i.e. it is the *natural* form of the permuted
    matrix, so every downstream reference can stay unchanged.
    """
    if width % (2 * chunk):
        raise ValueError(f"width {width} not divisible by 2*chunk {2 * chunk}")
    c = torch.arange(width)
    j = (c // chunk) % 2
    l = (c // (2 * chunk)) * chunk + (c % chunk)
    perm = torch.empty(width, dtype=torch.long)
    perm[j * (width // 2) + l] = c
    return perm


def permute_k(x: torch.Tensor, chunk: int = CHUNK) -> torch.Tensor:
    """Apply the gauge to a K-major tensor's last dimension."""
    return x[..., interleave_perm(x.shape[-1], chunk).to(x.device)].contiguous()


# --------------------------------------------------------------------------
# References
# --------------------------------------------------------------------------
def swiglu(z: torch.Tensor) -> torch.Tensor:
    i = z.shape[-1] // 2
    zf = z.float()
    return torch.nn.functional.silu(zf[:, :i]) * zf[:, i:]


def chain_a_planes(z: torch.Tensor, scheme, chunk: int = CHUNK,
                   out_dtype=torch.bfloat16) -> torch.Tensor:
    """``L_r(SwiGLU(Z))`` in the interleaved gauge, as ``(R*M2, K2)``.

    fp32 throughout with a single rounding at the end -- the same rounding
    structure the fused epilogue has, so a correct kernel matches this closely
    rather than approximately.
    """
    s = permute_k(swiglu(z), chunk)          # (M, I), gauge applied
    m, i = s.shape
    m2, k2 = m // scheme.p, i // scheme.q
    out = torch.empty((scheme.rank * m2, k2), dtype=out_dtype, device=z.device)
    for r, terms in enumerate(scheme.a_terms):
        acc = torch.zeros(m2, k2, dtype=torch.float32, device=z.device)
        for bi, bj, coeff in terms:
            acc += float(coeff) * s[bi * m2:(bi + 1) * m2, bj * k2:(bj + 1) * k2]
        out[r * m2:(r + 1) * m2] = acc.to(out_dtype)
    return out


def prepare_down_b(w: torch.Tensor, scheme, chunk: int = CHUNK) -> torch.Tensor:
    """``down``'s B-planes, with the gauge folded in.  Offline, once per layer.

    The permutation of the *static* operand is the entire cost of the gauge, so
    this is the whole of what a chained consumer does differently -- and it is
    called from :meth:`lcgemm.seams.down.DownPlan.prepare_weight` rather than
    restated there, because a gauge applied on one side only is a silently wrong
    answer rather than a crash.
    """
    return prepare_b(permute_k(w, chunk), scheme)
