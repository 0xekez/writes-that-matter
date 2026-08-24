"""The low-complexity GEMM operand-plane contract.

A rank-R scheme replaces the ``p*q*s`` block products of ``A @ B`` with R
products of *combined planes*::

    Atilde_r = sum_ik a[r][i,k] A[i][k]      Btilde_r = sum_kj b[r][k,j] B[k][j]

so every operand reaches the mainloop as R planes stacked along the leading
dimension -- one TMA descriptor per operand with the rank folded into the tile
coordinate.  This module is the whole of that contract:

* :func:`apply_terms` forms signed plane combinations inside fused producers;
* :func:`prepare_b` forms B planes once, offline, because weights are fixed.

The artifact intentionally contains no standalone A transform: both deployed
paths produce A planes directly from the operation preceding the GEMM.
"""

from __future__ import annotations

import cutlass
import torch

from lcgemm.scheme import Scheme

# ------------------------------------------------------------------ kernel side
def apply_terms(terms, frag):
    """``sum_ik coeff * frag[i, k]`` as a TensorSSA.

    ``frag`` is any mapping from a block index pair to a loaded fragment.  Lives
    here so the ``+-1`` fast path -- a signed sum, no multiplies, which is what
    makes these schemes nearly free once the operands are loaded -- exists in one
    place for every producer.  Ablating it (same loads, same stores, no adds)
    costs 0.1 us of the A transform's 24.9: it really is free.

    Indexable rather than a callback on purpose: a closure would be rejected by
    the DSL when this is called from inside a staged (dynamic) loop.
    """
    acc = None
    for i, k, coeff in terms:
        term = frag[i, k]
        if cutlass.const_expr(abs(coeff) != 1):
            term = term * float(abs(coeff))
        if cutlass.const_expr(acc is None):
            acc = -term if cutlass.const_expr(coeff < 0) else term
        elif cutlass.const_expr(coeff < 0):
            acc = acc - term
        else:
            acc = acc + term
    return acc


# --------------------------------------------------------------------- B side
def prepare_b(w: torch.Tensor, scheme: Scheme) -> torch.Tensor:
    """Stack the R combined weight planes.  ``w`` is ``(N, K)``, K-major.

    Offline: B is a fixed weight in this workload.  Summation is in fp32 with a
    single rounding to the storage dtype, so the planes carry no more error than
    one rounding of an exact combination.
    """
    n, k = w.shape
    n2, k2 = n // scheme.s, k // scheme.q
    out = torch.empty((scheme.rank * n2, k2), dtype=w.dtype, device=w.device)
    acc = torch.empty((n2, k2), dtype=torch.float32, device=w.device)
    for r in range(scheme.rank):
        acc.zero_()
        for kk, j, coeff in scheme.b_terms[r]:
            blk = w[j * n2 : (j + 1) * n2, kk * k2 : (kk + 1) * k2]
            acc.add_(blk.float(), alpha=float(coeff))
        out[r * n2 : (r + 1) * n2] = acc.to(w.dtype)
    return out
