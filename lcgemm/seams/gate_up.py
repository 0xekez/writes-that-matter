"""The MLP entry seam: ``pre_feedforward_layernorm(residual + delta) -> gate_up_proj``.

Two kernels:

1. :class:`~lcgemm.kernels.fused_norm.FusedNormResidual` reads ``residual``
   and ``delta``, forms their sum, normalises it and writes the R A-planes --
   plus the residual stream, which the rest of the layer needs.  The normalised
   ``X`` never lands.
2. the lcGEMM computes the R products and scatters them into ``Z``.

The GEMM's epilogue also emits the ``down`` seam's A-planes from
the finished ``Z`` tiles (``plane_scheme=``), which deletes the standalone
``L(SwiGLU)`` kernel from the next seam entirely.  That is the chain; see
:mod:`lcgemm.seams` for how it is selected and :mod:`lcgemm.chain_gauge` for the
interleaved-K gauge the planes come out in.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

import lcgemm.planes as lc_planes
from lcgemm.dlpack import dls
from lcgemm.kernels.hetero import LcGemmSm100Hetero
from lcgemm.kernels.fused_norm import FusedNormResidual
from lcgemm.kernels.sm100 import LcGemmSm100
from lcgemm.scheme import Scheme

# 624 CTA-pairs over 74 clusters is 8.43 waves, so the ninth runs 43% occupied;
# re-tiling it as a narrow second region recovers most of that (-38 us).  This
# is a property of the shape, not a preference -- `attn_merged` at 1.84 waves
# measures +17.2 us for the same split, and `o_proj` at 1.41 measures -27.
#
# 4 is also what the drain argument below wants, and at Muse 2048 -- the one
# shape that reaches the hetero path through this default -- it is what the
# occupancy rule picks anyway (region 1 is 296 tiles = 4 x 74 clusters).
# Measured there: persist 2 costs +8.7 us/layer in `gate_up`, +250 to +340
# us/pass overall, in both plan-construction orders.
PLANE_PERSIST = 4

# Exact-shape policies from the Muse and Qwen prefill campaigns.  At 8192, a one-wave persistent chained
# grid improves B-panel L2 reuse.  Qwen M=8192 instead fills the entire uniform
# N range in one p8 launch.  Unlisted shapes retain the established p4 and M-tile
# policies.
#
# Persistence also amortises the plane phase's *drain*: the last tile's SwiGLU
# and plane stores have no mainloop left to hide behind, and a CTA-pair pays that
# once however many tiles it walks.  At Qwen 2048 the drain is 8.4 us, so p1/p2/p4
# measure 463.1/444.1/435.7 us -- one drain apart each time -- and p4 is the
# largest persistence that still leaves 68 of the 74 clusters a tile.  Going to
# p4 there costs `down` +2.3 us/layer of L2 warmth and still wins by 8.3.
CHAIN_PERSIST = {
    (2048, 5120, 34816): 4,
    (4096, 5120, 34816): 4,
    (8192, 6656, 39936): 16,
    (8192, 5120, 34816): 8,
}

# Qwen3.8-27B: exact validated sizes use one uniform launch.  All Muse and
# other-model policy is unchanged.
QWEN_UNIFORM_SHAPES = {
    (2048, 5120, 34816),
    (4096, 5120, 34816),
    (8192, 5120, 34816),
}


class GateUpPlan:
    """The deployed ``gate_up`` producer, including its chained plane epilogue."""

    seam = "gate_up"

    def __init__(self, scheme: Scheme, shape: tuple[int, int, int], *,
                 plane_scheme: Scheme):
        self.scheme = scheme
        self.shape = shape                      # (M, K, N)
        self.plane_scheme = plane_scheme
        m, k, n = shape
        self.a = torch.empty((scheme.rank * (m // scheme.p), k // scheme.q),
                             device="cuda", dtype=torch.bfloat16)
        self.c = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        self.residual_out = torch.empty((m, k), device="cuda", dtype=torch.bfloat16)
        # ``down``'s A-planes, in the interleaved-K gauge, written by our GEMM's
        # epilogue.  ``N/2`` because SwiGLU halves the width.
        ps = plane_scheme
        self.planes = torch.empty(
            (ps.rank * (m // ps.p), (n // 2) // ps.q),
            device="cuda",
            dtype=torch.bfloat16,
        )
        # The consumer's output, when the consumer needs it cleared before it
        # accumulates into it; :func:`lcgemm.seams.build_chain` wires it up and
        # our epilogue clears it while the GEMM runs.  ``self.c`` means no clear.
        self.preclear = self.c
        self.stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        self._transform = self._gemm = None
        self._desc = "(not compiled)"

    # ------------------------------------------------------------------ offline
    def prepare_weight(self, w: torch.Tensor) -> torch.Tensor:
        return lc_planes.prepare_b(w, self.scheme)

    # -------------------------------------------------------------------- parts
    def a_planes(self, residual, delta, norm_weight, eps, *,
                 residual_out=None) -> torch.Tensor:
        """**The fused kernel.**  ``L_r(RMSNorm(residual + delta))``, ``(R*M2, K2)``."""
        r = self.residual_out if residual_out is None else residual_out
        if self._transform is None:
            fn = FusedNormResidual(self.scheme)
            self._transform = cute.compile(
                fn, *dls(delta, residual, norm_weight, self.a, r),
                cutlass.Float32(eps), self.stream)
            self._desc = fn.describe()
        self._transform(*dls(delta, residual, norm_weight, self.a, r),
                        cutlass.Float32(eps), self.stream)
        return self.a

    def gemm(self, a_planes, prepared_w, *, out=None) -> torch.Tensor:
        c = self.c if out is None else out
        if self._gemm is None:
            self._build_gemm(a_planes, prepared_w, c)
        self._gemm(*dls(a_planes, prepared_w, c, self.planes, self.preclear),
                   self.stream)
        return c

    def __call__(self, *pre_inputs, prepared_w, out=None, residual_out=None):
        a = self.a_planes(*pre_inputs, residual_out=residual_out)
        return self.gemm(a, prepared_w, out=out)

    def describe(self) -> str:
        return self._desc

    # ------------------------------------------------------------- construction
    def _build_gemm(self, a_planes, prepared_w, c) -> None:
        common = {
            "plane_scheme": self.plane_scheme,
            "persist": CHAIN_PERSIST.get(self.shape, PLANE_PERSIST),
            "preclear": self.preclear is not self.c,
        }
        if self.shape in QWEN_UNIFORM_SHAPES:
            g = LcGemmSm100(self.scheme, self.shape, **common)
            entry = g.call_planes
        else:
            g = LcGemmSm100Hetero(
                self.scheme,
                self.shape,
                **common,
            )
            entry = g.call_planes
        self._gemm = cute.compile(entry,
                                  *dls(a_planes, prepared_w, c, self.planes,
                                       self.preclear),
                                  self.stream)
        self._desc += "\n" + g.describe()
