"""The chained ``down`` seam: plane-consuming low-complexity GEMM.

    S = silu(Z[:, :I]) * Z[:, I:]      # never lands in HBM
    out = S @ W_down^T                 M=4096, K=19968, N=6656

The artifact ships one fixed chained plan. The ``gate_up`` epilogue has
already applied SwiGLU and emitted the A-planes, so the consumer launches no
activation transform.  Its static weight is permuted into the same interleaved-K
gauge once, during model installation.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch

import lcgemm.chain_gauge as chain_gauge
from lcgemm.dlpack import dls
from lcgemm.kernels.down import LcGemmDown, LcGemmDownHetero
from lcgemm.kernels.down_rank_split import LcGemmDownRankSplit
from lcgemm.scheme import Scheme

# Shapes whose ``down`` is narrow enough that rank has to carry the parallelism.
# Its contributions then come from different CTAs, so its output has to arrive
# zeroed -- and the producing ``gate_up`` epilogue does that while it runs,
# instead of a ``ZeroFill`` launch doing it in between.
#
# Measured on every shape in this set, paired: both plans alternating in one
# process over one set of prepared weights, cold L2 per pass, 300 passes a run,
# median of the per-round deltas.  Positive means the folded clear is faster.
#
#   Qwen 2048   +111 +- 49 us/pass  (3/3 runs)   gate_up +2.3, clear -5.1 us/layer
#   Qwen 4096   +169 +- 128         (4/5)        gate_up +5.0, clear -8.5
#   Muse 2048   +171 +- 46          (4/4)        gate_up +4.6, clear -6.5
#
# The clear costs 0.11-0.17 us/MB inside the GEMM against 0.20-0.24 as its own
# launch, so it moves at about twice the effective bandwidth once it is hidden
# there -- but the GEMM keeps most of that, and what reaches the pass is mainly
# the launch the chain no longer makes.  Muse 4096/8192 ``down`` is not
# rank-split and needs no clear at all.
RANK_SPLIT_SHAPES = {
    # M=512 and M=1024 are evaluation-only (see `lcgemm.integrate`); rank has
    # to carry the parallelism there for the same reason it does at 2048, only
    # more so -- the all-rank grid is 10 to 20 CTA-pairs, a quarter of a wave.
    (512, 19968, 6656),
    (512, 17408, 5120),
    (1024, 19968, 6656),
    (1024, 17408, 5120),
    (2048, 19968, 6656),
    (2048, 17408, 5120),
    (4096, 17408, 5120),
    (8192, 17408, 5120),
}


class DownPlan:
    """The deployed ``down`` plan, coupled to its ``gate_up`` producer."""

    seam = "down"

    def __init__(self, scheme: Scheme, shape: tuple[int, int, int], *, producer):
        self.scheme = scheme
        self.shape = shape                      # (M, K, N)
        self.producer = producer
        m, k, n = shape
        self.c = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        self.rank_split = shape in RANK_SPLIT_SHAPES
        self.stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        self._planes = self._gemm = None
        self._desc = "(not compiled)"

    # ------------------------------------------------------------------ offline
    def prepare_weight(self, w: torch.Tensor) -> torch.Tensor:
        """The R combined B-planes, ``(R*N/s, K/q)``.  Once per layer, untimed.

        Chained, the K-permutation of the *static* operand is the entire cost of
        the interleaved gauge: a decomposition's K-partition is a free gauge as
        long as A's and B's agree, so the consuming GEMM needs no change at all.
        """
        return chain_gauge.prepare_down_b(w, self.scheme, chain_gauge.CHUNK)

    # -------------------------------------------------------------------- parts
    def a_planes(self) -> torch.Tensor:
        """Return the planes already produced by the chained gate/up epilogue."""
        return self.producer.planes

    def gemm(self, a_planes, prepared_w, *, out=None) -> torch.Tensor:
        c = self.c if out is None else out
        if self._gemm is None:
            if self.rank_split:
                g = LcGemmDownRankSplit(self.scheme, self.shape)
            elif self.shape == (8192, 19968, 6656):
                g = LcGemmDown(self.scheme, self.shape, raster_order="n_fast")
            elif self.shape == (4096, 19968, 6656):
                g = LcGemmDownHetero(self.scheme, self.shape)
            else:
                raise ValueError(f"no validated down schedule for {self.shape}")
            self._gemm = cute.compile(g, *dls(a_planes, prepared_w, c), self.stream)
            self._desc = g.describe()
        self._gemm(*dls(a_planes, prepared_w, c), self.stream)
        return c

    def __call__(self, *, prepared_w, out=None):
        return self.gemm(self.a_planes(), prepared_w, out=out)

    def describe(self) -> str:
        return "  chained (a_planes are free)\n" + self._desc
