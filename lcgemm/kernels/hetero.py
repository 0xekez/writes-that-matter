"""Heterogeneous two-region schedule for the B200 lcGEMM.

The problem is wave quantization.  With a 256x256 MMA tile the block grid is
8x78 = 624 CTA-pairs against 74 resident clusters: 8.43 waves, so the machine
runs a ninth wave that is only 43% occupied.  Measured, the kernel's SMs are
active for 89.9% of elapsed cycles against cuBLAS's 94.0%.

This uses a two-region schedule analogous to what FalconGEMM calls Split-Group
Parallelism: stop making every tile the same size. Region 1 takes the largest
whole number of full waves at the wide
tile; region 2 mops up the remainder with a finer spatial tile, so the tail
wave is both fuller and individually cheaper.

Splitting spatially (rather than across ranks) is what keeps this simple and
correct.  The tail may split both M and N when a shape-specific schedule shows
that a fuller device wave repays the smaller per-CTA tile.  Each sub-tile still
owns all R of its rank contributions in one CTA,
so the "first contribution is a plain store, the rest are reduce-adds"
protocol needs no pre-cleared C and no cross-CTA ordering.

The two regions write disjoint N ranges of C and are launched back to back on
one stream from a single compiled entry point.
"""

from __future__ import annotations

from typing import Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from lcgemm.scheme import Scheme
from lcgemm.kernels.sm100 import LcGemmSm100, num_clusters


def plan_boundary(tiles_m: int, tiles_n: int, clusters: int) -> int:
    """The region-1/region-2 split, in wide-tile N units.

    Region 1 gets the largest whole number of full waves that lands on an N
    column boundary; whatever is left goes to region 2 at finer granularity.
    """
    full_waves = (tiles_m * tiles_n) // clusters
    return min(tiles_n, (full_waves * clusters) // tiles_m)


class LcGemmSm100Hetero:
    """Two-region launcher wrapping the single-region ``LcGemmSm100`` kernel."""

    def __init__(
        self,
        scheme: Scheme,
        mnk: Tuple[int, int, int],
        *,
        plane_scheme: Scheme,
        persist: int,
        preclear: bool = False,
    ):
        self.plane_scheme = plane_scheme
        # Persistence is region 1's business: region 2 is under one wave, so
        # walking it persistently would only serialise it.
        max_persist = persist
        self.scheme = scheme
        self.mnk = mnk
        self.split_n = 2
        self.split_m = 2 if mnk == (2048, 6656, 39936) else 1
        split_m = self.split_m
        tile_m, tile_n = 256, 256
        p, q, s = scheme.shape
        m2, n2 = mnk[0] // p, mnk[2] // s
        tiles_m, tiles_n = m2 // tile_m, n2 // tile_n

        n_boundary = plan_boundary(tiles_m, tiles_n, num_clusters())
        self.n_boundary = n_boundary
        self.tiles_n = tiles_n
        region1_pairs = tiles_m * n_boundary
        divisors = [p for p in range(max_persist, 0, -1)
                    if region1_pairs % p == 0]
        occupancy_safe = [p for p in divisors
                          if region1_pairs // p >= num_clusters()]
        persist = max(occupancy_safe or divisors)

        self.region1 = (
            LcGemmSm100(
                scheme,
                mnk,
                mma_tiler_mn=(tile_m, tile_n),
                tile_n_range=(0, n_boundary),
                plane_scheme=plane_scheme,
                persist=persist,
                preclear=preclear,
            )
            if n_boundary > 0
            else None
        )
        # Region 2 re-tiles the same output columns spatially.  N tile
        # indices are scaled by split_n; M is fully re-tiled at tile_m/split_m.
        # It inherits the pre-clear when region 1 is empty, which happens when
        # the shape has fewer than one full wave of wide tiles to give it.
        tail = tiles_n - n_boundary
        self.region2 = (
            LcGemmSm100(
                scheme,
                mnk,
                mma_tiler_mn=(tile_m // split_m, tile_n // self.split_n),
                tile_n_range=(n_boundary * self.split_n, tail * self.split_n),
                plane_scheme=plane_scheme,
                preclear=preclear and self.region1 is None,
            )
            if tail > 0
            else None
        )
        if preclear and self.region1 is None and self.region2 is None:
            raise ValueError(f"{mnk} has no region to clear the consumer from")

    def describe(self) -> str:
        parts = [f"LcGemmSm100Hetero[{self.scheme.name}] "
                 f"split_m={self.split_m} split_n={self.split_n} boundary={self.n_boundary}/{self.tiles_n} persist={self.region1.persist if self.region1 else 1}"]
        for name, region in (("region1", self.region1), ("region2", self.region2)):
            parts.append(f"  {name}: " + (region.describe().replace("\n", "\n  ")
                                          if region else "(empty)"))
        return "\n".join(parts)

    @cute.jit
    def call_planes(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
                    mS: cute.Tensor, mZ: cute.Tensor, stream: cuda.CUstream):
        """Both regions, each emitting the consumer's A-planes for its own N range.

        The regions own disjoint N columns, so they own disjoint plane columns
        too and the single-writer property survives the split.
        """
        if cutlass.const_expr(self.region1 is not None):
            self.region1.call_planes(mA, mB, mC, mS, mZ, stream)
        if cutlass.const_expr(self.region2 is not None):
            self.region2.call_planes(mA, mB, mC, mS, mZ, stream)
