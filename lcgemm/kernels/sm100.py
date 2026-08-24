"""Low-complexity GEMM for NVIDIA Blackwell SM100 (B200), in CuTe DSL.

Computes ``C = A @ B^T`` with ``A: (M, K)``, ``B: (N, K)`` (both K-major, i.e.
the layout ``torch.nn.functional.linear`` hands us) using a rank-R bilinear
decomposition of the block product instead of the naive ``p*q*s`` block
multiplications.  For the 2x2x2 rank-7 schemes in ``schemes/`` that is 7/8 of
the FLOPs.

Structure, and why:

* One CTA-pair owns one ``(m, n)`` position of the *block* tile grid and walks
  **all R ranks** there, and the grid is walked in near-square blocks so the
  tiles resident at any instant need as few operand panels as possible
  (``_group_m``).  Every contribution to a given output tile is then
  produced by a single CTA, so the first contribution can be a plain TMA store
  (which initialises the block -- C needs no pre-clear) and the rest TMA
  reduce-adds, with no cross-CTA ordering to arrange.
* The mainloop is the standard SM100 warp-specialised tcgen05 pipeline (TMA
  producer warp -> UMMA warp -> 4 epilogue warps) with 2-CTA MMA.  Two
  accumulator stages in TMEM let rank ``r+1``'s MMA overlap rank ``r``'s
  epilogue; one stage costs 92 us, and three do not fit.
* Destinations sharing a coefficient share one accumulator->smem staging pass.
* **Postsum CSE.**  If the scheme carries a plan (``Scheme.has_postsum``), a
  step may leave ``ACCUMULATE`` set so its product lands on top of the one two
  steps earlier -- the same TMEM stage.  A run of such steps is a chain whose
  prefix sums are the shared postsum subexpressions, summed for free in the
  accumulator that had to exist anyway, and only those prefixes are written
  out.  For ``2x2_postsum_cse`` that is 7 global writes instead of 12.
  The plane epilogue is indifferent to it: CSE changes *which* partial sums
  reach C and in what order, never the four (i, j) blocks a CTA-pair owns nor
  the fact that they are final by ``c_pipeline.producer_tail()``, which is the
  only thing the read-back depends on.

Everything scheme-specific arrives as ``lc_scheme.Scheme`` and is baked in at
trace time, so the rank loop and the scatter table unroll to straight-line code.
The *tile* loop of a persistent CTA does not: it rolls, so `persist` costs no
code.
See README.md for the measurements behind each choice, including the ones that
did not work.

The shipping plane epilogue adds a second phase after the rank loop: 16
dedicated warps re-read the four C tiles the CTA just finished, apply SwiGLU
across the gate/up pair, and emit the next GEMM's A-planes one tile behind the
mainloop. The research-only standalone GEMM mode is not part of this artifact.
"""

from __future__ import annotations

from typing import Optional, Tuple, Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import const_expr
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import LayoutEnum

from lcgemm.planes import apply_terms
from lcgemm.scheme import Scheme


def num_clusters(device: int = 0) -> int:
    """Resident 2-CTA MMA clusters: one per pair of SMs."""
    err, dev = cuda.cuDeviceGet(device)
    assert err == cuda.CUresult.CUDA_SUCCESS, err
    err, sms = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev
    )
    assert err == cuda.CUresult.CUDA_SUCCESS, err
    return sms // 2


def _group_m(tiles_m: int, clusters: int) -> int:
    """How many M tiles the walk keeps open at once.

    The machine holds `clusters` consecutive tiles of the walk resident, so the
    operands it needs at any instant are whatever rectangle those tiles span: a
    block of ``a`` M tiles by ``clusters / a`` N tiles needs ``R * (a +
    clusters / a)`` operand panels.  The MMA tile is square, so an M panel and
    an N panel are the same size and that sum is minimised at ``a =
    sqrt(clusters)`` -- 8.6 on a 148-SM B200.

    Plain M-fastest order is ``a = tiles_m``.  That is why this never mattered
    before: ``tiles_m`` is 4 at M=2048 and 8 at M=4096, already optimal.  At
    M=8192 it is 16, the resident set goes 158 -> 189 MB against 126 MB of L2,
    and the mainloop pays 54 us a layer for it (Qwen; 35 us on Muse's
    heterogeneous region 1).  Restricted to divisors of ``tiles_m`` because a
    partial block would give some CTAs a different N stride.
    """
    return min((d for d in range(1, tiles_m + 1) if tiles_m % d == 0),
               key=lambda d: d + clusters / d)


class LcGemmSm100:
    """Scheme-parameterised low-complexity GEMM for SM100."""

    # Warp roles: 4 epilogue warps, then the UMMA warp, then the TMA warp.
    EPI_WARPS = (0, 1, 2, 3)
    MMA_WARP = 4
    TMA_WARP = 5

    TMEM_ALLOC_BAR_ID = 3
    EPI_SYNC_BAR_ID = 4
    PLANE_SYNC_BAR_ID = 5

    # 128-bit accesses in the plane epilogue: 8 bf16 per thread per instruction.
    PLANE_VEC = 8
    PLANE_CHUNK = 64
    PLANE_WARPS = 16

    # bf16 in, bf16 out, fp32 accumulate.  Not a choice: bf16 is the dtype the
    # activations and the weights are already in, and fp32 is the only
    # accumulator a bf16 tcgen05 MMA has.
    AB_DTYPE: Type[cutlass.Numeric] = cutlass.BFloat16
    C_DTYPE: Type[cutlass.Numeric] = cutlass.BFloat16
    ACC_DTYPE: Type[cutlass.Numeric] = cutlass.Float32

    #: Accumulator (TMEM) stages.  Two is the only viable count: with one, rank
    #: ``r+1``'s MMA cannot overlap rank ``r``'s epilogue and the kernel costs
    #: **+92 us**; three do not fit in TMEM at a 256x256 tile.  Two is also what
    #: makes step ``t`` own stage ``t % 2``, which is the assumption every
    #: postsum CSE plan is searched under -- so this is load-bearing twice and
    #: is a constant rather than an argument.
    ACC_STAGE = 2

    #: Epilogue smem buffers the accumulator->smem staging passes rotate through.
    #: Re-swept under CSE after the merge: ``epi_stage=1`` measured **+1.3 us**
    #: on ``2x2_postsum_cse`` and -1.0 on ``2x2_locality_ordered`` (MERGE.md,
    #: "plane-epilogue knobs re-derived under CSE"), so 2 is kept.
    EPI_STAGE = 2

    #: K per MMA instruction.  The operand K-tile the SM100 bf16 tcgen05 atom
    #: takes; no other value has been run on this kernel.
    MMA_TILER_K = 64

    def __init__(
        self,
        scheme: Scheme,
        mnk: Tuple[int, int, int],
        plane_scheme: Scheme,
        mma_tiler_mn: Tuple[int, int] = (256, 256),
        tile_n_range: Optional[Tuple[int, int]] = None,
        persist: int = 1,
        preclear: bool = False,
    ):
        self.scheme = scheme
        self.plane_scheme = plane_scheme
        self.plane_chunk = self.PLANE_CHUNK
        # Sixteen dedicated warps overlap tile t's plane epilogue with tile
        # t+1's mainloop. Shared memory, not registers, limits occupancy here.
        self.m, self.k, self.n = mnk
        p, q, s = scheme.shape
        if self.m % p or self.k % q or self.n % s:
            raise ValueError(f"{mnk} not divisible by block grid {scheme.shape}")
        # Per-block extents: the shape every one of the R products actually has.
        self.m2, self.k2, self.n2 = self.m // p, self.k // q, self.n // s

        self.ab_dtype, self.c_dtype, self.acc_dtype = (
            self.AB_DTYPE, self.C_DTYPE, self.ACC_DTYPE)
        self.mma_tiler = (*mma_tiler_mn, self.MMA_TILER_K)
        self.acc_stage, self.epi_stage = self.ACC_STAGE, self.EPI_STAGE

        # The epilogue's per-step publication table, and whether each step opens
        # a fresh accumulator.  Both come off the scheme with no branch here:
        # the plain scatter *is* the postsum plan whose chains all have length
        # one, so ``Scheme`` derives one pair of tables for either kind and this
        # kernel is written once.  (They used to be re-derived here from
        # ``has_postsum``, ``postsum_groups``/``dest_groups`` and ``clear``,
        # which is the same 8 schemes' worth of tables computed twice.)
        #
        # With `persist > 1` the chains still close inside one tile: `t` and
        # `t + 2` share a TMEM stage whatever index the tile *starts* on (the
        # stages simply alternate), and `Scheme.validate_postsum` requires steps
        # 0 and 1 to clear, so tile `t+1`'s first two steps never inherit tile
        # `t`'s accumulator.  That holds for odd R, where the starting parity
        # flips every tile.  It is what `ACC_STAGE == 2` buys and the reason the
        # plan may assume step `t` owns TMEM stage `t % 2`.
        self.epi_groups = scheme.epi_groups
        self.acc_clear = scheme.acc_clear

        # 2-CTA MMA, i.e. a 2x1 cluster.  Wider clusters would engage TMA
        # multicast; that path measured both slower and wrong here, so the
        # cluster is not a parameter (see README, "measured negative results").
        self.atom_thr = 2
        self.cta_group = tcgen05.CtaGroup.TWO
        self.cluster_shape_mnk = (self.atom_thr, 1, 1)
        self.cta_tile = (self.mma_tiler[0] // self.atom_thr, *self.mma_tiler[1:])

        for label, extent, tile in (
            ("M2", self.m2, self.mma_tiler[0]),
            ("N2", self.n2, self.mma_tiler[1]),
            ("K2", self.k2, self.mma_tiler[2]),
        ):
            if extent % tile:
                raise ValueError(f"block extent {label}={extent} not divisible by tile {tile}")

        self.tiles_m = self.m2 // self.mma_tiler[0]
        self.tiles_n = self.n2 // self.mma_tiler[1]
        self.k_tiles = self.k2 // self.mma_tiler[2]
        # A launch may cover only a slice of the N tile range, so a second launch
        # with a narrower tile can mop up the partial last wave (lcgemm_hetero).
        self.n_start, self.n_count = tile_n_range or (0, self.tiles_n)
        if self.n_start < 0 or self.n_start + self.n_count > self.tiles_n:
            raise ValueError(f"tile_n_range {(self.n_start, self.n_count)} outside "
                             f"[0, {self.tiles_n})")
        # 1-D grid walked in bands of `group_m` rows: within a band every M tile
        # reads the same B panel, and the band is narrow enough that its A panels
        # survive beside them.  See `_group_m`.
        self.num_pairs = self.tiles_m * self.n_count
        self.group_m = _group_m(self.tiles_m, num_clusters())
        self.threads_per_cta = 32 * (len(self.EPI_WARPS) + 2 + self.PLANE_WARPS)
        # Persistent CTA-pairs: each walks `persist` tiles, strided by the grid
        # so that one stride is exactly one of today's waves -- the tile *order*,
        # and so B's L2 reuse, is unchanged.  This is what lets the plane phase
        # of tile t overlap the mainloop of tile t+1.
        self.persist = persist
        # Clearing the consumer's output from this kernel rather than from a
        # ``ZeroFill`` launch measures -3.2 us per layer on the gate_up + down
        # pair, and deletes a launch: 21 MB of stores disappear into a kernel
        # running at a quarter of DRAM peak.
        self.preclear = preclear
        if self.num_pairs % persist:
            raise ValueError(f"persist={persist} does not divide {self.num_pairs} tiles")
        self.grid_pairs = self.num_pairs // persist
        self._plane_setup()

    # ------------------------------------------------------- plane epilogue
    def _plane_setup(self):
        """Thread map for the ``SwiGLU -> next GEMM's A-planes`` phase.

        The phase is legal at all because of two coincidences and one gauge
        choice (implemented in :mod:`lcgemm.chain_gauge`):

        * ``s = 2`` splits this GEMM's N at exactly the gate/up boundary and
          ``p = 2`` splits M the same way the consumer does, so the CTA already
          owns C blocks ``(i, j)`` for every ``i, j`` at one ``(tile_m, tile_n)``;
        * the consumer's K-partition is a **free gauge**, so it is chosen as a
          chunk-interleave of width ``plane_chunk``, which puts the consumer's
          lattice partner of column ``c`` at ``c ^ plane_chunk`` -- inside this
          CTA's own N-tile instead of 9984 columns away.

        Together: every element of the consumer's A-planes has exactly one
        writer, so the phase needs plain stores, no pre-zeroing and no
        cross-CTA ordering.
        """
        ps, w, vec = self.plane_scheme, self.plane_chunk, self.PLANE_VEC
        if (self.scheme.p, self.scheme.s) != (2, 2):
            raise ValueError("the plane epilogue needs a 2x_x2 producer scheme "
                             f"(gate/up on the s axis), got {self.scheme.shape}")
        if (ps.p, ps.q) != (2, 2):
            raise ValueError(f"the plane epilogue needs a 2x2x_ consumer scheme, "
                             f"got {ps.shape}")
        tile_cols = self.cta_tile[1]
        if w % vec or tile_cols % (2 * w):
            raise ValueError(f"plane_chunk {w} must be a multiple of {vec} and "
                             f"2*{w} must divide the N tile {tile_cols}")
        # A slot is one 128-bit vector of one chunk; a thread owns the same slot
        # in both chunks of its pair, which is what makes it self-sufficient.
        self.plane_pairs = tile_cols // (2 * w)
        self.plane_vecs_per_chunk = w // vec
        self.plane_slots = self.plane_pairs * self.plane_vecs_per_chunk
        # The phase is covered by the dedicated warps. The rendezvous barrier
        # also includes the epilogue warps that publish C.
        threads = 32 * self.PLANE_WARPS
        self.plane_threads = threads
        self.plane_bar_threads = 32 * (len(self.EPI_WARPS) + self.PLANE_WARPS)
        if threads % self.plane_slots or self.cta_tile[0] % (threads // self.plane_slots):
            raise ValueError(f"{threads} threads do not tile {self.cta_tile[0]} rows "
                             f"x {self.plane_slots} slots")
        self.plane_rows_per_step = threads // self.plane_slots
        self.plane_steps = self.cta_tile[0] // self.plane_rows_per_step

    # ------------------------------------------------------------------ setup
    def _setup(self):
        """Derive layouts and stage counts.  Runs at trace time, inside a context."""
        self.tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.ab_dtype, self.ab_dtype, OperandMajorMode.K, OperandMajorMode.K,
            self.acc_dtype, self.cta_group, self.mma_tiler[:2],
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (self.tiled_mma.thr_id.shape,)
        )
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile, True, LayoutEnum.ROW_MAJOR, self.c_dtype
        )
        # Plain ints for reporting: the cute Layouts in epi_tile are only valid
        # to touch inside an MLIR context.
        self.epi_tile_mn = tuple(cute.size(t) for t in self.epi_tile)
        # Which epilogue smem buffer each store uses, keyed by its position in
        # the (rank, sub_m, sub_n, coefficient group) nest.  A table rather than
        # a running counter because the tile loop is dynamic: a counter assigned
        # inside it would become an ``scf.for`` iter_arg, i.e. a runtime value,
        # and an smem buffer index has to be a constant.  A persistent CTA
        # restarts the rotation on every tile, so it also has to *close* on the
        # tile boundary, or tile t+1's first store would reuse a buffer whose
        # TMA is still in flight.
        self.epi_bufs, stores = {}, 0
        for r in range(self.scheme.rank):
            for sm in range(self.cta_tile[0] // self.epi_tile_mn[0]):
                for sn in range(self.cta_tile[1] // self.epi_tile_mn[1]):
                    for gi in range(len(self.epi_groups[r])):
                        self.epi_bufs[r, sm, sn, gi] = stores % self.epi_stage
                        stores += 1
        if self.persist > 1 and stores % self.epi_stage:
            raise ValueError(f"{stores} epilogue stores per tile do not close the "
                             f"{self.epi_stage}-buffer rotation")

        def smem_ab(stages):
            return (
                sm100_utils.make_smem_layout_a(self.tiled_mma, self.mma_tiler,
                                               self.ab_dtype, stages),
                sm100_utils.make_smem_layout_b(self.tiled_mma, self.mma_tiler,
                                               self.ab_dtype, stages),
            )

        self.epi_smem_layout = sm100_utils.make_smem_layout_epi(
            self.c_dtype, LayoutEnum.ROW_MAJOR, self.epi_tile, self.epi_stage
        )
        # The AB pipeline gets every stage the epilogue buffers leave over, and
        # that is the whole rule -- there is no depth to choose.  Deeper always
        # wins here (starving it costs 31 us at 5 stages against 6) and the
        # ceiling is smem, so the ceiling *is* the answer: at the 256x256 tile
        # a stage is 32,768 B, six of them plus 16,384 B of epilogue and this
        # overhead reach 214,016 of the 232,448 sm_100 has, and a seventh would
        # need 246,784.  The narrow tail tile halves B and fits eight.  So this
        # is derived rather than requested, and the "you asked for more than
        # fits" error it used to raise is unreachable.
        ab_bytes = sum(cute.size_in_bytes(self.ab_dtype, layout) for layout in smem_ab(1))
        epi_bytes = cute.size_in_bytes(self.c_dtype, self.epi_smem_layout)
        overhead = 1024  # mbarriers + tmem holding buffer, rounded up
        capacity = utils.get_smem_capacity_in_bytes("sm_100")
        self.ab_stage = (capacity - epi_bytes - overhead) // ab_bytes
        if self.ab_stage < 1:
            raise ValueError(
                f"{epi_bytes} B of epilogue buffers leave no room for even one "
                f"{ab_bytes} B AB stage in {capacity} B of smem")
        self.smem_bytes = self.ab_stage * ab_bytes + epi_bytes + overhead

        self.a_smem_layout, self.b_smem_layout = smem_ab(self.ab_stage)
        self.num_tma_load_bytes = ab_bytes * self.atom_thr

        acc_cols = tcgen05.find_tmem_tensor_col_offset(
            self.tiled_mma.make_fragment_C(self.tiled_mma.partition_shape_C(self.mma_tiler[:2]))
        )
        # tcgen05 allocates a power-of-two column count, minimum 32.
        want = acc_cols * self.acc_stage
        self.num_tmem_cols = max(32, 1 << (want - 1).bit_length())
        max_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")
        if self.num_tmem_cols > max_cols:
            raise ValueError(f"{self.acc_stage} accumulator stages need {want} -> "
                             f"{self.num_tmem_cols} tmem columns, only {max_cols} available")

    def describe(self) -> str:
        return (
            f"LcGemmSm100[{self.scheme.name}] {self.m}x{self.k}x{self.n} "
            f"blocks {self.scheme.shape} rank {self.scheme.rank}\n"
            f"  mma_tiler={self.mma_tiler} cta_tile={self.cta_tile} "
            f"epi_tile={self.epi_tile_mn}\n"
            f"  ab_stage={self.ab_stage} acc_stage={self.acc_stage} "
            f"epi_stage={self.epi_stage} smem={self.smem_bytes}B "
            f"tmem_cols={self.num_tmem_cols}\n"
            f"  epilogue={sum(len(g) for step in self.epi_groups for _, g in step)} writes"
            f"{' (postsum CSE)' if self.scheme.has_postsum else ''}\n"
            f"  n=[{self.n_start},{self.n_start + self.n_count}) "
            f"({self.num_pairs} mma tiles, {self.num_pairs / 74:.2f} waves), "
            f"k_tiles={self.k_tiles}/rank, persist={self.persist}, "
            f"group_m={self.group_m}/{self.tiles_m}"
            f"{', clears the consumer output' if self.preclear else ''}"
        )

    # ------------------------------------------------------------------- host
    @cute.jit
    def call_planes(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,  # (M, N) this GEMM's output -- scratch, in the chain
        mS: cute.Tensor,  # (R' * M2, N2) the consumer's A-planes
        mZ: cute.Tensor,  # (M, N') the consumer's output, cleared iff `preclear`
        stream: cuda.CUstream,
    ):
        """The GEMM plus the ``SwiGLU -> consumer A-planes`` epilogue."""
        self._emit(mA, mB, mC, mS, mZ, stream)

    def _emit(self, mA, mB, mC, mS, mZ, stream):
        """Trace-time body shared by both entry points."""
        self._setup()

        stage0 = lambda layout: cute.slice_(layout, (None, None, None, 0))  # noqa: E731
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(self.cta_group), mA, stage0(self.a_smem_layout),
            self.mma_tiler, self.tiled_mma, self.cluster_layout_vmnk.shape,
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(self.cta_group), mB, stage0(self.b_smem_layout),
            self.mma_tiler, self.tiled_mma, self.cluster_layout_vmnk.shape,
        )

        # Two atoms over one C tensor: a plain store for each block's first
        # contribution and a reduce-add for the rest.  Both atoms are built from
        # the same tensor, tile and smem layout, so they share one TMA tensor and
        # one gmem partition -- only the copy op differs.
        epi_stage_layout = cute.slice_(self.epi_smem_layout, (None, None, 0))
        c_cta_v_layout = cute.composition(cute.make_identity_layout(mC.shape), self.epi_tile)
        tma_atom_c_store, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), mC, epi_stage_layout, c_cta_v_layout
        )
        tma_atom_c_add, _ = cpasync.make_tiled_tma_atom(
            cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.ADD),
            mC, epi_stage_layout, c_cta_v_layout,
        )

        # The plane epilogue reads C and writes the planes with ordinary
        # vectorised ld/st: both are perfectly coalesced at 128 B per 8 threads
        # and neither wants smem, so TMA would only add descriptors.
        vec = const_expr(self.PLANE_VEC)
        as_vectors = lambda t: cute.make_tensor(  # noqa: E731
            t.iterator,
            cute.make_layout((t.shape[0], t.shape[1] // vec, vec),
                             stride=(t.shape[1], vec, 1)),
        )
        gC_vec, gS_vec = as_vectors(mC), as_vectors(mS)
        gZ_vec = self._preclear_setup(mZ)
        plane_ld = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type,
                                       num_bits_per_copy=vec * mC.element_type.width)
        plane_st = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mS.element_type,
                                       num_bits_per_copy=vec * mS.element_type.width)

        self.kernel(
            self.tiled_mma, tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b,
            tma_atom_c_store, tma_atom_c_add, tma_tensor_c, self.cluster_layout_vmnk,
            self.a_smem_layout, self.b_smem_layout, self.epi_smem_layout, self.epi_tile,
            gC_vec, gS_vec, gZ_vec, plane_ld, plane_st,
        ).launch(
            grid=(self.grid_pairs * self.atom_thr, 1, 1),
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
        )

    # ----------------------------------------------------------------- device
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB: cute.Tensor,
        tma_atom_c_store: cute.CopyAtom,
        tma_atom_c_add: cute.CopyAtom,
        mC: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        epi_smem_layout: cute.ComposedLayout,
        epi_tile: cute.Tile,
        gC_vec: cute.Tensor,
        gS_vec: cute.Tensor,
        gZ_vec: cute.Tensor,
        plane_ld: cute.CopyAtom,
        plane_st: cute.CopyAtom,
    ):
        R = const_expr(self.scheme.rank)
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        if warp_idx == self.TMA_WARP:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % self.atom_thr
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

        # Tile position inside the block grid, M-fastest.  With persist > 1 this
        # CTA-pair walks `persist` of them, strided by the grid: stride
        # `grid_pairs` is exactly one of the non-persistent kernel's waves, so
        # the global tile order -- and B's L2 reuse with it -- is unchanged.
        # Every warp role walks its tiles in a **rolled** loop (`_tile_coord`
        # off a dynamic `t`).  Unrolling it at trace time measured the same
        # (../CHAIN.md, "the tile loop"), and rolling keeps the kernel one
        # tile's worth of code at any `persist`.
        base_pair = bidx // self.atom_thr

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.acc_stage * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len(self.EPI_WARPS) * self.atom_thr
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.TMEM_ALLOC_BAR_ID, num_threads=32 * (1 + len(self.EPI_WARPS))
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.EPI_WARPS[0],
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        alloc = lambda dtype, layout: smem.allocate_tensor(  # noqa: E731
            element_type=dtype, layout=layout.outer, byte_alignment=128, swizzle=layout.inner)
        sA = alloc(self.ab_dtype, a_smem_layout)
        sB = alloc(self.ab_dtype, b_smem_layout)
        sC = alloc(self.c_dtype, epi_smem_layout)

        # 2-CTA MMA splits A over the pair along M and B along N, so each operand
        # is delivered to both CTAs of the pair.
        a_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
        )
        b_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
        )

        # (bM, bK, RestM, RestK), with RestM spanning all R stacked A planes.
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(cute.local_tile(mA, cute.select(self.mma_tiler, [0, 2]),
                                                   (None, None)))
        tCgB = thr_mma.partition_B(cute.local_tile(mB, cute.select(self.mma_tiler, [1, 2]),
                                                   (None, None)))
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a, block_in_cluster_coord_vmnk[2],
            cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape),
            cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b, block_in_cluster_coord_vmnk[1],
            cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape),
            cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3),
        )

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(tiled_mma.partition_shape_C(self.mma_tiler[:2]), self.acc_stage)
        )

        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        # ---------------------------------------------------------- TMA warp
        if warp_idx == self.TMA_WARP:
            for t in cutlass.range(self.persist, unroll=1):
                tm, tn, _ = self._tile_coord(base_pair, t, mma_tile_coord_v)
                for r in cutlass.range_constexpr(R):
                    a_tile = const_expr(r * self.tiles_m) + tm
                    b_tile = const_expr(r * self.tiles_n) + tn
                    for kk in cutlass.range(self.k_tiles, unroll=1):
                        handle = ab_producer.acquire_and_advance()
                        cute.copy(tma_atom_a, tAgA[(None, a_tile, kk)],
                                  tAsA[(None, handle.index)],
                                  tma_bar_ptr=handle.barrier, mcast_mask=a_mcast_mask)
                        cute.copy(tma_atom_b, tBgB[(None, b_tile, kk)],
                                  tBsB[(None, handle.index)],
                                  tma_bar_ptr=handle.barrier, mcast_mask=b_mcast_mask)
            ab_producer.tail()

        # ---------------------------------------------------------- MMA warp
        if warp_idx == self.MMA_WARP:
            tmem.wait_for_alloc()
            tCtAcc_base = cute.make_tensor(tmem.retrieve_ptr(self.acc_dtype), tCtAcc_fake.layout)
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.acc_stage
            )
            num_kblocks = const_expr(cute.size(tCrA, mode=[2]))

            for t in cutlass.range(self.persist, unroll=1):
                for r in cutlass.range_constexpr(R):
                    if is_leader_cta:
                        acc_pipeline.producer_acquire(acc_producer_state)
                    tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]
                    # A chain continuation adds into the sum this stage already
                    # holds: the accumulator *is* the shared postsum
                    # subexpression, and producer_acquire is what orders it
                    # after the epilogue's read of the previous prefix.
                    tiled_mma.set(tcgen05.Field.ACCUMULATE,
                                  const_expr(not self.acc_clear[r]))
                    for kk in cutlass.range(self.k_tiles, unroll=1):
                        if is_leader_cta:
                            handle = ab_consumer.wait_and_advance()
                            for kb in cutlass.range_constexpr(num_kblocks):
                                crd = (None, None, kb, handle.index)
                                cute.gemm(tiled_mma, tCtAcc, tCrA[crd], tCrB[crd], tCtAcc)
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            handle.release()
                    if is_leader_cta:
                        acc_pipeline.producer_commit(acc_producer_state)
                    acc_producer_state.advance()
            acc_pipeline.producer_tail(acc_producer_state)

        # ----------------------------------------------------- epilogue warps
        if warp_idx < self.MMA_WARP:
            tmem.allocate(self.num_tmem_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            # tmem -> rmem.  (MMA_M, MMA_N, STAGE) -> (EPI_M, EPI_N, SUB_M, SUB_N, STAGE)
            tAcc_epi = cute.flat_divide(tCtAcc_base[((None, None), 0, 0, None)], epi_tile)
            tiled_copy_t2r = tcgen05.make_tmem_copy(
                sm100_utils.get_tmem_load_op(self.cta_tile, LayoutEnum.ROW_MAJOR, self.c_dtype,
                                             self.acc_dtype, epi_tile, True),
                tAcc_epi[(None, None, 0, 0, 0)],
            )
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tAcc_base = thr_copy_t2r.partition_S(tAcc_epi)
            tTR_cAcc = thr_copy_t2r.partition_D(
                cute.flat_divide(cute.make_identity_tensor(self.cta_tile[:2]), epi_tile)
            )
            tTR_rAcc = cute.make_rmem_tensor(tTR_cAcc[None, None, None, 0, 0].shape,
                                             self.acc_dtype)

            # rmem -> smem
            tiled_copy_r2s = cute.make_tiled_copy_D(
                sm100_utils.get_smem_store_op(LayoutEnum.ROW_MAJOR, self.c_dtype,
                                              self.acc_dtype, tiled_copy_t2r),
                tiled_copy_t2r,
            )
            tRS_sC = tiled_copy_r2s.get_slice(tidx).partition_D(sC)
            tRS_rC = tiled_copy_r2s.retile(
                cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            )

            # smem -> gmem.  The destination block only selects which Rest
            # coordinate we slice at use time, so one partition serves all four.
            bSG_sC, bSG_gC = cpasync.tma_partition(
                tma_atom_c_store, 0, cute.make_layout(1), cute.group_modes(sC, 0, 2),
                cute.group_modes(
                    cute.flat_divide(cute.local_tile(mC, self.cta_tile[:2], (None, None)),
                                     epi_tile),
                    0, 2,
                ),
            )

            epi_sync = pipeline.NamedBarrier(
                barrier_id=self.EPI_SYNC_BAR_ID, num_threads=32 * len(self.EPI_WARPS)
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.epi_stage,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, 32 * len(self.EPI_WARPS)
                ),
            )
            is_tma_warp = warp_idx == self.EPI_WARPS[0]

            rows_per_block = const_expr(self.m2 // self.cta_tile[0])
            cols_per_block = const_expr(self.n2 // self.cta_tile[1])
            num_sub_m = const_expr(cute.size(tTR_tAcc_base, mode=[3]))
            num_sub_n = const_expr(cute.size(tTR_tAcc_base, mode=[4]))

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.acc_stage
            )
            for t in cutlass.range(self.persist, unroll=1):
                _, tn, rib = self._tile_coord(base_pair, t, mma_tile_coord_v)
                for r in cutlass.range_constexpr(R):
                    acc_pipeline.consumer_wait(acc_consumer_state)
                    tTR_tAcc = tTR_tAcc_base[
                        (None, None, None, None, None, acc_consumer_state.index)
                    ]
                    for sub_m in cutlass.range_constexpr(num_sub_m):
                        for sub_n in cutlass.range_constexpr(num_sub_n):
                            cute.copy(tiled_copy_t2r,
                                      tTR_tAcc[(None, None, None, sub_m, sub_n)], tTR_rAcc)
                            acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                            for gi, (coeff, dests) in const_expr(
                                    tuple(enumerate(self.epi_groups[r]))):
                                buf = const_expr(self.epi_bufs[r, sub_m, sub_n, gi])
                                scaled = acc_vec if const_expr(coeff == 1) else acc_vec * coeff
                                tRS_rC.store(scaled.to(self.c_dtype))
                                cute.copy(tiled_copy_r2s, tRS_rC,
                                          tRS_sC[(None, None, None, buf)])
                                cute.arch.fence_proxy("async.shared", space="cta")
                                epi_sync.arrive_and_wait()
                                if is_tma_warp:
                                    for i, j, is_first in const_expr(dests):
                                        cute.copy(
                                            tma_atom_c_store if const_expr(is_first)
                                            else tma_atom_c_add,
                                            bSG_sC[(None, buf)],
                                            bSG_gC[(None, sub_m, sub_n,
                                                    const_expr(i * rows_per_block) + rib,
                                                    const_expr(j * cols_per_block) + tn)],
                                        )
                                    c_pipeline.producer_commit()
                                    c_pipeline.producer_acquire()
                                epi_sync.arrive_and_wait()
                    cute.arch.fence_view_async_tmem_load()
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                # Hand tile t to the plane warps. This CTA is the only writer
                # of the four C tiles they read, so ordering is intra-CTA:
                # producer_tail lands TMA stores, the fence publishes them,
                # and the rendezvous limits the lag to exactly one tile.
                c_pipeline.producer_tail()
                cute.arch.fence_proxy("async.global")
                self._plane_barrier().arrive_and_wait()

            self._plane_barrier().arrive_and_wait()   # drain the last tile
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        # ------------------------------------------------------- plane warps
        # One tile behind the epilogue warps: they run tile t's SwiGLU and plane
        # stores while the mainloop is already streaming tile t+1.
        if warp_idx > self.TMA_WARP:
            ptid = tidx - const_expr(32 * (len(self.EPI_WARPS) + 2))
            self._plane_barrier().arrive_and_wait()
            for t in cutlass.range(self.persist, unroll=1):
                _, tn, rib = self._tile_coord(base_pair, t, mma_tile_coord_v)
                self._clear_share(gZ_vec, plane_st, ptid, bidx, t)
                self._plane_epilogue(gC_vec, gS_vec, plane_ld, plane_st,
                                     ptid, rib, tn)
                self._plane_barrier().arrive_and_wait()

    def _tile_coord(self, base_pair, t, v):
        """``(tile_m, tile_n, row_in_block)`` of this pair's ``t``-th tile.

        The walk covers one ``group_m``-tall band of M across the whole N range
        before starting the next, so consecutive tiles -- and therefore the
        tiles resident at any instant -- span a near-square rectangle of the
        block grid.  Which CTA owns which output tile is all this changes, so
        the plane epilogue's single-writer property and the postsum CSE chains
        are untouched and the result stays bit-identical.

        ``row_in_block`` is this CTA's row within each C block in ``cta_tile``
        units -- block ``(i, j)`` shifts by whole blocks from there.
        """
        pid = base_pair + t * const_expr(self.grid_pairs)
        band, within = (pid // const_expr(self.group_m * self.n_count),
                        pid % const_expr(self.group_m * self.n_count))
        tm = band * const_expr(self.group_m) + within % const_expr(self.group_m)
        return (tm, const_expr(self.n_start) + within // const_expr(self.group_m),
                tm * const_expr(self.atom_thr) + v)

    def _plane_barrier(self):
        return pipeline.NamedBarrier(barrier_id=self.PLANE_SYNC_BAR_ID,
                                     num_threads=self.plane_bar_threads)

    # ---------------------------------------------------- consumer preclear
    def _preclear_setup(self, mZ) -> cute.Tensor:
        """Flatten the consumer's output and stripe it over (CTA, tile, thread).

        The consumer accumulates that output from several CTAs, so it has to
        arrive zeroed.  Striping it over every (CTA, tile step, thread) rather
        than dumping it in one place is what keeps all but one step's worth of
        it behind the mainloop instead of in the plane phase's drain.
        """
        vec = self.PLANE_VEC
        elems = mZ.shape[0] * mZ.shape[1]
        if elems % vec:
            raise ValueError(f"{elems} consumer elements are not a multiple of {vec}")
        stripes = self.grid_pairs * self.atom_thr * self.persist * self.plane_threads
        self.zero_vecs = elems // vec if self.preclear else 0
        self.zero_stride = stripes
        self.zero_steps, self.zero_rest = divmod(self.zero_vecs, stripes)
        return cute.make_tensor(
            mZ.iterator, cute.make_layout((elems // vec, vec), stride=(vec, 1)))

    # ``cute.jit`` for the same reason ``_plane_epilogue`` is: the bounds test on
    # the ragged last stripe has to become an ``scf.if``, not a Python branch.
    @cute.jit
    def _clear_share(self, gZ, st_atom, tidx, bidx, t):
        """Zero this thread's stripe of the consumer's output."""
        if const_expr(self.zero_vecs):
            z = cute.make_rmem_tensor(cute.make_layout(const_expr(self.PLANE_VEC)),
                                      gZ.element_type)
            z.store(cute.full_like(z.load(), 0.0))
            base = ((bidx * const_expr(self.persist) + t)
                    * const_expr(self.plane_threads) + tidx)
            for u in cutlass.range_constexpr(self.zero_steps):
                cute.copy(st_atom, z,
                          gZ[base + const_expr(u * self.zero_stride), None])
            if const_expr(self.zero_rest):
                last = base + const_expr(self.zero_steps * self.zero_stride)
                if last < const_expr(self.zero_vecs):
                    cute.copy(st_atom, z, gZ[last, None])

    # ------------------------------------------------- plane epilogue (device)
    # ``cute.jit`` rather than a plain method: only decorated sources get the
    # AST pass that turns ``cutlass.range`` into a dynamic loop.
    @cute.jit
    def _plane_epilogue(self, gC, gS, ld_atom, st_atom, tidx, row_in_block, tile_n):
        """SwiGLU over this CTA's finished C tiles -> the consumer's A-planes.

        The CTA owns C blocks ``(i, j)``, ``i, j in {0, 1}``, at rows
        ``i*M2 + row_in_block*cta_tile_m + [0, cta_tile_m)`` and columns
        ``j*N2 + tile_n*cta_tile_n + [0, cta_tile_n)``.  With ``j`` the gate/up
        axis, ``S = silu(C[i][0]) * C[i][1]`` is computable for both ``i``; with
        the consumer's K-gauge interleaved at ``plane_chunk``, the two S columns
        its lattice pairs are ``h = 0`` and ``h = 1`` of one chunk pair, which
        this CTA also owns.  So the eight-element lattice closes inside the CTA
        and every plane element is written exactly once.
        """
        ps = const_expr(self.plane_scheme)
        vec, w = const_expr(self.PLANE_VEC), const_expr(self.plane_chunk)

        slot = tidx % const_expr(self.plane_slots)
        pair = slot // const_expr(self.plane_vecs_per_chunk)
        vc = slot % const_expr(self.plane_vecs_per_chunk)
        row0 = (row_in_block * const_expr(self.cta_tile[0])
                + tidx // const_expr(self.plane_slots))

        # Vector-columns of C for gate/up block j and chunk h of this thread's
        # pair, and the one vector-column of the plane both chunks feed.
        col_of_tile_n = tile_n * const_expr(self.cta_tile[1] // vec)
        c_cols = {
            (j, h): const_expr(j * (self.n2 // vec) + h * (w // vec))
            + col_of_tile_n + pair * const_expr(2 * w // vec) + vc
            for j in range(2) for h in range(2)
        }
        s_col = (tile_n * const_expr(self.cta_tile[1] // (2 * vec))
                 + pair * const_expr(w // vec) + vc)

        z = {(i, j, h): cute.make_rmem_tensor(cute.make_layout(vec), gC.element_type)
             for i in range(2) for j in range(2) for h in range(2)}
        out = cute.make_rmem_tensor(cute.make_layout(vec), gS.element_type)

        # One row per step, unroll=1: both ways of giving a thread more loads in
        # flight -- unrolling the loop, and handing it 2 or 4 rows at once --
        # measured worse at every warp count (../CHAIN.md).  The phase is short
        # of registers, not of parallelism.
        for step in cutlass.range(const_expr(self.plane_steps), unroll=1):
            row = row0 + step * const_expr(self.plane_rows_per_step)
            # All eight loads first: this thread's whole share of memory-level
            # parallelism, and the four C tiles are L2-hot from its own scatter.
            for i in cutlass.range_constexpr(2):
                for j in cutlass.range_constexpr(2):
                    for h in cutlass.range_constexpr(2):
                        cute.copy(ld_atom,
                                  gC[const_expr(i * self.m2) + row, c_cols[j, h], None],
                                  z[i, j, h])
            s = {}
            for i in cutlass.range_constexpr(2):
                for h in cutlass.range_constexpr(2):
                    g = z[i, 0, h].load().to(cutlass.Float32)
                    u = z[i, 1, h].load().to(cutlass.Float32)
                    # silu as the tanh half-angle identity: one MUFU and no
                    # divide, 15 us cheaper than ``g / (1 + exp(-g))`` and 12
                    # cheaper than ``g * rcp(1 + ex2(-g log2 e))``, for one bf16
                    # ulp on the planes (../CHAIN.md, "the silu form").  It also
                    # cannot overflow, which ``exp(g)/(1+exp(g))`` does on the
                    # activations ``gate_up`` produces.
                    hg = g * cutlass.Float32(0.5)
                    s[i, h] = hg * (cute.math.tanh(hg, fastmath=True)
                                    + cutlass.Float32(1.0)) * u
            for r in cutlass.range_constexpr(ps.rank):
                out.store(apply_terms(const_expr(ps.a_terms[r]), s).to(gS.element_type))
                cute.copy(st_atom, out,
                          gS[const_expr(r * self.m2) + row, s_col, None])
