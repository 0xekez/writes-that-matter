"""Persistent rank-split ``down`` GEMM for NVIDIA Blackwell SM100.

Computes ``C = A @ B^T`` with ``A: (M, K)``, ``B: (N, K)`` (both K-major) using a
rank-R bilinear decomposition of the block product instead of the naive
``p*q*s`` block multiplications -- 7/8 of the FLOPs for the 2x2x2 schemes.

One persistent CTA-pair processes each ``(rank, m, n)`` work unit.  Splitting
rank exposes enough work to fill the B200 at the deployed Muse-2048 and
Qwen-2048/4096/8192 shapes.  The 74 resident CTA-pairs walk the work list so
unit ``i+1`` overlaps unit ``i``'s accumulator drain.

* A C block's contributions now come from different CTAs, so the "first
  contribution is a plain store" trick is gone: C arrives cleared -- the
  producing ``gate_up`` epilogue does it, striped across that kernel's tile
  steps -- and every contribution is a TMA reduce-add.
Work items are ordered rank-major, then n, then m-fastest, so one rank's whole
A plane stays L2-resident across its N sweep while B planes stream once.
"""

from __future__ import annotations

from typing import Tuple

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


class LcGemmDownRankSplit:
    """The fixed one-rank-per-CTA persistent schedule used by the artifact."""

    # Warp roles: 4 epilogue warps, then the UMMA warp, then the TMA warp.
    EPI_WARPS = (0, 1, 2, 3)
    MMA_WARP = 4
    TMA_WARP = 5

    TMEM_ALLOC_BAR_ID = 3
    EPI_SYNC_BAR_ID = 4
    MMA_TILER = (256, 256, 64)
    ACC_STAGE = 2
    EPI_STAGE = 2

    def __init__(
        self,
        scheme: Scheme,
        mnk: Tuple[int, int, int],
    ):
        self.scheme = scheme
        self.m, self.k, self.n = mnk
        p, q, s = scheme.shape
        if self.m % p or self.k % q or self.n % s:
            raise ValueError(f"{mnk} not divisible by block grid {scheme.shape}")
        # Per-block extents: the shape every one of the R products actually has.
        self.m2, self.k2, self.n2 = self.m // p, self.k // q, self.n // s

        self.ab_dtype = self.c_dtype = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32
        self.mma_tiler = self.MMA_TILER
        self.acc_stage, self.epi_stage = self.ACC_STAGE, self.EPI_STAGE

        # 2-CTA MMA, i.e. a 2x1 cluster.  Wider clusters would engage TMA
        # multicast; that path measured both slower and wrong on gate_up, so the
        # cluster is not a parameter here either.
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
        self.n_start, self.n_count = 0, self.tiles_n

        R = scheme.rank
        self.rank_groups = R

        # Work items, ordered rank-slowest / n / m-fastest: for a fixed n every M
        # tile reads the same B panel, and a whole A plane stays L2-resident for
        # its rank's n sweep.
        self.tiles_mn = self.tiles_m * self.n_count
        self.num_units = self.tiles_mn * self.rank_groups
        self.clusters = num_clusters()
        self.grid_pairs = min(self.num_units, self.clusters)
        self.iters = (self.num_units + self.grid_pairs - 1) // self.grid_pairs
        self.threads_per_cta = 32 * (len(self.EPI_WARPS) + 2)

    def _mn(self, mn):
        """Linear work index within a rank -> (tile_m, tile_n)."""
        return (mn % const_expr(self.tiles_m),
                const_expr(self.n_start) + mn // const_expr(self.tiles_m))

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
        # Give the AB pipeline every stage the epilogue buffers leave over.
        ab_bytes = sum(cute.size_in_bytes(self.ab_dtype, layout) for layout in smem_ab(1))
        epi_bytes = cute.size_in_bytes(self.c_dtype, self.epi_smem_layout)
        overhead = 1024  # mbarriers + tmem holding buffer, rounded up
        capacity = utils.get_smem_capacity_in_bytes("sm_100")
        max_stage = (capacity - epi_bytes - overhead) // ab_bytes
        self.ab_stage = max_stage
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
            f"LcGemmDownRankSplit[{self.scheme.name}] {self.m}x{self.k}x{self.n} "
            f"blocks {self.scheme.shape} rank {self.scheme.rank}\n"
            f"  mma_tiler={self.mma_tiler} cta_tile={self.cta_tile} "
            f"epi_tile={self.epi_tile_mn}\n"
            f"  ab_stage={self.ab_stage} acc_stage={self.acc_stage} "
            f"epi_stage={self.epi_stage} smem={self.smem_bytes}B "
            f"tmem_cols={self.num_tmem_cols}\n"
            f"  n=[{self.n_start},{self.n_start + self.n_count}) "
            f"one rank/CTA, persistent work list\n"
            f"  {self.num_units} units over {self.grid_pairs} CTA-pairs "
            f"({self.iters} iters, {self.num_units / self.clusters:.2f} waves), "
            f"k_tiles={self.k_tiles}/rank"
        )

    # ------------------------------------------------------------------- host
    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,  # (R * M2, K2) transformed A planes, K-major
        mB: cute.Tensor,  # (R * N2, K2) transformed B planes, K-major
        mC: cute.Tensor,  # (M, N) output, N-major
        stream: cuda.CUstream,
    ):
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

        # Contributions to one C block come from separate rank CTAs, so the
        # pre-cleared output always uses TMA reduce-add.
        epi_stage_layout = cute.slice_(self.epi_smem_layout, (None, None, 0))
        c_cta_v_layout = cute.composition(cute.make_identity_layout(mC.shape), self.epi_tile)
        tma_atom_c_add, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyReduceBulkTensorTileS2GOp(cpasync.ReductionOp.ADD),
            mC, epi_stage_layout, c_cta_v_layout,
        )

        self.kernel(
            self.tiled_mma, tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b,
            tma_atom_c_add, tma_tensor_c, self.cluster_layout_vmnk,
            self.a_smem_layout, self.b_smem_layout, self.epi_smem_layout, self.epi_tile,
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
        tma_atom_c_add: cute.CopyAtom,
        mC: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        epi_smem_layout: cute.ComposedLayout,
        epi_tile: cute.Tile,
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
        pair_id = bidx // self.atom_thr

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
            for it in cutlass.range(self.iters, unroll=1):
                unit = pair_id + it * const_expr(self.grid_pairs)
                if unit < const_expr(self.num_units):
                    rank0 = unit // const_expr(self.tiles_mn)
                    mn = unit % const_expr(self.tiles_mn)
                    tile_m, tile_n = self._mn(mn)
                    for r in cutlass.range_constexpr(1):
                        a_tile = (rank0 + const_expr(r)) * const_expr(self.tiles_m) + tile_m
                        b_tile = (rank0 + const_expr(r)) * const_expr(self.tiles_n) + tile_n
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

            for it in cutlass.range(self.iters, unroll=1):
                unit = pair_id + it * const_expr(self.grid_pairs)
                if unit < const_expr(self.num_units):
                    for r in cutlass.range_constexpr(1):
                        if is_leader_cta:
                            acc_pipeline.producer_acquire(acc_producer_state)
                        tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
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
                tma_atom_c_add, 0, cute.make_layout(1), cute.group_modes(sC, 0, 2),
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
            # The epilogue smem rotation is a runtime counter: with a persistent
            # grid the number of staging passes a CTA has issued depends on which
            # ranks its work items landed on, so it cannot be tracked at trace time.
            buf = cutlass.Int32(0)
            for it in cutlass.range(self.iters, unroll=1):
                unit = pair_id + it * const_expr(self.grid_pairs)
                if unit < const_expr(self.num_units):
                    rank0 = unit // const_expr(self.tiles_mn)
                    mn = unit % const_expr(self.tiles_mn)
                    tile_m, tile_n = self._mn(mn)
                    # This CTA's output tile row within a C block; block (i, j)
                    # just shifts by whole blocks.
                    row_in_block = tile_m * self.atom_thr + mma_tile_coord_v

                    for r in cutlass.range_constexpr(1):
                        acc_pipeline.consumer_wait(acc_consumer_state)
                        tTR_tAcc = tTR_tAcc_base[
                            (None, None, None, None, None, acc_consumer_state.index)
                        ]
                        for sub_m in cutlass.range_constexpr(num_sub_m):
                            for sub_n in cutlass.range_constexpr(num_sub_n):
                                cute.copy(tiled_copy_t2r,
                                          tTR_tAcc[(None, None, None, sub_m, sub_n)], tTR_rAcc)
                                acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                                for rr in cutlass.range_constexpr(R):
                                    dg = const_expr(self.scheme.dest_groups[rr])
                                    if rank0 == rr:
                                        for coeff, dests in dg:
                                            scaled = (acc_vec if const_expr(coeff == 1)
                                                      else acc_vec * coeff)
                                            tRS_rC.store(scaled.to(self.c_dtype))
                                            cute.copy(tiled_copy_r2s, tRS_rC,
                                                      tRS_sC[(None, None, None, buf)])
                                            cute.arch.fence_proxy("async.shared", space="cta")
                                            epi_sync.arrive_and_wait()
                                            if is_tma_warp:
                                                for i, j, _ in const_expr(dests):
                                                    cute.copy(
                                                        tma_atom_c_add,
                                                        bSG_sC[(None, buf)],
                                                        bSG_gC[(None, sub_m, sub_n,
                                                                const_expr(i * rows_per_block)
                                                                + row_in_block,
                                                                const_expr(j * cols_per_block)
                                                                + tile_n)],
                                                    )
                                                c_pipeline.producer_commit()
                                                c_pipeline.producer_acquire()
                                            epi_sync.arrive_and_wait()
                                            buf = (buf + 1) % const_expr(self.epi_stage)
                        cute.arch.fence_view_async_tmem_load()
                        with cute.arch.elect_one():
                            acc_pipeline.consumer_release(acc_consumer_state)
                        acc_consumer_state.advance()

            c_pipeline.producer_tail()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)
