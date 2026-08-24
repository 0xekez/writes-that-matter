"""Residual add and RMSNorm fused into the deployed A-plane transform.

The unfused path would write the residual sum, read it for RMSNorm, write the
normalised activation, then read it again to form the rank-R planes. Only the
residual stream and combined planes are wanted. :class:`FusedNormResidual`
produces both directly from the attention delta and incoming residual.

The fusion works because row `k` of `Atilde_r` depends only on row `k` of each
block row of A (see ``docs/IMPLEMENTATION.md``). So a CTA owns a *stack* of `p` rows --
rows `k, k + M/p, ..., k + (p-1) M/p` -- normalises each (each has its own
`rstd`) and emits row `k` of all R planes.  A is read once, the planes are
written once, and the normalised activations never leave the SM.

The residual variant's winning launch is one row group per CTA. A persistent
three-stage control kernel was retained in the research tree only; it measured
52.3 us here against 42.4 us for the shipped schedule and is intentionally not
part of this artifact.

The CTA shape is forced, not tuned
----------------------------------
A thread owns exactly one aligned 128-bit vector of every block and of every
output row, so the CTA is `(K/q)/vec` threads and that must be warp aligned.
`K/q = 3328 = 2^8 * 13` for the 2x2 schemes, so `vec=8` gives 416 = 13*32
threads and it is the only warp-aligned vector count; a 4x4 scheme's
`K/q = 1664` needs `vec=4` to land on 416 again.  `_vec` derives it, so neither
the vector width nor the block size is a tuning knob.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr

from lcgemm.planes import apply_terms
from lcgemm.scheme import Scheme

# ------------------------------------------------------------------ host views
def _vec(k2: int, elem_bits: int) -> int:
    """Elements per thread: the widest access that leaves a warp-aligned CTA.

    128 bits is the widest load the machine has, and `k2 // vec` is the CTA
    size, so this walks 8, 4, 2, 1 elements (at bf16) and takes the first that
    divides `k2` into a multiple of 32 threads.  For every scheme measured the
    widest legal width is also the fastest: at `2x2` it picks vec=8 (416
    threads), and vec=4 -- the only other legal choice -- costs 1.5 us.
    """
    widest = 128 // elem_bits
    for vec in (widest >> s for s in range(widest.bit_length())):
        threads = k2 // vec
        if k2 % vec == 0 and threads % cute.arch.WARP_SIZE == 0 and threads <= 1024:
            return vec
    raise ValueError(
        f"a block row of {k2} elements has no warp-aligned CTA: {k2}/vec threads "
        f"must be a multiple of {cute.arch.WARP_SIZE} and at most 1024 for some "
        f"vec <= {widest}"
    )


def _block_view(t: cute.Tensor, p: int, q: int, vec: int) -> cute.Tensor:
    """`(M, K)` seen as its `(p, q)` block grid: `(p, M2, q, NV, VEC)`.

    The modes are *block row, row within it, block column, vector, element*, so
    the kernel indexes `t[i, row, j, tidx, None]` and every offset the row-stack
    decomposition needs is a layout stride rather than index arithmetic.
    """
    m, k = t.shape
    m2, k2 = m // p, k // q
    return cute.make_tensor(
        t.iterator,
        cute.make_layout((p, m2, q, k2 // vec, vec), stride=(m2 * k, k, k2, vec, 1)),
    )


def _plan(scheme: Scheme, mW: cute.Tensor, mAt: cute.Tensor, *operands: cute.Tensor):
    """Validate the operands and build every view the kernels index.

    Returns `(views, gW, gAt, vec)`; `views` is one blocked view per operand, in
    the order given.  The first operand fixes `(M, K)`.
    """
    p, q = scheme.p, scheme.q
    m, k_dim = operands[0].shape
    m2, k2 = m // p, k_dim // q
    if const_expr(m2 * p != m or k2 * q != k_dim):
        raise ValueError(f"A ({m}, {k_dim}) must divide the {p}x{q} block grid")
    if const_expr(mW.shape[0] != k_dim):
        raise ValueError(f"norm weight must be ({k_dim},)")
    for t in operands:
        if const_expr(tuple(t.shape) != (m, k_dim)):
            raise ValueError(f"every operand must be ({m}, {k_dim}), got {tuple(t.shape)}")
        if const_expr(t.stride[1] != 1 or t.stride[0] != k_dim):
            raise ValueError("every operand must be contiguous and K-major")
        if const_expr(t.element_type is not mAt.element_type):
            raise ValueError("the operands and the A workspace must share a dtype")
    if const_expr(mAt.shape[0] != scheme.rank * m2 or mAt.shape[1] != k2):
        raise ValueError(f"A workspace must be ({scheme.rank}*{m2}, {k2})")
    if const_expr(mAt.stride[1] != 1):
        raise ValueError("the A workspace must be K-major")

    vec = const_expr(_vec(k2, mAt.element_type.width))
    gW = cute.make_tensor(
        mW.iterator, cute.make_layout((q, k2 // vec, vec), stride=(k2, vec, 1))
    )
    gAt = cute.make_tensor(
        mAt.iterator,
        cute.make_layout(
            (scheme.rank, m2, k2 // vec, vec),
            stride=(m2 * mAt.stride[0], mAt.stride[0], vec, 1),
        ),
    )
    views = tuple(_block_view(t, p, q, vec) for t in operands)
    return views, gW, gAt, vec


# ---------------------------------------------------------------- kernel parts
@cute.jit
def _weight(gW: cute.Tensor, tidx, q: int) -> dict:
    """This thread's vector of the norm weight, per block column.

    Loop invariant, `q` vectors, and L2-hot for every CTA.  The effective scale
    is `weight + 1`: the checkpoint stores the weight with a baked offset.
    """
    wv = {}
    for j in cutlass.range_constexpr(q):
        wv[j] = gW[j, tidx, None].load().to(Float32) + Float32(1.0)
    return wv


@cute.jit
def _emit(gAt: cute.Tensor, row, tidx, a_terms, y: dict) -> None:
    """Write row `row` of all R planes.  Every rank is a signed sum of `y`."""
    for r in cutlass.range_constexpr(len(a_terms)):
        gAt[r, row, tidx, None] = apply_terms(const_expr(a_terms[r]), y).to(
            gAt.element_type
        )


# --------------------------------------------------------------------- kernels
@cute.kernel
def fused_norm_residual_kernel(
    gX: cute.Tensor,  # (p, M2, q, NV, VEC)
    gR: cute.Tensor,  # the incoming residual stream
    gRo: cute.Tensor,  # the outgoing one; may alias gR
    gW: cute.Tensor,  # (q, NV, VEC)
    gAt: cute.Tensor,  # (R, M2, NV, VEC)
    eps: Float32,
    a_terms: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()  # one CTA per row group; the grid is M/p
    lane, warp = cute.arch.lane_idx(), cute.arch.warp_idx()
    p, _, q, nv, vec = gX.shape
    k_dim = const_expr(q * nv * vec)
    num_warps = const_expr(nv // cute.arch.WARP_SIZE)

    # The only smem the kernel needs: one partial sum per warp per block row.
    # One row group per CTA means one barrier per CTA, so there is no second
    # step to double buffer against.
    s_red = cutlass.utils.SmemAllocator().allocate_tensor(
        Float32, cute.make_layout((p, num_warps), stride=(num_warps, 1)),
        byte_alignment=8,
    )
    wv = _weight(gW, tidx, q)

    # Issue the whole group before consuming any of it: 2*p*q loads in flight is
    # this thread's entire share of memory-level parallelism, and at 2 CTAs/SM
    # it is what keeps both read streams saturated.
    xf, rf = {}, {}
    for i in cutlass.range_constexpr(p):
        for j in cutlass.range_constexpr(q):
            xf[i, j] = gX[i, row, j, tidx, None].load()
            rf[i, j] = gR[i, row, j, tidx, None].load()

    h, ss = {}, {}
    for i in cutlass.range_constexpr(p):
        acc = Float32(0.0)
        for j in cutlass.range_constexpr(q):
            # Round the sum before normalising: residual_out is what the next
            # layer reads, so the norm has to see exactly the value stored --
            # which is what vLLM's add_rms_norm does -- and keeping the group in
            # bf16 halves what stays live across the barrier.
            h[i, j] = (xf[i, j].to(Float32) + rf[i, j].to(Float32)).to(gRo.element_type)
            # Nothing below the barrier depends on it, so it goes out now and
            # streams while the CTA stalls there.  After the barrier costs 3 us.
            gRo[i, row, j, tidx, None] = h[i, j]
            hf = h[i, j].to(Float32)
            acc += (hf * hf).reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile=0)
        ss[i] = acc

    rstd = _rstd(s_red, ss, lane, warp, eps, p, num_warps, k_dim)

    # Scale once per block, then every rank is a signed sum.
    y = {}
    for i in cutlass.range_constexpr(p):
        for j in cutlass.range_constexpr(q):
            y[i, j] = h[i, j].to(Float32) * rstd[i] * wv[j]
    _emit(gAt, row, tidx, a_terms, y)


@cute.jit
def _rstd(s_red, ss: dict, lane, warp, eps, p, num_warps, k_dim) -> dict:
    """Finish the p sums of squares across the CTA and return `1/rms` each.

    A butterfly within each warp, one fp32 per warp per block row through smem,
    then a second butterfly over the `num_warps` partials -- which is why the
    partials are read at `lane` and zero-padded rather than reduced as a
    tensor: 13 warps is not a power of two, and the tensor form is 0.1 us
    slower and changes the summation order (so it is not bit-identical).

    Inlined into the kernel AST by the caller, so it may not branch on a
    runtime row index -- `lane` and `warp` are fine.
    """
    for i in cutlass.range_constexpr(p):
        ss[i] = cute.arch.warp_reduction_sum(ss[i])
    if lane == 0:
        for i in cutlass.range_constexpr(p):
            s_red[i, warp] = ss[i]
    cute.arch.barrier()
    rstd = {}
    for i in cutlass.range_constexpr(p):
        partial = Float32(0.0)
        if lane < num_warps:
            partial = s_red[i, lane]
        total = cute.arch.warp_reduction_sum(partial)
        rstd[i] = cute.math.rsqrt(total / Float32(k_dim) + eps, fastmath=True)
    return rstd


# --------------------------------------------------------------------- drivers
class FusedNormResidual:
    """``residual_out = X + residual``, then RMSNorm + A-plane transform.

    This is the deployed pre-GEMM kernel.  It writes the same ``(R*M2, K2)``
    planes described by :mod:`lcgemm.planes`, so the GEMM and its TMA
    descriptors are unchanged. It also writes the residual stream straight to
    HBM (a real write: the next block's skip connection closes with it). ``residual_out``
    may alias ``residual`` -- a CTA reads its whole row group before storing any
    of it, and no other CTA touches those rows.

    The scheme rides on the instance rather than through the JIT signature:
    passing the dataclass as a ``Constexpr`` makes the DSL marshal it as a
    runtime argument and fail to convert. It is also the only parameter: the vector width, the CTA size and
    the grid all follow from it, and residency is left to ptxas, which picks
    2 CTAs/SM on its own.
    """

    def __init__(self, scheme: Scheme):
        self.scheme = scheme

    def describe(self) -> str:
        return f"FusedNormResidual({self.scheme.name})"

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,  # (M, K) activations, K-major
        mResidual: cute.Tensor,  # (M, K) incoming residual stream
        mW: cute.Tensor,  # (K,) norm weight; the effective scale is weight + 1
        mAt: cute.Tensor,  # (R*M/p, K/q) combined A planes
        mResidualOut: cute.Tensor,  # (M, K) outgoing stream; may alias mResidual
        eps: Float32,
        stream: cuda.CUstream,
    ):
        (gX, gR, gRo), gW, gAt, vec = _plan(
            self.scheme, mW, mAt, mX, mResidual, mResidualOut
        )
        fused_norm_residual_kernel(gX, gR, gRo, gW, gAt, eps, self.scheme.a_terms).launch(
            grid=(cute.size(gX, mode=[1]), 1, 1),
            block=(cute.size(gX, mode=[3]), 1, 1),
            stream=stream,
        )
