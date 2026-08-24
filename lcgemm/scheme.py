"""The decomposition layer: a rank-R bilinear scheme and its epilogue plan.

A ``matmul-rabc-v1`` scheme is a rank-R bilinear decomposition of the block
matrix product.  Splitting A into ``p x q`` blocks and B into ``q x s`` blocks::

    Atilde_r = sum_{i,k} a[r][i,k] * A[i][k]
    Btilde_r = sum_{k,j} b[r][k,j] * B[k][j]
    P_r      = Atilde_r @ Btilde_r
    C[i][j]  = sum_r c[r][i,j] * P_r

R multiplications replace the naive ``p*q*s``.  For ``2x2x2`` the schemes here
have R=7 against 8, so 87.5% of the FLOPs.

The kernel consumes the derived tables on this class, never the raw JSON:
``a_terms``/``b_terms`` drive the operand transforms and ``epi_groups`` drives
the epilogue.

Postsum CSE
-----------
The plain epilogue scatters every product straight to its output blocks, so it
pays one global write per nonzero of ``C`` -- and every write past the first to
a block is a TMA reduce-add, a DRAM read *and* a write.

**The on-chip adder we already own is the MMA accumulator.**  ``tcgen05`` MMA
accumulates into TMEM; two products landing in one TMEM buffer without a clear
in between are summed for free.  So a shared partial sum is *a run of
consecutive products accumulated into one TMEM buffer*, and the values such a
chain can publish are exactly its **prefix sums**.

Writing ``sigma_t`` of step ``t``'s prefix gives output ``(i, j)`` the
coefficient ``sum_{u <= t}sigma_u`` on every product of the chain up to ``t``,
so ``sigma_t = c[t] - c[next(t)]`` -- **the level change down each column of
C**.  A column that does not change costs no write, which is the entire saving.

The accumulator pipeline has two stages and advances once per step, so step
``t`` owns stage ``t % 2`` and a chain occupies every other step.  That makes
the whole plan expressible as one flag per step: ``clear[t]``, whether step
``t`` starts a fresh sum rather than adding to the one at ``t - 2``.
Everything else is re-derived from ``clear`` and checked against ``C``, so a
plan cannot silently disagree with the identity it implements.

Two gauges are free and are what the offline search exploits:

* **product sign** -- negate a product's A plane and its C row together;
* **the K-partition** -- ``C = sum_j A^(.,j) B^(j,.)`` holds for any partition
  of the K index set into ``q`` equal parts, provided A's and B's agree.  That
  one is a permutation of real tensor columns rather than of these coefficient
  tables, so it lives with the operand transform, not here; see
  ``docs/IMPLEMENTATION.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import NamedTuple

_HERE = Path(__file__).resolve().parent
# A tree that owns its own scheme variants keeps them in a sibling ``schemes/``;
# otherwise the shared directory one level up wins.
SCHEME_DIR = _HERE / "schemes" if (_HERE / "schemes").is_dir() else _HERE.parent / "schemes"

FORMAT = "matmul-rabc-v1"


class Term(NamedTuple):
    """One coefficient of an operand transform: ``coeff * M[row][col]``.

    ``row``/``col`` index the *block grid* of the matrix the table belongs to:
    ``(i, k)`` in ``a_terms``, ``(k, j)`` in ``b_terms``, ``(i, j)`` in
    ``c_terms`` and ``postsum_terms``.  It is a ``tuple`` subclass, so
    ``for row, col, coeff in const_expr(terms)`` keeps working unchanged.

    ``__repr__`` deliberately prints as a plain tuple: the DSL builds its MLIR
    symbol names from the repr of captured constexpr values, and the generated
    field names would inflate them 2.7x (1726 -> 4750 chars for the 4x4 scheme's
    A terms) for no benefit.
    """

    row: int
    col: int
    coeff: int

    def __repr__(self) -> str:
        return f"({self.row}, {self.col}, {self.coeff})"


class Dest(NamedTuple):
    """An output block a staged tile is written to.

    ``is_first`` marks the lowest step contributing to this block: that write
    can be a plain TMA store, which initialises the block without a separate
    clear, while every later one must be a TMA reduce-add.
    """

    row: int
    col: int
    is_first: bool

    def __repr__(self) -> str:
        return f"({self.row}, {self.col}, {self.is_first})"


class Group(NamedTuple):
    """One accumulator->smem staging pass, and the blocks it feeds.

    The destinations in a group share one staged tile across their TMA ops, so
    the number of *groups* -- not of destinations -- sets the epilogue's
    register traffic.
    """

    coeff: int
    dests: tuple[Dest, ...]

    def __repr__(self) -> str:
        return f"({self.coeff}, {self.dests!r})"


@dataclass(frozen=True)
class Scheme:
    """A validated rank-R decomposition of the ``p x q x s`` block product."""

    name: str
    shape: tuple[int, int, int]  # (p, q, s)
    rank: int
    a: tuple[tuple[int, ...], ...]  # rank x (p*q), row-major over (i, k)
    b: tuple[tuple[int, ...], ...]  # rank x (q*s), row-major over (k, j)
    c: tuple[tuple[int, ...], ...]  # rank x (p*s), row-major over (i, j)
    # Postsum CSE plan, or () for the plain scatter epilogue.  ``clear[t]`` says
    # step ``t`` starts a new accumulator chain instead of adding to step t-2's
    # sum.  Everything else about the plan is derived from it.
    clear: tuple[int, ...] = ()

    # ---- block-grid extents -------------------------------------------------
    @property
    def p(self) -> int:
        return self.shape[0]

    @property
    def q(self) -> int:
        return self.shape[1]

    @property
    def s(self) -> int:
        return self.shape[2]

    # ---- operand transforms -------------------------------------------------
    @cached_property
    def a_terms(self) -> tuple[tuple[Term, ...], ...]:
        """Per product, the ``(i, k, coeff)`` terms of ``Atilde_r``."""
        return self._terms(self.a, self.q)

    @cached_property
    def b_terms(self) -> tuple[tuple[Term, ...], ...]:
        """Per product, the ``(k, j, coeff)`` terms of ``Btilde_r``."""
        return self._terms(self.b, self.s)

    @cached_property
    def c_terms(self) -> tuple[tuple[Term, ...], ...]:
        """Per product, the ``(i, j, coeff)`` output destinations of ``P_r``."""
        return self._terms(self.c, self.s)

    @staticmethod
    def _terms(rows, width) -> tuple[tuple[Term, ...], ...]:
        return tuple(
            tuple(Term(idx // width, idx % width, v) for idx, v in enumerate(row) if v)
            for row in rows
        )

    # ---- plain scatter epilogue ---------------------------------------------
    @cached_property
    def first_writer(self) -> dict[tuple[int, int], int]:
        """Lowest product contributing to each C block.

        Exactly one CTA owns all contributions to a given output tile, so no
        atomicity across CTAs is needed for the store/reduce-add split to be
        race-free.
        """
        return _first_writers(self.c_terms)

    @cached_property
    def dest_groups(self) -> tuple[tuple[Group, ...], ...]:
        """Per product, ``c_terms`` grouped by coefficient."""
        return _group(self.c_terms, self.first_writer)

    # ---- postsum CSE --------------------------------------------------------
    @property
    def has_postsum(self) -> bool:
        """Whether this scheme carries a CSE plan at all."""
        return bool(self.clear)

    @property
    def acc_clear(self) -> tuple[int, ...]:
        """Per step, whether the MMA must clear the accumulator before it.

        A scheme with no CSE plan clears at every step -- the plain scatter
        epilogue *is* the postsum plan whose chains all have length one -- so
        the kernel can read this unconditionally instead of branching.
        """
        return self.clear if self.has_postsum else (1,) * self.rank

    @cached_property
    def postsum_chains(self) -> tuple[tuple[int, ...], ...]:
        """The accumulator chains, as runs of step indices ``t, t+2, t+4, ...``.

        Step ``t`` owns TMEM stage ``t % 2``, so a chain is the run of
        same-parity steps that follows a clear without hitting the next one.
        This is the primitive; ``postsum_next`` and ``postsum_terms`` are read
        off it.  With no CSE plan every chain is a single step.
        """
        clear = self.acc_clear
        open_chain: dict[int, list[int]] = {}
        chains: list[list[int]] = []
        for t in range(self.rank):
            stage = t % 2
            if clear[t] or stage not in open_chain:
                open_chain[stage] = []
                chains.append(open_chain[stage])
            open_chain[stage].append(t)
        return tuple(tuple(ch) for ch in chains)

    @cached_property
    def postsum_next(self) -> tuple[int, ...]:
        """Per step, the next step of the same accumulator chain, or -1."""
        nxt = [-1] * self.rank
        for chain in self.postsum_chains:
            for u, t in enumerate(chain[:-1]):
                nxt[t] = chain[u + 1]
        return tuple(nxt)

    @cached_property
    def postsum_terms(self) -> tuple[tuple[Term, ...], ...]:
        """Per step, the ``(i, j, coeff)`` publications of that prefix sum.

        ``sigma_t = c[t] - c[next(t)]``: the level change down each column of C.
        """
        nxt = self.postsum_next
        out = []
        for t in range(self.rank):
            after = self.c[nxt[t]] if nxt[t] >= 0 else (0,) * (self.p * self.s)
            out.append(tuple(
                Term(idx // self.s, idx % self.s, sigma)
                for idx, sigma in enumerate(x - y for x, y in zip(self.c[t], after))
                if sigma
            ))
        return tuple(out)

    @cached_property
    def postsum_groups(self) -> tuple[tuple[Group, ...], ...]:
        """``postsum_terms`` grouped by coefficient, with the first-writer flag."""
        return _group(self.postsum_terms, _first_writers(self.postsum_terms))

    # ---- the plan the kernel will actually run ------------------------------
    @property
    def epi_groups(self) -> tuple[tuple[Group, ...], ...]:
        """``postsum_groups`` if this scheme carries a CSE plan, else ``dest_groups``.

        The kernel walks steps ``0..R-1`` and reads this either way, so it never
        branches on ``has_postsum`` in the epilogue. Without a plan the two are
        equal by construction.
        """
        return self.postsum_groups if self.has_postsum else self.dest_groups

    @property
    def num_staging_passes(self) -> int:
        """Accumulator->smem staging passes across the whole epilogue."""
        return sum(len(step) for step in self.epi_groups)

    @property
    def num_writes(self) -> int:
        """Global output writes the epilogue will issue."""
        return sum(len(g.dests) for step in self.epi_groups for g in step)

    @property
    def num_output_writes(self) -> int:
        """Writes the *plain scatter* epilogue would issue: the CSE baseline."""
        return sum(len(t) for t in self.c_terms)

    # ---- validation ---------------------------------------------------------
    def validate(self) -> None:
        """Check everything: shape, the tensor identity, and any CSE plan.

        This is the only thing standing between a mistyped coefficient and a
        kernel that is silently wrong everywhere, so it is exact integer
        arithmetic and it is not optional.
        """
        self._validate_shape()
        self._validate_identity()
        if self.has_postsum:
            self.validate_postsum()

    def _validate_shape(self) -> None:
        if len(self.shape) != 3 or any(x < 1 for x in self.shape):
            raise ValueError(
                f"scheme {self.name!r}: shape {self.shape} is not a (p, q, s) block grid")
        if self.rank < 1:
            raise ValueError(f"scheme {self.name!r}: rank {self.rank} must be at least 1")
        for label, mat, width, grid in (
            ("A", self.a, self.p * self.q, f"{self.p}x{self.q} blocks of the left operand"),
            ("B", self.b, self.q * self.s, f"{self.q}x{self.s} blocks of the right operand"),
            ("C", self.c, self.p * self.s, f"{self.p}x{self.s} blocks of the output"),
        ):
            if len(mat) != self.rank:
                raise ValueError(
                    f"scheme {self.name!r}: {label} has {len(mat)} rows but the "
                    f"decomposition has rank {self.rank} -- every product needs "
                    f"exactly one row of {label} coefficients")
            for r, row in enumerate(mat):
                if len(row) != width:
                    raise ValueError(
                        f"scheme {self.name!r}: {label} row for product {r} has "
                        f"{len(row)} coefficients, expected {width} (one per the "
                        f"{grid})")
        for r in range(self.rank):
            for label, terms in (("A", self.a_terms[r]), ("B", self.b_terms[r]),
                                 ("C", self.c_terms[r])):
                if not terms:
                    raise ValueError(
                        f"scheme {self.name!r}: product {r} has an all-zero {label} "
                        f"row, so it computes nothing and the rank is overstated")

    def _validate_identity(self) -> None:
        """Assert the exact tensor identity the decomposition claims::

            sum_r c[r][i,j] a[r][i',k'] b[r][k'',j''] = [i'=i][k'=k''][j''=j]

        Accumulated over the schemes' *nonzeros* rather than the dense block
        grid, so the cost tracks how sparse the decomposition is.
        """
        acc: dict[tuple[int, int, int, int, int, int], int] = {}
        for r in range(self.rank):
            for i, j, cv in self.c_terms[r]:
                for ip, kp, av in self.a_terms[r]:
                    w = cv * av
                    for kpp, jpp, bv in self.b_terms[r]:
                        key = (i, j, ip, kp, kpp, jpp)
                        acc[key] = acc.get(key, 0) + w * bv
        # Exactly the p*q*s wired-through products must survive, each with
        # coefficient 1; everything else must cancel.
        for i in range(self.p):
            for k in range(self.q):
                for j in range(self.s):
                    key = (i, j, i, k, k, j)
                    got = acc.pop(key, 0)
                    if got != 1:
                        raise ValueError(
                            f"scheme {self.name!r} is not a decomposition of the "
                            f"matmul: the term A[{i}][{k}]*B[{k}][{j}] reaches "
                            f"C[{i}][{j}] with coefficient {got}, but the product "
                            f"requires 1")
        for key, v in acc.items():
            if v:
                i, j, ip, kp, kpp, jpp = key
                why = ("the two operands' inner indices disagree"
                       if kp != kpp else "it lands in the wrong output block")
                raise ValueError(
                    f"scheme {self.name!r} is not a decomposition of the matmul: "
                    f"A[{ip}][{kp}]*B[{kpp}][{jpp}] leaks into C[{i}][{j}] with "
                    f"coefficient {v}, but must cancel to 0 -- {why}")

    def validate_postsum(self) -> None:
        """Check the CSE plan is well formed, and that it telescopes back to C.

        **What this can and cannot catch.**  ``sigma_t = c[t] - c[next(t)]`` is
        defined so that a chain's prefix sums telescope, so the reconstruction
        below returns ``C`` for *any* well-formed ``clear`` -- 221504 clear
        vectors over 10 schemes and the whole 6912-variant gauge orbit were
        tried and none was rejected.  Corrupting ``clear`` therefore does not
        make a scheme wrong, it makes it **more expensive**; the guardrail that
        catches that is the ``dests`` cross-check in ``load``, which pins the
        publication table the plan was searched for.  The reconstruction is kept
        because it is cheap and it does pin the ``sigma`` formula itself.

        The checks that do bite here are the well-formedness ones: a plan of the
        wrong length, a non-boolean flag, or a first step on a TMEM stage that
        continues a chain that never started.
        """
        if len(self.clear) != self.rank:
            raise ValueError(
                f"scheme {self.name!r}: the CSE plan has {len(self.clear)} step "
                f"flags but the decomposition has {self.rank} products -- one "
                f"'does this step clear the accumulator' flag per product")
        if any(v not in (0, 1) for v in self.clear):
            raise ValueError(
                f"scheme {self.name!r}: clear must be 0/1 per step, got {self.clear}")
        for t in range(min(2, self.rank)):
            if not self.clear[t]:
                raise ValueError(
                    f"scheme {self.name!r}: step {t} continues a chain, but it is "
                    f"the first step on TMEM stage {t % 2} -- there is no prior "
                    f"accumulator for it to add to")
        got = [[0] * (self.p * self.s) for _ in range(self.rank)]
        for chain in self.postsum_chains:
            for u, t in enumerate(chain):
                # Step t publishes its chain's prefix, so every product from the
                # chain's start through t receives the coefficient written.
                for i, j, sigma in self.postsum_terms[t]:
                    for prior in chain[: u + 1]:
                        got[prior][i * self.s + j] += sigma
        if got != [list(row) for row in self.c]:
            bad = next(r for r in range(self.rank) if got[r] != list(self.c[r]))
            raise ValueError(
                f"scheme {self.name!r}: the CSE plan does not reproduce C. Product "
                f"{bad} ends up with output coefficients {got[bad]}, but the "
                f"decomposition says {list(self.c[bad])} -- the prefix sums "
                f"published along its chain do not telescope to C")

    def summary(self) -> str:
        a_add = sum(len(t) - 1 for t in self.a_terms)
        b_add = sum(len(t) - 1 for t in self.b_terms)
        naive = self.p * self.q * self.s
        epi = (f"{self.num_writes} output writes (postsum CSE in "
               f"{len(self.postsum_chains)} chains, from {self.num_output_writes})"
               if self.has_postsum else f"{self.num_writes} output writes")
        return (
            f"{self.name}: {self.p}x{self.q}x{self.s} rank {self.rank}/{naive} "
            f"({self.rank / naive:.4f} FLOPs), A adds {a_add}, B adds {b_add}, "
            f"{self.num_staging_passes} epilogue staging passes, {epi}"
        )


def _first_writers(term_table) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    for t, terms in enumerate(term_table):
        for row, col, _ in terms:
            out.setdefault((row, col), t)
    return out


def _group(term_table, first) -> tuple[tuple[Group, ...], ...]:
    groups = []
    for t, terms in enumerate(term_table):
        by_coeff: dict[int, list[Dest]] = {}
        for row, col, v in terms:
            by_coeff.setdefault(v, []).append(Dest(row, col, first[(row, col)] == t))
        # Positive coefficients first, so the common all-positive schemes emit
        # their staging passes in a stable order.
        groups.append(tuple(Group(v, tuple(by_coeff[v]))
                            for v in sorted(by_coeff, reverse=True)))
    return tuple(groups)


def load(name: str) -> Scheme:
    """Load and validate a scheme by path, file name or stem from ``schemes/``."""
    path = Path(name)
    if not path.exists():
        path = SCHEME_DIR / (name if name.endswith(".json") else f"{name}.json")
    if not path.exists():
        raise FileNotFoundError(f"no scheme {name!r} here or in {SCHEME_DIR}")
    raw = json.loads(path.read_text())
    if raw.get("format") != FORMAT:
        raise ValueError(f"{path}: unsupported format {raw.get('format')!r}, "
                         f"expected {FORMAT!r}")
    plan = raw.get("postsum") or {}
    scheme = Scheme(
        name=path.stem,
        shape=tuple(raw["shape"]),
        rank=int(raw["r"]),
        a=tuple(tuple(int(v) for v in row) for row in raw["A"]),
        b=tuple(tuple(int(v) for v in row) for row in raw["B"]),
        c=tuple(tuple(int(v) for v in row) for row in raw["C"]),
        clear=tuple(int(v) for v in plan.get("clear", ())),
    )
    scheme.validate()
    if scheme.has_postsum:
        # This is the guardrail that actually bites: 'clear' alone always
        # telescopes back to C, so the only way to catch a corrupted plan is to
        # pin the publication table the search produced.
        stated = plan.get("dests")
        if stated is None:
            raise ValueError(
                f"{path}: the CSE plan has 'clear' but no 'dests' table. Without "
                f"it nothing pins which prefix sums the plan was searched for, "
                f"and a corrupted 'clear' would load as a silently costlier plan")
        want = tuple(tuple(Term(int(i), int(j), int(v)) for i, j, v in step)
                     for step in stated)
        if want != scheme.postsum_terms:
            bad = next(t for t in range(scheme.rank) if want[t] != scheme.postsum_terms[t])
            raise ValueError(
                f"{path}: the recorded 'dests' table disagrees with the plan "
                f"re-derived from 'clear'. Step {bad} is recorded as publishing "
                f"{list(want[bad])} but the chains imply "
                f"{list(scheme.postsum_terms[bad])}")
    return scheme


if __name__ == "__main__":
    import sys

    names = sys.argv[1:] or sorted(x.name for x in SCHEME_DIR.glob("2x2*.json"))
    for n in names:
        sch = load(n)
        print(sch.summary())
        for r in range(sch.rank):
            flag = ("CLEAR " if sch.clear[r] else "accum ") if sch.has_postsum else ""
            print(f"    r{r}: {flag}A{list(sch.a_terms[r])} B{list(sch.b_terms[r])} "
                  f"-> {list(sch.epi_groups[r])}")
