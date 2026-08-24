"""The single deployed MLP-chain configuration.

The development tree supported independently selectable seams and schemes. The
paper artifact intentionally exposes only the configuration used by the six
positive B200 shapes: postsum-CSE for ``gate_up`` and the
locality-ordered decomposition for ``down``.
"""

from __future__ import annotations

from lcgemm.scheme import load
from lcgemm.seams.down import DownPlan
from lcgemm.seams.gate_up import GateUpPlan

GATE_UP_SCHEME = "2x2_postsum_cse"
DOWN_SCHEME = "2x2_locality_ordered"


def build_chain(
    gate_up_shape: tuple[int, int, int],
    down_shape: tuple[int, int, int],
) -> tuple[GateUpPlan, DownPlan]:
    """Build the exact chained plans for one validated prefill shape."""
    gate_scheme = load(GATE_UP_SCHEME)
    down_scheme = load(DOWN_SCHEME)
    gate = GateUpPlan(
        gate_scheme,
        gate_up_shape,
        plane_scheme=down_scheme,
    )
    down = DownPlan(
        down_scheme,
        down_shape,
        producer=gate,
    )
    if down.rank_split:
        gate.preclear = down.c
    return gate, down


__all__ = ["DOWN_SCHEME", "GATE_UP_SCHEME", "build_chain"]
