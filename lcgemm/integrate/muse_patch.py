"""Install the validated Muse-Glimmer MLP chain into vLLM.

Call :func:`arm` before constructing the engine and :func:`install` after the
weights have loaded.  Only the requested exact prefill length uses lcGEMM;
profiling, dummy, and decode shapes take the functional reference inside the
opaque custom ops.
"""

from __future__ import annotations

import functools
import os
import sys
from collections import defaultdict
from typing import Tuple

import torch
import torch.nn.functional as F

from lcgemm.seams import build_chain

_CACHE_PRESET = os.environ.get("VLLM_DISABLE_COMPILE_CACHE") in ("1", "true", "True")
_VLLM_PREIMPORTED = "vllm" in sys.modules
os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"

_STATE = {"armed": False, "installed": False, "layers": 0, "tokens": 0}
_PLANS: dict[int, tuple[object, object]] = {}
_PREPARED: dict[int, tuple[int, torch.Tensor, torch.Tensor]] = {}
_DOWN_PREPARED: dict[int, torch.Tensor] = {}
_CALLS: dict[str, int] = defaultdict(int)
_RESIDUAL_RING: list[torch.Tensor] = []
_RESIDUAL_TURN = 0


def _residual_buf(like: torch.Tensor) -> torch.Tensor:
    """Return one of two buffers; a custom op may not return one of its inputs."""
    global _RESIDUAL_RING, _RESIDUAL_TURN
    if not _RESIDUAL_RING:
        _RESIDUAL_RING = [torch.empty_like(like), torch.empty_like(like)]
    out = _RESIDUAL_RING[_RESIDUAL_TURN]
    _RESIDUAL_TURN = 1 - _RESIDUAL_TURN
    return out


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    f = x.float()
    f *= torch.rsqrt(f.square().mean(-1, keepdim=True) + eps)
    return (f * (weight.float() + 1.0)).to(x.dtype)


@torch.library.custom_op("lcgemm_muse::norm_gateup", mutates_args=(), device_types="cuda")
def norm_gateup(
    delta: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    prepared = _PREPARED.get(gate_up_weight.data_ptr())
    plans = _PLANS.get(delta.shape[0])
    if plans is None or prepared is None:
        _CALLS["gate_up.reference"] += 1
        h = residual + delta
        return h, F.linear(_rmsnorm(h, norm_weight, eps), gate_up_weight)

    _CALLS["gate_up.fused"] += 1
    gate_plan, _ = plans
    residual_out = _residual_buf(delta)
    gate = gate_plan(
        residual,
        delta,
        norm_weight,
        eps,
        prepared_w=prepared[1],
        residual_out=residual_out,
    )
    return residual_out, gate


@norm_gateup.register_fake
def _(delta, residual, norm_weight, gate_up_weight, eps):
    return torch.empty_like(delta), delta.new_empty((delta.shape[0], gate_up_weight.shape[0]))


@torch.library.custom_op("lcgemm_muse::swiglu_down", mutates_args=(), device_types="cuda")
def swiglu_down(gate_up: torch.Tensor, down_weight: torch.Tensor) -> torch.Tensor:
    plans = _PLANS.get(gate_up.shape[0])
    prepared = _DOWN_PREPARED.get(down_weight.data_ptr())
    if plans is None or prepared is None:
        _CALLS["down.reference"] += 1
        gate, up = gate_up.chunk(2, dim=-1)
        return F.linear(F.silu(gate) * up, down_weight)

    _CALLS["down.fused"] += 1
    _, down_plan = plans
    return down_plan(prepared_w=prepared)


@swiglu_down.register_fake
def _(gate_up, down_weight):
    return gate_up.new_empty((gate_up.shape[0], down_weight.shape[0]))


def _make_forward():
    def forward(self, positions, hidden_states, residual):
        # Attention is byte-for-byte the model path; only the MLP entry/tail are
        # replaced by opaque custom ops.
        h = hidden_states
        x = self.input_layernorm(h)
        residual = h
        hidden_states = self.self_attn(positions=positions, hidden_states=x)
        hidden_states = self.post_attention_layernorm(hidden_states)

        residual, gate_up = torch.ops.lcgemm_muse.norm_gateup(
            hidden_states,
            residual,
            self.pre_feedforward_layernorm.weight,
            self.mlp.gate_up_proj.weight,
            self.pre_feedforward_layernorm.eps,
        )
        hidden_states = torch.ops.lcgemm_muse.swiglu_down(
            gate_up, self.mlp.down_proj.weight
        )
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return residual + hidden_states, residual

    return forward


def arm() -> None:
    """Patch the decoder before vLLM compiles it."""
    if _VLLM_PREIMPORTED and not _CACHE_PRESET:
        raise RuntimeError(
            "vLLM was imported before muse_patch with compile cache enabled"
        )
    from vllm.model_executor.models.muse_glimmer import MuseGlimmerDecoderLayer

    MuseGlimmerDecoderLayer.forward = _make_forward()
    _STATE["armed"] = True


def _dry_run(gate_plan, down_plan, gate_bw, down_bw) -> None:
    m, k, _ = gate_plan.shape
    delta = torch.zeros((m, k), device="cuda", dtype=torch.bfloat16)
    residual = torch.zeros_like(delta)
    norm = torch.zeros((k,), device="cuda", dtype=torch.bfloat16)
    gate_plan(residual, delta, norm, 1e-5, prepared_w=gate_bw)
    down_plan(prepared_w=down_bw)
    torch.cuda.synchronize()


def _prepare(model, tokens: int) -> dict:
    layers = [
        module
        for module in model.modules()
        if type(module).__name__ == "MuseGlimmerDecoderLayer"
    ]
    if len(layers) != 52:
        raise RuntimeError(f"expected 52 MuseGlimmerDecoderLayer modules, found {len(layers)}")

    gate0 = layers[0].mlp.gate_up_proj.weight.data
    down0 = layers[0].mlp.down_proj.weight.data
    h, two_i = gate0.shape[1], gate0.shape[0]
    i, out = down0.shape[1], down0.shape[0]
    if (h, i, two_i, out) != (6656, 19968, 39936, 6656):
        raise RuntimeError(f"unexpected Muse MLP shapes {(h, i, two_i, out)}")

    plans = build_chain((tokens, h, two_i), (tokens, i, out))
    _PLANS[tokens] = plans

    nbytes = 0
    first = None
    for layer in layers:
        gate_w = layer.mlp.gate_up_proj.weight.data
        down_w = layer.mlp.down_proj.weight.data
        gate_bw = plans[0].prepare_weight(gate_w)
        down_bw = plans[1].prepare_weight(down_w)
        _PREPARED[gate_w.data_ptr()] = (down_w.data_ptr(), gate_bw, down_bw)
        _DOWN_PREPARED[down_w.data_ptr()] = down_bw
        nbytes += (gate_bw.numel() + down_bw.numel()) * gate_bw.element_size()
        if first is None:
            first = gate_bw, down_bw

    _dry_run(*plans, *first)
    return {
        "layers": len(layers),
        "planes_gib": nbytes / 2**30,
        "tokens": tokens,
        "describe": {"gate_up": plans[0].describe(), "down": plans[1].describe()},
    }


def install(llm, tokens: int) -> dict:
    """Prepare the exact validated shape after the engine loads its weights."""
    if not _STATE["armed"]:
        raise RuntimeError("call muse_patch.arm() before constructing LLM")
    # 512 and 1024 are *evaluation-only*: they build so the benchmark can
    # measure the arm that loses there, and they are deliberately left on the
    # default schedules.  `lcgemm.qwen_policy` still deploys stock at both.
    if tokens not in (512, 1024, 2048, 4096, 8192):
        raise ValueError(f"no Muse LC shape for M={tokens}")
    info = llm.apply_model(functools.partial(_prepare, tokens=tokens))
    info = info[0] if isinstance(info, list) else info
    _STATE.update(installed=True, layers=info["layers"], tokens=tokens)
    return info


def counts() -> dict:
    return dict(_CALLS)


def check_calls(passes: int, since: dict | None = None) -> dict:
    if not _STATE["installed"]:
        raise RuntimeError("muse_patch.check_calls before install")
    base = since or {}
    fused = sum(
        _CALLS[key] - base.get(key, 0)
        for key in ("gate_up.fused", "down.fused")
    )
    reference = sum(
        _CALLS[key] - base.get(key, 0)
        for key in ("gate_up.reference", "down.reference")
    )
    expected = passes * _STATE["layers"] * 2
    if fused != expected or reference:
        raise RuntimeError(
            f"Muse chain calls fused={fused}, reference={reference}, expected={expected}"
        )
    return {"fused_calls": fused, "reference_calls": reference, "expected": expected}
