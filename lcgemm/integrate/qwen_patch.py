"""Qwen3.8-27B MLP-chain integration for vLLM.

Arm before engine construction, install after weights load. Only M values passed
to install use lcGEMM; dummy/profile/decode shapes take the functional reference
inside the opaque custom op.
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

_STATE = {"armed": False, "installed": False, "layers": 0, "shapes": ()}
_PLANS: dict[int, tuple[object, object]] = {}
_PREPARED: dict[int, tuple[int, torch.Tensor, torch.Tensor]] = {}
_CALLS: dict[str, int] = defaultdict(int)
_RESIDUAL_RING: list[torch.Tensor] = []
_RESIDUAL_TURN = 0
_ORIGINAL_FORWARD = None


def _residual_buf(like: torch.Tensor) -> torch.Tensor:
    global _RESIDUAL_RING, _RESIDUAL_TURN
    if not _RESIDUAL_RING:
        _RESIDUAL_RING = [torch.empty_like(like), torch.empty_like(like)]
    out = _RESIDUAL_RING[_RESIDUAL_TURN]
    _RESIDUAL_TURN = 1 - _RESIDUAL_TURN
    return out


def _reference(
    delta: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Same mathematical/rounding reference as qwen38_seams.py. The installed
    # M=4096 path never takes this branch; it exists for graph profiling/decode.
    h = (delta.float() + residual.float()).to(delta.dtype)
    hf = h.float()
    x = hf * torch.rsqrt(hf.square().mean(-1, keepdim=True) + eps)
    x = (x * (norm_weight.float() + 1.0)).to(delta.dtype)
    gate_up = F.linear(x, gate_up_weight)
    gate, up = gate_up.chunk(2, dim=-1)
    act = F.silu(gate.float()).to(gate.dtype) * up
    return F.linear(act, down_weight), h


def _dispatch(m: int, gate_up_weight: torch.Tensor, down_weight: torch.Tensor):
    plans = _PLANS.get(m)
    prepared = _PREPARED.get(gate_up_weight.data_ptr())
    if plans is None or prepared is None or prepared[0] != down_weight.data_ptr():
        return None
    return plans, prepared[1], prepared[2]


@torch.library.custom_op(
    "lcgemm_qwen::mlp_chain", mutates_args=(), device_types="cuda"
)
def mlp_chain(
    delta: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    selected = _dispatch(delta.shape[0], gate_up_weight, down_weight)
    if selected is None:
        _CALLS["chain.reference"] += 1
        return _reference(delta, residual, norm_weight, gate_up_weight, down_weight, eps)

    _CALLS["chain.fused"] += 1
    (gate_plan, down_plan), gate_bw, down_bw = selected
    residual_out = _residual_buf(delta)
    gate_plan(
        residual,
        delta,
        norm_weight,
        eps,
        prepared_w=gate_bw,
        residual_out=residual_out,
    )
    down_out = down_plan(prepared_w=down_bw)
    return down_out, residual_out


@mlp_chain.register_fake
def _(
    delta: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    eps: float,
):
    return (
        delta.new_empty((delta.shape[0], down_weight.shape[0])),
        torch.empty_like(delta),
    )


def _make_forward():
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor = None,
        **kwargs: object,
    ):
        # Qwen3.8 dense TP=1 has this false. Refuse to reinterpret the sequence-
        # parallel MoE path: the campaign is specifically the dense 27B model.
        if self.use_attn_reduce_scatter_for_moe:
            return _ORIGINAL_FORWARD(
                self,
                hidden_states=hidden_states,
                residual=residual,
                positions=positions,
                **kwargs,
            )

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states=hidden_states)
        elif self.layer_type == "full_attention":
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
            )
        else:
            raise ValueError(f"invalid Qwen layer type {self.layer_type}")

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype) + 1
                )

        hidden_states, residual = torch.ops.lcgemm_qwen.mlp_chain(
            hidden_states,
            residual,
            self.post_attention_layernorm.weight,
            self.mlp.gate_up_proj.weight,
            self.mlp.down_proj.weight,
            self.post_attention_layernorm.variance_epsilon,
        )

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype) + 1
                )
        return hidden_states, residual

    return forward


def arm() -> None:
    global _ORIGINAL_FORWARD
    if _VLLM_PREIMPORTED and not _CACHE_PRESET:
        raise RuntimeError(
            "vLLM was imported before qwen_patch with compile cache enabled; "
            "the stock graph could be reused silently"
        )
    from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer

    if _ORIGINAL_FORWARD is None:
        _ORIGINAL_FORWARD = Qwen3_5DecoderLayer.forward
    Qwen3_5DecoderLayer.forward = _make_forward()
    _STATE["armed"] = True


def _dry_run(gate_plan, down_plan, gate_bw, down_bw) -> None:
    m, k, _ = gate_plan.shape
    delta = torch.zeros((m, k), device="cuda", dtype=torch.bfloat16)
    residual = torch.zeros_like(delta)
    norm = torch.zeros((k,), device="cuda", dtype=torch.bfloat16)
    gate_plan(residual, delta, norm, 1e-6, prepared_w=gate_bw)
    down_plan(prepared_w=down_bw)
    torch.cuda.synchronize()


def _prepare(model, tokens: int) -> dict:
    layers = [
        module
        for module in model.modules()
        if type(module).__name__ == "Qwen3_5DecoderLayer"
    ]
    if len(layers) != 64:
        raise RuntimeError(f"expected 64 Qwen3_5DecoderLayer modules, found {len(layers)}")
    if any(layer.use_attn_reduce_scatter_for_moe for layer in layers):
        raise RuntimeError("Qwen MLP-chain campaign supports dense TP=1 only")
    if any(layer.layer_scale for layer in layers):
        raise RuntimeError("layer_scale was not validated for this campaign")

    gate0 = layers[0].mlp.gate_up_proj.weight.data
    down0 = layers[0].mlp.down_proj.weight.data
    h, two_i = gate0.shape[1], gate0.shape[0]
    i, out = down0.shape[1], down0.shape[0]
    if (h, i, two_i, out) != (5120, 17408, 34816, 5120):
        raise RuntimeError(f"unexpected Qwen MLP shapes {(h, i, two_i, out)}")

    _PLANS[tokens] = build_chain(
        (tokens, h, two_i),
        (tokens, i, out),
    )

    nbytes = 0
    first_prepared = None
    gate_prepare = _PLANS[tokens][0].prepare_weight
    down_prepare = _PLANS[tokens][1].prepare_weight
    for layer in layers:
        gate_w = layer.mlp.gate_up_proj.weight.data
        down_w = layer.mlp.down_proj.weight.data
        gate_bw = gate_prepare(gate_w)
        down_bw = down_prepare(down_w)
        _PREPARED[gate_w.data_ptr()] = (down_w.data_ptr(), gate_bw, down_bw)
        nbytes += gate_bw.numel() * gate_bw.element_size()
        nbytes += down_bw.numel() * down_bw.element_size()
        if first_prepared is None:
            first_prepared = (gate_bw, down_bw)

    gate_bw, down_bw = first_prepared
    _dry_run(*_PLANS[tokens], gate_bw, down_bw)

    return {
        "layers": len(layers),
        "planes_gib": nbytes / 2**30,
        "tokens": tokens,
        "schemes": {
            "gate_up": _PLANS[tokens][0].scheme.name,
            "down": _PLANS[tokens][1].scheme.name,
        },
        "describe": {
            "gate_up": _PLANS[tokens][0].describe(),
            "down": _PLANS[tokens][1].describe(),
        },
    }


def install(llm, tokens: int) -> dict:
    """Prepare one Qwen LC prefill shape: a validated one, or a losing one."""
    if not _STATE["armed"]:
        raise RuntimeError("call qwen_patch.arm() before constructing LLM")
    # 512 and 1024 are *evaluation-only*: they build so the benchmark can
    # measure the arm that loses there, and they are deliberately left on the
    # default schedules.  `lcgemm.qwen_policy` still deploys stock at both.
    if tokens not in (512, 1024, 2048, 4096, 8192):
        raise ValueError(f"no Qwen LC shape for M={tokens}")
    info = llm.apply_model(functools.partial(_prepare, tokens=tokens))
    info = info[0] if isinstance(info, list) else info
    _STATE.update(
        installed=True,
        layers=info["layers"],
        shapes=(tokens,),
    )
    return info


def counts() -> dict:
    return dict(_CALLS)


def check_calls(passes: int, since: dict | None = None) -> dict:
    if not _STATE["installed"]:
        raise RuntimeError("qwen_patch.check_calls before install")
    base = since or {}
    fused = _CALLS["chain.fused"] - base.get("chain.fused", 0)
    reference = _CALLS["chain.reference"] - base.get("chain.reference", 0)
    expected = passes * _STATE["layers"]
    if fused != expected or reference:
        raise RuntimeError(
            f"Qwen chain calls fused={fused}, reference={reference}, expected={expected}"
        )
    return {
        "fused_calls": fused,
        "reference_calls": reference,
        "expected": expected,
    }
