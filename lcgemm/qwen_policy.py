"""Graph-safe final Qwen prefill engine selection.

Stock sizes must use a process in which the decoder patch is never armed.
LC sizes use one separately constructed engine per exact prefill length; mixing
shapes in a single armed engine is intentionally not a supported deployment.
"""
from __future__ import annotations

import sys
from typing import Literal

Deployment = Literal["stock", "lc_chain"]

STOCK_TOKENS = frozenset((512, 1024))
LC_TOKENS = frozenset((2048, 4096, 8192))
SUPPORTED_TOKENS = STOCK_TOKENS | LC_TOKENS


def deployment_for(tokens: int) -> Deployment:
    """Return the validated engine kind for an exact batch-one prefill size."""
    if tokens in STOCK_TOKENS:
        return "stock"
    if tokens in LC_TOKENS:
        return "lc_chain"
    raise ValueError(
        f"no validated Qwen engine policy for M={tokens}; "
        f"supported={sorted(SUPPORTED_TOKENS)}"
    )


def arm_selected_engine(tokens: int):
    """Arm an exact LC engine before vLLM import, or leave stock truly unarmed.

    The returned patch module must be passed to install_selected_engine after
    model weights are loaded. A stock result is None and deliberately does not
    import lcgemm.integrate.qwen_patch.
    """
    selected = deployment_for(tokens)
    if selected == "stock":
        loaded = sys.modules.get("lcgemm.integrate.qwen_patch")
        if loaded is not None and getattr(loaded, "_STATE", {}).get("armed"):
            raise RuntimeError(
                "cannot construct a truly unarmed stock engine after the Qwen "
                "decoder patch was armed in this process; use a fresh process"
            )
        return None

    if "vllm" in sys.modules:
        raise RuntimeError("select and arm the LC engine before importing vLLM")
    from lcgemm.integrate import qwen_patch

    qwen_patch.arm()
    return qwen_patch


def install_selected_engine(patch, llm, tokens: int):
    """Install exactly one validated LC shape after the engine loads weights."""
    if deployment_for(tokens) != "lc_chain":
        raise ValueError(
            f"M={tokens} is a stock-only policy; do not arm or install the patch"
        )
    if patch is None:
        raise ValueError("the LC patch returned by arm_selected_engine is required")
    return patch.install(llm, tokens=tokens)
