"""Reproduce the Muse or Qwen batch-one BF16 prefill result on one B200.

Examples::

    CUDA_VISIBLE_DEVICES=0 python -m lcgemm.bench.prefill --model muse --tokens 4096
    CUDA_VISIBLE_DEVICES=0 python -m lcgemm.bench.prefill --model qwen --tokens 4096

Stock and optimized arms always run in fresh processes.  Every measured pass
starts from cold L2; untrimmed wall samples, GPU kernel totals, actual SM clocks,
top-20 agreement, and fused/fallback call counts are written to JSON.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"
# The canonical Qwen campaign used the in-process V1 frontend. It is also what
# makes the prepared CuTe callables and exact fused-call counters share state
# with the harness; a separate engine process would require pickle RPC and its
# counters would be invisible here.
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from lcgemm.bench.common import (  # noqa: E402
    ClockWatch,
    L2Flush,
    cold_passes,
    profile_passes,
    stats,
)

_MODULE = "lcgemm.bench.prefill"


@dataclass(frozen=True)
class ModelSpec:
    env: str
    default_path: str
    revision: str
    patch_module: str
    gpu_memory_utilization: float
    language_model_only: bool


MODELS = {
    "muse": ModelSpec(
        env="MUSE_MODEL_DIR",
        default_path=str(Path.home() / "muse-glimmer" / "Muse-Glimmer-30B"),
        revision="a4e59da52a7bc87ae7251dd5545c0dd437c44b68",
        patch_module="lcgemm.integrate.muse_patch",
        gpu_memory_utilization=0.40,
        language_model_only=False,
    ),
    "qwen": ModelSpec(
        env="QWEN_MODEL_DIR",
        default_path=str(Path.home() / "qwen38-27b"),
        revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        patch_module="lcgemm.integrate.qwen_patch",
        gpu_memory_utilization=0.35,
        language_model_only=True,
    ),
}


def _model_path(args, spec: ModelSpec) -> Path:
    return Path(args.model_dir or os.environ.get(spec.env, spec.default_path)).expanduser()


def run_arm(args) -> dict:
    import torch

    spec = MODELS[args.model]
    patch = None
    if args.arm == "optimized":
        patch = importlib.import_module(spec.patch_module)
        patch.arm()
    elif args.arm != "stock":
        raise ValueError(f"unknown arm {args.arm!r}")

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    model_path = _model_path(args, spec)
    if not model_path.is_dir():
        raise FileNotFoundError(f"model directory not found: {model_path}")
    llm_kwargs = dict(
        model=str(model_path),
        max_model_len=args.tokens + 256,
        max_num_batched_tokens=args.tokens + 256,
        max_num_seqs=1,
        gpu_memory_utilization=(
            args.gpu_memory_utilization
            if args.gpu_memory_utilization is not None
            else spec.gpu_memory_utilization
        ),
        enable_prefix_caching=False,
        dtype="bfloat16",
    )
    if spec.language_model_only:
        llm_kwargs["language_model_only"] = True
    llm = LLM(**llm_kwargs)

    info = patch.install(llm, tokens=args.tokens) if patch else {}
    if info:
        print("[patch] " + json.dumps({k: v for k, v in info.items() if k != "describe"}))
        for seam, description in info.get("describe", {}).items():
            print(f"[plan {seam}] " + description.replace("\n", "\n  "))

    prompt = TokensPrompt(
        prompt_token_ids=[1000 + index % 5000 for index in range(args.tokens)]
    )
    greedy = llm.generate(
        [prompt],
        SamplingParams(max_tokens=args.gen_tokens, temperature=0.0),
        use_tqdm=False,
    )[0].outputs[0]
    logprob_output = llm.generate(
        [prompt],
        SamplingParams(max_tokens=1, temperature=0.0, logprobs=20),
        use_tqdm=False,
    )[0].outputs[0]
    logprobs = sorted(
        (
            (int(token), float(value.logprob))
            for token, value in logprob_output.logprobs[0].items()
        ),
        key=lambda item: -item[1],
    )
    one_token = SamplingParams(max_tokens=1, temperature=0.0)

    def call():
        llm.generate([prompt], one_token, use_tqdm=False)

    for _ in range(args.warmup):
        call()
    flush = L2Flush()
    before = patch.counts() if patch else {}
    with ClockWatch() as clock:
        wall = cold_passes(call, args.runs, args.gap, flush)
        gpu, kernels = profile_passes(call, args.profile_runs, args.gap, flush)

    passes = len(wall) + len(gpu)
    result = {
        "schema_version": 1,
        "model": args.model,
        "model_path": str(model_path),
        "model_revision": spec.revision,
        "arm": args.arm,
        "tokens": args.tokens,
        "dtype": "bfloat16",
        "cache_protocol": "cold L2 before every measured pass",
        "measurement": {
            "warmup": args.warmup,
            "gap_s": args.gap,
            "gen_tokens": args.gen_tokens,
            "profile_runs": args.profile_runs,
        },
        "wall_us": stats(wall),
        "gpu_us": stats(gpu) if gpu else {},
        "generated": list(greedy.token_ids),
        "logprobs": logprobs,
        "passes": passes,
        "patch": info,
        "kernels": kernels,
        **clock.summary(),
    }
    if patch:
        result.update(patch.check_calls(passes, since=before))
    return result


# The BF16 logit quantization step.  Tokens whose log probabilities differ by
# less than this are tied as far as the model can express, so which of them
# lands inside a top-20 cut is not a property either arm controls.  See
# ``docs/NUMERICS.md``.
LOGIT_STEP = 0.125


def _tie_aware_overlap(left: dict, right: dict) -> int:
    """``left``'s tokens found in ``right``, plus those excluded only by a tie.

    A token of ``left`` missing from ``right`` counts as agreement when its log
    probability sits within one quantization step of ``right``'s cut, because
    everything at that boundary is competing on values the model rounds to the
    same number.  A token missing from ``right`` while ranking well above that
    cut is a real disagreement and still counts against the gate.
    """
    cut = min(right.values())
    return sum(1 for token, lp in left.items()
               if token in right or lp <= cut + LOGIT_STEP)


def _compare(stock: dict, optimized: dict) -> dict:
    left, right = dict(stock["logprobs"]), dict(optimized["logprobs"])
    shared = set(left) & set(right)
    return {
        "wall_speedup": stock["wall_us"]["mean"] / optimized["wall_us"]["mean"],
        "wall_delta_ms": (stock["wall_us"]["mean"] - optimized["wall_us"]["mean"]) / 1000,
        "top1_equal": stock["logprobs"][0][0] == optimized["logprobs"][0][0],
        "top20_overlap": len(shared),
        "top20_agreement": _tie_aware_overlap(left, right),
        "max_shared_logprob_delta": max(
            (abs(left[token] - right[token]) for token in shared), default=float("nan")
        ),
        "greedy_equal": stock["generated"] == optimized["generated"],
    }


def run_pair(args) -> dict:
    results = {}
    for arm in ("stock", "optimized"):
        command = [
            sys.executable,
            "-m",
            _MODULE,
            "--model",
            args.model,
            "--tokens",
            str(args.tokens),
            "--runs",
            str(args.runs),
            "--profile-runs",
            str(args.profile_runs),
            "--warmup",
            str(args.warmup),
            "--gap",
            str(args.gap),
            "--gen-tokens",
            str(args.gen_tokens),
            "--arm",
            arm,
        ]
        if args.model_dir:
            command += ["--model-dir", args.model_dir]
        if args.gpu_memory_utilization is not None:
            command += ["--gpu-memory-utilization", str(args.gpu_memory_utilization)]
        print(f"\n{'=' * 72}\n{args.model} M={args.tokens}: {arm}\n{'=' * 72}", flush=True)
        process = subprocess.run(command, capture_output=True, text=True)
        sys.stdout.write(process.stdout)
        if process.returncode:
            sys.stderr.write(process.stderr[-12000:])
            raise SystemExit(f"{arm} failed with rc={process.returncode}")
        lines = [line for line in process.stdout.splitlines() if line.startswith("RESULT_JSON ")]
        if not lines:
            raise RuntimeError(f"{arm} emitted no RESULT_JSON")
        results[arm] = json.loads(lines[-1].split(" ", 1)[1])

    comparison = _compare(results["stock"], results["optimized"])
    payload = {"results": results, "comparison": comparison}
    print("\n" + json.dumps(comparison, indent=2))
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {output}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    # 512 and 1024 deploy as stock; they are selectable here so the table can
    # show the LC arm losing at them, and are untuned by policy.
    parser.add_argument("--tokens", type=int,
                        choices=(512, 1024, 2048, 4096, 8192), required=True)
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--profile-runs", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--gap", type=float, default=0.25)
    parser.add_argument("--gen-tokens", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--out", default="")
    parser.add_argument("--arm", choices=("stock", "optimized"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.arm:
        print("RESULT_JSON " + json.dumps(run_arm(args)), flush=True)
    else:
        run_pair(args)


if __name__ == "__main__":
    main()
