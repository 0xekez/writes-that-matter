#!/usr/bin/env python3
"""Reproduce Figure 1's matched-round B200 prefill measurements."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lcgemm.bench.headline import (
    ARMS,
    MODELS,
    TOKENS,
    agreement,
    arm_path,
    arm_summary,
    compare_headlines,
    distribution,
    expected_order,
    load_arm,
)

REFERENCE = ROOT / "results/headline/summary.json"


def run(command: list[str], env: dict[str, str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def run_arm(
    *,
    raw: Path,
    logs: Path,
    model: str,
    tokens: int,
    round_number: int,
    arm: str,
    runs: int,
    warmup: int,
    gap: float,
    gen_tokens: int,
    env: dict[str, str],
    model_path: Path,
) -> dict:
    output = arm_path(raw, model, tokens, round_number, arm)
    if output.exists():
        result = load_arm(
            output,
            model=model,
            tokens=tokens,
            arm=arm,
            runs=runs,
            warmup=warmup,
            gap=gap,
            gen_tokens=gen_tokens,
        )
        print(f"  reuse {output}", flush=True)
        return result

    command = [
        sys.executable,
        "-m",
        "lcgemm.bench.prefill",
        "--model",
        model,
        "--tokens",
        str(tokens),
        "--runs",
        str(runs),
        "--profile-runs",
        "0",
        "--warmup",
        str(warmup),
        "--gap",
        str(gap),
        "--gen-tokens",
        str(gen_tokens),
        "--arm",
        arm,
        "--model-dir",
        str(model_path),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    log = logs / output.relative_to(raw).with_suffix(".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"  run {arm}", flush=True)
    process = subprocess.run(
        command, cwd=ROOT, env=env, capture_output=True, text=True
    )
    log.write_text(process.stdout + "\n--- STDERR ---\n" + process.stderr)
    if process.returncode:
        raise RuntimeError(f"{arm} failed with rc={process.returncode}; see {log}")
    lines = [
        line for line in process.stdout.splitlines() if line.startswith("RESULT_JSON ")
    ]
    if not lines:
        raise RuntimeError(f"{arm} emitted no RESULT_JSON; see {log}")
    result = json.loads(lines[-1].split(" ", 1)[1])
    # Validate before publishing the file. A failed arm remains in its log only.
    temp = output.with_suffix(".pending.json")
    temp.write_text(json.dumps(result, indent=2) + "\n")
    try:
        load_arm(
            temp,
            model=model,
            tokens=tokens,
            arm=arm,
            runs=runs,
            warmup=warmup,
            gap=gap,
            gen_tokens=gen_tokens,
        )
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    temp.replace(output)
    return result


def collect_case(args, raw: Path, logs: Path, env: dict[str, str], model: str, tokens: int):
    rows = []
    for round_number in range(1, args.rounds + 1):
        order = expected_order(round_number)
        print(
            f"{model}_{tokens} round {round_number}/{args.rounds}: "
            + "/".join(order),
            flush=True,
        )
        arms = {}
        for arm in order:
            arms[arm] = run_arm(
                raw=raw,
                logs=logs,
                model=model,
                tokens=tokens,
                round_number=round_number,
                arm=arm,
                runs=args.runs,
                warmup=args.warmup,
                gap=args.gap,
                gen_tokens=args.gen_tokens,
                env=env,
                model_path=(args.muse_model if model == "muse" else args.qwen_model),
            )
        agreed = agreement(arms["stock"], arms["optimized"])
        if not agreed["top1_equal"] or agreed["top20_agreement"] < 19:
            raise RuntimeError(
                f"next-token agreement failed for {model}_{tokens} round "
                f"{round_number}: {agreed}"
            )
        summaries = {arm: arm_summary(result) for arm, result in arms.items()}
        rows.append(
            {
                "round": round_number,
                "order": list(order),
                "arms": summaries,
                "speedup_percent": 100
                * (
                    summaries["stock"]["median_us"]
                    / summaries["optimized"]["median_us"]
                    - 1
                ),
                "agreement": agreed,
            }
        )
    return {
        "model": model,
        "tokens": tokens,
        "rounds": rows,
        "speedup_percent": distribution([row["speedup_percent"] for row in rows]),
    }


def write_summary(path: Path, summary: dict) -> None:
    path.write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True, help="one exclusive physical B200 index")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--runs", type=int, default=40, help="cold-L2 passes per arm")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--gap", type=float, default=0.25)
    parser.add_argument("--gen-tokens", type=int, default=64)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--tokens", nargs="+", type=int, choices=TOKENS, default=TOKENS)
    parser.add_argument(
        "--muse-model",
        type=Path,
        default=Path.home() / "muse-glimmer/Muse-Glimmer-30B",
    )
    parser.add_argument(
        "--qwen-model", type=Path, default=Path.home() / "qwen38-27b"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="one 4096-token process pair per model; validates wiring, not speedup",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--absolute-tolerance", type=float, default=1.0)
    args = parser.parse_args()
    if min(args.rounds, args.runs, args.warmup + 1, args.gen_tokens) <= 0:
        parser.error("rounds, runs, warmup, and gen-tokens must be non-negative/positive")
    if args.gap < 0:
        parser.error("gap must be non-negative")
    if args.smoke:
        args.models = MODELS
        args.tokens = (4096,)
        args.rounds, args.runs, args.warmup, args.gen_tokens = 1, 1, 1, 8

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output_dir or ROOT / "results/runs" / stamp).resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw, logs = output / "raw", output / "logs"
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": args.gpu,
        "PYTHONPATH": str(ROOT),
    }
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    args.muse_model = args.muse_model.expanduser().resolve()
    args.qwen_model = args.qwen_model.expanduser().resolve()

    run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], env)
    if not args.skip_preflight:
        run(
            [
                sys.executable,
                "scripts/preflight.py",
                "--muse-model",
                str(args.muse_model),
                "--qwen-model",
                str(args.qwen_model),
                "--json",
                str(output / "environment.json"),
            ],
            env,
        )

    summary = {
        "schema_version": 1,
        "estimator": "median arm time within each fresh-process block; ratio within block",
        "iqr_unit": "matched round/block speedups",
        "rounds": args.rounds,
        "cold_l2_passes_per_arm_per_round": args.runs,
        "gpu": args.gpu,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "headline": {},
    }
    for model in args.models:
        for tokens in args.tokens:
            key = f"{model}_{tokens}"
            summary["headline"][key] = collect_case(
                args, raw, logs, env, model, tokens
            )
            write_summary(output / "summary.json", summary)
    summary["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_summary(output / "summary.json", summary)

    canonical = (
        not args.smoke
        and tuple(args.models) == MODELS
        and tuple(args.tokens) == TOKENS
        and args.rounds == 10
        and args.runs == 40
    )
    if canonical:
        reference = json.loads(REFERENCE.read_text())
        failures = compare_headlines(
            summary["headline"], reference["headline"], args.absolute_tolerance
        )
        if failures:
            raise SystemExit("\n".join(failures))
        print(
            f"all Figure 1 medians are within {args.absolute_tolerance:.1f} "
            "percentage point of the reference",
            flush=True,
        )
    else:
        print("custom/smoke run complete; no headline speedup claim was tested", flush=True)
    print(f"results: {output}", flush=True)


if __name__ == "__main__":
    main()
