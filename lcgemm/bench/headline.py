"""Pure validation and aggregation for the paper's headline experiment.

The unit of replication is a matched fresh-process round.  Each arm contributes
the median of its cold-L2 wall samples, and the round contributes the ratio of
those two medians.  Figure 1 reports the median and interquartile range of the
ten round-level ratios.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


MODELS = ("muse", "qwen")
TOKENS = (1024, 2048, 4096, 8192)
ARMS = ("stock", "optimized")
LOGIT_STEP = 0.125
MODEL_REVISIONS = {
    "muse": "a4e59da52a7bc87ae7251dd5545c0dd437c44b68",
    "qwen": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
}


def quantile(values: list[float], probability: float) -> float:
    """Linearly interpolate at ``(n - 1) * probability`` (NumPy default)."""
    if not values:
        raise ValueError("cannot take a quantile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values: list[float]) -> dict:
    return {
        "n": len(values),
        "q1": quantile(values, 0.25),
        "median": quantile(values, 0.50),
        "q3": quantile(values, 0.75),
        "values": values,
    }


def agreement(stock: dict, candidate: dict) -> dict:
    left = dict(stock["logprobs"])
    right = dict(candidate["logprobs"])
    if not left or not right:
        raise ValueError("both arms must retain next-token log probabilities")
    right_cut = min(right.values())
    tie_aware = sum(
        1
        for token, logprob in left.items()
        if token in right or logprob <= right_cut + LOGIT_STEP
    )
    return {
        "top1_equal": stock["logprobs"][0][0] == candidate["logprobs"][0][0],
        "top20_overlap": len(set(left) & set(right)),
        "top20_agreement": tie_aware,
        "greedy_equal": stock["generated"] == candidate["generated"],
    }


def expected_order(round_number: int) -> tuple[str, str]:
    return ARMS if round_number % 2 else tuple(reversed(ARMS))


def arm_path(raw: Path, model: str, tokens: int, round_number: int, arm: str) -> Path:
    return raw / f"{model}_{tokens}" / f"round_{round_number:02d}_{arm}.json"


def load_arm(
    path: Path,
    *,
    model: str,
    tokens: int,
    arm: str,
    runs: int,
    warmup: int | None = None,
    gap: float | None = None,
    gen_tokens: int | None = None,
) -> dict:
    try:
        result = json.loads(path.read_text())
    except FileNotFoundError:
        raise FileNotFoundError(f"missing arm artifact: {path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error
    validate_arm(
        result,
        path=path,
        model=model,
        tokens=tokens,
        arm=arm,
        runs=runs,
        warmup=warmup,
        gap=gap,
        gen_tokens=gen_tokens,
    )
    return result


def validate_arm(
    result: dict,
    *,
    path: Path,
    model: str,
    tokens: int,
    arm: str,
    runs: int,
    warmup: int | None = None,
    gap: float | None = None,
    gen_tokens: int | None = None,
) -> None:
    expected = {
        "schema_version": 1,
        "model": model,
        "model_revision": MODEL_REVISIONS[model],
        "arm": arm,
        "tokens": tokens,
        "dtype": "bfloat16",
        "cache_protocol": "cold L2 before every measured pass",
        "passes": runs,
    }
    errors = [
        f"{key}={result.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if result.get(key) != value
    ]
    wall = result.get("wall_us", {})
    samples = wall.get("samples", ())
    if wall.get("n") != runs or len(samples) != runs:
        errors.append(
            f"wall sample count={wall.get('n')!r}/{len(samples)}, expected {runs}"
        )
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
           for value in samples):
        errors.append("wall samples must be finite and positive")
    clock = result.get("sm_clock_mean_mhz")
    if not isinstance(clock, (int, float)) or not math.isfinite(clock) or clock <= 0:
        errors.append("missing or invalid SM-clock telemetry")
    if arm == "optimized":
        if result.get("fused_calls") != result.get("expected"):
            errors.append(
                f"fused calls={result.get('fused_calls')}, expected={result.get('expected')}"
            )
        if result.get("reference_calls") != 0:
            errors.append(f"reference fallback calls={result.get('reference_calls')}")
    requested_measurement = {
        "warmup": warmup,
        "gap_s": gap,
        "gen_tokens": gen_tokens,
        "profile_runs": 0 if warmup is not None else None,
    }
    if any(value is not None for value in requested_measurement.values()):
        observed_measurement = result.get("measurement")
        if observed_measurement is None:
            errors.append("missing measurement-control metadata")
        else:
            for key, value in requested_measurement.items():
                if value is not None and observed_measurement.get(key) != value:
                    errors.append(
                        f"measurement.{key}={observed_measurement.get(key)!r}, "
                        f"expected {value!r}"
                    )
    if errors:
        raise ValueError(f"invalid arm artifact {path}:\n  " + "\n  ".join(errors))


def arm_summary(result: dict) -> dict:
    samples = result["wall_us"]["samples"]
    return {
        "median_us": statistics.median(samples),
        "mean_us": statistics.mean(samples),
        "sm_clock_mean_mhz": result["sm_clock_mean_mhz"],
    }


def build_case(raw: Path, model: str, tokens: int, rounds: int, runs: int) -> dict:
    round_rows = []
    for round_number in range(1, rounds + 1):
        arms = {
            arm: load_arm(
                arm_path(raw, model, tokens, round_number, arm),
                model=model,
                tokens=tokens,
                arm=arm,
                runs=runs,
            )
            for arm in ARMS
        }
        agreed = agreement(arms["stock"], arms["optimized"])
        if not agreed["top1_equal"] or agreed["top20_agreement"] < 19:
            raise ValueError(
                f"next-token agreement failed for {model}_{tokens} round "
                f"{round_number}: {agreed}"
            )
        summaries = {arm: arm_summary(result) for arm, result in arms.items()}
        speedup = 100 * (
            summaries["stock"]["median_us"]
            / summaries["optimized"]["median_us"]
            - 1
        )
        round_rows.append(
            {
                "round": round_number,
                "order": list(expected_order(round_number)),
                "arms": summaries,
                "speedup_percent": speedup,
                "agreement": agreed,
            }
        )
    return {
        "model": model,
        "tokens": tokens,
        "rounds": round_rows,
        "speedup_percent": distribution(
            [row["speedup_percent"] for row in round_rows]
        ),
    }


def build_headline(
    raw: Path,
    *,
    models: tuple[str, ...] = MODELS,
    tokens: tuple[int, ...] = TOKENS,
    rounds: int = 10,
    runs: int = 40,
) -> dict:
    return {
        f"{model}_{token_count}": build_case(
            raw, model, token_count, rounds, runs
        )
        for model in models
        for token_count in tokens
    }


def compare_headlines(candidate: dict, reference: dict, tolerance: float) -> list[str]:
    failures = []
    if set(candidate) != set(reference):
        return [
            "case set differs: "
            f"candidate={sorted(candidate)}, reference={sorted(reference)}"
        ]
    for key in sorted(reference):
        observed = candidate[key]["speedup_percent"]["median"]
        target = reference[key]["speedup_percent"]["median"]
        if observed and target and (observed > 0) != (target > 0):
            failures.append(f"{key}: sign changed ({observed:+.4f}% vs {target:+.4f}%)")
        if abs(observed - target) > tolerance:
            failures.append(
                f"{key}: median differs by {observed - target:+.4f} percentage "
                f"points (limit {tolerance:.4f})"
            )
    return failures
