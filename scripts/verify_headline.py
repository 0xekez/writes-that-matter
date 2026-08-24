#!/usr/bin/env python3
"""Validate raw Figure 1 evidence and compare a rerun with the reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lcgemm.bench.headline import MODELS, TOKENS, build_headline, compare_headlines

REFERENCE = ROOT / "results/headline/summary.json"


def verify_checksums(root: Path) -> None:
    checksum_file = root / "SHA256SUMS"
    failures = []
    for line in checksum_file.read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        path = root / relative.lstrip("* ")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            failures.append(str(path))
    if failures:
        raise ValueError("checksum mismatch: " + ", ".join(failures))


def print_table(candidate: dict, reference: dict) -> None:
    print("model  tokens  reference median [IQR]     observed median [IQR]      delta")
    for model in MODELS:
        for tokens in TOKENS:
            key = f"{model}_{tokens}"
            ref = reference[key]["speedup_percent"]
            got = candidate[key]["speedup_percent"]
            print(
                f"{model:5s} {tokens:6d}  "
                f"{ref['median']:+7.3f}% [{ref['q1']:+6.3f}, {ref['q3']:+6.3f}]  "
                f"{got['median']:+7.3f}% [{got['q1']:+6.3f}, {got['q3']:+6.3f}]  "
                f"{got['median'] - ref['median']:+7.3f} pp"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path, nargs="?", default=REFERENCE)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0)
    parser.add_argument(
        "--skip-raw",
        action="store_true",
        help="do not recompute a candidate summary from its sibling raw/ directory",
    )
    args = parser.parse_args()
    candidate = json.loads(args.candidate.read_text())
    reference = json.loads(args.reference.read_text())

    if not args.skip_raw:
        raw = args.candidate.parent / "raw"
        recomputed = build_headline(
            raw,
            models=MODELS,
            tokens=TOKENS,
            rounds=candidate["rounds"],
            runs=candidate["cold_l2_passes_per_arm_per_round"],
        )
        if recomputed != candidate["headline"]:
            raise SystemExit("summary does not exactly match its raw arm artifacts")
        checksum_file = args.candidate.parent / "SHA256SUMS"
        if checksum_file.exists():
            verify_checksums(args.candidate.parent)

    print_table(candidate["headline"], reference["headline"])
    failures = compare_headlines(
        candidate["headline"], reference["headline"], args.absolute_tolerance
    )
    if failures:
        raise SystemExit("\n".join(failures))
    print(
        f"verified: every median is within {args.absolute_tolerance:.1f} "
        "percentage point of the reference and has the same sign"
    )


if __name__ == "__main__":
    main()
