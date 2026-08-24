#!/usr/bin/env python3
"""Check Figure 1's TeX point commands against the headline summary JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "results/headline/summary.json"
DEFAULT_SOURCE = ROOT / "tex/main.tex"
MODELS = (
    ("muse", "leftorigin"),
    ("qwen", "rightorigin"),
)
TOKENS = (1024, 2048, 4096, 8192)
POSITIONS = ("0.65", "2.15", "3.65", "5.15")


def expected_commands(summary: dict) -> list[str]:
    cases = summary["headline"]
    commands = []
    for model, origin in MODELS:
        for x, tokens in zip(POSITIONS, TOKENS):
            row = cases[f"{model}_{tokens}"]["speedup_percent"]
            commands.append(
                f"\\speeduppoint{{\\{origin}}}{{{x}}}"
                f"{{{row['q1']:.4f}}}{{{row['median']:.4f}}}{{{row['q3']:.4f}}}"
                f"{{${row['median']:.1f}\\%$}}"
            )
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    source = args.source.read_text()
    failures = [
        command
        for command in expected_commands(summary)
        if source.count(command) != 1
    ]
    if failures:
        rendered = "\n  ".join(failures)
        raise SystemExit(
            "Figure 1 commands in main.tex do not match summary.json:\n  " + rendered
        )
    print(f"Figure 1 commands match the headline summary: {args.source}")


if __name__ == "__main__":
    main()
