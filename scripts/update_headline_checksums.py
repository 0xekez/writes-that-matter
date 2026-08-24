#!/usr/bin/env python3
"""Write deterministic SHA-256 checksums for the published headline evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "results/headline"


def main() -> None:
    paths = [ROOT / "environment.json", ROOT / "summary.json", *sorted((ROOT / "raw").rglob("*.json"))]
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(ROOT)}"
        for path in paths
    ]
    output = ROOT / "SHA256SUMS"
    output.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(paths)} checksums to {output}")


if __name__ == "__main__":
    main()
