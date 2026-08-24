#!/usr/bin/env python3
"""Validate and record the B200 runtime before a reproduction run."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


EXPECTED = {
    "torch": "2.13.0",
    "triton": "3.7.1",
    "cuda-python": "13.3.1",
    "nvidia-cutlass-dsl": "4.6.2",
    "vllm": "0.26.1rc1.dev608+g99a10304d.precompiled",
    "transformers": "5.15.0",
    "flashinfer-python": "0.6.16.post3",
    "huggingface-hub": "1.27.0",
    "safetensors": "0.8.0",
    "numpy": "2.3.5",
    "ninja": "1.13.0",
    "nvidia-ml-py": "13.610.43",
}

EXPECTED_PYTHON = "3.12.13"
EXPECTED_CUDA = "13.0"
EXPECTED_DRIVER = "580.126.09"
ROOT = Path(__file__).resolve().parents[1]


def version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "MISSING"


def snapshot_revision(model_dir: Path) -> str | None:
    metadata_file = model_dir / ".cache/huggingface/download/config.json.metadata"
    if not metadata_file.is_file():
        return None
    return metadata_file.read_text().splitlines()[0].strip()


def source_revision() -> dict:
    top = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    if top.returncode or Path(top.stdout.strip()).resolve() != ROOT:
        return {"git_revision": None, "git_dirty": None}
    revision = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return {"git_revision": revision, "git_dirty": bool(dirty)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--muse-model", type=Path)
    parser.add_argument("--qwen-model", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="make the exact validated driver and Python patch release mandatory",
    )
    args = parser.parse_args()

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise SystemExit("set CUDA_VISIBLE_DEVICES to one exclusive B200")
    if "," in os.environ["CUDA_VISIBLE_DEVICES"]:
        raise SystemExit("exactly one GPU must be visible")

    import torch
    import pynvml

    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    pynvml.nvmlInit()
    driver = pynvml.nvmlSystemGetDriverVersion()
    if isinstance(driver, bytes):
        driver = driver.decode()
    packages = {name: version(name) for name in EXPECTED}
    physical = int(os.environ["CUDA_VISIBLE_DEVICES"])
    handle = pynvml.nvmlDeviceGetHandleByIndex(physical)
    report = {
        "python": platform.python_version(),
        "driver": driver,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "gpu": props.name,
        "sm_count": props.multi_processor_count,
        "memory_total_mib": total // 2**20,
        "memory_free_mib": free // 2**20,
        "power_limit_w": pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000,
        "max_sm_clock_mhz": pynvml.nvmlDeviceGetMaxClockInfo(
            handle, pynvml.NVML_CLOCK_SM
        ),
        "torch_cuda": torch.version.cuda,
        "packages": packages,
        "models": {},
        "source": source_revision(),
    }
    errors = []
    warnings = []
    if platform.python_version().split(".")[:2] != EXPECTED_PYTHON.split(".")[:2]:
        errors.append(f"Python: expected 3.12.x, got {platform.python_version()}")
    elif platform.python_version() != EXPECTED_PYTHON:
        warnings.append(
            f"Python patch: validated {EXPECTED_PYTHON}, got {platform.python_version()}"
        )
    if torch.version.cuda != EXPECTED_CUDA:
        errors.append(f"CUDA: expected {EXPECTED_CUDA}, got {torch.version.cuda}")
    if driver != EXPECTED_DRIVER:
        warnings.append(f"driver: validated {EXPECTED_DRIVER}, got {driver}")
    if shutil.which("ninja") is None:
        errors.append("ninja executable is not on PATH")
    if props.name != "NVIDIA B200" or props.multi_processor_count != 148:
        errors.append(f"expected NVIDIA B200 with 148 SMs, got {props.name!r}/{props.multi_processor_count}")
    if free < 170 * 2**30:
        errors.append(f"GPU is not sufficiently idle: only {free / 2**30:.1f} GiB free")
    for name, expected in EXPECTED.items():
        actual = packages[name]
        if name == "torch":
            actual = actual.split("+")[0]
        if actual != expected:
            errors.append(f"{name}: expected {expected}, got {packages[name]}")

    expected_models = (
        ("muse", args.muse_model, "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"),
        ("qwen", args.qwen_model, "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"),
    )
    for name, path, expected in expected_models:
        if path is None:
            continue
        path = path.expanduser().resolve()
        revision = snapshot_revision(path)
        report["models"][name] = {"path": str(path), "revision": revision}
        if revision != expected:
            errors.append(f"{name} checkpoint revision: expected {expected}, got {revision}")

    if args.strict:
        errors.extend(warnings)
        warnings = []
    report["warnings"] = warnings
    report["errors"] = errors
    print(json.dumps(report, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    if errors:
        raise SystemExit("preflight failed")


if __name__ == "__main__":
    main()
