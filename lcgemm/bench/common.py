"""Shared cold-L2 timing and profiler helpers for the paper benchmarks."""

from __future__ import annotations

import json
import os
import re
import statistics
import tempfile
import threading
import time
from dataclasses import dataclass

import torch

FLUSH_RE = re.compile(r"FillFunctor|Memset", re.I)


@dataclass(frozen=True)
class KernelEvent:
    name: str
    ts: float
    dur: float


def kernels_from_trace(path: str) -> list[KernelEvent]:
    with open(path) as f:
        trace = json.load(f)
    kernels = []
    for event in trace.get("traceEvents", []):
        if event.get("ph") != "X":
            continue
        if (event.get("cat") or "").lower() not in (
            "kernel", "gpu_memset", "gpu_memcpy"
        ):
            continue
        if event.get("dur") is None or event.get("ts") is None:
            continue
        kernels.append(
            KernelEvent(event["name"], float(event["ts"]), float(event["dur"]))
        )
    return sorted(kernels, key=lambda event: event.ts)


def split_on_idle(kernels: list[KernelEvent], min_gap_us: float):
    windows, current, previous_end = [], [], None
    for kernel in kernels:
        if previous_end is not None and kernel.ts - previous_end > min_gap_us:
            if current:
                windows.append(current)
            current = []
        current.append(kernel)
        end = kernel.ts + kernel.dur
        previous_end = end if previous_end is None else max(previous_end, end)
    if current:
        windows.append(current)
    return windows


class L2Flush:
    """Evict L2 with a buffer twice the device's reported cache size."""

    def __init__(self):
        props = torch.cuda.get_device_properties("cuda")
        nbytes = 2 * int(getattr(props, "L2_cache_size", 128 << 20))
        self.buffer = torch.empty(nbytes, dtype=torch.int8, device="cuda")

    def __call__(self) -> None:
        self.buffer.zero_()


class ClockWatch:
    """Sample the physical GPU's SM clock while a measurement runs."""

    def __init__(self, period_s: float = 0.05):
        self.period_s = period_s
        self.samples: list[int] = []
        self._stop = threading.Event()
        try:
            import pynvml

            pynvml.nvmlInit()
            physical = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(physical)
            self.nvml = pynvml
        except Exception:
            self.nvml = None

    def _loop(self):
        while not self._stop.wait(self.period_s):
            self.samples.append(
                self.nvml.nvmlDeviceGetClockInfo(self.handle, self.nvml.NVML_CLOCK_SM)
            )

    def __enter__(self):
        if self.nvml:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        if self.nvml:
            self._stop.set()
            self._thread.join()

    def summary(self) -> dict:
        if not self.samples:
            return {}
        return {
            "sm_clock_min_mhz": min(self.samples),
            "sm_clock_mean_mhz": statistics.mean(self.samples),
            "sm_clock_max_mhz": max(self.samples),
        }


def cold_passes(call, runs: int, gap_s: float, flush: L2Flush) -> list[float]:
    samples = []
    for _ in range(runs):
        torch.cuda.synchronize()
        time.sleep(gap_s)
        flush()
        torch.cuda.synchronize()
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6)
    return samples


def profile_passes(call, runs: int, gap_s: float, flush: L2Flush):
    if runs == 0:
        return [], {}
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        for _ in range(runs):
            torch.cuda.synchronize()
            time.sleep(gap_s)
            flush()
            torch.cuda.synchronize()
            call()
            torch.cuda.synchronize()

    with tempfile.NamedTemporaryFile(suffix=".json") as temp:
        profiler.export_chrome_trace(temp.name)
        kernels = kernels_from_trace(temp.name)
    windows = split_on_idle(kernels, min_gap_us=gap_s * 1e6 * 0.5)
    windows = [[k for k in window if not FLUSH_RE.search(k.name)] for window in windows]
    windows = [window for window in windows if window]
    totals = [sum(kernel.dur for kernel in window) for window in windows]
    if totals:
        median = statistics.median(totals)
        kept = [(total, window) for total, window in zip(totals, windows) if total > median / 2]
        totals = [total for total, _ in kept]
        windows = [window for _, window in kept]

    per_kernel: dict[str, float] = {}
    for window in windows:
        for kernel in window:
            per_kernel[kernel.name] = per_kernel.get(kernel.name, 0.0) + kernel.dur
    divisor = max(1, len(windows))
    return totals, {
        name: value / divisor
        for name, value in sorted(per_kernel.items(), key=lambda item: -item[1])
    }


def stats(samples: list[float]) -> dict:
    return {
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "stddev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "n": len(samples),
        # Keep the evidence, not just aggregates. Reproduction and confidence
        # intervals must be recomputable without rerunning a 30B model.
        "samples": samples,
    }
