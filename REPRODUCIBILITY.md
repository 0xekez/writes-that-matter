# Figure 1 reproduction protocol

## Validated platform

The reference run used one 182632 MiB NVIDIA B200 with 148 SMs, driver
580.126.09, CUDA 13.0, Ubuntu 24.04.3, and Python 3.12.13. Exact Python package,
vLLM, and model revisions are recorded in `environment.json`. The benchmark
uses vLLM internals, so changing its revision changes the system under test.

`scripts/preflight.py` rejects the wrong GPU, fewer than 170 GiB of free device
memory, package mismatches, CUDA mismatches, and checkpoint-revision mismatches.
It records the observed environment next to every run. A different driver or
Python 3.12 patch release is recorded as a warning because it can affect timing
but need not make the experiment invalid; pass `--strict` to make those exact
matches mandatory.

## Experimental unit and estimator

The unit of replication is one matched fresh-process round. One arm runs stock
vLLM and the other installs lcGEMM before constructing the engine. Odd rounds
run stock then optimized; even rounds reverse the order.

Within each arm:

1. construct the engine and run the correctness probes;
2. run three untimed warmups;
3. synchronize the GPU and wait 250 ms;
4. overwrite a buffer twice the device's reported L2 size;
5. synchronize, time one generated token after the fixed prefill, and
   synchronize again;
6. repeat steps 3--5 forty times without trimming any stored sample.

The arm estimate is the median of its 40 wall samples. The round estimate is
`100 * (stock arm median / optimized arm median - 1)`. Figure 1 reports the
median, first quartile, and third quartile of ten round estimates, using linear
interpolation at `(n - 1) * p` (the NumPy default quantile convention).

The within-arm median is deliberate: a few preserved 40-pass blocks contain a
late lazy-compilation or host-scheduling outlier even after warmup. All samples
remain in the JSON, so readers can inspect or replace this estimator.

## Isolation and exact-shape dispatch

Every arm is a new Python process. `VLLM_DISABLE_COMPILE_CACHE=1` and
`VLLM_ENABLE_V1_MULTIPROCESSING=0` are set before vLLM import. This prevents an
optimized process from reusing a stock compiled graph and keeps prepared CuTe
callables and fused-call counters in the measured process.

`max_model_len` and `max_num_batched_tokens` are derived from the requested
prefill length. Prefix caching is disabled. The optimized process installs only
the exact requested shape. M=1024 is available for evaluation but is not chosen
by the deployment policy because Figure 1 shows that it loses.

## Correctness gates

The optimized process records counters inside each opaque custom operation.
Every measured pass must invoke every eligible fused MLP operation, and the
reference fallback count after the counter snapshot must be zero.

Each arm also records a greedy 64-token continuation and the top 20 next-token
log probabilities. Each matched round requires equal top-1 and at least 19/20
tie-aware top-token agreement. The raw overlap and full recorded continuation
remain in each artifact. See `NUMERICS.md` for why ties at the BF16 top-20
boundary need special handling.

## Clocks and interpretation

SM clocks are sampled throughout every arm and retained with the round. The
paper reports observed wall time without clock normalization. Dynamic power and
clock behavior are therefore part of the measured deployment result. Use an
exclusive B200 and avoid concurrent inference, profiling, or clock management.

The 1.0-percentage-point reproduction guard is a practical alarm for a materially
different result, not a statistical confidence interval. Publication-quality
comparisons should retain every process, report the process-level distribution,
and avoid treating the 40 correlated passes inside one process as 40 independent
replicates.

## Provenance

`results/headline/summary.json` is a deterministic aggregation of the 160 JSON
files below `results/headline/raw/`. `scripts/verify_headline.py` rebuilds it
from those files and validates `SHA256SUMS`. `scripts/check_headline_tex.py`
then checks the exact Figure 1 point commands in `tex/main.tex` against the
summary without modifying the paper. `main.tex` is the artifact's only TeX
source file.
