Code for reproducing results in
[this](https://percisely.xyz/writes-that-matter) blog post. Namely,
speeding up large prefills on the recent Muse Glimmer and Qwen3.8 27B
models using low-complexity matrix multiplication. You will need a
B200 GPU to do this.

| Model | Prefill tokens | Median speedup | IQR |
|---|---:|---:|---:|
| Muse Glimmer 30B | 2,048 | +3.487% | [+3.238%, +4.027%] |
| Muse Glimmer 30B | 4,096 | +4.676% | [+4.606%, +4.761%] |
| Muse Glimmer 30B | 8,192 | +1.959% | [+1.833%, +2.010%] |
| Qwen3.8 27B | 2,048 | +3.755% | [+3.315%, +3.899%] |
| Qwen3.8 27B | 4,096 | +4.094% | [+3.333%, +4.500%] |
| Qwen3.8 27B | 8,192 | +2.741% | [+2.590%, +2.852%] |

Speedup is with respect to vLLM. The table reports the median and interquartile
range (IQR) over ten rounds; each arm time within a round is the median of 40
cold-L2 passes.

## 1. Create the validated environment

Start from Ubuntu 24.04 on an otherwise idle B200. Run these commands from this
repository's root. The exact validated hardware and package versions are in
[`environment.json`](environment.json).

```bash
uv venv --python 3.12.13 --seed .venv
source .venv/bin/activate

git clone https://github.com/vllm-project/vllm.git ../vllm-pinned
git -C ../vllm-pinned checkout --detach 99a10304dce8945119bd0b1a072297803c52a749
VLLM_USE_PRECOMPILED=1 uv pip install --editable ../vllm-pinned --torch-backend=auto

uv pip install --requirements requirements-b200.txt --torch-backend=auto
uv pip install --no-deps --editable .
```

The pinned vLLM distribution identifies itself as
`0.26.1rc1.dev608+g99a10304d.precompiled`. If its precompiled wheel is no
longer available from the upstream wheel service, build that exact revision
against CUDA 13.0.

## 2. Download the exact checkpoints

The Muse checkpoint may require accepting the model terms and authenticating
with Hugging Face first.

```bash
hf download meta-models/Muse-Glimmer-30B \
  --revision a4e59da52a7bc87ae7251dd5545c0dd437c44b68 \
  --local-dir "$HOME/muse-glimmer/Muse-Glimmer-30B"

hf download Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --local-dir "$HOME/qwen38-27b"
```

Use `--muse-model` or `--qwen-model` if the snapshots live elsewhere. The
preflight checks their revisions before loading either model.

## 3. Run the artifact

Choose an idle physical GPU index from `nvidia-smi`. First run the four-process
wiring check (one stock/optimized pair per model):

```bash
python scripts/reproduce_headline.py --gpu 0 --smoke
```

The smoke run validates the environment, model revisions, exact-shape dispatch,
zero reference fallbacks, and next-token agreement. Its single timing sample is
not a speedup result.

The following command reproduces all eight Figure 1 points with the publication
protocol:

```bash
python scripts/reproduce_headline.py --gpu 0
```

This launches 160 fresh model processes (two arms x ten rounds x eight points)
and took about 4.25 hours on the validated machine. Results are written to a
timestamped directory under `results/runs/`. The runner alternates arm order,
is resumable, and never silently reuses an artifact whose model, revision,
shape, arm, protocol, or sample count differs.

At completion it compares every reproduced median with the shipped reference.
The automated reproduction guard requires the same sign and an absolute
difference no larger than 1.0 percentage point.

To validate the shipped evidence without a GPU:

```bash
python scripts/verify_headline.py
python scripts/check_headline_tex.py
```

The first command recomputes every round, quartile, and median from the raw
files, validates all SHA-256 checksums, and checks the headline guard.

## Measurement protocol

- One synthetic prompt, batch size one, BF16 weights and activations.
- A complete model prefill and one generated token through a real vLLM engine.
- Stock and optimized arms in separate processes, with alternating order.
- Three warmups, then 40 timed passes with a 250 ms idle gap and an L2 eviction
  before each pass.
- The median pass time represents an arm; the ratio of matched arm medians
  represents a round; Figure 1 summarizes ten round-level ratios.
- Prefix caching and the vLLM compile cache are disabled before vLLM import.
- The optimized arm must report exact per-layer fused-call counts, zero
  reference fallbacks, equal top-1, and at least 19/20 tie-aware top-token
  agreement.
- SM clocks are sampled for each arm and retained, but wall times are not clock
  normalized. Raw wall time is the deployment-facing measurement.

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for more details.
