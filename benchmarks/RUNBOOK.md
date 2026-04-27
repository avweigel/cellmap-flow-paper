# Benchmark runbook

Step-by-step for running B1, B3, and B6 on Janelia infrastructure. Read this end-to-end before launching anything; the configs reference real `/nrs/cellmap` and `/groups/cellmap` paths and will fail outside Janelia's network.

## Prerequisites

- Login on a Janelia head node with `bsub` available.
- A GPU node for B1 (interactive); B3 and B6 dispatch via `bsub` on their own.
- A Python ≥3.11 environment with `cellmap-flow==0.2.3` installed:
  ```sh
  pip install "cellmap-flow==0.2.3" pyyaml
  ```
- Verify the CLI entrypoints are on PATH:
  ```sh
  which cellmap_flow_yaml cellmap_flow_blockwise
  ```

## Sanity check (no Janelia required)

The harness has an offline smoke test against an HTTP stub:

```sh
cd /path/to/cellmap-flow-paper
python -m benchmarks.b1_interactive_latency.smoke_test
```

Expect `SMOKE OK` with median ≈ injected delay (25 ms). Run this once after any harness change.

---

## B1 — Interactive chunk-request latency

Pre-filled config: [configs/jrc_mus-salivary-1_mito_s1.yaml](b1_interactive_latency/configs/jrc_mus-salivary-1_mito_s1.yaml). It mirrors `example/sal_1_mito.yaml` from the cellmap-flow repo (mito distance model, jrc_mus-salivary-1 at s1).

### Run

In **terminal A** on a GPU node (e.g., `bsub -Is -q gpu_h100 -gpu "num=1" /bin/bash`):

```sh
cellmap_flow_yaml \
    benchmarks/b1_interactive_latency/configs/jrc_mus-salivary-1_mito_s1.yaml
# note the host:port the server prints, e.g. http://h09u01.int.janelia.org:8000
```

In **terminal B** (anywhere with HTTP access to that host):

```sh
python -m benchmarks.b1_interactive_latency.run \
    --server http://<host>:<port> \
    --dataset jrc_mus-salivary-1.zarr \
    --scale 1 \
    --chunk-grid 32 32 32 \
    --n-warmup 20 \
    --n-measure 200 \
    --output benchmarks/b1_interactive_latency/results/jrc_mus-salivary-1_mito_s1_h100.json \
    --label "jrc_mus-salivary-1 mito_distance_16 s1 H100"
```

### Sweep

Repeat with different YAMLs to cover the regime map promised in the paper:

| Sweep axis | How to vary |
|---|---|
| Chunk size | Edit the model config / `chunk_shape` in the YAML; `--chunk-grid` follows from `volume_shape // chunk_shape` |
| Model size | Use a different checkpoint or `type` (a small ScriptModel, a medium fly model, a large dacapo run) |
| GPU type | Change `-q gpu_h100` to `-q gpu_a100` / `gpu_v100` when launching the server |
| FP16 | Toggle `use_half_prediction: true` in the server YAML |

One JSON per cell of the sweep, distinct filenames so the aggregator picks them all up.

---

## B3 — Cluster strong scaling

Pre-filled config: [configs/jrc_mus-salivary-1_mito.yaml](b3_strong_scaling/configs/jrc_mus-salivary-1_mito.yaml). Same model, same dataset, but blockwise output. **Edit `output_path` to a fresh location before each sweep** — re-using a path means previously-written blocks get skipped and timings will be wrong.

### Run

From a host that can submit to LSF:

```sh
python -m benchmarks.b3_strong_scaling.run \
    --config benchmarks/b3_strong_scaling/configs/jrc_mus-salivary-1_mito.yaml \
    --workers 1 4 16 64 128 \
    --output-dir benchmarks/b3_strong_scaling/results/jrc_mus-salivary-1_mito/
```

The harness rewrites the YAML's `workers` field per run, dispatches `cellmap_flow_blockwise`, captures wall time, and writes one JSON per N to the output directory.

### Notes

- N=128 will saturate the queue if other people are running. Coordinate with the lab.
- Use `--dry-run` to print the per-N command without executing.
- Set the output Zarr to a path on `/nrs/cellmap/<your-user>/...` to avoid stomping on production paths.

---

## B6 — Hand-rolled PyTorch baseline vs. cellmap-flow

This benchmark requires **the same model loadable both ways**: as a TorchScript file (for `run_baseline.py`) and as a cellmap-flow `script` model (for `run_cellmapflow.py`). The pre-filled `fly`-type checkpoint used in B1 and B3 is not directly loadable as TorchScript; one of these two:

1. **Convert** the fly checkpoint to TorchScript by tracing the cellmap-flow model wrapper:
   ```python
   from cellmap_flow.models.models_config import FlyModelConfig
   import torch
   m = FlyModelConfig(checkpoint="...", scale="s1", resolution=16, classes=["mito"])
   ts = torch.jit.script(m.load())  # or torch.jit.trace with a dummy input
   ts.save("mito_distance_16.ts")
   ```
2. Or **pick** a TorchScript-native model from the cellmap-models registry or HuggingFace and use it on both sides.

Once you have a TorchScript file:

- Edit [configs/template.yaml](b6_baseline_comparison/configs/template.yaml) to point at the TorchScript file (`model_checkpoint`) and the corresponding cellmap-flow `script` YAML.
- Run:
  ```sh
  python -m benchmarks.b6_baseline_comparison.run_baseline \
      --config benchmarks/b6_baseline_comparison/configs/template.yaml \
      --output benchmarks/b6_baseline_comparison/results/baseline.json

  python -m benchmarks.b6_baseline_comparison.run_baseline \
      --config benchmarks/b6_baseline_comparison/configs/template.yaml \
      --first-view-only \
      --output benchmarks/b6_baseline_comparison/results/baseline_first_view.json

  python -m benchmarks.b6_baseline_comparison.run_cellmapflow \
      --config benchmarks/b6_baseline_comparison/configs/template.yaml \
      --output benchmarks/b6_baseline_comparison/results/cellmapflow.json
  ```

---

## After all runs: regenerate paper tables

```sh
python benchmarks/regenerate_paper_tables.py \
    --results-dir benchmarks/ \
    --out figures/benchmark_tables.tex
```

Then in `main.tex` (or wherever you want the tables to appear), add:

```latex
\input{figures/benchmark_tables}
```

Recompile in Overleaf and the numbers flow into the paper.

---

## Open questions before launching the real runs

- [ ] Confirm the pre-filled `jrc_mus-salivary-1` + `mito_distance_16` is the dataset/model you want as the running example. If not, swap paths in the YAMLs.
- [ ] Confirm you have write access to the `output_path` set in the B3 config; pick a fresh path each sweep.
- [ ] Decide which sweep axes for B1 are essential for v1 of the paper (recommended: chunk size × GPU type, three of each = 9 cells).
- [ ] For B6, decide TorchScript-conversion vs. picking a different model. Conversion is one-time prep but lets B6 use the same model as B1/B3 for cross-benchmark consistency.
