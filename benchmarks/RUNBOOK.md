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

Two pre-filled configs, both public S3 + public HuggingFace, covering cell-culture vs tissue:

- [configs/jrc_hela-2_fly_organelles.yaml](b1_interactive_latency/configs/jrc_hela-2_fly_organelles.yaml) — HeLa cell line (cell culture)
- [configs/jrc_mus-liver_fly_organelles.yaml](b1_interactive_latency/configs/jrc_mus-liver_fly_organelles.yaml) — mouse liver (tissue)

Both use `cellmap/fly_organelles_run07_432000` so the model is held fixed; latency differences across the two datasets reflect I/O and chunk-content variance, not model variance. Both datasets were published in Heinrich et al., Nature 2021.

### Run

Repeat for each dataset. In **terminal A** on a GPU node (e.g., `bsub -Is -q gpu_h100 -gpu "num=1" /bin/bash`):

```sh
# cell culture
cellmap_flow_yaml benchmarks/b1_interactive_latency/configs/jrc_hela-2_fly_organelles.yaml
# ... or, for the tissue dataset:
cellmap_flow_yaml benchmarks/b1_interactive_latency/configs/jrc_mus-liver_fly_organelles.yaml
# note the host:port the server prints, e.g. http://h09u01.int.janelia.org:8000
```

In **terminal B** (anywhere with HTTP access to that host):

```sh
# cell culture
python -m benchmarks.b1_interactive_latency.run \
    --server http://<host>:<port> \
    --dataset jrc_hela-2.zarr \
    --scale 1 \
    --chunk-grid 32 32 32 \
    --n-warmup 20 \
    --n-measure 200 \
    --output benchmarks/b1_interactive_latency/results/jrc_hela-2_fly_organelles_s1_h100.json \
    --label "jrc_hela-2 fly_organelles_run07_432000 s1 H100"

# tissue
python -m benchmarks.b1_interactive_latency.run \
    --server http://<host>:<port> \
    --dataset jrc_mus-liver.zarr \
    --scale 1 \
    --chunk-grid 32 32 32 \
    --n-warmup 20 \
    --n-measure 200 \
    --output benchmarks/b1_interactive_latency/results/jrc_mus-liver_fly_organelles_s1_h100.json \
    --label "jrc_mus-liver fly_organelles_run07_432000 s1 H100"
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

Pre-filled configs (both datasets, blockwise output):

- [configs/jrc_hela-2_fly_organelles.yaml](b3_strong_scaling/configs/jrc_hela-2_fly_organelles.yaml)
- [configs/jrc_mus-liver_fly_organelles.yaml](b3_strong_scaling/configs/jrc_mus-liver_fly_organelles.yaml)

**Replace `<user>` in `output_path` and pick a fresh path before each sweep** — re-using a path means previously-written blocks get skipped and timings will be wrong.

### Run

From a host that can submit to LSF (run for each dataset):

```sh
python -m benchmarks.b3_strong_scaling.run \
    --config benchmarks/b3_strong_scaling/configs/jrc_hela-2_fly_organelles.yaml \
    --workers 1 4 16 64 128 \
    --output-dir benchmarks/b3_strong_scaling/results/jrc_hela-2_fly_organelles/

python -m benchmarks.b3_strong_scaling.run \
    --config benchmarks/b3_strong_scaling/configs/jrc_mus-liver_fly_organelles.yaml \
    --workers 1 4 16 64 128 \
    --output-dir benchmarks/b3_strong_scaling/results/jrc_mus-liver_fly_organelles/
```

The harness rewrites the YAML's `workers` field per run, dispatches `cellmap_flow_blockwise`, captures wall time, and writes one JSON per N to the output directory.

### Notes

- N=128 will saturate the queue if other people are running. Coordinate with the lab.
- Use `--dry-run` to print the per-N command without executing.
- Set the output Zarr to a path on `/nrs/cellmap/<your-user>/...` to avoid stomping on production paths.

---

## B6 — Hand-rolled PyTorch baseline vs. cellmap-flow

This benchmark requires **the same model loadable both ways**: as a TorchScript file (for `run_baseline.py`) and as a cellmap-flow model (for `run_cellmapflow.py`). The cellmap HuggingFace models ship with a TorchScript artifact alongside the PyTorch one, so this is a one-time download:

```python
from huggingface_hub import hf_hub_download
ts_path = hf_hub_download(
    repo_id="cellmap/fly_organelles_run07_432000",
    filename="model.ts",  # adjust filename to whatever the repo provides
)
print(ts_path)  # use this path in the B6 config
```

If `model.ts` isn't shipped, export it once:

```python
from cellmap_models.model_export.cellmap_model import CellmapModel
import torch
m = CellmapModel("cellmap/fly_organelles_run07_432000")
torch.jit.script(m.model).save("/path/to/cellmap_flow_paper/benchmarks/b6_baseline_comparison/fly_organelles_run07_432000.ts")
```

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

- [ ] Decide which sweep axes for B1 are essential for v1 of the paper (recommended: chunk size × GPU type, three of each = 9 cells).
- [ ] Pick a writable `output_path` in the B3 config (replace `<user>` placeholder); use a fresh path per sweep so cached blocks don't distort timing.
- [ ] For B6, run the one-time TorchScript export (snippet above) and point both runners at the resulting `.ts` file.
