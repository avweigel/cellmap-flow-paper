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

Both use `cellmap/fly_organelles_run07_700000` so the model is held fixed; latency differences across the two datasets reflect I/O and chunk-content variance, not model variance. Both datasets were published in Heinrich et al., Nature 2021.

### Run

Repeat for each dataset. In **terminal A** on a GPU node (e.g., `bsub -Is -q gpu_h100 -gpu "num=1" /bin/bash`).

Launch the chunk-serving server **directly** with `cellmap_flow_server` — NOT
`cellmap_flow_yaml`. `cellmap_flow_yaml` submits one bsub job per model and then
busy-loops; it does not itself serve chunks on the local node. `cellmap_flow_server`
is a click group whose subcommand is the model type (`huggingface`, `fly`,
`dacapo`, …), so the model is given on the command line rather than via the YAML:

```sh
# cell culture
cellmap_flow_server huggingface \
    --repo cellmap/fly_organelles_run07_700000 \
    -d s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.zarr/recon-1/em/fibsem-uint8/s1 \
    -p 0   # 0 = auto-pick a free port
# ... or, for the tissue dataset, swap the -d path to jrc_mus-liver/.../s1
# note the host:port the server prints, e.g. http://127.0.0.1:22541
```

The pre-filled `configs/*_fly_organelles.yaml` capture the same data_path + model
for the record and for the `cellmap_flow_yaml` cluster-dispatch path; the
interactive B1 measurement above talks to a `cellmap_flow_server` on the node.

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
    --label "jrc_hela-2 fly_organelles_run07_700000 s1 H100"

# tissue
python -m benchmarks.b1_interactive_latency.run \
    --server http://<host>:<port> \
    --dataset jrc_mus-liver.zarr \
    --scale 1 \
    --chunk-grid 32 32 32 \
    --n-warmup 20 \
    --n-measure 200 \
    --output benchmarks/b1_interactive_latency/results/jrc_mus-liver_fly_organelles_s1_h100.json \
    --label "jrc_mus-liver fly_organelles_run07_700000 s1 H100"
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

The harness rewrites the YAML's `workers` field per run (and gives each N its own `_n<NNNN>.zarr` container), dispatches `cellmap_flow_blockwise`, captures wall time, and writes one JSON per N to the output directory. The orchestrator stays alive and `bsub`-submits N GPU workers via daisy, so run it on the submit host (or under `nohup`/`bsub` on a CPU queue), not on a login node that may reap long processes.

**Calibrate the bounded volume first.** Both configs ship with a `bounding_boxes` cube (8192³ nm). Run a single `--workers 1` job and check the wall time and the daisy block count (in `daisy_logs/`): you want N=1 to finish in tens of minutes and enough blocks that N=128 still keeps every worker busy. Scale the cube edge up/down and re-run before committing to the full `1 4 16 64 128` sweep. Keep the box identical across the two datasets.

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
    repo_id="cellmap/fly_organelles_run07_700000",
    filename="model.ts",  # adjust filename to whatever the repo provides
)
print(ts_path)  # use this path in the B6 config
```

If `model.ts` isn't shipped, export it once:

```python
from cellmap_models.model_export.cellmap_model import CellmapModel
import torch
m = CellmapModel("cellmap/fly_organelles_run07_700000")
torch.jit.script(m.model).save("/path/to/cellmap_flow_paper/benchmarks/b6_baseline_comparison/fly_organelles_run07_700000.ts")
```

Once you have a TorchScript file, set `model_checkpoint` to its path in **both** shared configs (the same export drives both datasets):

1. **Two datasets, one geometry.** B6 runs on a cell-culture and a tissue volume, exactly like B1/B3:
   - [configs/template.yaml](b6_baseline_comparison/configs/template.yaml) — jrc\_hela-2; blockwise [configs/cf_blockwise_jrc_hela-2.yaml](b6_baseline_comparison/configs/cf_blockwise_jrc_hela-2.yaml).
   - [configs/template_jrc_mus-liver.yaml](b6_baseline_comparison/configs/template_jrc_mus-liver.yaml) — jrc\_mus-liver; blockwise [configs/cf_blockwise_jrc_mus-liver.yaml](b6_baseline_comparison/configs/cf_blockwise_jrc_mus-liver.yaml).
   - Each config carries a `dataset:` label that tags its result JSON, so the aggregator groups the two datasets into separate table blocks automatically.
   - **Calibrate the baseline chunking:** `output_block_shape + 2*context` must be a valid input size for the model (the fly U-Net wants a 178³ input at 8 nm ⇒ 56³ output, context 61/side). If the hand-rolled run errors on a shape mismatch, this is why. Both configs are pre-filled with these.
   - **Keep the regions aligned:** the baseline `full_volume_roi_shape` (s1 voxels) and the blockwise `bounding_boxes` (nm) must cover the same sub-volume. Defaults already match (512 voxels × 8 nm = 4096 nm) and are identical across both datasets so their numbers are comparable.
   - The cellmap-flow first-view server is launched by `run_cellmapflow.py` itself as `cellmap_flow_server huggingface --repo <hf_repo> -d <data_path> -p <port>` (the same entrypoint as B1); no separate server YAML is needed.
2. **Run both datasets.** The baseline writes two distinct filenames per dataset — the aggregator distinguishes first-view from full-volume by the `first_view_only` flag, so don't collapse them. `run_cellmapflow.py` measures time-to-first-view once and then sweeps the blockwise completion over `--workers` (one result JSON per count, dataset-tagged); the **N>1 points are what show the completion-time crossover** against the single-process baseline:
  ```sh
  for CFG in template template_jrc_mus-liver; do
    C=benchmarks/b6_baseline_comparison/configs/$CFG.yaml
    R=benchmarks/b6_baseline_comparison/results

    # baseline: full-volume completion, then first-view-only
    python -m benchmarks.b6_baseline_comparison.run_baseline \
        --config $C --output $R/baseline_${CFG}_full.json
    python -m benchmarks.b6_baseline_comparison.run_baseline \
        --config $C --first-view-only --output $R/baseline_${CFG}_first_view.json

    # cellmap-flow: first-view once + blockwise completion swept over N workers
    python -m benchmarks.b6_baseline_comparison.run_cellmapflow \
        --config $C --output-dir $R --workers 1 4 16
  done
  ```
  Delete the pre-existing single-dataset result files (`baseline_full.json`, `baseline_first_view.json`, `cellmapflow.json`) once the dataset-tagged runs exist — the aggregator reads *every* `results/*.json`, so leaving the old untagged pair in place would double-count jrc\_hela-2.

---

## After all runs: regenerate paper tables

```sh
# run as a module from the repo root so `benchmarks` is importable
python -m benchmarks.regenerate_paper_tables \
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
