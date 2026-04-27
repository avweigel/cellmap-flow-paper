# cellmap-flow paper benchmarks

Benchmark suite for the cellmap-flow software paper. This directory lives **in the paper repository**, not in `janelia-cellmap/cellmap-flow`. The benchmarks exercise an installed `cellmap-flow` package via its public CLI (`cellmap_flow_yaml`, `cellmap_flow_blockwise`); they do not import internal cellmap-flow modules. Pin a specific cellmap-flow version in your environment so the numbers reported in the paper map to a known release.

Each `bN_*` directory is a self-contained experiment with its own `run.py`, configuration files, and result JSONs. Aggregate reporting is in `regenerate_paper_tables.py`.

## Layout

```
benchmarks/
├── _common/                  # shared timing / reporting / env utilities
├── b1_interactive_latency/   # HTTP chunk-request latency
├── b3_strong_scaling/        # cluster strong scaling (N workers, fixed volume)
├── b6_baseline_comparison/   # cellmap-flow vs hand-rolled PyTorch
└── regenerate_paper_tables.py
```

The full benchmark plan (B1–B10) is described in the paper's Benchmarks section. This directory currently implements **B1, B3, B6** — the recommended starting subset. B2/B4/B5/B7–B10 protocols are documented in the paper but not yet scripted; their stubs land here as they are developed.

## Running

Run from the paper repository root so that `benchmarks` is importable as a package:

```sh
cd /path/to/cellmap-flow-paper
python -m benchmarks.b1_interactive_latency.run \
    --server http://localhost:8000 \
    --dataset jrc_mus-salivary-1.zarr \
    --chunk-grid 32 32 32 \
    --output benchmarks/b1_interactive_latency/results/h100_unet_medium.json
```

Results are written as JSON with both raw timings and the env metadata (git SHA, hostname, GPU model, package versions) needed to reproduce them. The `results/` directories under each benchmark are gitignored.

## Aggregating for the paper

```sh
python benchmarks/regenerate_paper_tables.py \
    --results-dir benchmarks/ \
    --out figures/benchmark_tables.tex
```

The output `figures/benchmark_tables.tex` is included from the paper's benchmarks section via `\input{figures/benchmark_tables}`.

## What we expect from each benchmark

| ID | Question | Primary metric |
|----|----------|----------------|
| B1 | Interactive chunk latency vs.\ chunk size and model | ms/chunk (median, p99) |
| B3 | Strong scaling on cluster | speedup vs.\ ideal |
| B6 | Comparison vs.\ hand-rolled PyTorch + offline export | end-to-end wall time |

Add new benchmarks by copying an existing `bN_*/` directory and editing.
