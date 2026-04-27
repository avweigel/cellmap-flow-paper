# B6: Comparison vs hand-rolled PyTorch + offline export

**Question.** What does cellmap-flow actually save a user, compared with the workflow it replaces?

**What it does.** Defines one task ("produce viewable predictions for organelle X across volume Y") and runs it two ways:

1. **Baseline (`run_baseline.py`)**: a hand-written PyTorch chunked-inference loop that writes output to Zarr and registers it with Neuroglancer's precomputed adapter.
2. **cellmap-flow (`run_cellmapflow.py`)**: the same model and dataset configured via a YAML, then run in server mode (for the first interactive view) and blockwise mode (for full-volume export).

Two metrics matter:
- **Time-to-first-view**: from cold start to the user seeing a predicted chunk. cellmap-flow should win here by skipping the offline-export step.
- **Time-to-completion**: full-volume wall time. Should be roughly comparable since both exercise chunked inference.

A secondary metric is **lines of code**, captured by counting non-blank, non-comment lines in each entrypoint.

## Inputs

- A single model checkpoint, single input dataset, and (for full-volume mode) a target output path.
- A test ROI defining what "the user views first" — a small bounding box that approximates an initial Neuroglancer view (e.g., 4096³ around the volume center at scale s2).

## Output

```
results/
├── baseline_<run_id>.json
└── cellmapflow_<run_id>.json
```

Each JSON contains both metrics, the env capture, and a path to the produced Zarr (so a separate validation script can confirm both pipelines produced equivalent output).

## Running

```sh
python -m benchmarks.b6_baseline_comparison.run_baseline    --config configs/jrc_mus-salivary-1_mito.yaml
python -m benchmarks.b6_baseline_comparison.run_cellmapflow --config configs/jrc_mus-salivary-1_mito.yaml
```

Both share a config so the comparison is apples-to-apples.

## Notes

- For a fair comparison, both pipelines must use the same chunk size, padding, and FP16 setting.
- The baseline deliberately uses no cellmap-flow modules; it is a clean reimplementation of "what a user would write themselves."
- The first-view experiment is intentionally generous to the baseline: we time only the smallest sub-volume the baseline can produce, not a full volume the user would not actually wait for.
