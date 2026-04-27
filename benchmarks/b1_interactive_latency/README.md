# B1: Interactive chunk-request latency

**Question.** How long does cellmap-flow take to respond to a Neuroglancer-style chunk request, as a function of chunk size and model size?

**What it does.** Boots a `cellmap_flow_yaml` server (or attaches to a running one), then sends N chunk-request HTTP GETs at random spatial offsets within the dataset's ROI. Times each request end-to-end and reports median / p95 / p99 latency.

## Inputs

- A pipeline YAML — same format as `cellmap_flow_yaml` accepts.
- A target dataset path declared inside the YAML (must already exist on disk or be reachable).
- Number of warm-up requests (default 20) and measured requests (default 200).

## Output

A single JSON file with raw timings, the timing summary, and the full env capture (git SHA, GPU model, package versions).

## Usage

```sh
# 1. In one terminal, start a server
cellmap_flow_yaml configs/h100_unet_medium.yaml

# 2. In another terminal, run the benchmark client
python -m benchmarks.b1_interactive_latency.run \
    --server http://localhost:8000 \
    --dataset jrc_mus-salivary-1.zarr \
    --scale 0 \
    --chunk-shape 128 128 128 \
    --n-warmup 20 \
    --n-measure 200 \
    --output benchmarks/b1_interactive_latency/results/h100_unet_medium_128.json
```

## Sweep recipe

To produce the regime map promised in the paper, repeat with:

- chunk shapes: `64³`, `128³`, `256³`
- model sizes: small (~1M params), medium (~10M), large (~50M)
- GPU types: H100, A100, V100, CPU

Each run produces one JSON; the aggregator script picks them all up.
