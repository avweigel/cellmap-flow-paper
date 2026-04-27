# B3: Strong scaling on cluster

**Question.** For a fixed-size volume, how does total wall time scale with the number of GPU workers?

**What it does.** Drives `cellmap_flow_blockwise` with worker counts `N ∈ {1, 4, 16, 64, 128}` over the same input volume and same model, captures wall time per run, and reports speedup vs. ideal.

## Inputs

- A blockwise YAML with a fixed sub-volume defined (e.g., a centered ROI of ~4096³ voxels — large enough that scheduling overhead is amortized but small enough that the run completes in tens of minutes at large worker counts).
- A worker-count list (default `1 4 16 64 128`).

## Output

One JSON per worker-count point under `results/`, each holding:
- `wall_time_s` for the blockwise run
- the `cellmap_flow_blockwise` invocation it used
- env capture (Daisy version, LSF queue, etc.)

A driver script aggregates these into a `b3_strong_scaling.json` summary that the paper-table generator picks up.

## Usage

```sh
# Configure the YAML to point at the target dataset, model, and a fixed ROI.
$EDITOR configs/jrc_mus-salivary-1.yaml

# Drive the sweep. This must run on a host that can submit to LSF.
python -m benchmarks.b3_strong_scaling.run \
    --config configs/jrc_mus-salivary-1.yaml \
    --workers 1 4 16 64 128 \
    --output-dir results/jrc_mus-salivary-1/
```

## Notes

- The YAML is rewritten in-memory per run with the `workers` field set to N; everything else held fixed.
- Daisy retries are surfaced in the per-run JSON; high retry rates indicate file-system contention or bad block sizing.
- Plot speedup vs. N to see the scaling curve; fit Amdahl's law for a serial-fraction estimate.
