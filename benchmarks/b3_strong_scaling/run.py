"""B3: cluster strong scaling.

For a fixed sub-volume and model, runs `cellmap_flow_blockwise` with several
worker counts and records wall time. Each worker-count point is one JSON file;
aggregation happens in `regenerate_paper_tables.py`.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from benchmarks._common import capture_env, write_result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="base blockwise YAML")
    p.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=[1, 4, 16, 64, 128],
        help="worker counts to sweep",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="directory to write per-N result JSON files into",
    )
    p.add_argument(
        "--blockwise-cmd",
        default="cellmap_flow_blockwise",
        help="entrypoint to invoke; override if cellmap-flow is installed under a different name",
    )
    p.add_argument(
        "--label",
        default="",
        help="dataset label for this sweep (defaults to the config file stem); "
        "the aggregator groups scaling rows by this so multiple datasets render "
        "as separate curves",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the per-N command without executing",
    )
    return p.parse_args()


def per_n_output_path(output_path: str, n_workers: int) -> str:
    """Insert an _n<NNNN> tag before `.zarr` so each worker-count writes to its
    own container. cellmap_flow_blockwise derives the container by splitting on
    `.zarr`, so re-using one path across the sweep risks skipped/overwritten
    blocks and distorted timings; a fresh container per N avoids that."""
    if ".zarr" not in output_path:
        raise ValueError(f"output_path must contain '.zarr': {output_path!r}")
    container, _, rest = output_path.partition(".zarr")
    return f"{container}_n{n_workers:04d}.zarr{rest}"


def make_per_run_yaml(base_path: Path, n_workers: int, config_dir: Path) -> tuple[Path, Path]:
    """Write a per-N config and return (yaml_path, progress_tmp_dir).

    Each worker-count gets its OWN output container and its OWN progress
    `tmp_dir`. The installed cellmap-flow blockwise always tracks block progress
    via marker files under `<tmp_dir>/tmp_flow_daisy_progress_<task_name>` and
    skips any block already marked done -- so re-using one tmp_dir across the
    sweep would make every run after N=1 skip blocks and report a fake speedup.
    The per-N tmp_dir is wiped before each run (in main) for a clean full pass."""
    with base_path.open() as f:
        cfg = yaml.safe_load(f)
    if "tmp_dir" not in cfg:
        raise ValueError(
            "config must set tmp_dir: the installed cellmap-flow blockwise "
            "requires it (block-progress tracking is mandatory)."
        )
    cfg["workers"] = n_workers
    cfg["output_path"] = per_n_output_path(cfg["output_path"], n_workers)
    per_n_tmp = f"{str(cfg['tmp_dir']).rstrip('/')}/n{n_workers:04d}"
    cfg["tmp_dir"] = per_n_tmp
    out = config_dir / f"config_n{n_workers}.yaml"
    with out.open("w") as f:
        yaml.safe_dump(cfg, f)
    return out, Path(per_n_tmp)


def run_blockwise(cmd: str, yaml_path: Path) -> tuple[int, float]:
    t0 = time.perf_counter()
    proc = subprocess.run([cmd, str(yaml_path)], capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        print(proc.stdout, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
    return proc.returncode, elapsed


def main() -> int:
    args = parse_args()
    base = Path(args.config).resolve()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "_configs"
    tmp.mkdir(exist_ok=True)

    label = args.label or base.stem

    if shutil.which(args.blockwise_cmd) is None and not args.dry_run:
        print(f"warning: {args.blockwise_cmd} not on PATH", file=sys.stderr)

    for n in args.workers:
        per_run_yaml, progress_tmp = make_per_run_yaml(base, n, tmp)
        cmd_str = f"{args.blockwise_cmd} {per_run_yaml}"
        print(f"\n=== N={n} workers ===\n{cmd_str}", file=sys.stderr)
        if args.dry_run:
            continue
        # Wipe any stale block-progress markers so this is a clean full pass.
        if progress_tmp.exists():
            shutil.rmtree(progress_tmp, ignore_errors=True)
        rc, wall = run_blockwise(args.blockwise_cmd, per_run_yaml)
        payload = {
            "benchmark": "b3_strong_scaling",
            "label": label,
            "n_workers": n,
            "wall_time_s": wall,
            "return_code": rc,
            "config_used": str(per_run_yaml),
            "base_config": str(base),
            "env": capture_env(),
        }
        write_result(out_dir / f"n{n:04d}.json", payload)
        print(f"  wall={wall:.1f}s rc={rc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
