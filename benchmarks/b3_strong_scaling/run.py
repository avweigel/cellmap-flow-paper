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
        "--dry-run",
        action="store_true",
        help="print the per-N command without executing",
    )
    return p.parse_args()


def make_per_run_yaml(base_path: Path, n_workers: int, tmp_dir: Path) -> Path:
    with base_path.open() as f:
        cfg = yaml.safe_load(f)
    cfg["workers"] = n_workers
    out = tmp_dir / f"config_n{n_workers}.yaml"
    with out.open("w") as f:
        yaml.safe_dump(cfg, f)
    return out


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

    if shutil.which(args.blockwise_cmd) is None and not args.dry_run:
        print(f"warning: {args.blockwise_cmd} not on PATH", file=sys.stderr)

    for n in args.workers:
        per_run_yaml = make_per_run_yaml(base, n, tmp)
        cmd_str = f"{args.blockwise_cmd} {per_run_yaml}"
        print(f"\n=== N={n} workers ===\n{cmd_str}", file=sys.stderr)
        if args.dry_run:
            continue
        rc, wall = run_blockwise(args.blockwise_cmd, per_run_yaml)
        payload = {
            "benchmark": "b3_strong_scaling",
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
