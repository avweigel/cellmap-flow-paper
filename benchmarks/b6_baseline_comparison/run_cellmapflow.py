"""B6 cellmap-flow side: drives the same task through cellmap-flow's CLI.

Times two stages:
  1. Time-to-first-view: from server start to the first chunk response
     (measured once; it is a single-chunk server response, worker-independent).
  2. Time-to-completion: blockwise full-volume export, optionally swept over
     several worker counts so the comparison shows the point at which
     cellmap-flow's native multi-worker mode overtakes the single-process
     baseline (the completion-time crossover).

The first-view stage launches the same server entrypoint B1 uses --
`cellmap_flow_server <type> <model-flags> -d <data_path> -p <port>` -- and polls
the chunk URL until it answers. (Note: `cellmap_flow_yaml` is NOT a server; it
submits one bsub job per model and busy-loops, so it cannot be driven here.)

The completion stage runs `cellmap_flow_blockwise <yaml>` on the same model +
dataset as run_baseline.py, so it is apples-to-apples. When more than one worker
count is requested, each gets its own output container and its own progress
tmp_dir (wiped first) so every point does a clean full pass -- the same
discipline B3 uses -- and emits one result JSON tagged with dataset + workers.
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from urllib.error import URLError

import yaml

from benchmarks._common import capture_env, write_result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="shared B6 config")
    p.add_argument(
        "--output-dir",
        required=True,
        help="directory to write per-worker-count result JSON files into",
    )
    p.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=[1],
        help="blockwise worker counts to sweep (e.g. 1 4 16); the "
        "single-process baseline has no equivalent, so this axis is what "
        "exposes the completion-time crossover",
    )
    p.add_argument(
        "--server-cmd",
        default="cellmap_flow_server",
        help="server entrypoint (a click group; a model-type subcommand is appended)",
    )
    p.add_argument(
        "--blockwise-cmd",
        default="cellmap_flow_blockwise",
        help="blockwise entrypoint",
    )
    p.add_argument("--server-host", default="127.0.0.1")
    p.add_argument(
        "--server-port",
        type=int,
        default=0,
        help="preferred port; 0 (default) picks a free one",
    )
    return p.parse_args()


def free_port(host: str, preferred: int) -> int:
    """Return `preferred` if free, else an OS-assigned free port. Avoids
    collisions on shared GPU hosts."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            if preferred:
                s.bind((host, preferred))
                return preferred
            raise OSError
        except OSError:
            s.bind((host, 0))
            return s.getsockname()[1]


def build_server_argv(cfg: dict, server_cmd: str, host: str, port: int) -> list[str]:
    """cellmap_flow_server huggingface --repo <repo> -d <data_path> -p <port>.

    The model subcommand/flags come from the shared config so the cellmap-flow
    side stays in lockstep with the dataset/model used by the baseline."""
    model_type = cfg.get("cf_model_type", "huggingface")
    argv = [server_cmd, model_type]
    for flag, key in (("--repo", "hf_repo"),):
        if cfg.get(key):
            argv += [flag, str(cfg[key])]
    # extra free-form server flags, e.g. {"--name": "fly"}; optional.
    for flag, value in (cfg.get("cf_server_extra_flags") or {}).items():
        argv += [str(flag), str(value)]
    argv += ["-d", str(cfg["data_path"]), "-p", str(port)]
    return argv


def time_to_first_view(
    server_argv: list[str], chunk_url: str, boot_timeout: float
) -> tuple[float, subprocess.Popen]:
    """Boot the server and time how long until the first chunk request returns."""
    t0 = time.perf_counter()
    proc = subprocess.Popen(server_argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    deadline = t0 + boot_timeout
    while True:
        if proc.poll() is not None:
            err = proc.stderr.read().decode() if proc.stderr else ""
            raise RuntimeError(f"server exited early: rc={proc.returncode}\n{err}")
        if time.perf_counter() > deadline:
            proc.terminate()
            raise TimeoutError(
                f"server did not answer {chunk_url} within {boot_timeout:.0f}s"
            )
        try:
            with urllib.request.urlopen(chunk_url, timeout=5.0) as resp:
                resp.read()
            break
        except (URLError, ConnectionResetError):
            time.sleep(0.25)
    return time.perf_counter() - t0, proc


def per_n_output_path(output_path: str, n_workers: int) -> str:
    """Insert an _n<NNNN> tag before `.zarr` so each worker-count writes to its
    own container. cellmap_flow_blockwise derives the container by splitting on
    `.zarr`, so re-using one path across a sweep risks skipped/overwritten
    blocks and distorted timings; a fresh container per N avoids that."""
    if ".zarr" not in output_path:
        raise ValueError(f"output_path must contain '.zarr': {output_path!r}")
    container, _, rest = output_path.partition(".zarr")
    return f"{container}_n{n_workers:04d}.zarr{rest}"


def make_per_run_yaml(base_path: Path, n_workers: int, config_dir: Path, ds_tag: str) -> tuple[Path, Path]:
    """Write a per-N blockwise config and return (yaml_path, progress_tmp_dir).

    Mirrors B3: each worker-count gets its own output container and its own
    progress tmp_dir (wiped before the run in main) so every point does a clean
    full pass. Re-using one tmp_dir would make runs after the first skip blocks
    already marked done and report a fake speedup."""
    cfg = yaml.safe_load(base_path.read_text())
    if "tmp_dir" not in cfg:
        raise ValueError(
            "blockwise config must set tmp_dir: the installed cellmap-flow "
            "blockwise requires it (block-progress tracking is mandatory)."
        )
    cfg["workers"] = n_workers
    cfg["output_path"] = per_n_output_path(cfg["output_path"], n_workers)
    per_n_tmp = f"{str(cfg['tmp_dir']).rstrip('/')}/n{n_workers:04d}"
    cfg["tmp_dir"] = per_n_tmp
    out = config_dir / f"cf_blockwise_{ds_tag}_n{n_workers:04d}.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out, Path(per_n_tmp)


def time_full_volume(blockwise_cmd: str, blockwise_yaml: Path) -> tuple[float, int]:
    t0 = time.perf_counter()
    proc = subprocess.run(
        [blockwise_cmd, str(blockwise_yaml)], capture_output=True, text=True
    )
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        print(proc.stdout, file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
    return elapsed, proc.returncode


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    dataset = cfg.get("dataset")
    ds_tag = dataset or "run"  # filename tag so multiple datasets don't collide
    blockwise_yaml = Path(cfg["cellmap_flow_blockwise_yaml"])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_dir = out_dir / "_configs"
    config_dir.mkdir(exist_ok=True)

    ng_dataset = cfg.get("server_dataset", "data")
    scale = cfg.get("server_scale", 0)
    first_chunk = cfg.get("server_first_chunk", "0.0.0")
    boot_timeout = float(cfg.get("server_boot_timeout_s", 600))

    # --- Stage 1: time-to-first-view (once; worker-independent) ---
    port = free_port(args.server_host, args.server_port)
    server_argv = build_server_argv(cfg, args.server_cmd, args.server_host, port)
    chunk_url = f"http://{args.server_host}:{port}/{ng_dataset}/s{scale}/{first_chunk}"
    print("== B6 cellmap-flow: time-to-first-view ==", file=sys.stderr)
    print(f"  $ {shlex.join(server_argv)}", file=sys.stderr)
    print(f"  polling {chunk_url}", file=sys.stderr)
    ttfv, server_proc = time_to_first_view(server_argv, chunk_url, boot_timeout)
    server_proc.terminate()
    try:
        server_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server_proc.kill()
    print(f"  ttfv = {ttfv:.2f}s", file=sys.stderr)

    # --- Stage 2: time-to-completion, swept over worker counts ---
    for i, n in enumerate(args.workers):
        per_run_yaml, progress_tmp = make_per_run_yaml(blockwise_yaml, n, config_dir, ds_tag)
        print(f"\n== B6 cellmap-flow: time-to-completion (N={n}) ==", file=sys.stderr)
        print(f"  $ {args.blockwise_cmd} {per_run_yaml}", file=sys.stderr)
        if progress_tmp.exists():
            shutil.rmtree(progress_tmp, ignore_errors=True)
        full_wall, rc = time_full_volume(args.blockwise_cmd, per_run_yaml)
        print(f"  N={n}: full={full_wall:.1f}s rc={rc}", file=sys.stderr)

        payload = {
            "benchmark": "b6_baseline_comparison",
            "variant": "cellmapflow",
            "dataset": dataset,
            "workers": n,
            # first-view is worker-independent; record it only against the first
            # sweep point so it is not double-counted across rows.
            "time_to_first_view_s": ttfv if i == 0 else None,
            "time_to_completion_s": full_wall,
            "blockwise_return_code": rc,
            "server_argv": server_argv,
            "chunk_url": chunk_url,
            "config": str(args.config),
            "blockwise_config_used": str(per_run_yaml),
            "lines_of_code_total": _self_loc(),
            "env": capture_env(),
        }
        write_result(out_dir / f"cellmapflow_{ds_tag}_n{n:04d}.json", payload)

    return 0


def _self_loc() -> int:
    src = Path(__file__).read_text().splitlines()
    return sum(1 for line in src if line.strip() and not line.strip().startswith("#"))


if __name__ == "__main__":
    sys.exit(main())
