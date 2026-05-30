"""B6 cellmap-flow side: drives the same task through cellmap-flow's CLI.

Times two stages:
  1. Time-to-first-view: from server start to the first chunk response.
  2. Time-to-completion: blockwise full-volume export.

The first-view stage launches the same server entrypoint B1 uses --
`cellmap_flow_server <type> <model-flags> -d <data_path> -p <port>` -- and polls
the chunk URL until it answers. (Note: `cellmap_flow_yaml` is NOT a server; it
submits one bsub job per model and busy-loops, so it cannot be driven here.)

The completion stage runs `cellmap_flow_blockwise <yaml>` on the same model +
dataset, so it is apples-to-apples with run_baseline.py.
"""

from __future__ import annotations

import argparse
import shlex
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
    p.add_argument("--output", required=True, help="path for the result JSON")
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

    blockwise_yaml = Path(cfg["cellmap_flow_blockwise_yaml"])
    dataset = cfg.get("server_dataset", "data")
    scale = cfg.get("server_scale", 0)
    first_chunk = cfg.get("server_first_chunk", "0.0.0")
    boot_timeout = float(cfg.get("server_boot_timeout_s", 600))

    port = free_port(args.server_host, args.server_port)
    server_argv = build_server_argv(cfg, args.server_cmd, args.server_host, port)
    chunk_url = f"http://{args.server_host}:{port}/{dataset}/s{scale}/{first_chunk}"

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

    print("== B6 cellmap-flow: time-to-completion (blockwise) ==", file=sys.stderr)
    full_wall, rc = time_full_volume(args.blockwise_cmd, blockwise_yaml)
    print(f"  full = {full_wall:.1f}s rc={rc}", file=sys.stderr)

    payload = {
        "benchmark": "b6_baseline_comparison",
        "variant": "cellmapflow",
        "time_to_first_view_s": ttfv,
        "time_to_completion_s": full_wall,
        "blockwise_return_code": rc,
        "server_argv": server_argv,
        "chunk_url": chunk_url,
        "config": str(args.config),
        "lines_of_code_total": _self_loc(),
        "env": capture_env(),
    }
    write_result(args.output, payload)
    return 0


def _self_loc() -> int:
    src = Path(__file__).read_text().splitlines()
    return sum(1 for line in src if line.strip() and not line.strip().startswith("#"))


if __name__ == "__main__":
    sys.exit(main())
