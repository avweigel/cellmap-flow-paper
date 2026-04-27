"""B6 cellmap-flow side: drives the same task through cellmap_flow's CLI.

Times two stages:
  1. Time-to-first-view: from server start to the first chunk response.
  2. Time-to-completion: blockwise full-volume export.

Both use the cellmap-flow YAML referenced by the shared config, ensuring an
apples-to-apples comparison with run_baseline.py.
"""

from __future__ import annotations

import argparse
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
        default="cellmap_flow_yaml",
        help="server entrypoint",
    )
    p.add_argument(
        "--blockwise-cmd",
        default="cellmap_flow_blockwise",
        help="blockwise entrypoint",
    )
    p.add_argument("--server-host", default="127.0.0.1")
    p.add_argument("--server-port", type=int, default=8765)
    return p.parse_args()


def free_port(host: str, port: int) -> int:
    """Return port if free, else find another. Avoids collisions on shared hosts."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return port
        except OSError:
            s.bind((host, 0))
            return s.getsockname()[1]


def time_to_first_view(
    server_cmd: str, server_yaml: Path, host: str, port: int, first_chunk_url_path: str
) -> tuple[float, subprocess.Popen]:
    """Boot the server and time how long until the first chunk request returns."""
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        [server_cmd, str(server_yaml), "--host", host, "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    chunk_url = f"http://{host}:{port}/{first_chunk_url_path}"
    while True:
        if proc.poll() is not None:
            raise RuntimeError(
                f"server exited early: rc={proc.returncode} stderr={proc.stderr.read().decode() if proc.stderr else ''}"
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
    server_yaml = Path(cfg["cellmap_flow_server_yaml"])
    blockwise_yaml = Path(cfg["cellmap_flow_blockwise_yaml"])
    first_chunk_url_path = cfg["first_chunk_url_path"]

    port = free_port(args.server_host, args.server_port)

    print("== B6 cellmap-flow: time-to-first-view ==", file=sys.stderr)
    ttfv, server_proc = time_to_first_view(
        args.server_cmd, server_yaml, args.server_host, port, first_chunk_url_path
    )
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
