"""Smoke test for B1: validates the harness end-to-end against a stub server.

Does NOT require cellmap-flow to be installed. Spins up a tiny HTTP stub that
mimics cellmap-flow's chunk-URL contract with a fixed artificial delay, runs
the B1 client against it, and checks that the JSON output is well-formed and
that the measured median is in the right ballpark.

Run from the paper repo root:

    python -m benchmarks.b1_interactive_latency.smoke_test
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from benchmarks._common.stub_server import StubServer

EXPECTED_DELAY_MS = 25.0
TOLERANCE_MS = 25.0  # generous: thread scheduling + HTTP overhead


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    with StubServer(payload_bytes=8192, fixed_delay_ms=EXPECTED_DELAY_MS) as srv:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fout:
            out_path = Path(fout.name)
        cmd = [
            sys.executable,
            "-m",
            "benchmarks.b1_interactive_latency.run",
            "--server",
            srv.url,
            "--dataset",
            "smoke.zarr",
            "--scale",
            "0",
            "--chunk-grid",
            "8",
            "8",
            "8",
            "--n-warmup",
            "5",
            "--n-measure",
            "30",
            "--output",
            str(out_path),
            "--label",
            "smoke",
        ]
        proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
        if proc.returncode != 0:
            print("client failed:", proc.stderr, file=sys.stderr)
            return 1

    payload = json.loads(out_path.read_text())
    summary = payload["summary"]
    median = summary["median_ms"]
    fails = payload["failures"]

    print(f"smoke median={median:.1f}ms expected~={EXPECTED_DELAY_MS}ms p99={summary['p99_ms']:.1f}ms failures={fails}")

    ok = (
        summary["count"] == 30
        and abs(median - EXPECTED_DELAY_MS) < TOLERANCE_MS
        and fails == 0
        and payload["bytes_per_chunk"]["min"] == 8192
    )
    if not ok:
        print("SMOKE FAILED: harness output looks wrong", file=sys.stderr)
        return 2
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
