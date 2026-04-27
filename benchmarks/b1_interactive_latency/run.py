"""B1: interactive chunk-request latency.

Sends N chunk-request HTTP GETs to a running cellmap_flow server and records
per-request wall time. Assumes the server has already been started (e.g. via
`cellmap_flow_yaml <config>` in another shell). The server-side YAML defines
the dataset and model; this script only chooses where in the volume to
sample.
"""

from __future__ import annotations

import argparse
import random
import sys
import urllib.request
from urllib.error import HTTPError, URLError

from benchmarks._common import Timer, capture_env, summarize, write_result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server", required=True, help="server base URL, e.g. http://localhost:8000")
    p.add_argument("--dataset", required=True, help="dataset path component in the URL")
    p.add_argument("--scale", type=int, default=0, help="multiscale level (s0, s1, ...)")
    p.add_argument(
        "--chunk-grid",
        type=int,
        nargs=3,
        required=True,
        metavar=("NZ", "NY", "NX"),
        help="size of the chunk grid (number of chunks per axis); used to pick random chunk indices",
    )
    p.add_argument("--n-warmup", type=int, default=20)
    p.add_argument("--n-measure", type=int, default=200)
    p.add_argument("--seed", type=int, default=0xC0FFEE)
    p.add_argument("--timeout", type=float, default=120.0, help="per-request timeout in seconds")
    p.add_argument("--output", required=True, help="path for output JSON")
    p.add_argument(
        "--label",
        default="",
        help="free-form label for this run (e.g. 'h100_unet_medium_128')",
    )
    return p.parse_args()


def request_chunk(server: str, dataset: str, scale: int, z: int, y: int, x: int, timeout: float) -> int:
    url = f"{server.rstrip('/')}/{dataset}/s{scale}/{z}.{y}.{x}"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return len(resp.read())


def random_chunk(rng: random.Random, grid: tuple[int, int, int]) -> tuple[int, int, int]:
    return (rng.randrange(grid[0]), rng.randrange(grid[1]), rng.randrange(grid[2]))


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    grid = tuple(args.chunk_grid)

    print(f"warming up: {args.n_warmup} requests", file=sys.stderr)
    failures = 0
    for _ in range(args.n_warmup):
        z, y, x = random_chunk(rng, grid)
        try:
            request_chunk(args.server, args.dataset, args.scale, z, y, x, args.timeout)
        except (HTTPError, URLError) as exc:
            failures += 1
            print(f"warmup failure {z}.{y}.{x}: {exc}", file=sys.stderr)

    print(f"measuring: {args.n_measure} requests", file=sys.stderr)
    timer = Timer(label=args.label)
    bytes_returned = []
    for i in range(args.n_measure):
        z, y, x = random_chunk(rng, grid)
        try:
            with timer.measure():
                n_bytes = request_chunk(args.server, args.dataset, args.scale, z, y, x, args.timeout)
            bytes_returned.append(n_bytes)
        except (HTTPError, URLError) as exc:
            failures += 1
            print(f"measure failure {z}.{y}.{x}: {exc}", file=sys.stderr)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{args.n_measure} done", file=sys.stderr)

    payload = {
        "benchmark": "b1_interactive_latency",
        "label": args.label,
        "args": vars(args),
        "summary": summarize(timer.samples_ms),
        "samples_ms": timer.samples_ms,
        "bytes_per_chunk": {
            "min": min(bytes_returned) if bytes_returned else 0,
            "max": max(bytes_returned) if bytes_returned else 0,
        },
        "failures": failures,
        "env": capture_env(),
    }
    write_result(args.output, payload)

    s = payload["summary"]
    print(
        f"\n{args.label or 'B1'}: median={s['median_ms']:.1f}ms "
        f"p95={s['p95_ms']:.1f}ms p99={s['p99_ms']:.1f}ms "
        f"failures={failures}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
