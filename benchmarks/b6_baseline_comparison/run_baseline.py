"""B6 baseline: hand-rolled chunked PyTorch inference with offline Zarr export.

The point of this script is to look like what a user would write themselves
without cellmap-flow. It is intentionally not factored, not parameterised
beyond a small CLI, and uses no cellmap_flow imports. This is the shape of the
workflow the paper claims to replace; its run time is the comparison baseline.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
import zarr

from benchmarks._common import capture_env, write_result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="shared B6 config")
    p.add_argument("--output", required=True, help="path for the result JSON")
    p.add_argument(
        "--first-view-only",
        action="store_true",
        help="only produce the small first-view ROI, then stop (used to time time-to-first-view)",
    )
    return p.parse_args()


def load_model(path: Path, device: torch.device) -> torch.nn.Module:
    model = torch.jit.load(str(path), map_location=device)
    model.eval()
    return model


def chunked_inference(
    model: torch.nn.Module,
    input_arr: zarr.Array,
    output_arr: zarr.Array,
    roi_offset: tuple[int, int, int],
    roi_shape: tuple[int, int, int],
    chunk_shape: tuple[int, int, int],
    context: tuple[int, int, int],
    device: torch.device,
) -> int:
    n_chunks = 0
    for z in range(roi_offset[0], roi_offset[0] + roi_shape[0], chunk_shape[0]):
        for y in range(roi_offset[1], roi_offset[1] + roi_shape[1], chunk_shape[1]):
            for x in range(roi_offset[2], roi_offset[2] + roi_shape[2], chunk_shape[2]):
                # context-padded read
                in_slice = (
                    slice(z - context[0], z + chunk_shape[0] + context[0]),
                    slice(y - context[1], y + chunk_shape[1] + context[1]),
                    slice(x - context[2], x + chunk_shape[2] + context[2]),
                )
                x_in = np.ascontiguousarray(input_arr[in_slice]).astype(np.float32) / 255.0
                x_in = torch.from_numpy(x_in[None, None]).to(device)
                with torch.no_grad():
                    y_out = model(x_in)
                y_np = y_out[0, 0].cpu().numpy()
                # crop context
                cropped = y_np[
                    context[0] : context[0] + chunk_shape[0],
                    context[1] : context[1] + chunk_shape[1],
                    context[2] : context[2] + chunk_shape[2],
                ]
                output_arr[
                    z : z + chunk_shape[0],
                    y : y + chunk_shape[1],
                    x : x + chunk_shape[2],
                ] = cropped
                n_chunks += 1
    return n_chunks


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(cfg["model_checkpoint"]), device)

    input_arr = zarr.open(cfg["data_path"], mode="r")
    output_path = (
        cfg["first_view_output_path"] if args.first_view_only else cfg["output_path"]
    )
    roi_offset = tuple(
        cfg["first_view_roi_offset"] if args.first_view_only else cfg["full_volume_roi_offset"]
    )
    roi_shape = tuple(
        cfg["first_view_roi_shape"] if args.first_view_only else cfg["full_volume_roi_shape"]
    )
    output_arr = zarr.open(
        output_path, mode="w", shape=tuple(input_arr.shape), chunks=tuple(cfg["chunk_shape"]),
        dtype=cfg.get("output_dtype", "float32"),
    )

    t0 = time.perf_counter()
    n_chunks = chunked_inference(
        model,
        input_arr,
        output_arr,
        roi_offset,
        roi_shape,
        tuple(cfg["chunk_shape"]),
        tuple(cfg["context"]),
        device,
    )
    wall = time.perf_counter() - t0

    payload = {
        "benchmark": "b6_baseline_comparison",
        "variant": "baseline",
        "first_view_only": args.first_view_only,
        "wall_time_s": wall,
        "n_chunks": n_chunks,
        "config": str(args.config),
        "lines_of_code": _self_loc(),
        "env": capture_env(),
    }
    write_result(args.output, payload)
    print(f"baseline {'first-view' if args.first_view_only else 'full'}: {wall:.1f}s, {n_chunks} chunks", file=sys.stderr)
    return 0


def _self_loc() -> int:
    """Count non-blank, non-comment lines in this file as the LoC metric."""
    src = Path(__file__).read_text().splitlines()
    return sum(1 for line in src if line.strip() and not line.strip().startswith("#"))


if __name__ == "__main__":
    sys.exit(main())
