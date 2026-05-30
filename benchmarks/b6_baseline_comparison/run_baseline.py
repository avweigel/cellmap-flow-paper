"""B6 baseline: hand-rolled chunked PyTorch inference with offline Zarr export.

The point of this script is to look like what a user would write themselves
without cellmap-flow: load a TorchScript model, loop over output blocks, read a
context-padded input window for each, run inference, write the result to Zarr.
It is intentionally not factored and uses no cellmap_flow imports. This is the
shape of the workflow the paper claims to replace; its run time is the baseline.

The fly_organelles model is a valid-convolution U-Net: it maps a 178^3 input to
a 56^3 x 8-channel output (output is smaller than input, not same-size), so the
loop steps by the OUTPUT block and reads an input window grown by the context on
every side. Block shapes come from the config.
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
        help="only produce the small first-view ROI, then stop (times time-to-first-view)",
    )
    return p.parse_args()


def read_padded(arr: zarr.Array, starts, size) -> np.ndarray:
    """Read a `size`-shaped window starting at `starts`, zero-padding out-of-bounds
    edges so the model always sees a full-size input."""
    out = np.zeros(size, dtype=np.float32)
    src_slices, dst_slices = [], []
    for start, n, dim in zip(starts, size, arr.shape):
        lo, hi = max(start, 0), min(start + n, dim)
        if lo >= hi:
            return out  # window entirely outside the array
        src_slices.append(slice(lo, hi))
        dst_slices.append(slice(lo - start, hi - start))
    out[tuple(dst_slices)] = arr[tuple(src_slices)].astype(np.float32)
    return out


def chunked_inference(
    model, input_arr, output_arr, roi_offset, roi_shape,
    in_block, out_block, out_channel, device,
) -> int:
    context = tuple((i - o) // 2 for i, o in zip(in_block, out_block))
    n_blocks = 0
    for z in range(roi_offset[0], roi_offset[0] + roi_shape[0], out_block[0]):
        for y in range(roi_offset[1], roi_offset[1] + roi_shape[1], out_block[1]):
            for x in range(roi_offset[2], roi_offset[2] + roi_shape[2], out_block[2]):
                starts = (z - context[0], y - context[1], x - context[2])
                x_in = read_padded(input_arr, starts, in_block)
                t = torch.from_numpy(x_in[None, None]).to(device)
                with torch.no_grad():
                    y_out = model(t)
                pred = y_out[0, out_channel].cpu().numpy()  # out_block^3
                # write at output-relative coords; clip overhang at the ROI edge
                oz, oy, ox = z - roi_offset[0], y - roi_offset[1], x - roi_offset[2]
                ze = min(oz + out_block[0], output_arr.shape[0])
                ye = min(oy + out_block[1], output_arr.shape[1])
                xe = min(ox + out_block[2], output_arr.shape[2])
                output_arr[oz:ze, oy:ye, ox:xe] = pred[: ze - oz, : ye - oy, : xe - ox]
                n_blocks += 1
    return n_blocks


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(str(cfg["model_checkpoint"]), map_location=device).eval()

    input_arr = zarr.open(cfg["data_path"], mode="r")
    in_block = tuple(cfg["input_block_shape"])
    out_block = tuple(cfg["output_block_shape"])
    out_channel = int(cfg.get("out_channel", 0))

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
        output_path, mode="w", shape=roi_shape, chunks=out_block,
        dtype=cfg.get("output_dtype", "float32"),
    )

    t0 = time.perf_counter()
    n_blocks = chunked_inference(
        model, input_arr, output_arr, roi_offset, roi_shape,
        in_block, out_block, out_channel, device,
    )
    wall = time.perf_counter() - t0

    payload = {
        "benchmark": "b6_baseline_comparison",
        "variant": "baseline",
        "first_view_only": args.first_view_only,
        "wall_time_s": wall,
        "n_blocks": n_blocks,
        "config": str(args.config),
        "lines_of_code": _self_loc(),
        "env": capture_env(),
    }
    write_result(args.output, payload)
    print(f"baseline {'first-view' if args.first_view_only else 'full'}: {wall:.1f}s, {n_blocks} blocks", file=sys.stderr)
    return 0


def _self_loc() -> int:
    """Count non-blank, non-comment lines in this file as the LoC metric."""
    src = Path(__file__).read_text().splitlines()
    return sum(1 for line in src if line.strip() and not line.strip().startswith("#"))


if __name__ == "__main__":
    sys.exit(main())
