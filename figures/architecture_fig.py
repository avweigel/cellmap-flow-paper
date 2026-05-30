"""Generate the cellmap-flow architecture schematic (Fig. 1) as a vector PDF.

Three loosely-coupled layers left-to-right -- data access, compute, service --
tied together by a YAML config. Pure matplotlib (no LaTeX) so it can be rendered
and visually checked without a TeX toolchain.

    python figures/architecture_fig.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

BLUE = "#2b6ca3"
LBLUE = "#dceaf5"
GREEN = "#1a9850"
LGREEN = "#e0f0e2"
ORANGE = "#d8821f"
LORANGE = "#fbebd6"
GREY = "#555555"


def _layer(ax, x, y, w, h, title, edge, face):
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.6, edgecolor=edge, facecolor=face, zorder=2,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h - 0.05, title, ha="center", va="top",
            fontsize=11, fontweight="bold", color=edge, zorder=3)


def _lines(ax, x, y, items, color="#222"):
    for i, t in enumerate(items):
        ax.text(x, y - i * 0.052, t, ha="left", va="top", fontsize=8.0,
                color=color, zorder=3)


def _arrow(ax, xy_from, xy_to, color=GREY, style="-|>", lw=1.8):
    ax.add_patch(FancyArrowPatch(
        xy_from, xy_to, arrowstyle=style, mutation_scale=14,
        linewidth=lw, color=color, zorder=4,
        shrinkA=2, shrinkB=2,
    ))


def main() -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0.02, 0.95)
    ax.axis("off")

    top, h = 0.30, 0.60
    w = 0.285
    xa, xb, xc = 0.025, 0.355, 0.690

    # (A) data-access
    _layer(ax, xa, top, w, h, "(A) Data access", BLUE, LBLUE)
    ax.text(xa + w / 2, top + h - 0.10, "ImageDataInterface", ha="center",
            va="top", fontsize=8.5, style="italic", color=BLUE)
    _lines(ax, xa + 0.02, top + h - 0.17, [
        "Zarr  •  N5  •  OME-Zarr",
        "S3  •  precomputed://",
        "— via TensorStore —",
        "multiscale + ROI indexing",
        "chunks aligned to model",
        "input shape / voxel size",
    ])

    # (B) compute
    _layer(ax, xb, top, w, h, "(B) Compute", GREEN, LGREEN)
    ax.text(xb + w / 2, top + h - 0.10, "Inferencer", ha="center", va="top",
            fontsize=8.5, style="italic", color=GREEN)
    _lines(ax, xb + 0.02, top + h - 0.17, [
        "adapters: TorchScript,",
        "DaCapo, BioImage.io,",
        "HuggingFace, cellmap-models",
        "normalize → GPU forward",
        "(FP16, context padding)",
        "→ output post-process",
    ])

    # (C) service
    _layer(ax, xc, top, w, h, "(C) Service", ORANGE, LORANGE)
    _lines(ax, xc + 0.02, top + h - 0.11, [
        "Server mode:",
        "  Flask → virtual Zarr over",
        "  HTTP → Neuroglancer",
        "  (on-demand chunks)",
        "Blockwise mode:",
        "  Daisy → LSF GPU workers",
        "  → output Zarr",
    ])

    # flow arrows A -> B -> C
    ymid = top + h / 2
    _arrow(ax, (xa + w, ymid), (xb, ymid))
    _arrow(ax, (xb + w, ymid), (xc, ymid))

    # YAML config bar tying all three together
    yb_y, yb_h = 0.05, 0.15
    yb_x, yb_w = xa, (xc + w) - xa
    box = FancyBboxPatch(
        (yb_x, yb_y), yb_w, yb_h, boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.4, edgecolor=GREY, facecolor="#f2f2f2", zorder=2,
    )
    ax.add_patch(box)
    ax.text(yb_x + yb_w / 2, yb_y + yb_h / 2,
            "YAML configuration  —  single source of truth "
            "(input · model · normalizers · post-processors · output / workers)",
            ha="center", va="center", fontsize=8.6, color="#333")
    for xc_center in (xa + w / 2, xb + w / 2, xc + w / 2):
        _arrow(ax, (xc_center, yb_y + yb_h), (xc_center, top), color=GREY,
               style="-|>", lw=1.2)

    fig.tight_layout(pad=0.4)
    out = Path(__file__).parent / "fig1_architecture.pdf"
    fig.savefig(out)
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
