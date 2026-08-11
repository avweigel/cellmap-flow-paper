"""Generate the "Conventional workflow vs. CellMap Flow" before/after figure.

Left panel: the slow linear pipeline a researcher runs today -- a monotonous
top-to-bottom chain of steps with a single "repeat from scratch" return loop.
Right panel: CellMap Flow's continuous four-stage feedback loop.

Pure matplotlib (no LaTeX, no external assets) so it renders to a vector PDF and
a PNG that can be visually checked without a TeX toolchain. This replaces an
auto-traced SVG whose text and arrows were baked into 557 uneditable paths.

    python figures/fig1_workflow.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

# Palette (shared with architecture_fig.py): cool greys on the "before" side,
# a blue->teal family on the "after" side, green chips, one warm accent.
GREY_EDGE = "#8a8f96"
GREY_FILL = "#e9ebed"
GREY_TEXT = "#3b4046"
BLUE = "#2b6ca3"
TEAL = "#1f7a70"
GREEN = "#2e8b57"
LGREEN = "#dcefe1"
ORANGE = "#d8721f"
INK = "#2b2f34"
BG = "#f4f2ea"

STEPS = [
    "Choose model",
    "Crop small region",
    "Submit to cluster",
    "Wait (hours)",
    "Export to disk",
    "Tune post-processing",
    "Serve to viewer",
    "Build share links",
]

# Right-panel loop: (label, angle in degrees). Clockwise from the top.
NODES = [
    ("Choose\nmodel", 90),
    ("Serve\npredictions", 0),
    ("View\nlive", -90),
    ("Adjust", 180),
]
CHIPS = [  # (text, angle) placed just outside the loop, one per quadrant
    ("no offline export", 135),
    ("automated", 45),
    ("real-time", -45),
    ("same YAML", -135),
]


def rbox(ax, cx, cy, w, h, label, *, edge, fill, tcolor, fs=10.5, bold=False):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.02",
        linewidth=1.3, edgecolor=edge, facecolor=fill, zorder=3,
    ))
    ax.text(cx, cy, label, ha="center", va="center", zorder=4,
            fontsize=fs, color=tcolor, fontweight="bold" if bold else "normal")


def clock(ax, cx, cy, r):
    ax.add_patch(Circle((cx, cy), r, fill=False, lw=1.2, edgecolor=GREY_TEXT, zorder=5))
    ax.plot([cx, cx], [cy, cy + r * 0.62], lw=1.1, color=GREY_TEXT, zorder=5)
    ax.plot([cx, cx + r * 0.5], [cy, cy], lw=1.1, color=GREY_TEXT, zorder=5)


def main() -> None:
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    ax.set_xlim(0, 2.0)
    ax.set_ylim(0, 1.08)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.add_patch(plt.Rectangle((0, 0), 2.0, 1.08, facecolor=BG, edgecolor="none", zorder=0))

    # ---- header strip ----
    ax.text(0.50, 1.03, "Conventional workflow", ha="center", va="center",
            fontsize=14, fontweight="bold", color=GREY_TEXT)
    ax.text(1.50, 1.03, "CellMap Flow", ha="center", va="center",
            fontsize=14, fontweight="bold", color=BLUE)
    ax.plot([1.0, 1.0], [0.06, 0.98], lw=1.0, color="#cfccc2", zorder=1)

    # ================= LEFT: linear pipeline =================
    bx, bw, bh = 0.50, 0.46, 0.086
    ys = [0.90 - i * 0.108 for i in range(len(STEPS))]  # top -> bottom
    for step, y in zip(STEPS, ys):
        rbox(ax, bx, y, bw, bh, step, edge=GREY_EDGE, fill=GREY_FILL,
             tcolor=GREY_TEXT, fs=10.0)
    # clean downward connectors between consecutive steps
    for y_top, y_bot in zip(ys[:-1], ys[1:]):
        ax.add_patch(FancyArrowPatch(
            (bx, y_top - bh / 2), (bx, y_bot + bh / 2),
            arrowstyle="-|>", mutation_scale=11, lw=1.3, color=GREY_EDGE,
            shrinkA=0, shrinkB=0, zorder=2))

    # single full-height "repeat from scratch" return arrow on the left
    ax.add_patch(FancyArrowPatch(
        (bx - bw / 2, ys[-1]), (bx - bw / 2, ys[0]),
        connectionstyle="arc3,rad=-0.55", arrowstyle="-|>", mutation_scale=20,
        lw=3.0, color=ORANGE, shrinkA=6, shrinkB=6, zorder=2))
    ax.text(0.045, (ys[0] + ys[-1]) / 2, "repeat from scratch", rotation=90,
            ha="center", va="center", fontsize=9.5, color=ORANGE, fontweight="bold")

    # clock icon + "hours of waiting" tag on the Wait step (index 3), on the right
    yw = ys[3]
    clock(ax, bx + bw / 2 - 0.05, yw, 0.022)
    ax.add_patch(FancyArrowPatch(
        (0.82, yw), (bx + bw / 2, yw), arrowstyle="-", lw=1.0,
        color=GREY_EDGE, shrinkA=2, shrinkB=2, zorder=2))
    ax.text(0.90, yw, "hours of\nwaiting", ha="center", va="center",
            fontsize=9.0, color=GREY_TEXT, style="italic")

    # ================= RIGHT: feedback loop =================
    import numpy as np
    cx, cy, R, nr = 1.52, 0.50, 0.30, 0.135
    pos = {}
    for label, deg in NODES:
        a = np.radians(deg)
        pos[label] = (cx + R * np.cos(a), cy + R * np.sin(a))

    # clockwise connecting arcs between the four nodes
    order = [n[0] for n in NODES]
    for i in range(4):
        p, q = pos[order[i]], pos[order[(i + 1) % 4]]
        ax.add_patch(FancyArrowPatch(
            p, q, connectionstyle="arc3,rad=-0.28", arrowstyle="-|>",
            mutation_scale=18, lw=3.2, color=TEAL,
            shrinkA=nr * 78, shrinkB=nr * 78, zorder=2))

    # nodes
    for label, deg in NODES:
        x, y = pos[label]
        ax.add_patch(Circle((x, y), nr, facecolor=BLUE, edgecolor="white",
                            lw=1.5, zorder=4))
        ax.text(x, y, label, ha="center", va="center", color="white",
                fontsize=9.5, fontweight="bold", zorder=5, linespacing=0.95)

    # center label
    ax.text(cx, cy, "Continuous\nfeedback loop", ha="center", va="center",
            fontsize=11, fontweight="bold", color=INK, linespacing=1.0)

    # green check chips, one per quadrant just outside the ring
    for text, deg in CHIPS:
        a = np.radians(deg)
        x, y = cx + (R + 0.20) * np.cos(a), cy + (R + 0.20) * np.sin(a)
        ax.add_patch(FancyBboxPatch(
            (x - 0.11, y - 0.028), 0.22, 0.056,
            boxstyle="round,pad=0.004,rounding_size=0.03",
            linewidth=0, facecolor=LGREEN, zorder=4))
        ax.text(x, y, f"✓ {text}", ha="center", va="center",
                fontsize=8.5, color=GREEN, fontweight="bold", zorder=5)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    out = Path(__file__).parent / "fig1_workflow.pdf"
    fig.savefig(out, facecolor=BG)
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
