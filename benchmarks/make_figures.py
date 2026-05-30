"""Generate publication figures from benchmark result JSONs.

Mirrors regenerate_paper_tables.py but emits vector PDFs into figures/ for
\\includegraphics in the paper. Reads the same results/*.json the aggregator
does, so it stays in sync with whatever has been run.

    python -m benchmarks.make_figures --results-dir benchmarks --out-dir figures
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

from benchmarks._common.reporting import load_results  # noqa: E402


def _clean_label(label: str) -> str:
    """jrc_hela-2_fly_organelles_s1_h100 -> jrc_hela-2 (drop model/scale/gpu tail)."""
    for token in ("_fly_organelles", "_s1", "_s0", "_h100", "_a100", "_v100"):
        label = label.replace(token, "")
    return label or "?"


def figure_b1(results, out_dir: Path) -> str | None:
    rows = [r for r in results if r.get("benchmark") == "b1_interactive_latency"]
    rows = [r for r in rows if r.get("samples_ms")]
    if not rows:
        return None
    rows.sort(key=lambda r: r.get("label", ""))
    labels = [_clean_label(r.get("label", "?")) for r in rows]
    data = [r["samples_ms"] for r in rows]

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False, widths=0.5, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#cfe3f2")
        patch.set_edgecolor("#2b6ca3")
    for med in bp["medians"]:
        med.set_color("#b2182b")
        med.set_linewidth(1.6)
    # annotate p99
    for i, r in enumerate(rows, start=1):
        p99 = r["summary"]["p99_ms"]
        ax.plot(i, p99, marker="D", color="#444", markersize=4)
        ax.annotate(f"p99 {p99:.0f}", (i, p99), textcoords="offset points",
                    xytext=(8, 0), va="center", fontsize=7, color="#444")
    ax.set_ylabel("chunk-request latency (ms)")
    ax.set_title("B1: interactive chunk latency (single H100)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    out = out_dir / "b1_latency.pdf"
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def figure_b3(results, out_dir: Path) -> str | None:
    rows = [
        r for r in results
        if r.get("benchmark") == "b3_strong_scaling" and r.get("return_code", 0) == 0
    ]
    if not rows:
        return None
    groups = defaultdict(list)
    for r in rows:
        label = r.get("label") or Path(r.get("base_config", "unknown")).stem
        groups[label].append(r)

    fig, (ax_w, ax_e) = plt.subplots(1, 2, figsize=(8.0, 3.4))
    colors = ["#2b6ca3", "#b2182b", "#1a9850", "#762a83"]
    for ci, label in enumerate(sorted(groups)):
        grp = sorted(groups[label], key=lambda r: r["n_workers"])
        ns = [r["n_workers"] for r in grp]
        walls = [r["wall_time_s"] for r in grp]
        base = next((r for r in grp if r["n_workers"] == 1), grp[0])
        c = colors[ci % len(colors)]
        clean = _clean_label(label)
        # wall time vs N (log-log) + ideal
        ax_w.plot(ns, walls, marker="o", color=c, label=clean)
        ideal = [base["wall_time_s"] / (n / base["n_workers"]) for n in ns]
        ax_w.plot(ns, ideal, ls="--", color=c, alpha=0.4)
        # parallel efficiency vs N
        eff = [(base["wall_time_s"] / w) / n for w, n in zip(walls, ns)]
        ax_e.plot(ns, eff, marker="o", color=c, label=clean)

    ax_w.set_xscale("log", base=2)
    ax_w.set_yscale("log")
    ax_w.set_xlabel("workers $N$")
    ax_w.set_ylabel("wall time (s)")
    ax_w.set_title("Strong scaling (dashed = ideal)")
    ax_w.set_xticks([n for g in groups.values() for n in [r["n_workers"] for r in g]])
    ax_w.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_w.grid(which="both", ls=":", alpha=0.5)
    ax_w.legend(fontsize=8)

    ax_e.axhline(1.0, ls="--", color="#888", alpha=0.6)
    ax_e.set_xscale("log", base=2)
    ax_e.set_xlabel("workers $N$")
    ax_e.set_ylabel("parallel efficiency")
    ax_e.set_title("Parallel efficiency")
    ax_e.set_ylim(0, 1.05)
    ax_e.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_e.set_xticks([n for g in groups.values() for n in [r["n_workers"] for r in g]])
    ax_e.grid(which="both", ls=":", alpha=0.5)
    ax_e.legend(fontsize=8)

    fig.suptitle("B3: cluster strong scaling (fixed sub-volume, H100 workers)")
    fig.tight_layout()
    out = out_dir / "b3_scaling.pdf"
    fig.savefig(out)
    plt.close(fig)
    return str(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default="benchmarks")
    p.add_argument("--out-dir", default="figures")
    args = p.parse_args()

    results = load_results(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    made = [figure_b1(results, out_dir), figure_b3(results, out_dir)]
    made = [m for m in made if m]
    for m in made:
        print(f"wrote {m}", file=sys.stderr)
    if not made:
        print("no figures produced (no matching results)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
