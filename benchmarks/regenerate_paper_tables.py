"""Aggregate per-benchmark JSON results into LaTeX tables for the paper.

Reads every `*/results/*.json` under benchmarks/, groups by benchmark ID, and
emits one LaTeX file per benchmark plus an `_all.tex` master include.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from benchmarks._common.reporting import load_results


def _esc(s) -> str:
    """Escape underscores for LaTeX text mode."""
    return str(s).replace("_", r"\_")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-dir",
        default="benchmarks",
        help="root containing each bN_*/results/*.json",
    )
    p.add_argument(
        "--out",
        required=True,
        help="output .tex file (single file with all generated tables)",
    )
    return p.parse_args()


def render_b1(rows: list[dict]) -> str:
    if not rows:
        return "% B1: no results yet\n"
    lines = [
        "\\begin{table}[tbhp]",
        "\\centering",
        "\\caption{B1 interactive chunk-request latency. Median / p95 / p99 in ms over the measured request set.}",
        "\\label{tab:b1_results}",
        "\\small",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Configuration & N & median (ms) & p95 (ms) & p99 (ms) \\\\",
        "\\midrule",
    ]
    for r in sorted(rows, key=lambda r: r.get("label", "")):
        s = r["summary"]
        lines.append(
            f"{_esc(r.get('label', '?'))} & {s['count']} & {s['median_ms']:.1f} & "
            f"{s['p95_ms']:.1f} & {s['p99_ms']:.1f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def _b3_label(r: dict) -> str:
    """Dataset label for grouping; falls back to the config stem for rows
    written before the harness recorded an explicit label."""
    if r.get("label"):
        return r["label"]
    from pathlib import Path as _P
    return _P(r.get("base_config", "unknown")).stem


def render_b3(rows: list[dict]) -> str:
    if not rows:
        return "% B3: no results yet\n"
    # A nonzero return code means cellmap_flow_blockwise crashed; its wall time is
    # whatever it took to fail, not a valid scaling point. Dropping these avoids
    # fabricating a huge speedup from a fast failure.
    dropped = sorted(
        {(_b3_label(r), r["n_workers"]) for r in rows if r.get("return_code", 0) != 0}
    )
    rows = [r for r in rows if r.get("return_code", 0) == 0]
    if not rows:
        return "% B3: all runs failed (nonzero return_code); no valid scaling points\n"

    # Group into one curve per dataset; speedup is relative to each dataset's own N=1.
    groups = defaultdict(list)
    for r in rows:
        groups[_b3_label(r)].append(r)

    lines = []
    if dropped:
        lines.append(f"% B3: dropped failed (label, N) runs: {dropped}")
    lines += [
        "\\begin{table}[tbhp]",
        "\\centering",
        "\\caption{B3 cluster strong scaling. Wall time, speedup, and parallel "
        "efficiency vs.\\ $N{=}1$ at each worker count, for a fixed sub-volume.}",
        "\\label{tab:b3_results}",
        "\\small",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Dataset & $N$ & wall (s) & speedup & efficiency \\\\",
        "\\midrule",
    ]
    for gi, label in enumerate(sorted(groups)):
        grp = sorted(groups[label], key=lambda r: r["n_workers"])
        base = next((r for r in grp if r["n_workers"] == 1), grp[0])
        if gi:
            lines.append("\\midrule")
        for j, r in enumerate(grp):
            speedup = base["wall_time_s"] / r["wall_time_s"] if r["wall_time_s"] > 0 else 0.0
            eff = speedup / r["n_workers"] if r["n_workers"] > 0 else 0.0
            ds = _esc(label) if j == 0 else ""
            lines.append(
                f"{ds} & {r['n_workers']} & {r['wall_time_s']:.1f} & "
                f"{speedup:.2f}$\\times$ & {eff:.2f} \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def render_b6(rows: list[dict]) -> str:
    if not rows:
        return "% B6: no results yet\n"
    # run_baseline.py emits two rows, both variant="baseline", distinguished by
    # the `first_view_only` flag; run_cellmapflow.py emits one variant="cellmapflow".
    baselines = [r for r in rows if r.get("variant") == "baseline"]
    bl_first = next((r for r in baselines if r.get("first_view_only")), None)
    bl_full = next((r for r in baselines if not r.get("first_view_only")), None)
    cf = next((r for r in rows if r.get("variant") == "cellmapflow"), None)
    missing = [
        name
        for name, row in [
            ("baseline first-view", bl_first),
            ("baseline full-volume", bl_full),
            ("cellmapflow", cf),
        ]
        if row is None
    ]
    if missing:
        return f"% B6: missing runs to render: {', '.join(missing)}\n"
    lines = [
        "\\begin{table}[tbhp]",
        "\\centering",
        "\\caption{B6 \\cellmapflow{} vs.\\ hand-rolled PyTorch baseline on the same task. Times in seconds.}",
        "\\label{tab:b6_results}",
        "\\small",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Pipeline & first-view (s) & full-volume (s) & LoC \\\\",
        "\\midrule",
        f"Hand-rolled PyTorch baseline & {bl_first.get('wall_time_s', 0):.1f} & "
        f"{bl_full.get('wall_time_s', 0):.1f} & {bl_full.get('lines_of_code', 0)} \\\\",
        f"\\cellmapflow{{}} & {cf.get('time_to_first_view_s', 0):.1f} & "
        f"{cf.get('time_to_completion_s', 0):.1f} & {cf.get('lines_of_code_total', 0)} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


RENDERERS = {
    "b1_interactive_latency": render_b1,
    "b3_strong_scaling": render_b3,
    "b6_baseline_comparison": render_b6,
}


def main() -> int:
    args = parse_args()
    results = load_results(args.results_dir)

    grouped = defaultdict(list)
    for r in results:
        grouped[r.get("benchmark", "unknown")].append(r)

    out_lines = ["% Auto-generated by benchmarks/regenerate_paper_tables.py", ""]
    for bench_id, renderer in RENDERERS.items():
        out_lines.append(f"% --- {bench_id} ---")
        out_lines.append(renderer(grouped.get(bench_id, [])))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(out_lines))
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
