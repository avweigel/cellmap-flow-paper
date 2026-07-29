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


def _clean_ds(label: str) -> str:
    """Trim model/scale/GPU suffixes off a result label so the table shows just
    the dataset (e.g. jrc_hela-2_fly_organelles_s1_h100 -> jrc_hela-2). The
    stripped detail lives in the table caption instead, which keeps the rows
    narrow enough to fit the column."""
    for tok in ("_fly_organelles", "_s0", "_s1", "_s2", "_h100", "_a100", "_v100", "_cpu"):
        i = label.find(tok)
        if i != -1:
            label = label[:i]
    return label or "?"


def _count_loc(path) -> int | None:
    """Non-blank, non-comment (#) lines of a file. Works for both .py and .yaml.
    Returns None if the file can't be read."""
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return None
    return sum(1 for ln in lines if ln.strip() and not ln.strip().startswith("#"))


def _b6_user_loc(cf_row: dict) -> tuple[int | None, int | None]:
    """User-facing lines of code for the B6 comparison: the hand-rolled
    inference *script* a user must write for the baseline, vs the *YAML config*
    a user writes to drive cellmap-flow (plus one CLI line). This is the real
    operational-effort comparison -- not the line counts of the benchmark
    harness scripts, which the stored *_loc fields record. Falls back to those
    stored values if the artifacts can't be read."""
    here = Path(__file__).parent / "b6_baseline_comparison"
    baseline_loc = _count_loc(here / "run_baseline.py")
    cf_loc = None
    try:
        import yaml  # noqa: PLC0415
        shared = yaml.safe_load(Path(cf_row.get("config", "")).read_text())
        bw_yaml = shared.get("cellmap_flow_blockwise_yaml")
        n = _count_loc(bw_yaml)
        if n is not None:
            cf_loc = n + 1  # + the single CLI invocation
    except Exception:
        cf_loc = None
    return baseline_loc, cf_loc


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
        "\\begin{table*}[tbhp]",
        "\\centering",
        "\\caption{B1 interactive chunk-request latency, model \\texttt{cellmap/fly\\_organelles\\_run07\\_700000} at scale s1 on a single H100. Median / p95 / p99 in ms over the measured request set.}",
        "\\label{tab:b1_results}",
        "\\small",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Dataset & N & median (ms) & p95 (ms) & p99 (ms) \\\\",
        "\\midrule",
    ]
    for r in sorted(rows, key=lambda r: r.get("label", "")):
        s = r["summary"]
        lines.append(
            f"{_esc(_clean_ds(r.get('label', '?')))} & {s['count']} & {s['median_ms']:.1f} & "
            f"{s['p95_ms']:.1f} & {s['p99_ms']:.1f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
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
        "\\begin{table*}[tbhp]",
        "\\centering",
        "\\caption{B3 cluster strong scaling (\\texttt{fly\\_organelles} model, "
        "fixed $8192^3$~nm sub-volume, single-GPU H100 workers). Wall time, "
        "speedup, and parallel efficiency vs.\\ $N{=}1$ at each worker count.}",
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
            ds = _esc(_clean_ds(label)) if j == 0 else ""
            lines.append(
                f"{ds} & {r['n_workers']} & {r['wall_time_s']:.1f} & "
                f"{speedup:.2f}$\\times$ & {eff:.2f} \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    return "\n".join(lines)


def _b6_dataset(row: dict) -> str:
    """Dataset label for a B6 result row: the explicit `dataset` field if the run
    recorded one, else the `dataset`/`data_path` of its shared config, else
    'unknown'. The config fallback keeps older result files -- written before the
    field existed -- grouping correctly."""
    ds = row.get("dataset")
    if ds:
        return ds
    cfg_path = row.get("config")
    if cfg_path:
        try:
            import re  # noqa: PLC0415
            import yaml  # noqa: PLC0415
            cfg = yaml.safe_load(Path(cfg_path).read_text())
            if cfg.get("dataset"):
                return cfg["dataset"]
            m = re.search(r"jrc[_-][A-Za-z0-9-]+", str(cfg.get("data_path", "")))
            if m:
                return m.group(0)
        except Exception:
            pass
    return "unknown"


def render_b6(rows: list[dict]) -> str:
    if not rows:
        return "% B6: no results yet\n"
    # Group by dataset. Within a dataset: run_baseline.py emits two baseline rows
    # (first_view_only True/False); run_cellmapflow.py emits one cellmapflow row
    # per blockwise worker count. The worker-count axis is what exposes the
    # completion-time crossover (single-process baseline vs. cellmap-flow at N>1).
    by_ds: dict[str, dict] = defaultdict(
        lambda: {"bl_first": None, "bl_full": None, "cf": {}}
    )
    for r in rows:
        bucket = by_ds[_b6_dataset(r)]
        if r.get("variant") == "baseline":
            key = "bl_first" if r.get("first_view_only") else "bl_full"
            bucket[key] = r
        elif r.get("variant") == "cellmapflow":
            bucket["cf"][int(r.get("workers", 1) or 1)] = r

    def _fmt(v) -> str:
        return f"{v:.1f}" if isinstance(v, (int, float)) else "---"

    body = []
    datasets = sorted(by_ds)
    for di, ds in enumerate(datasets):
        b = by_ds[ds]
        ds_tex = _esc(ds)
        any_cf = next((b["cf"][n] for n in sorted(b["cf"])), None)
        bl_loc, cf_loc = _b6_user_loc(any_cf) if any_cf else (None, None)
        if bl_loc is None and b["bl_full"]:
            bl_loc = b["bl_full"].get("lines_of_code")
        if cf_loc is None and any_cf:
            cf_loc = any_cf.get("lines_of_code_total")

        if b["bl_first"] or b["bl_full"]:
            fv = _fmt(b["bl_first"].get("wall_time_s")) if b["bl_first"] else "---"
            full = _fmt(b["bl_full"].get("wall_time_s")) if b["bl_full"] else "---"
            body.append(
                f"Hand-rolled PyTorch baseline & {ds_tex} & {fv} & {full} & "
                f"{bl_loc if bl_loc is not None else '---'} \\\\"
            )
        multi = len(b["cf"]) > 1
        for i, n in enumerate(sorted(b["cf"])):
            cf = b["cf"][n]
            label = f"\\cellmapflow{{}} ($N{{=}}{n}$)" if multi else "\\cellmapflow{}"
            # first-view is a worker-independent single-chunk response; show it
            # only against the first (smallest-N) row so it is not double-counted.
            fv = _fmt(cf.get("time_to_first_view_s")) if i == 0 else "---"
            full = _fmt(cf.get("time_to_completion_s"))
            body.append(
                f"{label} & {ds_tex} & {fv} & {full} & "
                f"{cf_loc if cf_loc is not None else '---'} \\\\"
            )
        if di != len(datasets) - 1:
            body.append("\\midrule")

    lines = [
        "\\begin{table*}[tbhp]",
        "\\centering",
        "\\caption{B6 \\cellmapflow{} vs.\\ a hand-rolled PyTorch baseline on the "
        "same task, model, and hardware (one H100 per worker). Times in seconds; "
        "user LoC is the hand-written inference script vs.\\ the cellmap-flow YAML "
        "config plus its CLI invocation. For cellmap-flow, $N$ is the number of "
        "blockwise workers; the single-process baseline cannot scale out. "
        "First-view is a single-chunk server response and is worker-independent.}",
        "\\label{tab:b6_results}",
        "\\small",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Pipeline & Dataset & first-view (s) & full-volume (s) & user LoC \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
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
