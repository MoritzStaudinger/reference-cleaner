"""
Two figures, no bar charts:

  fig_1  time-per-reference distribution per config (strip + box, one
         point per paper; reveals the long tail and the per-config
         spread that summary stats hide)
  fig_2  gold-truth verdict matrix per config (categorical heatmap;
         shows exactly what the pipeline did on each labelled
         hallucination, not just a single recall %)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

mpl.rcParams.update({
    "font.size":       10,
    "axes.titlesize":  11,
    "axes.labelsize":  10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":      120,
    "savefig.dpi":     200,
    "savefig.bbox":    "tight",
    "savefig.pad_inches": 0.08,
})

CONFIGS = [
    ("grobid_cold",                       "grobid"),
    ("hybrid_cold",                       "hybrid"),
    ("hybrid_llm_qwen-3_6-35b_cold",      "hybrid+LLM"),
]
COLORS = {"grobid": "#4477AA", "hybrid": "#EE6677", "hybrid+LLM": "#228833"}


def load_configs(rundir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for cfg_file, label in CONFIGS:
        p = rundir / f"results_{cfg_file}.json"
        if p.exists():
            out[label] = json.loads(p.read_text())
    return out


def _save(fig, outdir: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}")
    plt.close(fig)
    print(f"  ✓ {name}")


# =====================================================================
# Figure 1 — time per reference, distribution
# =====================================================================
def fig_time_per_ref(data, outdir):
    cfgs = list(data.keys())
    per_cfg_sec_per_ref: dict[str, list[float]] = {}
    for cfg in cfgs:
        vals = []
        for r in data[cfg]["results"]:
            n = r.get("n_refs") or 0
            t = r.get("lookup_s")
            if n and isinstance(t, (int, float)) and t > 0:
                vals.append(t / n)
        per_cfg_sec_per_ref[cfg] = sorted(vals)

    fig, ax = plt.subplots(figsize=(9, 4.8))

    rng = np.random.default_rng(0)
    for i, cfg in enumerate(cfgs):
        vals = np.array(per_cfg_sec_per_ref[cfg])
        if not len(vals):
            continue
        y = i + 1 + rng.uniform(-0.18, 0.18, size=len(vals))   # jitter
        ax.scatter(vals, y, s=12, alpha=0.45, color=COLORS[cfg],
                   edgecolor="none", rasterized=True)

        # Median = solid vertical tick.  Mean = dotted vertical tick.
        # Same visual idiom for both so readers can compare position at
        # a glance; line style distinguishes which is which.
        med  = np.percentile(vals, 50)
        mean = float(np.mean(vals))
        ax.plot([med, med], [i + 0.78, i + 1.22],
                color="black", linewidth=2.5, solid_capstyle="butt")
        ax.plot([mean, mean], [i + 0.78, i + 1.22],
                color="black", linewidth=1.6, linestyle=(0, (1.5, 1.5)))
        ax.annotate(f"median {med:.2f}s",
                    xy=(med, i + 1.32), ha="center", fontsize=9,
                    color="#222", fontweight="bold")
        ax.annotate(f"mean {mean:.2f}s",
                    xy=(mean, i + 0.62), ha="center", fontsize=9,
                    color="#555")

    ax.set_yticks(range(1, len(cfgs) + 1))
    ax.set_yticklabels(cfgs)
    ax.set_xlabel("Lookup time per reference (seconds, log scale)")
    ax.set_xscale("log")
    ax.set_xlim(0.05, 50)
    ax.set_ylim(0.4, len(cfgs) + 0.7)
    ax.grid(axis="x", linestyle=":", alpha=0.5, which="both")
    fig.tight_layout()
    _save(fig, outdir, "01_time_per_reference")


# =====================================================================
# Figure 2 — gold-truth verdict matrix
# =====================================================================
def fig_gold_truth_matrix(data, outdir, rundir):
    """
    For each config, classify each of the 295 labelled hallucinations
    into one of four bins:
      caught_mismatch  pipeline flagged it as wrong-paper (precise hit)
      caught_missing   pipeline flagged it as not-findable (soft hit)
      not_caught       pipeline confirmed it as a match (FALSE NEGATIVE)
      parser_miss      parser couldn't even extract the labelled ref
    Show as a categorical heatmap — clear "what % went where".
    """
    from cache import cache_key  # noqa: F401  (kept for downstream use)
    from report import _overall_status

    gold_path = ROOT / "benchmark/hallucitation_table7.json"
    gold = {g["paper_id"]: g for g in json.loads(gold_path.read_text())}

    cfgs = list(data.keys())
    bins = ["caught_mismatch", "caught_missing", "not_caught", "parser_miss"]
    pretty = {
        "caught_mismatch": "Pipeline flagged\nas WRONG PAPER",
        "caught_missing":  "Pipeline flagged\nas NOT FOUND",
        "not_caught":      "Pipeline confirmed\nas a MATCH (false neg.)",
        "parser_miss":     "Parser failed\nto extract the ref",
    }
    bin_colors = {
        "caught_mismatch": "#117733",
        "caught_missing":  "#88CCEE",
        "not_caught":      "#CC3311",
        "parser_miss":     "#888888",
    }

    matrix: dict[str, dict[str, int]] = {c: {b: 0 for b in bins} for c in cfgs}

    for cfg in cfgs:
        cfg_dir = {
            "grobid":     "results_grobid_cold_intermediates",
            "hybrid":     "results_hybrid_cold_intermediates",
            "hybrid+LLM": "results_hybrid_llm_qwen-3_6-35b_cold_intermediates",
        }[cfg]
        inter_dir = rundir / cfg_dir
        for pid in gold:
            paper_path = inter_dir / f"{pid}.json"
            if not paper_path.exists():
                continue
            try:
                d = json.loads(paper_path.read_text())
            except Exception:
                continue
            tgt = d.get("target_idx")
            if tgt is None:
                matrix[cfg]["parser_miss"] += 1
                continue
            lookups = d.get("lookup_results", []) or []
            if tgt >= len(lookups):
                matrix[cfg]["parser_miss"] += 1
                continue
            st = _overall_status(lookups[tgt])
            if st == "mismatch":
                matrix[cfg]["caught_mismatch"] += 1
            elif st == "missing":
                matrix[cfg]["caught_missing"] += 1
            elif st == "match":
                matrix[cfg]["not_caught"] += 1
            else:
                matrix[cfg]["parser_miss"] += 1

    # Render as a clean stacked-row visualisation — each config a row
    # whose total = 295, broken into the four bins side by side.
    fig, ax = plt.subplots(figsize=(11, 3.4))
    cumulative = {c: 0 for c in cfgs}
    for bin_name in bins:
        for yi, cfg in enumerate(cfgs):
            n = matrix[cfg][bin_name]
            left = cumulative[cfg]
            ax.barh(yi, n, left=left, color=bin_colors[bin_name],
                    edgecolor="white", linewidth=1.2)
            if n >= 8:
                ax.text(left + n / 2, yi, f"{n}",
                        ha="center", va="center", fontsize=10.5,
                        fontweight="bold",
                        color="white" if bin_name != "caught_missing" else "#222")
            cumulative[cfg] += n

    ax.set_yticks(range(len(cfgs)))
    ax.set_yticklabels(cfgs, fontsize=11, fontweight="bold")
    ax.set_xlim(0, 295)
    ax.set_xlabel("Number of labelled hallucinations (out of 295)")
    ax.invert_yaxis()

    # Legend with the verdicts
    legend_handles = [
        mpl_patch(bin_colors[b], pretty[b])
        for b in bins
    ]
    ax.legend(handles=legend_handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.45), ncol=4, frameon=False,
              handletextpad=0.6, columnspacing=1.4, handlelength=2)

    ax.set_title("What the pipeline did on each of the 295 labelled-bad citations",
                 fontsize=11.5, pad=10, loc="left", fontweight="bold")

    # Headline annotation: pipeline-correct rate per config
    for yi, cfg in enumerate(cfgs):
        n_correct = matrix[cfg]["caught_mismatch"] + matrix[cfg]["caught_missing"]
        ax.text(298, yi, f"  {n_correct}/295 = {100*n_correct/295:.1f}%",
                va="center", ha="left", fontsize=10.5, fontweight="bold")

    fig.text(0.01, -0.18,
             "GREEN = pipeline confidently called the labelled hallucination a different paper.\n"
             "BLUE  = pipeline couldn't find it; flagged for review.   "
             "RED   = pipeline confirmed it as a real paper — TRUE FALSE NEGATIVE.   "
             "GREY  = parser couldn't even extract the ref to evaluate.",
             fontsize=9.5, color="#222")
    fig.tight_layout()
    _save(fig, outdir, "02_gold_truth_verdicts")

    return matrix


def mpl_patch(color, label):
    """Helper: build a Patch handle for the legend."""
    import matplotlib.patches as mpatches
    return mpatches.Patch(color=color, label=label)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="from_dir", required=True)
    ap.add_argument("--outdir", default="")
    args = ap.parse_args()

    rundir = Path(args.from_dir).resolve()
    outdir = Path(args.outdir) if args.outdir else (rundir / "plots")
    outdir.mkdir(parents=True, exist_ok=True)

    data = load_configs(rundir)
    if not data:
        print("No config results found"); sys.exit(1)

    # Wipe all old plots — clean slate
    for p in outdir.glob("0?_*"):
        p.unlink()

    print("Generating figures:")
    fig_time_per_ref(data, outdir)
    matrix = fig_gold_truth_matrix(data, outdir, rundir)

    print(f"\nGold-truth verdict counts (each row sums to 295):")
    for cfg, m in matrix.items():
        print(f"  {cfg:<14} {m}")
    print(f"\nPlots → {outdir}")


if __name__ == "__main__":
    main()
