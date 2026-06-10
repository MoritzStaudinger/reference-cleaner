"""
Quantify which databases actually carry the load.

For each ref that overall-status'd as `match`, identify which sources
confirmed it.  Two views:

  PRIMARY     — counts each ref ONCE, attributed to its single
                "best confirming source" (highest-similarity match).
                Answers "if I could only run one source, which?"
  SUPPORTING  — counts each ref once per confirming source.  Answers
                "how often does each source corroborate?"

Also reports:
  - sources that flag wrong_doi / wrong_arxiv_id (the high-value precision
    signal — only doi.org / Crossref / S2 / arXiv)
  - overlap matrix: of refs confirmed by S2, what fraction also confirmed
    by Crossref?  Useful for justifying / pruning the source list.

Usage:
    python benchmark/analysis/source_contribution.py --from <rundir>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

VALIDATION_SOURCES = ("doi_org", "semantic_scholar", "openalex", "crossref",
                      "arxiv", "dblp", "acl_anthology", "scholarly", "openlibrary")


def _iter_intermediates(rundir: Path):
    for results_path in sorted(rundir.glob("results_*.json")):
        config = results_path.stem.replace("results_", "")
        inter_dir = rundir / f"{results_path.stem}_intermediates"
        if not inter_dir.is_dir():
            continue
        for paper_path in sorted(inter_dir.glob("*.json")):
            try:
                d = json.loads(paper_path.read_text())
            except Exception:
                continue
            for lr in (d.get("lookup_results") or []):
                yield config, paper_path.stem, lr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="from_dir", required=True)
    ap.add_argument("--config", default="",
                    help="Restrict to a single config (e.g. hybrid_llm_cold)")
    ap.add_argument("--plot", action="store_true",
                    help="Also emit a PDF/PNG plot in <from>/plots/")
    args = ap.parse_args()
    rundir = Path(args.from_dir).resolve()

    n_total           = 0       # all refs across all selected configs
    n_match           = 0
    primary_counts    = Counter()
    supporting_counts = Counter()
    flag_counts       = Counter()   # wrong_doi + wrong_arxiv_id by source
    by_config         = defaultdict(lambda: {"primary": Counter(),
                                             "supporting": Counter(),
                                             "n_match": 0, "n_total": 0})

    # For overlap: per ref, the set of sources that returned label=="match"
    overlap_pairs: Counter = Counter()
    sole_source: Counter   = Counter()   # ref confirmed by ONLY this source

    for config, paper_id, lr in _iter_intermediates(rundir):
        if args.config and config != args.config:
            continue
        if not lr:
            continue
        n_total += 1
        by_config[config]["n_total"] += 1

        # wrong-id flags can be set on any "found" source
        for src in VALIDATION_SOURCES:
            r = lr.get(src, {})
            if r.get("wrong_doi") or r.get("wrong_arxiv_id"):
                flag_counts[src] += 1

        # which sources confirmed
        confirming = []
        for src in VALIDATION_SOURCES:
            r = lr.get(src, {})
            if r.get("status") == "found" and r.get("label") == "match":
                confirming.append((src, r.get("similarity") or 0.0))
        if not confirming:
            continue
        n_match += 1
        by_config[config]["n_match"] += 1

        # primary attribution: highest-sim source wins
        primary = max(confirming, key=lambda kv: kv[1])[0]
        primary_counts[primary] += 1
        by_config[config]["primary"][primary] += 1

        sources = [s for s, _ in confirming]
        for s in sources:
            supporting_counts[s] += 1
            by_config[config]["supporting"][s] += 1
        if len(sources) == 1:
            sole_source[sources[0]] += 1

        for i, a in enumerate(sources):
            for b in sources[i+1:]:
                pair = tuple(sorted((a, b)))
                overlap_pairs[pair] += 1

    print(f"Scanned {n_total} refs across {len(by_config)} config(s); "
          f"{n_match} overall-status=match.\n")

    def _print_counter(title: str, c: Counter, denom: int):
        print(title)
        print(f"  {'source':<20}{'n':>8}{'%':>8}")
        for src, n in c.most_common():
            pct = (100 * n / denom) if denom else 0
            print(f"  {src:<20}{n:>8}{pct:>7.1f}%")
        print()

    _print_counter("PRIMARY confirming source (one count per matched ref):",
                   primary_counts, n_match)
    _print_counter("SUPPORTING (every confirming source counts, refs can multi-count):",
                   supporting_counts, n_match)
    _print_counter("Sole confirming source (only this source said 'match'):",
                   sole_source, n_match)
    _print_counter("Identifier-mismatch flags (wrong_doi + wrong_arxiv_id):",
                   flag_counts, n_total)

    if overlap_pairs:
        print("Pairwise co-confirmation (top 10 — refs where BOTH sources said 'match'):")
        for (a, b), n in overlap_pairs.most_common(10):
            print(f"  {a:<18} ∩ {b:<18}  {n}")
        print()

    if len(by_config) > 1:
        print("Per-config primary attribution:")
        configs = sorted(by_config.keys())
        srcs = sorted({s for c in by_config.values() for s in c["primary"]})
        header = f"  {'source':<20}" + "".join(f"{c[:18]:>20}" for c in configs)
        print(header)
        for s in srcs:
            row = f"  {s:<20}"
            for c in configs:
                n = by_config[c]["primary"].get(s, 0)
                row += f"{n:>20}"
            print(row)

    if args.plot:
        _emit_plot(rundir, by_config)


# ----------------------------------------------------------------------
# Plot — per-source contribution across configs, two panels:
#   left  = primary (each matched ref attributed to one source)
#   right = sole (this source was the only one to confirm the match)
# Configs ordered (grobid → hybrid → hybrid+LLM) and colored consistently.
# ----------------------------------------------------------------------
_CFG_DISPLAY = {
    "grobid_cold": "grobid",
    "hybrid_cold": "hybrid",
    "hybrid_llm_qwen-3_6-35b_cold": "hybrid+LLM",
}
_CFG_COLORS = {"grobid": "#4477AA", "hybrid": "#EE6677", "hybrid+LLM": "#228833"}
_SRC_DISPLAY = {
    "semantic_scholar": "Semantic Scholar",
    "crossref":         "Crossref",
    "acl_anthology":    "ACL Anthology",
    "arxiv":            "arXiv",
    "dblp":             "DBLP",
    "openalex":         "OpenAlex",
    "doi_org":          "doi.org",
    "openlibrary":      "OpenLibrary",
    "scholarly":        "Google Scholar",
}


def _emit_plot(rundir: Path, by_config: dict) -> None:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np

    outdir = rundir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    cfgs = [c for c in ("grobid_cold", "hybrid_cold",
                        "hybrid_llm_qwen-3_6-35b_cold") if c in by_config]
    if not cfgs:
        print("No known configs in data; skipping plot")
        return

    # Per-config coverage counter — number of matched refs where each
    # source returned label=match (regardless of whether it was the
    # tiebreaker winner).  Recomputed here from the intermediates.
    cov_per_config: dict[str, Counter] = {}
    for cfg in cfgs:
        cov = Counter()
        inter = rundir / f"results_{cfg}_intermediates"
        for paper_path in sorted(inter.glob("*.json")):
            try:
                d = json.loads(paper_path.read_text())
            except Exception:
                continue
            for lr in (d.get("lookup_results") or []):
                if not lr:
                    continue
                conf = [s for s in VALIDATION_SOURCES
                        if lr.get(s, {}).get("status") == "found"
                        and lr.get(s, {}).get("label") == "match"]
                if not conf:
                    continue
                for s in conf:
                    cov[s] += 1
        cov_per_config[cfg] = cov

    # Source ordering: by hybrid+LLM coverage (descending) — so the
    # high-coverage but low-primary sources (ACL Anthology) appear near
    # the top where their gap between the two panels is most visible.
    rank_cfg = "hybrid_llm_qwen-3_6-35b_cold" if "hybrid_llm_qwen-3_6-35b_cold" in cfgs else cfgs[-1]
    rank_n = by_config[rank_cfg]["n_match"] or 1
    sources_ranked = sorted(
        (s for s in _SRC_DISPLAY if s != "scholarly"),
        key=lambda s: -(cov_per_config[rank_cfg].get(s, 0) / rank_n)
    )
    src_labels = [_SRC_DISPLAY[s] for s in sources_ranked]

    mpl.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 10, "legend.fontsize": 9.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    })

    # Single plot — all sources, one x-axis.  The small tail bars stay
    # tiny visually but that IS the data: the long tail is negligible
    # compared to the top sources.  Numeric labels next to every bar
    # make even the tiny ones legible.
    denoms = {cfg: by_config[cfg]["n_match"] for cfg in cfgs}

    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    y = np.arange(len(sources_ranked))
    h = 0.26
    offsets = {cfg: (i - (len(cfgs) - 1) / 2) * h for i, cfg in enumerate(cfgs)}
    labels_y = [_SRC_DISPLAY[s] for s in sources_ranked]
    xmax = 80
    for cfg in cfgs:
        denom = denoms[cfg] or 1
        vals = [100 * cov_per_config[cfg].get(s, 0) / denom
                for s in sources_ranked]
        label = _CFG_DISPLAY[cfg]
        bars = ax.barh(y + offsets[cfg], vals, h,
                       color=_CFG_COLORS[label], label=label,
                       edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v >= 0.05:
                fmt = f"{v:.1f}" if v >= 1 else f"{v:.2f}"
                ax.text(v + xmax * 0.008,
                        bar.get_y() + bar.get_height() / 2,
                        fmt, va="center", fontsize=8.5, color="#222")
    ax.set_yticks(y)
    ax.set_yticklabels(labels_y)
    ax.invert_yaxis()
    ax.set_xlabel("% of matched references")
    ax.set_xlim(0, xmax)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.legend(loc="lower center", frameon=False, ncol=len(cfgs),
              bbox_to_anchor=(0.5, -0.25))
    fig.tight_layout()
    print()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"03_source_coverage.{ext}")
    plt.close(fig)
    print(f"Plot → {outdir / '03_source_coverage.pdf'}")


if __name__ == "__main__":
    main()
