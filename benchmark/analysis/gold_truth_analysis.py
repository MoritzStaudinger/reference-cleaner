"""
Gold-truth analysis using the HalluCitation Table-7 labels.

Cross-references per-paper results from the three configs with:
  (a) the labelled hallucination (one per paper, from the gold set)
  (b) the CORE rescue results — when CORE "finds" a known-fabricated
      citation, that's a precision regression in CORE; when CORE finds
      a labelled-bad citation that the pipeline correctly flagged as
      missing, it's evidence CORE hallucinates / mis-matches.

Outputs:
  - Per-config recall decomposition (parser, lookup, pipeline)
  - Of refs the pipeline marked `missing`, what fraction were:
      - LABELLED hallucinations → CORE-rescuing these is a FALSE POSITIVE
      - NOT labelled → could be legit findings
  - CORE precision proxy: rescued / labelled-missing
  - CORE total rescue rate: rescued / all-processed-missing

Usage:
    python benchmark/analysis/gold_truth_analysis.py \\
        --from   benchmark/runs/latest
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from cache import cache_key as _cache_key
from report import _overall_status


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5
    return ((c - s) / denom, (c + s) / denom)


def _pct(k, n):
    return 100.0 * k / n if n else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="from_dir", required=True)
    args = ap.parse_args()
    rundir = Path(args.from_dir).resolve()

    # Load gold set
    gold = json.loads((ROOT / "benchmark/hallucitation_table7.json").read_text())
    gold_by_paper = {g["paper_id"]: g for g in gold}
    print(f"Gold set: {len(gold)} labelled hallucinations")

    # Load rescue results (cache_key → result)
    rescue_path = rundir / "scholar_core_rescue.json"
    rescue_by_key: dict[str, dict] = {}
    if rescue_path.exists():
        rd = json.loads(rescue_path.read_text())
        for r in rd.get("results", []):
            rescue_by_key[r["cache_key"]] = r
        print(f"Rescue progress: {len(rescue_by_key)} refs processed so far\n")

    # Per-config analysis
    configs = [
        ("grobid_cold",                       "grobid"),
        ("hybrid_cold",                       "hybrid"),
        ("hybrid_llm_qwen-3_6-35b_cold",      "hybrid+qwen"),
    ]

    print("=" * 90)
    print("PER-CONFIG GOLD-TRUTH ANALYSIS")
    print("=" * 90)

    summary: dict[str, dict] = {}

    for cfg_dir, cfg_label in configs:
        inter_dir = rundir / f"results_{cfg_dir}_intermediates"
        if not inter_dir.is_dir():
            print(f"  (skip {cfg_label}: no intermediates)")
            continue

        n_papers = 0
        n_parser_found_target = 0
        n_caught_pipeline     = 0
        n_target_status_missing = 0     # caught BECAUSE status=missing
        n_target_status_mismatch = 0    # caught BECAUSE status=mismatch
        n_uncaught_pipeline   = 0
        n_uncaught_rescued_by_core = 0  # PRECISION REGRESSION proxy
        labelled_missing_seen: list[str] = []  # cache_keys of labelled refs marked missing

        # Rescue stats: of the rescued refs, how many are labelled hallucinations?
        n_rescued_total = 0
        n_rescued_labelled = 0

        # Iterate per paper
        for paper_path in sorted(inter_dir.glob("*.json")):
            pid = paper_path.stem
            gold_entry = gold_by_paper.get(pid)
            if gold_entry is None:
                continue
            try:
                d = json.loads(paper_path.read_text())
            except Exception:
                continue

            refs    = d.get("llm_refs") or d.get("parser_refs") or []
            lookups = d.get("lookup_results") or []
            target_idx = d.get("target_idx")

            n_papers += 1
            if target_idx is None:
                continue

            n_parser_found_target += 1
            ref = refs[target_idx]
            lr  = lookups[target_idx]
            status = _overall_status(lr)

            # Pipeline-level "caught"
            if status in ("missing", "mismatch"):
                n_caught_pipeline += 1
                if status == "missing":
                    n_target_status_missing += 1
                    # Check whether CORE rescued this labelled-bad ref
                    key = _cache_key(ref)
                    if key:
                        labelled_missing_seen.append(key)
                        rescue_r = rescue_by_key.get(key)
                        if rescue_r and rescue_r.get("new_status") in ("match", "fuzzy_rescue"):
                            n_uncaught_rescued_by_core += 1
                            # Precision regression: CORE confidently found
                            # what HalluCitation labels as fabricated
                else:
                    n_target_status_mismatch += 1
            else:
                n_uncaught_pipeline += 1

            # Rescue stats on all this config's missing refs
            for r, lr_i in zip(refs, lookups):
                if _overall_status(lr_i) != "missing":
                    continue
                k = _cache_key(r)
                if k and k in rescue_by_key:
                    rr = rescue_by_key[k]
                    if rr.get("new_status") in ("match", "fuzzy_rescue"):
                        n_rescued_total += 1

        # Wilson CIs
        pr_lo, pr_hi   = _wilson(n_parser_found_target, n_papers)
        pi_lo, pi_hi   = _wilson(n_caught_pipeline,    n_papers)
        lo_lo, lo_hi   = _wilson(n_caught_pipeline,    n_parser_found_target)

        print(f"\n--- {cfg_label} ({n_papers} papers) ---")
        print(f"  Parser found target:  {n_parser_found_target}/{n_papers} = "
              f"{_pct(n_parser_found_target, n_papers):.1f}% "
              f"(95%CI {pr_lo*100:.1f}–{pr_hi*100:.1f})")
        print(f"  Pipeline caught:      {n_caught_pipeline}/{n_papers} = "
              f"{_pct(n_caught_pipeline, n_papers):.1f}% "
              f"(95%CI {pi_lo*100:.1f}–{pi_hi*100:.1f})")
        print(f"  Lookup-recall (conditional): {n_caught_pipeline}/{n_parser_found_target} = "
              f"{_pct(n_caught_pipeline, n_parser_found_target):.1f}% "
              f"(95%CI {lo_lo*100:.1f}–{lo_hi*100:.1f})")
        print(f"  Of caught: status=missing {n_target_status_missing}, "
              f"status=mismatch {n_target_status_mismatch}")
        print()
        print(f"  Of {n_target_status_missing} labelled-bad refs marked 'missing':")
        print(f"    CORE 'rescued' (FALSE POSITIVE in CORE):  "
              f"{n_uncaught_rescued_by_core}/{n_target_status_missing} = "
              f"{_pct(n_uncaught_rescued_by_core, n_target_status_missing):.1f}%")
        print(f"  CORE-rescue distribution on this config's missing refs:")
        print(f"    Total CORE rescues:          {n_rescued_total}")
        print(f"    Of which labelled hallucinations:  {n_uncaught_rescued_by_core} "
              f"({_pct(n_uncaught_rescued_by_core, n_rescued_total):.1f}%)")

        summary[cfg_label] = {
            "n_papers":                n_papers,
            "parser_found":            n_parser_found_target,
            "caught":                  n_caught_pipeline,
            "uncaught":                n_uncaught_pipeline,
            "target_missing":          n_target_status_missing,
            "target_mismatch":         n_target_status_mismatch,
            "core_rescued_labelled":   n_uncaught_rescued_by_core,
            "core_rescued_total":      n_rescued_total,
            "parser_recall_pct":       _pct(n_parser_found_target, n_papers),
            "pipeline_recall_pct":     _pct(n_caught_pipeline,    n_papers),
            "lookup_recall_pct":       _pct(n_caught_pipeline,    n_parser_found_target),
        }

    # Headline cross-config table
    print()
    print("=" * 90)
    print("HEADLINE — PIPELINE RECALL ± CORE-RESCUE IMPACT")
    print("=" * 90)
    print(f"{'config':<14}{'parser%':>10}{'lookup%':>10}{'pipeline%':>12}"
          f"{'CORE-rescues':>14}{'  (of which labelled)':>22}")
    print("-" * 90)
    for cfg, s in summary.items():
        rescued_pct = (100.0 * s["core_rescued_labelled"] / s["core_rescued_total"]
                       if s["core_rescued_total"] else 0)
        print(f"{cfg:<14}{s['parser_recall_pct']:>9.1f}%"
              f"{s['lookup_recall_pct']:>9.1f}%"
              f"{s['pipeline_recall_pct']:>11.1f}%"
              f"{s['core_rescued_total']:>14}"
              f"   {s['core_rescued_labelled']:>4} ({rescued_pct:>4.1f}% of rescues)")

    # Interpretation note
    n_processed = len(rescue_by_key)
    if n_processed:
        rd_summary = (rundir / "scholar_core_rescue.json")
        d = json.loads(rd_summary.read_text())
        print(f"\nRescue progress: {d['summary']['ran']}/{d['summary']['total_pending']} refs processed.")
        print(f"  Headline CORE-rescue rate: "
              f"{d['summary']['rescued_match'] + d['summary']['rescued_fuzzy']}/{d['summary']['ran']} = "
              f"{_pct(d['summary']['rescued_match'] + d['summary']['rescued_fuzzy'], d['summary']['ran']):.1f}%")

    # Save summary JSON for the plot script
    out_path = rundir / "gold_truth_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSummary written to {out_path}")


if __name__ == "__main__":
    main()
