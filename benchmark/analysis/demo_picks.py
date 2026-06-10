"""
Pick papers from the benchmark that produce visually dramatic outputs,
suitable for a live conference demo.

For each paper, score it on three demo-readiness criteria:
  - has a wrong_arxiv_id catch (most photogenic — explicit red badge)
  - has a title_authors_mismatch catch (the "looks like the right paper
    but authors disagree" story — second-best demo moment)
  - has at least one clean green-across-the-board match (so the demo
    isn't all red)
  - reasonable size (10-60 refs — enough to look real, not so many the
    audience can't track what's happening)
  - lookup time under p75 (so the audience doesn't watch a spinner)

Prints a short-list ranked by total score.  Use the top 3-4 as your
pre-warmed demo papers.

Usage:
    python benchmark/analysis/demo_picks.py --from <rundir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from report import _overall_status, _categorise_suspicious


def _score_paper(d: dict) -> tuple[float, dict]:
    """Returns (score, signal_dict) for one paper's intermediates dict."""
    refs    = d.get("llm_refs") or d.get("parser_refs") or []
    lookups = d.get("lookup_results") or []
    n_refs  = len(refs)

    score = 0.0
    sig: dict = {"n_refs": n_refs, "has_wrong_arxiv": 0, "has_wrong_doi": 0,
                 "has_title_authors": 0, "has_clean_match": 0,
                 "n_match": 0, "n_missing": 0, "n_mismatch": 0}

    for lr in lookups:
        status = _overall_status(lr)
        cat    = _categorise_suspicious(lr)
        if status == "match":
            sig["n_match"] += 1
            sig["has_clean_match"] = 1
        elif status == "missing":
            sig["n_missing"] += 1
        elif status == "mismatch":
            sig["n_mismatch"] += 1
        if cat == "wrong_arxiv_id":  sig["has_wrong_arxiv"] += 1
        if cat == "wrong_doi":       sig["has_wrong_doi"]   += 1
        if cat == "title_authors_mismatch": sig["has_title_authors"] += 1

    # Scoring rubric — weights tuned so a paper with one of each
    # photogenic catch + a healthy denominator + size in the sweet spot
    # scores ~10.  Adjust if you disagree.
    if sig["has_wrong_arxiv"]:   score += 4.0    # the killer demo moment
    if sig["has_wrong_doi"]:     score += 2.5
    if sig["has_title_authors"]: score += 2.0
    if sig["has_clean_match"]:   score += 1.0    # need some green for balance
    # Size sweet spot
    if 10 <= n_refs <= 60:       score += 1.5
    elif n_refs > 60:            score += 0.5
    # Diversity — penalise all-red or all-green papers
    if sig["n_match"] and (sig["n_missing"] or sig["n_mismatch"]):
        score += 1.0
    return score, sig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="from_dir", required=True)
    ap.add_argument("--config", default="hybrid_llm_cold",
                    help="Which config's intermediates to score (default hybrid_llm_cold)")
    ap.add_argument("--top", type=int, default=10,
                    help="How many papers to print (default 10)")
    args = ap.parse_args()

    rundir = Path(args.from_dir).resolve()
    inter_dir = rundir / f"results_{args.config}_intermediates"
    if not inter_dir.is_dir():
        print(f"No intermediates directory at {inter_dir}")
        sys.exit(1)

    results_path = rundir / f"results_{args.config}.json"
    paper_meta: dict[str, dict] = {}
    if results_path.exists():
        try:
            for r in json.loads(results_path.read_text()).get("results", []):
                paper_meta[r["paper_id"]] = r
        except Exception:
            pass

    scored: list[tuple[float, str, dict]] = []
    for p in sorted(inter_dir.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        score, sig = _score_paper(d)
        scored.append((score, p.stem, sig))

    scored.sort(key=lambda t: -t[0])

    print(f"Top {args.top} demo-ready papers (config={args.config}):\n")
    print(f"  {'#':>3}  {'paper_id':<30}{'score':>6}{'refs':>5}"
          f"{'wAID':>5}{'wDOI':>5}{'T/A':>5}{'lkup':>7}")
    for i, (score, pid, sig) in enumerate(scored[:args.top], 1):
        meta = paper_meta.get(pid, {})
        lk   = meta.get("lookup_s", "")
        lk_s = f"{lk}s" if isinstance(lk, (int, float)) else "-"
        print(f"  {i:>3}  {pid:<30}"
              f"{score:>6.1f}"
              f"{sig['n_refs']:>5}"
              f"{sig['has_wrong_arxiv']:>5}"
              f"{sig['has_wrong_doi']:>5}"
              f"{sig['has_title_authors']:>5}"
              f"{lk_s:>7}")

    print()
    print("Columns: score | refs | wAID=wrong-arxiv-id catches | wDOI | "
          "T/A=title-authors mismatch | lkup=wall lookup seconds")
    print("\nTo pre-warm the cache for a demo paper, just run lookup_all() on it once "
          "before the demo (the SQLite cache persists).")


if __name__ == "__main__":
    main()
