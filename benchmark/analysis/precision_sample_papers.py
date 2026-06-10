"""
Sample 50 papers, list every ref the pipeline flagged in those papers,
and write a CSV that's ready for manual annotation.

The goal: HalluCitation only labels ONE hallucinated citation per paper.
That gives us recall-against-the-labelled-set but tells us nothing
about how many OTHER bad citations we caught (true positives) or how
many of our flags are false positives.

Manual annotation on a paper-sampled basis gives us the missing data:
for each flagged ref in the sampled papers, the annotator judges
whether the ref is actually wrong.  From the judgements we can then
compute:
   - **Precision**:  TP / (TP + FP)
   - **Per-paper precision**: same per-paper to get a CI
   - **Recall on the sampled papers** (an honest version, not just
     against HalluCitation's single labelled hallucination per paper)

Strategy:
   - Sample 50 papers uniformly from each config's results.
   - For each sampled paper, write ONE CSV row per flagged ref (i.e.
     status ∈ {missing, mismatch}), including all info needed to judge.
   - Annotator fills the `judgement` column with TP / FP / skip.
   - Run --score on the filled CSV to compute precision + per-paper
     recall + CIs.

Usage:
    # Pick the papers and emit the annotation CSV
    python benchmark/analysis/precision_sample_papers.py sample \\
        --from   benchmark/runs/latest \\
        --config hybrid_llm_qwen-3_6-35b_cold \\
        --n 50 \\
        --out benchmark/runs/latest/manual_precision_50papers.csv

    # [annotate the CSV by hand]

    # Compute precision + per-paper recall
    python benchmark/analysis/precision_sample_papers.py score \\
        benchmark/runs/latest/manual_precision_50papers.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from report import _overall_status, _categorise_suspicious

DEFAULT_CONFIG = "hybrid_llm_qwen-3_6-35b_cold"


def _best_found(lr: dict) -> tuple[str, dict]:
    if not lr:
        return "", {}
    cands = [(s, lr.get(s, {})) for s in
             ("doi_org", "semantic_scholar", "openalex", "crossref",
              "arxiv", "dblp", "acl_anthology")
             if lr.get(s, {}).get("status") == "found"]
    if not cands:
        return "", {}
    s, r = max(cands, key=lambda kv: (kv[1].get("similarity") or 0.0))
    return s, r


def _sources_summary(lr: dict) -> str:
    parts = []
    for s in ("doi_org", "semantic_scholar", "openalex", "crossref",
              "arxiv", "dblp", "acl_anthology"):
        r = (lr or {}).get(s, {})
        st = r.get("status", "")
        if st == "found":
            sim = r.get("similarity")
            parts.append(f"{s}={r.get('label','?')}/{sim:.0%}"
                         if sim is not None else f"{s}={r.get('label','?')}")
        elif st == "not_found":
            parts.append(f"{s}=nf")
        elif st == "error":
            parts.append(f"{s}=err")
    return " ".join(parts)


def _is_judgeable_ref(ref: dict) -> tuple[bool, str]:
    """
    Return (keep, reason).  False = this ref is not a clean-enough
    citation for a human annotator to fairly judge as TP/FP.

    The annotation task is "is THIS citation correct?".  That task is
    only well-defined when the ref has a recognisable title in its
    structured field — without one, the annotator is either:
      (a) being asked to do extraction work that isn't their job, or
      (b) judging body text fragments mis-classified as citations,
          which conflates parser robustness with pipeline precision.

    Both are interesting but separate; we exclude them from precision
    annotation and report parser-failure rates separately.

    Rejection criteria:
      - empty title
      - title < 15 chars OR title is a year token
      - title is just stop-word noise (single word, all common words)
    """
    title   = (ref.get("title")   or "").strip()
    if not title:
        return False, "no_title"
    if title[:4].isdigit() and len(title) <= 6:
        return False, "year_as_title"
    if len(title) < 15:
        return False, "title_too_short"
    # Single short word — likely a name fragment mis-extracted as title
    if " " not in title and len(title) < 25:
        return False, "title_single_word"
    return True, "has_title"


def cmd_sample(args: argparse.Namespace) -> None:
    rundir   = Path(args.from_dir).resolve()
    inter_dir = rundir / f"results_{args.config}_intermediates"
    if not inter_dir.is_dir():
        print(f"No such intermediates directory: {inter_dir}")
        sys.exit(1)

    # Gold set so we can mark which ref (if any) is the HalluCitation
    # labelled hallucination — annotator can use that as a reference
    # point for their own judgement.
    gold = {g["paper_id"]: g
            for g in json.loads((ROOT / "benchmark/hallucitation_table7.json").read_text())}

    rng = random.Random(args.seed)
    all_papers = sorted(p.stem for p in inter_dir.glob("*.json"))
    rng.shuffle(all_papers)
    sampled = all_papers[: args.n]
    print(f"Sampled {len(sampled)} papers (seed={args.seed}) from "
          f"{args.config} ({len(all_papers)} total)")

    rows: list[dict] = []
    n_flagged_total = 0
    n_dropped       = 0
    drop_reasons: dict[str, int] = {}
    for pid in sampled:
        try:
            d = json.loads((inter_dir / f"{pid}.json").read_text())
        except Exception:
            continue
        refs    = d.get("llm_refs") or d.get("parser_refs") or []
        lookups = d.get("lookup_results") or []
        target_idx = d.get("target_idx")
        target_title = (gold.get(pid, {}) or {}).get("hallucinated_citation", "")[:200]

        for i, (ref, lr) in enumerate(zip(refs, lookups)):
            status = _overall_status(lr)
            if status not in ("missing", "mismatch"):
                continue
            n_flagged_total += 1
            # Filter out refs the annotator can't fairly judge (parser /
            # LLM extraction failures with no recoverable title).  Unless
            # --no-filter is set.
            if not args.no_filter:
                keep, reason = _is_judgeable_ref(ref)
                if not keep:
                    n_dropped += 1
                    drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
                    continue
            best_src, best_r = _best_found(lr)
            rows.append({
                "paper_id":      pid,
                "ref_index":     i,
                "is_gold_labelled_hallucination":
                                 "Y" if i == target_idx else "",
                "pipeline_status": status,
                "category":      _categorise_suspicious(lr) or "",
                "cited_title":   (ref.get("title") or "")[:240],
                "cited_authors": (ref.get("authors") or "")[:240],
                "cited_venue":   (ref.get("venue") or "")[:200],
                "cited_year":    ref.get("year") or "",
                "raw_excerpt":   (ref.get("raw") or "")[:400].replace("\n", " "),
                "best_match_source": best_src,
                "best_match_title":  (best_r.get("found_title") or "")[:200],
                "best_match_sim":    f"{best_r.get('similarity'):.2f}"
                                     if best_r.get("similarity") is not None else "",
                "best_match_authors": (best_r.get("found_authors") or "")[:200],
                "all_sources":   _sources_summary(lr),
                "gold_labelled_hallucinated_citation_for_paper":
                                 target_title,
                # Annotation columns — left empty for human input
                "judgement":     "",
                "notes":         "",
            })

    fields = ["paper_id", "ref_index", "is_gold_labelled_hallucination",
              "pipeline_status", "category",
              "cited_title", "cited_authors", "cited_venue", "cited_year",
              "raw_excerpt",
              "best_match_source", "best_match_title", "best_match_sim",
              "best_match_authors", "all_sources",
              "gold_labelled_hallucinated_citation_for_paper",
              "judgement", "notes"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\nFlagged refs (raw): {n_flagged_total} in {len(sampled)} papers")
    if not args.no_filter:
        print(f"Dropped as unjudgeable: {n_dropped}")
        for reason, n in sorted(drop_reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {reason:<24} {n}")
    print(f"Rows written: {len(rows)}  (mean {len(rows)/len(sampled):.1f}/paper)")
    print(f"Written to {out_path}")
    print()
    print("Now: open the CSV in a spreadsheet.  For each row, look at:")
    print("  - cited_title / cited_authors / raw_excerpt   (the ref as cited)")
    print("  - best_match_title / best_match_authors       (what we found)")
    print("  - all_sources                                  (what each DB said)")
    print()
    print("Then fill the `judgement` column with one of:")
    print("  TP   — citation is actually wrong / unfindable (we were right)")
    print("  FP   — citation is actually correct (we cried wolf)")
    print("  skip — can't decide / not enough info")
    print()
    print("Save the file and run:")
    print(f"  python {Path(__file__).name} score {out_path}")


def cmd_exhaustive(args: argparse.Namespace) -> None:
    """
    Emit ALL refs (not just flagged ones) for N papers.  Unlocks
    natural-data recall and false-negative rate — metrics the
    flagged-only sample cannot produce.

    Papers are picked from the same shuffled sequence as `sample`
    (same seed → first --n of them), so a 20-paper exhaustive run
    overlaps fully with a 50-paper sample's first 20 papers.
    """
    rundir    = Path(args.from_dir).resolve()
    inter_dir = rundir / f"results_{args.config}_intermediates"
    if not inter_dir.is_dir():
        print(f"No such intermediates directory: {inter_dir}")
        sys.exit(1)

    gold = {g["paper_id"]: g
            for g in json.loads((ROOT / "benchmark/hallucitation_table7.json").read_text())}

    rng = random.Random(args.seed)
    all_papers = sorted(p.stem for p in inter_dir.glob("*.json"))
    rng.shuffle(all_papers)
    sampled = all_papers[: args.n]
    print(f"Picked {len(sampled)} papers (seed={args.seed}) from "
          f"{args.config} ({len(all_papers)} total)")
    print(f"Paper IDs:")
    for pid in sampled:
        print(f"  {pid}")

    rows: list[dict] = []
    n_total = 0
    n_dropped = 0
    drop_reasons: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for pid in sampled:
        try:
            d = json.loads((inter_dir / f"{pid}.json").read_text())
        except Exception:
            continue
        refs       = d.get("llm_refs") or d.get("parser_refs") or []
        lookups    = d.get("lookup_results") or []
        target_idx = d.get("target_idx")
        target_title = (gold.get(pid, {}) or {}).get("hallucinated_citation", "")[:200]

        for i, (ref, lr) in enumerate(zip(refs, lookups)):
            n_total += 1
            status = _overall_status(lr)
            status_counts[status] = status_counts.get(status, 0) + 1

            if not args.no_filter:
                keep, reason = _is_judgeable_ref(ref)
                if not keep:
                    n_dropped += 1
                    drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
                    continue

            best_src, best_r = _best_found(lr)
            rows.append({
                "paper_id":      pid,
                "ref_exists_human": "",
                "Correct_match": "",
                "comment":       "",
                "ref_index":     i,
                "is_gold_labelled_hallucination":
                                 "Y" if i == target_idx else "",
                "pipeline_status": status,
                "category":      _categorise_suspicious(lr) or "",
                "cited_title":   (ref.get("title") or "")[:240],
                "cited_authors": (ref.get("authors") or "")[:240],
                "cited_venue":   (ref.get("venue") or "")[:200],
                "cited_year":    ref.get("year") or "",
                "raw_excerpt":   (ref.get("raw") or "")[:400].replace("\n", " "),
                "best_match_source": best_src,
                "best_match_title":  (best_r.get("found_title") or "")[:200],
                "best_match_sim":    f"{best_r.get('similarity'):.2f}"
                                     if best_r.get("similarity") is not None else "",
                "best_match_authors": (best_r.get("found_authors") or "")[:200],
                "all_sources":   _sources_summary(lr),
                "gold_labelled_hallucinated_citation_for_paper":
                                 target_title,
                "judgement":     "",
                "notes":         "",
            })

    fields = ["paper_id",
              "ref_exists_human", "Correct_match", "comment",
              "ref_index", "is_gold_labelled_hallucination",
              "pipeline_status", "category",
              "cited_title", "cited_authors", "cited_venue", "cited_year",
              "raw_excerpt",
              "best_match_source", "best_match_title", "best_match_sim",
              "best_match_authors", "all_sources",
              "gold_labelled_hallucinated_citation_for_paper",
              "judgement", "notes"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print()
    print(f"Total refs across {len(sampled)} papers: {n_total}  "
          f"(mean {n_total/len(sampled):.1f}/paper)")
    print(f"Pipeline status distribution:")
    for st, n in sorted(status_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {st:<14} {n:<4}  ({100*n/n_total:.0f}%)")
    if not args.no_filter:
        print(f"Dropped as unjudgeable: {n_dropped}")
        for reason, n in sorted(drop_reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {reason:<24} {n}")
    print(f"Rows written: {len(rows)}")
    print(f"Written to {out_path}")
    print()
    print("Verify each ref via the Streamlit app:")
    print("  streamlit run app.py")
    print("Open each paper above, look at the highlighted refs, fill the CSV with:")
    print("  ref_exists_human  Y / N      (does the cited paper exist at all?)")
    print("  Correct_match     correct / mismatch / missing")
    print("  comment           free text")


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5
    return ((c - s) / d, (c + s) / d)


def cmd_score(args: argparse.Namespace) -> None:
    path = Path(args.csv)
    with path.open() as f:
        rows = list(csv.DictReader(f))

    judged = [r for r in rows if r.get("judgement", "").strip().upper() in ("TP", "FP")]
    tp = sum(1 for r in judged if r["judgement"].strip().upper() == "TP")
    fp = sum(1 for r in judged if r["judgement"].strip().upper() == "FP")
    skip = sum(1 for r in rows if r.get("judgement", "").strip().lower() == "skip")
    blank = sum(1 for r in rows if not r.get("judgement", "").strip())
    print(f"Rows: {len(rows)}  judged: {len(judged)} (TP={tp}, FP={fp})  "
          f"skip={skip}  blank={blank}")

    if not judged:
        print("\nNo judgements yet — open the CSV and fill the `judgement` column.")
        return

    lo, hi = _wilson(tp, len(judged))
    print(f"\nPrecision (on judged flags): {tp}/{len(judged)} = "
          f"{100*tp/len(judged):.1f}%  (95% CI {lo*100:.1f}–{hi*100:.1f}%)")

    # Per-paper TP/FP counts — supports computing per-paper recall if
    # the annotator also marks gold-labelled rows.
    by_paper: dict[str, dict[str, int]] = {}
    for r in judged:
        pid = r["paper_id"]
        d = by_paper.setdefault(pid, {"TP": 0, "FP": 0, "gold_TP": 0})
        v = r["judgement"].strip().upper()
        d[v] += 1
        if v == "TP" and r.get("is_gold_labelled_hallucination", "").upper() == "Y":
            d["gold_TP"] += 1

    n_papers = len(by_paper)
    n_papers_with_gold_caught = sum(1 for d in by_paper.values() if d["gold_TP"] > 0)
    print(f"\nPer-paper stats (across {n_papers} judged papers):")
    print(f"  Papers where annotator confirmed at least one TP:  "
          f"{sum(1 for d in by_paper.values() if d['TP'] > 0)}/{n_papers}")
    print(f"  Papers where gold-labelled hallucination was confirmed as TP:  "
          f"{n_papers_with_gold_caught}/{n_papers}")

    # By category breakdown — small samples per category but useful
    from collections import defaultdict
    by_cat: dict[str, dict[str, int]] = defaultdict(lambda: {"TP": 0, "FP": 0})
    for r in judged:
        by_cat[r.get("category", "")][r["judgement"].strip().upper()] += 1
    if len(by_cat) > 1:
        print(f"\nPer-category precision (small samples, indicative only):")
        for cat, d in sorted(by_cat.items(), key=lambda kv: -(kv[1]["TP"] + kv[1]["FP"])):
            n = d["TP"] + d["FP"]
            if n < 2:
                continue
            lo, hi = _wilson(d["TP"], n)
            print(f"  {cat:<26} {d['TP']}/{n} = {100*d['TP']/n:.0f}%  "
                  f"(95% CI {lo*100:.0f}–{hi*100:.0f}%)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="Pick papers + emit annotation CSV")
    s.add_argument("--from", dest="from_dir", required=True,
                   help="Run directory containing results_*_intermediates dirs")
    s.add_argument("--config", default=DEFAULT_CONFIG,
                   help=f"Which config to sample from (default {DEFAULT_CONFIG})")
    s.add_argument("--n", type=int, default=50, help="Number of papers (default 50)")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--no-filter", action="store_true",
                   help="Disable the title-quality filter and include "
                        "parser-extraction-failure rows.  Default behaviour "
                        "drops rows where the title is missing or too short "
                        "for a human to fairly judge.")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_sample)

    e = sub.add_parser("exhaustive",
                       help="Emit ALL refs for N papers (unlocks recall + FN rate)")
    e.add_argument("--from", dest="from_dir", required=True)
    e.add_argument("--config", default=DEFAULT_CONFIG)
    e.add_argument("--n", type=int, default=20)
    e.add_argument("--seed", type=int, default=42,
                   help="Use same seed as `sample` to overlap with that subset")
    e.add_argument("--no-filter", action="store_true")
    e.add_argument("--out", required=True)
    e.set_defaults(func=cmd_exhaustive)

    g = sub.add_parser("score", help="Read filled-in judgements")
    g.add_argument("csv")
    g.set_defaults(func=cmd_score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
