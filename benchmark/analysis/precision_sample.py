"""
Sample flagged refs uniformly at random for manual precision validation.

Reads the per-paper intermediates that smoke_test.py wrote alongside the
results JSON, picks N refs whose overall status is "missing" or
"mismatch", and writes a CSV with one row per ref ready for human review:

    paper_id  ref_index  category  cited_title  cited_authors  raw_excerpt
    found_title  found_authors  sources_consulted  judgement  notes

Open the CSV in a spreadsheet, fill the `judgement` column with
"TP" / "FP" / "skip", and run:

    python benchmark/analysis/precision_sample.py --score sample.csv

to compute precision with a Wilson 95% CI.

Strategy:
    --sample 50 --from RUNDIR     # write sample.csv
    [open in sheet, judge each, save]
    --score sample.csv            # print precision + CI

Honest precision is the only number that grounds the "X% of refs were
flagged" claim — every other approach reduces to "trust that the gold
labels are complete", which HalluCitation Table 7 explicitly isn't.
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


def _load_run(rundir: Path) -> list[dict]:
    """Yield (paper_id, ref_index, ref, lr) tuples for every flagged ref
    in every results_*.json file in the given run directory."""
    items: list[dict] = []
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
            paper_id = paper_path.stem
            refs    = d.get("llm_refs") or d.get("parser_refs") or []
            lookups = d.get("lookup_results") or []
            for i, (ref, lr) in enumerate(zip(refs, lookups)):
                status = _overall_status(lr)
                if status not in ("missing", "mismatch"):
                    continue
                items.append({
                    "config":     config,
                    "paper_id":   paper_id,
                    "ref_index":  i,
                    "category":   _categorise_suspicious(lr) or "unclassified",
                    "status":     status,
                    "cited_title":   (ref.get("title") or "")[:240],
                    "cited_authors": (ref.get("authors") or "")[:240],
                    "raw_excerpt":   (ref.get("raw") or "")[:400].replace("\n", " "),
                    "found_title":   _best_found_title(lr),
                    "found_authors": _best_found_authors(lr),
                    "sources_consulted": _sources_summary(lr),
                })
    return items


def _best_found(lr):
    cands = [(s, lr.get(s, {})) for s in
             ("doi_org","semantic_scholar","openalex","crossref","arxiv","dblp","acl_anthology")
             if lr.get(s, {}).get("status") == "found"]
    if not cands:
        return None, None
    s, r = max(cands, key=lambda kv: kv[1].get("similarity") or 0.0)
    return s, r

def _best_found_title(lr):
    _, r = _best_found(lr)
    return (r.get("found_title") if r else "") or ""

def _best_found_authors(lr):
    _, r = _best_found(lr)
    return ((r.get("found_authors") if r else "") or "")[:240]

def _sources_summary(lr):
    parts = []
    for s in ("doi_org","semantic_scholar","openalex","crossref","arxiv","dblp","acl_anthology"):
        r = lr.get(s, {})
        st = r.get("status", "")
        if st == "found":
            lbl = r.get("label", "?")
            sim = r.get("similarity")
            parts.append(f"{s}={lbl}/{sim:.0%}" if sim is not None else f"{s}={lbl}")
        elif st == "not_found":
            parts.append(f"{s}=nf")
        elif st == "error":
            parts.append(f"{s}=err")
    return " ".join(parts)


def cmd_sample(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    rundir = Path(args.from_dir).resolve()
    items = _load_run(rundir)
    # Deduplicate by (paper_id, ref_index) across configs so the same
    # flagged ref isn't sampled twice with different config labels.
    seen, dedup = set(), []
    for it in items:
        key = (it["paper_id"], it["ref_index"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(it)
    if args.stratified:
        # Stratify by category so rare failure modes (wrong_doi etc.) get
        # represented in the sample — straight uniform sampling will be
        # dominated by the most common bucket.
        from collections import defaultdict
        buckets: dict[str, list] = defaultdict(list)
        for it in dedup:
            buckets[it["category"]].append(it)
        per_bucket = max(1, args.sample // max(1, len(buckets)))
        sample: list = []
        for cat, pool in buckets.items():
            random.shuffle(pool)
            sample.extend(pool[:per_bucket])
        random.shuffle(sample)
        sample = sample[:args.sample]
        print(f"Stratified sample: drew {len(sample)} from {len(buckets)} categories"
              f" ({per_bucket}/category)")
    else:
        random.shuffle(dedup)
        sample = dedup[:args.sample]
        print(f"Uniform sample: drew {len(sample)} from {len(dedup)} flagged refs")

    out = Path(args.out)
    fields = ["config", "paper_id", "ref_index", "category", "status",
              "cited_title", "cited_authors", "raw_excerpt",
              "found_title", "found_authors", "sources_consulted",
              "judgement", "notes"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for it in sample:
            it = {**it, "judgement": "", "notes": ""}
            w.writerow({k: it.get(k, "") for k in fields})
    print(f"Written {out}")
    print()
    print("Now: open the CSV, fill the `judgement` column with one of:")
    print("    TP   — citation is actually wrong (we were right)")
    print("    FP   — citation is actually correct (we cried wolf)")
    print("    skip — can't decide / not enough info")
    print("Save it, then run:")
    print(f"    python {Path(__file__).name} score {out}")


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z*z/n
    c = p + z*z/(2*n)
    s = z * (p*(1-p)/n + z*z/(4*n*n))**0.5
    return ((c - s) / denom, (c + s) / denom)


def cmd_score(args: argparse.Namespace) -> None:
    path = Path(args.csv)
    with path.open() as f:
        rows = list(csv.DictReader(f))

    tp = sum(1 for r in rows if r.get("judgement","").strip().upper() == "TP")
    fp = sum(1 for r in rows if r.get("judgement","").strip().upper() == "FP")
    skip = sum(1 for r in rows if r.get("judgement","").strip().lower() == "skip")
    blank = sum(1 for r in rows if not r.get("judgement","").strip())
    judged = tp + fp
    print(f"Rows: {len(rows)}  judged: {judged} (TP={tp}, FP={fp})  skip={skip}  blank={blank}")
    if judged == 0:
        print("No judgements yet — nothing to score.")
        return
    lo, hi = _wilson(tp, judged)
    print(f"\nManual precision: {tp}/{judged} = {100*tp/judged:.1f}% "
          f"(95% CI {lo*100:.1f}–{hi*100:.1f}%)")

    # Per-category breakdown — answers "what are we *good* at flagging?"
    from collections import defaultdict
    by_cat: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        j = r.get("judgement","").strip().upper()
        if j not in ("TP","FP"):
            continue
        by_cat[r["category"]].append(j == "TP")
    if len(by_cat) > 1:
        print("\nPer-category precision (limited statistical power — small n):")
        for cat, hits in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
            n = len(hits)
            k = sum(hits)
            lo, hi = _wilson(k, n)
            print(f"  {cat:<24} {k}/{n} = {100*k/n:.0f}% "
                  f"({lo*100:.0f}–{hi*100:.0f}%)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="Draw a random sample to annotate")
    s.add_argument("--from", dest="from_dir", required=True,
                   help="Run directory containing results_*.json + intermediates dirs")
    s.add_argument("--sample", type=int, default=50,
                   help="How many refs to sample (default 50)")
    s.add_argument("--stratified", action="store_true",
                   help="Stratify the sample by failure-mode category so rare ones appear")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", default="precision_sample.csv")
    s.set_defaults(func=cmd_sample)

    g = sub.add_parser("score", help="Read filled-in judgements and compute precision")
    g.add_argument("csv")
    g.set_defaults(func=cmd_score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
