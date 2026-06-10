"""
Reconstruct a results JSON + CSV + summary block for a benchmark run
that was killed mid-way.  Uses the per-paper intermediates that
smoke_test.py wrote (one file per completed paper) plus the per-paper
one-line summaries from the run's stdout log.

Usage:
    python benchmark/salvage_partial_run.py \\
        --log    benchmark/runs/latest/results_hybrid_llm_qwen-3_6-35b_cold.log \\
        --inter  benchmark/runs/latest/results_hybrid_llm_qwen-3_6-35b_cold_intermediates \\
        --out    benchmark/runs/latest/results_hybrid_llm_qwen-3_6-35b_cold.json

Recovers everything the original summary block computes EXCEPT
`http_calls` (which is only held in memory between batch resets and
was never written to disk).  All recall/precision/timing/category
numbers reconstruct exactly because lookup_results + statuses +
suspicious-categories are derived deterministically from the
intermediates that ARE on disk.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lookup import _cited_arxiv_id, _extract_arxiv_id, _extract_doi
from report import _overall_status, _categorise_suspicious, SUSPICIOUS_CATEGORIES

LOG_RE = re.compile(
    r"^---\s+\[(\d+)/(\d+)\]\s+#(\d+):\s+(\S+)\s+---\s*\n"
    r"\s+refs=\s*(\d+)\s+target_found=([YN])\s+labelled_status=(\S+)\s+"
    r"caught=(\S+)\s+parse=([\d.]+)s\s+lookup=([\d.]+)s",
    re.MULTILINE,
)


def _pct(p, q):
    return 100.0 * p / q if q else 0.0


def _wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5
    return ((c - s) / denom, (c + s) / denom)


def _summarise_times(values):
    if not values:
        return {}
    vs = sorted(values)
    n = len(vs)
    def _q(p):
        if n == 1:
            return vs[0]
        k = p * (n - 1)
        f = int(k)
        c = min(f + 1, n - 1)
        return vs[f] + (vs[c] - vs[f]) * (k - f)
    return {"n": n, "mean": round(sum(vs)/n, 2),
            "median": round(_q(0.50), 2),
            "p95": round(_q(0.95), 2),
            "max":  round(vs[-1], 2)}


def parse_log(log_path: Path) -> dict[str, dict]:
    """Parse per-paper one-liners from the smoke_test stdout log.
    Keyed by paper_id (which is unique in the corpus)."""
    text = log_path.read_text()
    by_pid: dict[str, dict] = {}
    for m in LOG_RE.finditer(text):
        n_paper, n_total, eid, pid, n_refs, tfnd, ls, caught, parse_s, lookup_s = m.groups()
        caught_v = {"True": True, "False": False, "None": None}.get(caught, None)
        by_pid[pid] = {
            "event_id":      int(eid),
            "paper_id":      pid,
            "n_refs":        int(n_refs),
            "target_found_in_extract": tfnd == "Y",
            "labelled_status": ls if ls != "n/a" else "n/a",
            "caught":        caught_v,
            "parse_s":       float(parse_s),
            "lookup_s":      float(lookup_s),
        }
    return by_pid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log",   required=True)
    ap.add_argument("--inter", required=True)
    ap.add_argument("--out",   required=True)
    args = ap.parse_args()

    log_path   = Path(args.log).resolve()
    inter_dir  = Path(args.inter).resolve()
    out_path   = Path(args.out).resolve()

    log_by_pid = parse_log(log_path)
    print(f"Log: parsed {len(log_by_pid)} per-paper lines from {log_path.name}")
    inter_files = sorted(inter_dir.glob("*.json"))
    print(f"Intermediates: {len(inter_files)} files in {inter_dir.name}")

    results: list[dict] = []
    for ip in inter_files:
        pid = ip.stem
        log_entry = log_by_pid.get(pid)
        if log_entry is None:
            print(f"  ⚠️  intermediate without log line: {pid}")
            continue
        try:
            d = json.loads(ip.read_text())
        except Exception as exc:
            print(f"  ⚠️  failed to read {pid}: {exc}")
            continue

        lookups = d.get("lookup_results") or []
        refs    = d.get("llm_refs") or d.get("parser_refs") or []
        statuses = [_overall_status(lr) for lr in lookups]
        categories = [_categorise_suspicious(lr) for lr in lookups]

        # Cited-identifier distribution — recomputable from refs
        id_dist = {"doi": 0, "arxiv": 0, "title_only": 0}
        cited_dist = {"arxiv": 0, "other": 0}
        for r in refs:
            if _extract_arxiv_id(r):
                id_dist["arxiv"] += 1
            elif _extract_doi(r):
                id_dist["doi"] += 1
            else:
                id_dist["title_only"] += 1
            if _cited_arxiv_id(r):
                cited_dist["arxiv"] += 1
            else:
                cited_dist["other"] += 1

        out = dict(log_entry)
        out["status_counts"] = {s: statuses.count(s) for s in
                                ("match", "missing", "mismatch", "pending")}
        out["suspicious_category_counts"] = {
            c: categories.count(c) for c in SUSPICIOUS_CATEGORIES
            if categories.count(c) > 0
        }
        out["labelled_category"] = (categories[d["target_idx"]]
                                    if d.get("target_idx") is not None else None)
        out["cited_id_distribution"]        = id_dist
        out["user_cited_arxiv_distribution"] = cited_dist
        out["paper_wall_s"] = round(out["parse_s"] + out["lookup_s"], 2)
        # llm_repair_s and http_calls were not persisted per-paper to disk
        # for this run — leave them absent rather than fabricate.
        results.append(out)

    # ----- Summary block -----
    n_papers = len(results)
    located  = [r for r in results if r.get("target_found_in_extract")]
    caught   = [r for r in results if r.get("caught") is True]
    parse_fail = [r for r in results if "error" in r]
    n_loc, n_caught = len(located), len(caught)

    parser_lo, parser_hi = _wilson(n_loc, n_papers)
    cond_lo, cond_hi     = _wilson(n_caught, n_loc)
    joint_lo, joint_hi   = _wilson(n_caught, n_papers)

    agg = {"match": 0, "missing": 0, "mismatch": 0, "pending": 0}
    for r in results:
        for k, v in (r.get("status_counts") or {}).items():
            agg[k] = agg.get(k, 0) + v
    total_refs = sum(agg.values())
    flagged = agg["missing"] + agg["mismatch"]

    cat_agg = {c: 0 for c in SUSPICIOUS_CATEGORIES}
    for r in results:
        for k, v in (r.get("suspicious_category_counts") or {}).items():
            cat_agg[k] = cat_agg.get(k, 0) + v

    labelled_cats: dict[str, int] = {}
    for r in results:
        if r.get("caught") and r.get("labelled_category"):
            labelled_cats[r["labelled_category"]] = labelled_cats.get(r["labelled_category"], 0) + 1

    id_agg = {"doi": 0, "arxiv": 0, "title_only": 0}
    for r in results:
        for k, v in (r.get("cited_id_distribution") or {}).items():
            id_agg[k] = id_agg.get(k, 0) + v

    timing = {
        "parse_s":      _summarise_times([r["parse_s"]      for r in results if "parse_s"      in r]),
        "lookup_s":     _summarise_times([r["lookup_s"]     for r in results if "lookup_s"     in r]),
        "paper_wall_s": _summarise_times([r["paper_wall_s"] for r in results if "paper_wall_s" in r]),
    }

    print()
    print("=" * 70)
    print(f"SALVAGED SUMMARY — {n_papers} papers (run was killed at {n_papers}/295)")
    print("=" * 70)
    print(f"Papers attempted (completed):  {n_papers}")
    print(f"Parse failures:                {len(parse_fail)}")
    print()
    print("RECALL DECOMPOSITION (Wilson 95% CI):")
    print(f"  Parser-recall:    {n_loc}/{n_papers}  = "
          f"{_pct(n_loc, n_papers):.1f}% ({parser_lo*100:.1f}–{parser_hi*100:.1f}%)")
    print(f"  Lookup-recall:    {n_caught}/{n_loc} = "
          f"{_pct(n_caught, n_loc):.1f}% ({cond_lo*100:.1f}–{cond_hi*100:.1f}%)")
    print(f"  Pipeline-recall:  {n_caught}/{n_papers}  = "
          f"{_pct(n_caught, n_papers):.1f}% ({joint_lo*100:.1f}–{joint_hi*100:.1f}%)")
    print()
    print(f"Across all {total_refs} refs:")
    for k, v in agg.items():
        print(f"  {k:<9} {v:>6}  ({_pct(v, total_refs):5.1f}%)")
    print(f"\nSuspicious-ref breakdown ({flagged} flagged refs):")
    for cat in SUSPICIOUS_CATEGORIES:
        n = cat_agg.get(cat, 0)
        if not n:
            continue
        print(f"  {cat:<24} {n:>5}  ({_pct(n, flagged):5.1f}%)")
    print()
    print(f"Cited-identifier distribution across {total_refs} refs:")
    print(f"  arXiv-ID:     {id_agg['arxiv']:>5}  ({_pct(id_agg['arxiv'], total_refs):5.1f}%)")
    print(f"  DOI:          {id_agg['doi']:>5}  ({_pct(id_agg['doi'], total_refs):5.1f}%)")
    print(f"  Title-only:   {id_agg['title_only']:>5}  ({_pct(id_agg['title_only'], total_refs):5.1f}%)")
    print()
    print("Timing:")
    print(f"  stage         n    mean   median    p95    max")
    for stage, t in timing.items():
        if not t:
            continue
        print(f"  {stage:<13} {t['n']:>4}  {t['mean']:>5.1f}   {t['median']:>5.1f}  "
              f"{t['p95']:>5.1f}  {t['max']:>5.1f}")

    # ----- Write JSON in the smoke_test.py schema for downstream tooling -----
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {"parser": "hybrid", "use_llm": True,
                   "llm_backend": "aqueduct", "llm_model": "qwen-3.6-35b",
                   "scholarly_budget": 0, "cold_cache": True,
                   "save_intermediates": True,
                   "intermediates_dir": str(inter_dir),
                   "n": n_papers,
                   "salvaged_from_partial_run": True,
                   "salvage_note": "run was killed at paper 265/295; this JSON "
                                   "was reconstructed from on-disk intermediates "
                                   "and the per-paper one-line log entries. "
                                   "http_calls and llm_repair_s are not recovered."},
        "elapsed_sec": None,
        "summary": {
            "papers": n_papers,
            "parse_failures": len(parse_fail),
            "hallucination_located": n_loc,
            "caught": n_caught,
            "parser_recall_pct":   round(_pct(n_loc, n_papers), 2),
            "parser_recall_ci95":  [round(parser_lo*100,2), round(parser_hi*100,2)],
            "lookup_recall_pct":   round(_pct(n_caught, n_loc), 2),
            "lookup_recall_ci95":  [round(cond_lo*100,2), round(cond_hi*100,2)],
            "pipeline_recall_pct": round(_pct(n_caught, n_papers), 2),
            "pipeline_recall_ci95":[round(joint_lo*100,2), round(joint_hi*100,2)],
            "labelled_recall_pct": round(_pct(n_caught, n_loc), 2),
            "labelled_recall_ci95": [round(cond_lo*100,2), round(cond_hi*100,2)],
            "agg_status_counts": agg,
            "aggregate_suspicious_pct": round(_pct(flagged, total_refs), 2),
            "suspicious_breakdown": {c: cat_agg.get(c, 0) for c in SUSPICIOUS_CATEGORIES if cat_agg.get(c, 0) > 0},
            "labelled_caught_by_category": labelled_cats,
            "cited_id_distribution": id_agg,
            "timing": timing,
        },
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nWritten {out_path}")

    # CSV companion — same columns smoke_test.py emits
    csv_path = out_path.with_suffix(".csv")
    fields = ["event_id", "paper_id", "n_refs", "target_found_in_extract",
              "labelled_status", "labelled_category", "caught",
              "parse_s", "llm_repair_s", "lookup_s", "paper_wall_s",
              "n_match", "n_missing", "n_mismatch",
              "n_refs_with_doi", "n_refs_with_arxiv_id", "n_refs_title_only",
              "n_wrong_arxiv_id", "n_wrong_doi",
              "n_title_authors_mismatch", "n_uncertain_match",
              "n_weak_title_match", "n_not_in_any_index", "error"]
    counted_cols = {
        "n_match","n_missing","n_mismatch",
        "n_refs_with_doi","n_refs_with_arxiv_id","n_refs_title_only",
        "n_wrong_arxiv_id","n_wrong_doi",
        "n_title_authors_mismatch","n_uncertain_match",
        "n_weak_title_match","n_not_in_any_index",
    }
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            sc  = r.get("status_counts") or {}
            cat = r.get("suspicious_category_counts") or {}
            id_d = r.get("cited_id_distribution") or {}
            row = {**{k: r.get(k) for k in fields if k not in counted_cols},
                   "n_match":   sc.get("match", 0),
                   "n_missing": sc.get("missing", 0),
                   "n_mismatch":sc.get("mismatch", 0),
                   "n_refs_with_doi":           id_d.get("doi", 0),
                   "n_refs_with_arxiv_id":      id_d.get("arxiv", 0),
                   "n_refs_title_only":         id_d.get("title_only", 0),
                   "n_wrong_arxiv_id":          cat.get("wrong_arxiv_id", 0),
                   "n_wrong_doi":               cat.get("wrong_doi", 0),
                   "n_title_authors_mismatch":  cat.get("title_authors_mismatch", 0),
                   "n_uncertain_match":         cat.get("uncertain_match", 0),
                   "n_weak_title_match":        cat.get("weak_title_match", 0),
                   "n_not_in_any_index":        cat.get("not_in_any_index", 0)}
            w.writerow(row)
    print(f"Written {csv_path}")


if __name__ == "__main__":
    main()
