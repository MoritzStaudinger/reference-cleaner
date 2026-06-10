"""
Download the host paper PDFs listed in hallucitation_table7.json from the ACL
Anthology. Each row's `paper_id` (e.g. `2025.acl-long.42`) maps to
`https://aclanthology.org/<paper_id>.pdf`.

Usage:
    python benchmark/download_hosts.py             # download all 295
    python benchmark/download_hosts.py --limit 20  # only first 20
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT      = Path(__file__).parent
DATASET   = ROOT / "hallucitation_table7.json"
PDF_DIR   = ROOT / "pdfs"
ANTHOLOGY = "https://aclanthology.org/{}.pdf"


def _fetch(paper_id: str, dest: Path) -> tuple[str, str]:
    if dest.exists() and dest.stat().st_size > 1000:
        return paper_id, "skip"
    try:
        r = requests.get(ANTHOLOGY.format(paper_id), timeout=30, stream=True)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return paper_id, "ok"
    except Exception as e:
        return paper_id, f"fail: {e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    PDF_DIR.mkdir(exist_ok=True)
    entries = json.loads(DATASET.read_text())
    if args.limit:
        entries = entries[: args.limit]

    print(f"Downloading {len(entries)} PDFs to {PDF_DIR} (workers={args.workers})")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_fetch, e["paper_id"], PDF_DIR / f"{e['paper_id']}.pdf"): e
            for e in entries
        }
        ok = skip = fail = 0
        for f in as_completed(futures):
            pid, status = f.result()
            if status == "ok":   ok += 1
            elif status == "skip": skip += 1
            else: fail += 1; print(f"  ! {pid}: {status}")

    print(f"\nDone in {time.time()-t0:.1f}s — ok={ok} skip={skip} fail={fail}")


if __name__ == "__main__":
    main()
