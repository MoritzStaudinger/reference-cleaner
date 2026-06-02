# ReferenceCleaner

A tool for extracting references from PDFs and validating them against academic databases. Built to surface hallucinated citations (wrong paper cited, fabricated DOIs, venue mismatches) before they reach reviewers.

## Pipeline

1. **Parse** the PDF — `pymupdf4llm`, `docling`, or GROBID.
2. **Extract** structured fields (title, authors, year, venue, DOI, URL).
3. **Validate** each reference through a tiered API pipeline:
   - **S2 batch pre-warm** — every ref carrying a DOI or arXiv ID is collapsed into one `POST /paper/batch` request to Semantic Scholar; the per-ref pipeline then reads those results from cache.
   - **Phase 1 — Source router.** Pick the most likely API based on identifiers/venue: arXiv ID → arxiv, DOI → Crossref, CS-conference venue → DBLP API, book signal → Open Library. Identifier-based routes are authoritative — a `not_found` answer stops the pipeline.
   - **Phase 2 — Escalation.** Refs the router couldn't classify, or that the routed (non-identifier) source missed, fan out to the remaining APIs.
   - **Phase 3 — ACL Anthology backfill** from S2 external IDs / ACL DOIs.
   - **Phase 4 — Open Library** only when venue/publisher signals say the ref is a book.
4. **Score** each reference (verified / venue mismatch / wrong DOI / wrong paper / not found) and render a Streamlit UI plus a Markdown report.

All HTTP calls go through a single `requests.Session()` with a 20-connection keep-alive pool, so TCP+TLS handshakes amortise across the ~200 calls per paper. A small SQLite cache at `data/cache.sqlite` retains `found` / `not_found` results across runs.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # set S2_API_KEY, ANTHROPIC_API_KEY

streamlit run app.py
```

No local data dumps are required — everything is queried live against the source APIs.

## Benchmark

`benchmark/` contains a 295-paper labelled benchmark derived from Sakai et al. 2026 ("HalluCitation Matters"). Each row pairs a host PDF on the ACL Anthology with one known fabricated reference inside it.

```bash
python benchmark/download_hosts.py    # ~570 MB of host PDFs
python benchmark/smoke_test.py --n 5  # run pipeline on 5 papers
```

## Limitations

- **Network dependency.** With local indexes removed, every reference now goes through a live API. Cold-cache runs are slower than they were with on-disk DBLP/ACL dumps, but the persistent cache means repeat runs hit only the truly new refs.
- **Rate limits.** Concurrency is capped at 5 refs in flight per paper. Bumping it higher triggers S2's 429 + exponential backoff and ends up *slower*. If you have a higher-tier S2 key, raise `max_concurrent` in `lookup_all`.
- **Title-only matching for fallback search.** When refs lack a DOI/arXiv ID, every source falls back to title search, which is noisier than ID-based lookup. Author/venue confirmation kicks in at the composite-label step.
- **Parser quality.** Reference extraction with `pymupdf4llm` occasionally misclassifies year tokens (e.g. `2024b`) as titles. A built-in junk filter catches the most common patterns; GROBID produces cleaner output but needs a running service.
