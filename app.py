import html as _html
import json
import streamlit as st
import tempfile
import os
from typing import Any

from dotenv import load_dotenv
load_dotenv()

from parsers import (
    dedupe_references,
    extract_references_docling,
    extract_references_pymupdf4llm,
    extract_references_grobid,
    extract_references_hybrid,
    grobid_is_available,
)
from highlighter import highlight_references
from lookup import lookup_all

st.set_page_config(
    page_title="Reference Cleaner",
    page_icon="📚",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* Reference cards */
.ref-card {
    border-left: 5px solid;
    border-radius: 4px;
    padding: 10px 14px;
    margin: 6px 0 2px 0;
    font-family: sans-serif;
    font-size: 0.9em;
}
.ref-card.match   { border-color: #28a745; background: rgba(40,  167,  69, 0.08); }
.ref-card.mismatch{ border-color: #dc3545; background: rgba(220,  53,  69, 0.08); }
.ref-card.missing { border-color: #e6a817; background: rgba(255, 193,   7, 0.08); }
.ref-card.pending { border-color: #868e96; background: rgba(108, 117, 125, 0.06); }

.ref-card summary {
    cursor: pointer;
    font-weight: 600;
    font-size: 1em;
    list-style: none;         /* hide default arrow in Firefox */
}
.ref-card summary::-webkit-details-marker { display: none; }
.ref-card summary::before {
    content: "▶ ";
    font-size: 0.7em;
    vertical-align: middle;
}
details[open] > summary::before { content: "▼ "; }

.ref-raw {
    background: rgba(0,0,0,0.06);
    border-radius: 3px;
    padding: 6px 8px;
    font-family: monospace;
    font-size: 0.82em;
    white-space: pre-wrap;
    word-break: break-word;
    margin-top: 8px;
}
.ref-fields {
    margin-top: 6px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2px 12px;
    font-size: 0.88em;
}
.ref-lookup {
    margin-top: 8px;
    padding-top: 6px;
    border-top: 1px solid rgba(128,128,128,0.2);
    font-size: 0.88em;
}
.ref-lookup ul { margin: 4px 0 0 0; padding-left: 18px; }
.ref-lookup li { margin: 2px 0; }
.ref-lookup a  { color: inherit; }
/* tighten up the Go-to button */
div[data-testid="stButton"] > button[kind="secondary"] {
    padding: 2px 10px;
    font-size: 0.8em;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("📚 Reference Cleaner")
st.caption("Upload a PDF to extract and validate its references.")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
_ANTHROPIC_KEY_PRESENT = bool(os.getenv("ANTHROPIC_API_KEY"))
_AQUEDUCT_KEY_PRESENT  = bool(os.getenv("AQUEDUCT_API_KEY"))
_LLM_KEY_PRESENT       = _ANTHROPIC_KEY_PRESENT or _AQUEDUCT_KEY_PRESENT
_LLM_BACKEND           = (os.getenv("LLM_BACKEND") or
                          ("aqueduct" if _AQUEDUCT_KEY_PRESENT else "anthropic")).lower()
_LLM_MODEL             = os.getenv("LLM_MODEL") or (
    "qwen-3.6-35b" if _LLM_BACKEND == "aqueduct" else "claude-haiku-4-5"
)
_GROBID_AVAILABLE = grobid_is_available()

_PARSER_OPTIONS = ["pymupdf4llm", "docling"]
_PARSER_HELP = {
    "hybrid":      "Best quality (recommended). Merges GROBID's full-doc parse "
                   "with pymupdf-located section refs structured by GROBID's "
                   "/api/processCitationList — catches refs each method misses alone.",
    "grobid":      "GROBID full-doc layout parse only. Fast, clean, may miss "
                   "refs in oddly-formatted bibliographies.",
    "pymupdf4llm": "Fastest (~1 s). Regex-based field extraction; brittle on "
                   "complex bibliographies — body text can leak in.",
    "docling":     "Slow (minutes). Same regex extractor as pymupdf4llm but "
                   "better layout analysis on some PDFs.",
}
if _GROBID_AVAILABLE:
    _PARSER_OPTIONS = ["hybrid", "grobid"] + _PARSER_OPTIONS

with st.sidebar:
    st.header("Settings")
    parser = st.radio(
        "PDF Parser",
        options=_PARSER_OPTIONS,
        index=0,
        help=" | ".join(f"**{k}**: {v}" for k, v in _PARSER_HELP.items() if k in _PARSER_OPTIONS),
    )
    if not _GROBID_AVAILABLE:
        st.caption("💡 Run `docker compose up -d grobid` to unlock `grobid` / `hybrid`.")

    _llm_label = (
        f"Enhance with LLM ({_LLM_MODEL} via {_LLM_BACKEND})"
        if _LLM_KEY_PRESENT
        else "Enhance with LLM (no key set)"
    )
    use_llm = st.checkbox(
        _llm_label,
        value=_LLM_KEY_PRESENT,
        disabled=not _LLM_KEY_PRESENT,
        help=(
            f"Repairs garbled fields and flags body-text refs.  Backend is "
            f"controlled by env vars LLM_BACKEND / LLM_MODEL. Active: "
            f"**{_LLM_BACKEND}** / **{_LLM_MODEL}**."
            if _LLM_KEY_PRESENT
            else "Set ANTHROPIC_API_KEY (paid) or AQUEDUCT_API_KEY (free) in .env."
        ),
    )
    show_debug = st.checkbox("Show raw markdown (debug)", value=False)

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is None:
    st.info("Upload a PDF to get started.")
    st.stop()

pdf_bytes = uploaded_file.read()

# ---------------------------------------------------------------------------
# Parse & highlight (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_parser(pdf_bytes: bytes, parser: str):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        if parser == "hybrid":
            return extract_references_hybrid(tmp_path, return_markdown=True)
        if parser == "docling":
            return extract_references_docling(tmp_path, return_markdown=True)
        if parser == "grobid":
            return extract_references_grobid(tmp_path, return_markdown=True)
        return extract_references_pymupdf4llm(tmp_path, return_markdown=True)
    finally:
        os.unlink(tmp_path)


@st.cache_data(show_spinner=False)
def run_llm_enhancement(refs_json: str, backend: str, model: str) -> str:
    """
    Run LLM field extraction on parsed references. Returns enhanced refs as JSON.

    ``backend`` and ``model`` are folded into the cache key so switching LLM
    providers (or topping up credits after a failed run) automatically
    re-triggers extraction instead of serving stale enhanced refs.
    """
    _ = (backend, model)   # part of cache key, not used inside
    from llm_parser import parse_references_with_llm
    refs = json.loads(refs_json)
    enhanced = parse_references_with_llm(refs)
    return json.dumps(enhanced)


@st.cache_data(show_spinner=False)
def run_highlighter(pdf_bytes: bytes, refs_json: str, statuses_json: str = "[]"):
    refs = json.loads(refs_json)
    statuses = json.loads(statuses_json) or None
    return highlight_references(pdf_bytes, refs, statuses)


# --- Session state for user-added references -----------------------------
# Kept per-file so re-uploading a different PDF doesn't carry over.  Done
# BEFORE the processing block so the sidebar "add ref" UI can read it.
file_id = (uploaded_file.name, len(pdf_bytes))
if st.session_state.get("manual_refs_file") != file_id:
    st.session_state.manual_refs = []
    st.session_state.manual_refs_file = file_id

# --- Sidebar: add a reference the parser missed --------------------------
with st.sidebar:
    with st.expander(f"✏️ Add a missed reference ({len(st.session_state.manual_refs)})"):
        manual_text = st.text_area(
            "Paste the citation text",
            placeholder="e.g. John Hewitt and Christopher D. Manning. 2019. A structural probe for finding syntax in word representations. NAACL.",
            height=110,
            key="manual_ref_input",
        )
        if st.button("➕ Add to references", disabled=not manual_text.strip()):
            from parsers import _grobid_process_citation_list
            try:
                parsed = _grobid_process_citation_list([manual_text.strip()])
                if parsed:
                    new_ref = dict(parsed[0])
                    new_ref["_user_added"] = True
                    new_ref["raw"] = manual_text.strip()
                    st.session_state.manual_refs.append(new_ref)
                    # Invalidate lookup cache so the new ref gets queried.
                    st.session_state.lookup_results = None
                    st.success(
                        f"Added: {(new_ref.get('title') or manual_text.strip())[:70]}"
                    )
                    st.rerun()
                else:
                    st.error("GROBID couldn't structure that as a citation. "
                             "Check it has author + year + title.")
            except Exception as e:
                st.error(f"Couldn't parse: {e}")

        if st.session_state.manual_refs:
            st.divider()
            st.caption("Manually added (click ✕ to remove):")
            for i, mr in enumerate(list(st.session_state.manual_refs)):
                cols = st.columns([10, 1])
                cols[0].markdown(
                    f"**{i+1}.** {(mr.get('title') or mr.get('raw') or '')[:80]}"
                )
                if cols[1].button("✕", key=f"rm_manual_{i}", help="Remove"):
                    st.session_state.manual_refs.pop(i)
                    st.session_state.lookup_results = None
                    st.rerun()

# --- Multi-stage processing with visible progress ------------------------
# Show the multi-stage status block only the FIRST TIME we process a
# particular file (or after the user adds/removes manual refs).  On
# subsequent reruns — page-change events, ref clicks, toggle flips —
# the underlying functions are all @st.cache_data and return instantly,
# so re-displaying the status block with "0.0s" timings is noise.
import time as _time

_processed_key = (file_id, parser, use_llm, len(st.session_state.manual_refs))
_already_processed = (st.session_state.get("processed_key") == _processed_key)


def _silent_process():
    """Run the full processing pipeline without a status block.  Used on
    subsequent reruns when results are already cached."""
    references, debug_md = run_parser(pdf_bytes, parser)
    if use_llm:
        refs_json = run_llm_enhancement(json.dumps(references), _LLM_BACKEND, _LLM_MODEL)
        references = json.loads(refs_json)
    if st.session_state.manual_refs:
        references = list(references) + list(st.session_state.manual_refs)
    references, _ = dedupe_references(references)
    annotated_pdf, enriched_refs, orphan_blocks = run_highlighter(
        pdf_bytes, json.dumps(references))
    return references, debug_md, annotated_pdf, enriched_refs, orphan_blocks


if _already_processed:
    # Silent path: cached calls fill the variables in milliseconds.
    references, debug_md, annotated_pdf, enriched_refs, orphan_blocks = _silent_process()

    if show_debug:
        debug_lang = "xml" if parser == "grobid" else "markdown"
        debug_label = "Raw TEI XML from GROBID" if parser == "grobid" else "Raw markdown from parser"
        with st.expander(debug_label, expanded=False):
            st.code(debug_md, language=debug_lang)
else:
    _t_total = _time.time()
    with st.status("Processing paper …", expanded=True) as _stage:
        # Stage 1 — Parse
        _t = _time.time()
        st.write(f"📄 **Parsing references** with `{parser}` …")
        references, debug_md = run_parser(pdf_bytes, parser)
        _stage_t = _time.time() - _t
        if not references:
            st.write("   ✗ No references could be extracted")
            _stage.update(label="❌ No references found", state="error", expanded=True)
            st.warning(
                "No references could be extracted. Enable **Show raw markdown** "
                "in the sidebar to inspect the parser output."
            )
            st.stop()
        st.write(f"   ✓ Extracted **{len(references)} references** in {_stage_t:.1f}s")

        if show_debug:
            debug_lang = "xml" if parser == "grobid" else "markdown"
            debug_label = "Raw TEI XML from GROBID" if parser == "grobid" else "Raw markdown from parser"
            with st.expander(debug_label, expanded=False):
                st.code(debug_md, language=debug_lang)

        # Stage 2 — LLM repair (optional)
        if use_llm:
            _t = _time.time()
            st.write(f"🧠 **LLM repair** via `{_LLM_MODEL}` ({_LLM_BACKEND}) …")
            refs_json = run_llm_enhancement(json.dumps(references), _LLM_BACKEND, _LLM_MODEL)
            references = json.loads(refs_json)
            st.write(f"   ✓ Repaired in {_time.time() - _t:.1f}s")

        # User-added refs go in after LLM repair (they're already citation-form)
        if st.session_state.manual_refs:
            n_manual = len(st.session_state.manual_refs)
            references = list(references) + list(st.session_state.manual_refs)
            st.write(f"   ➕ Including **{n_manual}** user-added reference{'s' if n_manual != 1 else ''}")

        # Dedupe near-duplicate refs
        references, n_merged = dedupe_references(references)
        if n_merged:
            st.write(f"   🪡 Merged **{n_merged}** duplicate reference{'s' if n_merged != 1 else ''}")

        # Stage 3 — Locate in PDF + initial highlight
        _t = _time.time()
        st.write("🎨 **Locating references in the PDF** …")
        annotated_pdf, enriched_refs, orphan_blocks = run_highlighter(pdf_bytes, json.dumps(references))
        n_found = sum(1 for e in enriched_refs if e.get("found"))
        st.write(f"   ✓ Located **{n_found} / {len(enriched_refs)}** in {_time.time() - _t:.1f}s")
        if orphan_blocks:
            st.write(f"   📋 Found **{len(orphan_blocks)}** possible missed reference(s) "
                     f"in the bibliography region — see panel below the PDF")

        _stage.update(
            label=f"✓ Ready — {len(references)} refs, {n_found} located in PDF "
                  f"({_time.time() - _t_total:.1f}s)",
            state="complete",
            expanded=False,
        )

    # Record that we've shown the status block for this file-config combo
    # so subsequent reruns skip straight to _silent_process.
    st.session_state.processed_key = _processed_key

# ---------------------------------------------------------------------------
# Session state (file_id was set earlier when initialising manual_refs)
# ---------------------------------------------------------------------------
if st.session_state.get("lookup_file") != file_id:
    st.session_state.lookup_results = None
    st.session_state.lookup_file = file_id
if "selected_ref" not in st.session_state:
    st.session_state.selected_ref = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALIDATION_SOURCES = ("semantic_scholar", "acl_anthology", "dblp", "openalex", "crossref", "openlibrary", "scholarly", "arxiv")


def _overall_status(lr: dict | None) -> str:
    """'match' | 'mismatch' | 'missing' | 'pending'

    match    (green)  – at least one source returned a confirmed match
    mismatch (red)    – at least one source found a paper at this title
                        but the authors clearly disagree (= wrong paper /
                        mis-attribution).  Surfacing this aggressively
                        is important: the user wants to see the citation
                        errors, not have them buried.
    missing  (yellow) – not found anywhere, or only weak/fuzzy results

    Note: composite_label already decides per-source mismatch only when
    we have positive evidence of a wrong paper (high title + bad authors),
    never just from a low-tsim closest-match.  So lifting "mismatch" to
    the overall here is safe.
    """
    if lr is None:
        return "pending"

    found = [
        lr.get(src, {})
        for src in _VALIDATION_SOURCES
        if lr.get(src, {}).get("status") == "found"
    ]
    labels = [r.get("label") for r in found]

    if not found:
        return "missing"
    if "match" in labels:
        return "match"
    if "mismatch" in labels:
        return "mismatch"   # at least one source is sure it's a wrong paper
    return "missing"        # only fuzzy / weak results — couldn't confidently find


STATUS_ICON = {"match": "✅", "mismatch": "❌", "missing": "🟡", "pending": "⬜"}


def _status_reason(lr: dict | None) -> str:
    """One-line plain-English reason for the overall status, displayed
    next to the ref summary so the user knows *why* it's red/yellow/green
    without expanding the audit card."""
    if lr is None:
        return ""

    found = [(src, lr.get(src, {})) for src in _VALIDATION_SOURCES
             if lr.get(src, {}).get("status") == "found"]
    if not found:
        return "not found in any database"

    matches = [src for src, r in found if r.get("label") == "match"]
    if matches:
        # Friendly source names
        nice = {"semantic_scholar": "Semantic Scholar", "openalex": "OpenAlex",
                "dblp": "DBLP", "crossref": "Crossref", "arxiv": "arXiv",
                "scholarly": "Google Scholar", "acl_anthology": "ACL",
                "openlibrary": "Open Library"}
        names = [nice.get(s, s) for s in matches[:2]]
        return f"confirmed by {', '.join(names)}"

    mismatches = [(src, r) for src, r in found if r.get("label") == "mismatch"]
    if mismatches:
        # Pick the most specific reason available across all mismatch sources
        for src, r in mismatches:
            if r.get("wrong_arxiv_id"):
                return "cited arXiv ID points to a different paper"
            if r.get("wrong_doi"):
                return "cited DOI points to a different paper"
        # Otherwise it's a title/author disagreement
        for src, r in mismatches:
            asim = r.get("authors_sim")
            sim  = r.get("similarity")
            if asim is not None and asim < 0.35 and sim is not None and sim >= 0.70:
                return "title matches but the cited authors don't"
        return "found a paper at this title but details disagree"

    # Only fuzzy results
    return "no confident match found"


def _lookup_rows_html(lr: dict, extracted_title: str | None, extracted_venue: str | None) -> str:
    rows = []

    # --- Validation sources (title + venue matching) ---
    for label, key in [
        ("Semantic Scholar", "semantic_scholar"),
        ("ACL Anthology",    "acl_anthology"),
        ("DBLP",             "dblp"),
        ("OpenAlex",         "openalex"),
        ("Crossref",         "crossref"),
        ("arXiv",            "arxiv"),
        ("Open Library",     "openlibrary"),
        ("Google Scholar",   "scholarly"),
    ]:
        r = lr.get(key, {})
        status = r.get("status", "")
        # Only show sources that actually contributed a result.  Hide
        # skipped, not-queried, not-found, and errored sources — the
        # user only cares about the sources that successfully looked
        # the paper up.
        if status != "found":
            continue

        sim     = r.get("similarity")

        # Title
        icon    = {"match": "✅", "fuzzy": "🟡", "mismatch": "❌"}.get(r.get("label"), "❓")
        url     = r.get("url", "") or ""
        # For Google Scholar, fall back to a search URL when scholarly
        # didn't return a `pub_url`.  Better to give the user *some*
        # clickable handle than a dead label.
        if not url and key == "scholarly" and extracted_title:
            import urllib.parse as _up
            url = "https://scholar.google.com/scholar?q=" + _up.quote_plus(extracted_title)
        url     = _html.escape(url)
        sim_str = f"{sim:.0%}" if sim is not None else ""
        link    = f'<a href="{url}" target="_blank">{label}</a>' if url else label
        found_t = _html.escape(r.get("found_title", ""))
        note    = f" &mdash; <i>{found_t}</i>" if r.get("label") != "match" and found_t else ""
        rows.append(f"<li>{icon} <b>{link}</b> {sim_str}{note}</li>")

        # Wrong DOI warning
        if r.get("wrong_doi"):
            wrong = _html.escape(r["wrong_doi"])
            rows.append(f'<li>⚠️ DOI in reference (<code>{wrong}</code>) points to a different paper</li>')

        # Wrong arXiv ID warning — same precision tell as wrong_doi.
        if r.get("wrong_arxiv_id"):
            wrong = _html.escape(r["wrong_arxiv_id"])
            rows.append(
                f'<li>⚠️ arXiv ID in reference (<code>arXiv:{wrong}</code>) '
                f'points to a different paper</li>'
            )

        # Authors disagree — the most reliable "wrong paper" signal when
        # title similarity is high.  Surfacing the source's actual author
        # list tells the user *why* the match is yellow despite a 100%
        # title hit (classic fabrication / mis-attribution tell).
        asim = r.get("authors_sim")
        fa   = r.get("found_authors", "")
        if (asim is not None and asim < 0.35 and fa
                and sim is not None and sim >= 0.70):
            rows.append(
                f'<li style="margin-left:14px;opacity:.85">❗ authors: '
                f'<i>{_html.escape(fa)}</i> '
                f'({asim:.0%} overlap with cited authors)</li>'
            )

        # Venue check — only when we have both sides AND the cited
        # venue isn't a preprint marker.  A citation that says "arXiv
        # preprint" or "ResearchGate" will always disagree with the
        # canonical published venue; the discrepancy is uninformative.
        vsim   = r.get("venue_sim")
        vlabel = r.get("venue_label")
        fv     = r.get("found_venue", "")
        from lookup import _is_preprint_venue
        if (fv and extracted_venue and vlabel in ("fuzzy", "mismatch")
                and not _is_preprint_venue(extracted_venue)):
            vicon = "🟡" if vlabel == "fuzzy" else "❌"
            rows.append(
                f'<li style="margin-left:14px;opacity:.85">{vicon} venue: '
                f'<i>{_html.escape(fv)}</i>'
                + (f" ({vsim:.0%})" if vsim is not None else "")
                + "</li>"
            )

    # arXiv is rendered inline in the main validation loop above (it now
    # title+author-verifies the paper at the cited ID).

    # Show the reference's own URL as a clickable link when it isn't already
    # covered by arXiv or a doi.org link shown via validation sources.
    source_url = lr.get("_source_url", "")
    if source_url:
        rows.append(
            f'<li>🔗 <b><a href="{_html.escape(source_url)}" target="_blank">'
            f'Source URL</a></b></li>'
        )

    if not rows:
        return ""
    return '<div class="ref-lookup"><b>Lookup results</b><ul>' + "".join(rows) + "</ul></div>"


def _fields_html(ref: dict) -> str:
    keys = ("authors", "title", "year", "venue", "doi")
    items = [(k.capitalize(), ref[k]) for k in keys if ref.get(k)]
    if not items:
        return ""
    cells = "".join(
        f"<div><b>{_html.escape(k)}:</b> {_html.escape(str(v))}</div>"
        for k, v in items
    )
    return f'<div class="ref-fields">{cells}</div>'


def _render_reference(i: int, ref: dict, lr: dict | None, expanded: bool = False):
    status   = _overall_status(lr)
    icon     = STATUS_ICON[status]
    page     = ref.get("page")
    page_tag = f"p. {page}" if page else "not located"
    title    = _html.escape(ref.get("title") or ref.get("raw", "")[:100])
    raw      = _html.escape(ref.get("raw", ""))
    reason   = _status_reason(lr) if lr is not None else ""

    fields_h = _fields_html(ref)
    lookup_h = _lookup_rows_html(lr, ref.get("title"), ref.get("venue")) if lr else ""
    open_attr = " open" if expanded else ""

    # Build the second line in the summary: page + reason badge.  The
    # reason gives the user the answer at-a-glance without expanding.
    meta_bits = [f"({page_tag})"]
    if reason:
        meta_bits.append(f"&middot; {_html.escape(reason)}")
    meta_html = " ".join(meta_bits)

    card_html = f"""
<details class="ref-card {status}"{open_attr}>
  <summary>{icon} [{i+1}] {title} <small style="font-weight:400;opacity:.7;">{meta_html}</small></summary>
  <div class="ref-raw">{raw}</div>
  {fields_h}
  {lookup_h}
</details>"""

    st.markdown(card_html, unsafe_allow_html=True)

    # Interactive button must live outside the HTML block
    if ref.get("found") and page:
        if st.button(f"Go to p. {page}", key=f"goto_{i}"):
            st.session_state.selected_ref = i
            st.rerun()


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
col_refs, col_pdf = st.columns([2, 3], gap="medium")

with col_refs:
    found_count = sum(1 for r in enriched_refs if r.get("found"))
    st.subheader(f"References ({len(enriched_refs)}, {found_count} located in PDF)")

    lookup_results = st.session_state.get("lookup_results")

    # When loading lookup_results from session_state, recompute the
    # per-source author_sim, venue_sim, and composite label using the
    # ref's CURRENT extracted fields + current similarity logic.  This
    # is the heavyweight refresh (vs just relabelling): it picks up
    # post-lookup improvements to author_similarity / preprint-venue
    # handling without forcing a re-run of the API calls.
    if lookup_results:
        from lookup import refresh_result_against_ref
        _SRC_KEYS = ("semantic_scholar", "acl_anthology", "dblp", "openalex",
                     "crossref", "openlibrary", "scholarly", "arxiv")
        for lr_idx, lr in enumerate(lookup_results):
            if not isinstance(lr, dict) or lr_idx >= len(enriched_refs):
                continue
            ref = enriched_refs[lr_idx]
            for src in _SRC_KEYS:
                r = lr.get(src)
                if isinstance(r, dict):
                    lr[src] = refresh_result_against_ref(r, ref)

    if lookup_results is None:
        if st.button("🔍 Look up all references", type="primary"):
            _t_lookup = _time.time()
            with st.status("Querying academic databases …", expanded=True) as _lk:
                st.write("📡 **Batch pre-warming** S2, OpenAlex, arXiv (one round-trip each) …")
                progress_bar = st.progress(0.0, text="Looking up references …")
                _done_counter = {"n": 0}

                def _on_progress(done, total):
                    if done == 1 and _done_counter["n"] == 0:
                        st.write("🔍 **Per-reference lookups** running in parallel …")
                    _done_counter["n"] = done
                    progress_bar.progress(done / total, text=f"{done} / {total} refs")

                results = lookup_all(enriched_refs, progress_cb=_on_progress)
                progress_bar.empty()
                statuses = [_overall_status(r) for r in results]
                n_match = statuses.count("match")
                n_miss  = statuses.count("missing")
                n_mis   = statuses.count("mismatch")
                st.write(
                    f"   ✓ Done in {_time.time() - _t_lookup:.1f}s — "
                    f"**{n_match}** verified · **{n_mis}** wrong · "
                    f"**{n_miss}** not found"
                )
                _lk.update(
                    label=f"✓ Looked up {len(results)} references in {_time.time() - _t_lookup:.1f}s",
                    state="complete",
                    expanded=False,
                )
            st.session_state.lookup_results = results
            st.rerun()
    else:
        col_btn, col_stat = st.columns([1, 2])
        with col_btn:
            if st.button("🔄 Re-run lookups"):
                st.session_state.lookup_results = None
                st.rerun()
        with col_stat:
            statuses = [_overall_status(r) for r in lookup_results]
            st.caption(
                f"✅ {statuses.count('match')} match · "
                f"❌ {statuses.count('mismatch')} mismatch · "
                f"🟡 {statuses.count('missing')} not found"
            )

        # Re-highlight with per-reference colours derived from lookup results.
        # Wrap the post-lookup processing so a single bad ref doesn't take
        # down the whole page — surface the traceback instead of vanishing.
        try:
            # Length-match defence: if lookup_results somehow desynced from
            # enriched_refs (e.g. a stale rerun after manual refs were added),
            # pad / truncate statuses to the current ref count.
            n_refs = len(enriched_refs)
            statuses = [_overall_status(r) for r in lookup_results]
            if len(statuses) < n_refs:
                statuses.extend(["pending"] * (n_refs - len(statuses)))
            elif len(statuses) > n_refs:
                statuses = statuses[:n_refs]

            with st.spinner("Updating PDF highlights …"):
                annotated_pdf, enriched_refs, orphan_blocks = run_highlighter(
                    pdf_bytes,
                    json.dumps(enriched_refs),    # use the enriched list as the
                                                  # canonical ref source — its
                                                  # length matches `statuses`
                    json.dumps(statuses),
                )

            from report import generate_report
            report_md = generate_report(
                enriched_refs,
                lookup_results,
                pdf_name=uploaded_file.name,
                parser=parser,
            )
            with st.expander("📋 Quality Report", expanded=False):
                st.markdown(report_md)
            st.download_button(
                label="⬇️ Download report (.md)",
                data=report_md,
                file_name=uploaded_file.name.rsplit(".", 1)[0] + "_reference_report.md",
                mime="text/markdown",
            )
        except Exception as _exc:
            import traceback as _tb
            st.error(
                "⚠️ The post-lookup view crashed. "
                "Please paste this traceback in the chat so I can patch it."
            )
            st.code(_tb.format_exc(), language="python")
            st.button(
                "🔄 Reset lookup results and try again",
                on_click=lambda: st.session_state.update({"lookup_results": None}),
            )

    st.divider()

    # --- Page filter --------------------------------------------------------
    # The PDF component emits a `page` event when the user clicks ← / →
    # in the viewer; we filter the reference list to refs located on that
    # page.  A "All pages" toggle disables the filter.  Refs that have no
    # `page` (unlocated) always show when the filter is off, and they're
    # bucketed together under the "Unlocated" pseudo-page.
    current_page = st.session_state.get("pdf_current_page")
    all_pages = sorted({r.get("page") for r in enriched_refs if r.get("page")})
    n_unlocated = sum(1 for r in enriched_refs if not r.get("page"))

    # OPT-IN page filter.  When OFF (default), the PDF component does NOT
    # emit page-change events on scroll, so scrolling stays smooth — no
    # Streamlit rerun, no surrounding-UI re-render.  When ON, the JS starts
    # emitting; the refs list filters to whichever page is currently in
    # the viewport.
    sync_with_pdf = st.toggle(
        "🔗 Filter references to current PDF page",
        value=False,
        help=(
            "When enabled, the PDF component reports the visible page back "
            "to the app on scroll and this list filters accordingly.  "
            "Disabled by default because each page change triggers a "
            "Streamlit rerun, which can feel like a reload."
        ),
    )
    # Make sync_with_pdf visible to the PDF component code further below
    st.session_state._sync_with_pdf = sync_with_pdf

    if sync_with_pdf and not current_page and all_pages:
        current_page = all_pages[0]
        st.session_state.pdf_current_page = current_page

    if not sync_with_pdf:
        visible_indices = list(range(len(enriched_refs)))
    else:
        visible_indices = [
            i for i, r in enumerate(enriched_refs) if r.get("page") == current_page
        ]
        st.caption(
            f"Showing **{len(visible_indices)}** of {len(enriched_refs)} refs "
            f"on page {current_page}.  Scroll the PDF to switch pages."
        )

    # When the user clicks a highlighted annotation in the PDF, narrow the
    # list to just that ref so it auto-shows expanded at the top — no
    # scroll-to needed.  A "Show all" pill clears the focus.
    focused_idx = st.session_state.get("focused_ref")
    if isinstance(focused_idx, int) and 0 <= focused_idx < len(enriched_refs):
        cols = st.columns([6, 1])
        with cols[0]:
            st.markdown(f"📍 **Focused on reference [{focused_idx + 1}]**")
        with cols[1]:
            if st.button("✕ Clear", key="unfocus", help="Show all references again"):
                st.session_state.focused_ref = None
                st.rerun()
        visible_indices = [focused_idx]
        _render_focused_expanded = True
    else:
        _render_focused_expanded = False

    # Scrollable container for the reference cards — keeps the page-filter
    # controls visible at the top while the long list scrolls internally.
    with st.container(height=820, border=False):
        for i in visible_indices:
            ref = enriched_refs[i]
            lr = lookup_results[i] if lookup_results else None
            _render_reference(
                i, ref, lr,
                expanded=(_render_focused_expanded and i == focused_idx),
            )

    if sync_with_pdf and n_unlocated:
        st.caption(
            f"💡 {n_unlocated} reference{'s' if n_unlocated != 1 else ''} "
            f"could not be located in the PDF — turn **🔗 Filter references to "
            f"current PDF page** OFF to see them."
        )

with col_pdf:
    st.subheader("PDF")

    # Only consume `selected_ref` ONCE per click — otherwise every
    # downstream rerun would re-pass scroll_to_page and snap the PDF back
    # to that page after the user has scrolled away.
    scroll_page = None
    sel_ref = st.session_state.get("selected_ref")
    if sel_ref is not None and sel_ref != st.session_state.get("last_scrolled_ref"):
        scroll_page = enriched_refs[sel_ref].get("page")
        st.session_state.last_scrolled_ref = sel_ref

    # Toggle between the legacy streamlit_pdf_viewer and our new
    # selection-capable component.  Default to the new one; flip via env
    # var if it misbehaves.
    use_selector = os.getenv("REFCLEANER_PDF_SELECTOR", "1") != "0"
    if use_selector:
        from pdf_selector import pdf_selector
        # Build annotations for the component overlay.
        # PDF.js doesn't render PDF annotation objects on its canvas, so
        # we paint each matched ref as a coloured DOM-overlay rectangle
        # via the component.  Orphan auto-highlights are intentionally
        # omitted — the user adds missed refs manually via text selection.
        _STATUS_TO_COLOR = {
            "match":    "rgba(46, 200, 90, 0.30)",   # green
            "mismatch": "rgba(237, 70, 70, 0.30)",   # red
            "missing":  "rgba(255, 217, 51, 0.35)",  # yellow
            "pending":  "rgba(250, 244, 140, 0.45)", # pale yellow
        }
        component_annots: list[dict[str, Any]] = []
        for i, er in enumerate(enriched_refs):
            if not er.get("rect") or not er.get("page"):
                continue
            stat = er.get("status") or "pending"
            component_annots.append({
                "page":      er["page"],
                "rect":      er["rect"],
                "color":     _STATUS_TO_COLOR.get(stat, _STATUS_TO_COLOR["pending"]),
                "opacity":   0.40,
                "kind":      "filled",
                "label":     f"[{i+1}] {(er.get('title') or '')[:80]} · {stat}  (click to jump to details)",
                "ref_index": i,    # makes the box clickable; emits ref_click event
            })
        # Key MUST be stable across reruns within the same file — including
        # mutable state in the key remounts the component (and re-fetches the
        # PDF) on every change.
        selection = pdf_selector(
            pdf_bytes         = annotated_pdf,
            annotations       = component_annots,
            height            = 900,
            scroll_to_page    = scroll_page,
            emit_page_changes = bool(st.session_state.get("_sync_with_pdf", False)),
            key               = f"pdfsel_{file_id[0]}",
        )

        # The component returns either a text selection (user clicked
        # the floating "Add" button) or a page-change event (user clicked
        # ← / → in the nav bar).  Dispatch on `kind`.
        if selection and isinstance(selection, dict):
            kind = selection.get("kind")
            event_id = selection.get("id")
            last_event_id = st.session_state.get("last_pdf_event_id")

            if event_id and event_id != last_event_id:
                st.session_state.last_pdf_event_id = event_id

                if kind == "page":
                    new_page = selection.get("page")
                    if new_page and st.session_state.get("pdf_current_page") != new_page:
                        st.session_state.pdf_current_page = new_page
                        st.rerun()

                elif kind == "ref_click":
                    # User clicked a highlighted annotation in the PDF —
                    # focus the matching ref card on the left.
                    idx = selection.get("ref_index")
                    if isinstance(idx, int) and 0 <= idx < len(enriched_refs):
                        st.session_state.focused_ref = idx
                        st.rerun()

                elif kind == "selection":
                    from parsers import _grobid_process_citation_list
                    sel_text = (selection.get("text") or "").strip()
                    if sel_text:
                        try:
                            parsed = _grobid_process_citation_list([sel_text])
                            if parsed:
                                new_ref = dict(parsed[0])
                                new_ref["_user_added"] = True
                                new_ref["raw"] = sel_text
                                new_ref["page"] = selection.get("page")
                                st.session_state.manual_refs.append(new_ref)
                                st.session_state.lookup_results = None
                                st.toast(
                                    f"➕ Added: {(new_ref.get('title') or sel_text)[:80]}"
                                )
                                st.rerun()
                            else:
                                st.warning(
                                    "GROBID couldn't structure that selection as a "
                                    "citation. Try selecting the whole reference "
                                    "(authors + year + title)."
                                )
                        except Exception as e:
                            st.error(f"Couldn't parse selection: {e}")
    else:
        from streamlit_pdf_viewer import pdf_viewer
        pdf_viewer(input=annotated_pdf, height=900, scroll_to_page=scroll_page, render_text=True)

    # --- "Possible missed references" — orphan blocks from the bib region ---
    if orphan_blocks:
        # Hide orphans that match anything we've already user-added (so
        # they don't reappear after the user clicks Add).
        manual_raws = [
            (m.get("raw") or "").strip().lower()
            for m in st.session_state.manual_refs
        ]
        visible_orphans = [
            o for o in orphan_blocks
            if (o["text"] or "").strip().lower() not in manual_raws
        ]
        if visible_orphans:
            st.markdown(
                f"#### 📋 Possible missed references ({len(visible_orphans)})"
            )
            st.caption(
                "These are text blocks on the same pages as your parsed "
                "references that didn't get claimed — usually citations the "
                "parser missed.  Click ➕ to add one to the reference list."
            )
            for j, orph in enumerate(visible_orphans):
                cols = st.columns([10, 1])
                cols[0].markdown(
                    f"_p. {orph['page']}_ — {orph['text'][:240]}"
                    + ("…" if len(orph["text"]) > 240 else "")
                )
                if cols[1].button("➕", key=f"add_orphan_{j}",
                                  help="Add as reference"):
                    from parsers import _grobid_process_citation_list
                    try:
                        parsed = _grobid_process_citation_list([orph["text"].strip()])
                        if parsed:
                            new_ref = dict(parsed[0])
                            new_ref["_user_added"] = True
                            new_ref["raw"] = orph["text"].strip()
                            new_ref["page"] = orph["page"]
                            st.session_state.manual_refs.append(new_ref)
                            st.session_state.lookup_results = None
                            st.rerun()
                        else:
                            st.error("GROBID couldn't structure this block as a citation.")
                    except Exception as e:
                        st.error(f"Parse error: {e}")
