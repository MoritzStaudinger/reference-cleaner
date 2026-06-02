import html as _html
import json
import streamlit as st
import tempfile
import os

from dotenv import load_dotenv
load_dotenv()

from parsers import (
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


with st.spinner(f"Parsing with **{parser}** …"):
    references, debug_md = run_parser(pdf_bytes, parser)

if show_debug:
    debug_lang = "xml" if parser == "grobid" else "markdown"
    debug_label = "Raw TEI XML from GROBID" if parser == "grobid" else "Raw markdown from parser"
    with st.expander(debug_label, expanded=False):
        st.code(debug_md, language=debug_lang)

if not references:
    st.warning(
        "No references could be extracted. "
        "Enable **Show raw markdown** in the sidebar to inspect the parser output."
    )
    st.stop()

if use_llm:
    with st.spinner(f"Enhancing field extraction with {_LLM_MODEL} ({_LLM_BACKEND}) …"):
        refs_json = run_llm_enhancement(json.dumps(references), _LLM_BACKEND, _LLM_MODEL)
        references = json.loads(refs_json)

with st.spinner("Locating references in PDF and adding highlights …"):
    # Initial highlight pass with pending (yellow) colour so the PDF
    # is immediately visible while lookups haven't run yet.
    annotated_pdf, enriched_refs = run_highlighter(pdf_bytes, json.dumps(references))

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
file_id = (uploaded_file.name, len(pdf_bytes))
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

    match    (green)  – at least one source found the paper with high similarity
    missing  (yellow) – not found anywhere, OR only fuzzy title matches
    mismatch (red)    – found in a source but title similarity is low (wrong paper)
                        OR title matches but venue clearly wrong
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
        return "match"   # composite label already accounts for venue + authors
    if "fuzzy" in labels:
        return "missing"   # uncertain → yellow
    return "mismatch"      # all found results are clearly wrong


STATUS_ICON = {"match": "✅", "mismatch": "❌", "missing": "🟡", "pending": "⬜"}


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
        if status in ("skipped", "not_in_anthology"):
            continue
        if status == "not_found":
            rows.append(f"<li>❓ <b>{label}</b>: not found</li>")
            continue
        if status == "error":
            err = _html.escape(str(r.get("error", ""))[:120])
            detail = f" — <code>{err}</code>" if err else ""
            rows.append(f"<li>⚠️ <b>{label}</b>: error{detail}</li>")
            continue

        # Title
        icon    = {"match": "✅", "fuzzy": "🟡", "mismatch": "❌"}.get(r.get("label"), "❓")
        url     = _html.escape(r.get("url", ""))
        sim     = r.get("similarity")
        sim_str = f"{sim:.0%}" if sim is not None else ""
        link    = f'<a href="{url}" target="_blank">{label}</a>' if url else label
        found_t = _html.escape(r.get("found_title", ""))
        note    = f" &mdash; <i>{found_t}</i>" if r.get("label") != "match" and found_t else ""
        rows.append(f"<li>{icon} <b>{link}</b> {sim_str}{note}</li>")

        # Wrong DOI warning
        if r.get("wrong_doi"):
            wrong = _html.escape(r["wrong_doi"])
            rows.append(f'<li>⚠️ DOI in reference (<code>{wrong}</code>) points to a different paper</li>')

        # Venue check (only when we have both sides and they differ)
        vsim   = r.get("venue_sim")
        vlabel = r.get("venue_label")
        fv     = r.get("found_venue", "")
        if fv and extracted_venue and vlabel in ("fuzzy", "mismatch"):
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


def _render_reference(i: int, ref: dict, lr: dict | None):
    status   = _overall_status(lr)
    icon     = STATUS_ICON[status]
    page     = ref.get("page")
    page_tag = f"p. {page}" if page else "not located"
    title    = _html.escape(ref.get("title") or ref.get("raw", "")[:100])
    raw      = _html.escape(ref.get("raw", ""))

    fields_h = _fields_html(ref)
    lookup_h = _lookup_rows_html(lr, ref.get("title"), ref.get("venue")) if lr else ""

    card_html = f"""
<details class="ref-card {status}">
  <summary>{icon} [{i+1}] {title} <small style="font-weight:400;opacity:.7;">({page_tag})</small></summary>
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

    if lookup_results is None:
        if st.button("🔍 Look up all references", type="primary"):
            progress_bar = st.progress(0.0, text="Looking up references …")

            def _on_progress(done, total):
                progress_bar.progress(done / total, text=f"Looking up … {done}/{total}")

            results = lookup_all(enriched_refs, progress_cb=_on_progress)
            st.session_state.lookup_results = results
            progress_bar.empty()
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

        # Re-highlight with per-reference colours derived from lookup results
        statuses = [_overall_status(r) for r in lookup_results]
        with st.spinner("Updating PDF highlights …"):
            annotated_pdf, enriched_refs = run_highlighter(
                pdf_bytes,
                json.dumps(references),
                json.dumps(statuses),
            )

        # Report generation
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

    st.divider()

    for i, ref in enumerate(enriched_refs):
        lr = lookup_results[i] if lookup_results else None
        _render_reference(i, ref, lr)

with col_pdf:
    st.subheader("PDF")
    from streamlit_pdf_viewer import pdf_viewer

    scroll_page = None
    if st.session_state.selected_ref is not None:
        scroll_page = enriched_refs[st.session_state.selected_ref].get("page")

    pdf_viewer(input=annotated_pdf, height=900, scroll_to_page=scroll_page, render_text=True)
