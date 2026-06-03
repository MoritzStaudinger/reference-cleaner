"""
Generate a Markdown quality report for a set of validated references.

Severity tiers
--------------
Major error   – overall status is "mismatch": a paper was found in a database
                but the title is clearly wrong (wrong paper cited).
Moderate error– a wrong_doi flag: the DOI in the reference resolves to a
                different paper on Semantic Scholar.
Minor concern – title matched but venue/conference differs significantly
                (venue_label == "mismatch" in at least one validation source).
Not found     – no validation source could confirm the paper (books, theses,
                preprints not indexed anywhere are expected here).
Verified      – confirmed correct (title match, venue plausible).
"""

from __future__ import annotations

import html as _html
from datetime import date
from typing import Any

_VALIDATION_SOURCES = ("semantic_scholar", "acl_anthology", "dblp", "openalex", "crossref", "openlibrary", "scholarly", "arxiv")


# ---------------------------------------------------------------------------
# Internal helpers (mirror app.py logic without importing Streamlit)
# ---------------------------------------------------------------------------

def _overall_status(lr: dict | None) -> str:
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
        return "mismatch"
    return "missing"


def _best_found(lr: dict) -> dict | None:
    """Return the validation result with the highest title similarity."""
    candidates = [
        lr.get(src, {})
        for src in _VALIDATION_SOURCES
        if lr.get(src, {}).get("status") == "found"
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.get("similarity") or 0.0)


def _has_wrong_doi(lr: dict) -> str | None:
    """Return the wrong DOI string if flagged, else None."""
    return lr.get("semantic_scholar", {}).get("wrong_doi") or None


def _venue_concern(lr: dict) -> tuple[str, str] | None:
    """
    Return (extracted_venue, found_venue) if the title matched but the venue
    is clearly wrong in at least one source, else None.
    """
    for src in _VALIDATION_SOURCES:
        r = lr.get(src, {})
        if r.get("status") == "found" and r.get("label") == "match":
            if r.get("venue_label") == "mismatch":
                return r.get("found_venue", ""), r.get("found_venue", "")
    return None


def _classify(
    refs: list[dict[str, Any]],
    lookup_results: list[dict[str, Any]],
) -> tuple[list, list, list, list, list]:
    """
    Returns five lists: (major, moderate, venue_concern, not_found, verified).
    Each element is (1-based index, ref dict, lookup_result dict).
    """
    major, moderate, venue, not_found, verified = [], [], [], [], []

    for i, (ref, lr) in enumerate(zip(refs, lookup_results)):
        status   = _overall_status(lr)
        entry    = (i + 1, ref, lr)
        wrong_doi = _has_wrong_doi(lr)

        if status == "mismatch":
            major.append(entry)
        elif status == "missing":
            not_found.append(entry)
        elif status == "match":
            if wrong_doi:
                moderate.append(entry)
            else:
                # Check venue concern even for clean title matches
                has_venue_issue = any(
                    lr.get(src, {}).get("status") == "found"
                    and lr.get(src, {}).get("label") == "match"
                    and lr.get(src, {}).get("venue_label") == "mismatch"
                    for src in _VALIDATION_SOURCES
                )
                if has_venue_issue:
                    venue.append(entry)
                else:
                    verified.append(entry)
        else:
            not_found.append(entry)

    return major, moderate, venue, not_found, verified


# ---------------------------------------------------------------------------
# Markdown builders
# ---------------------------------------------------------------------------

def _ref_header(n: int, ref: dict) -> str:
    title = ref.get("title") or ref.get("raw", "")[:100]
    page  = ref.get("page")
    loc   = f" — p. {page}" if page else ""
    return f"### [{n}] {title}{loc}"


def _ref_raw(ref: dict) -> str:
    raw = ref.get("raw", "").strip()
    if not raw:
        return ""
    return f"> {raw}\n"


def _ref_found_line(lr: dict) -> str:
    best = _best_found(lr)
    if not best:
        return ""
    sim  = best.get("similarity")
    url  = best.get("url", "")
    ft   = best.get("found_title", "")
    sim_s = f" ({sim:.0%} title match)" if sim is not None else ""
    link  = f"[{ft}]({url})" if url else ft
    return f"**Found in database:** {link}{sim_s}\n"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    refs: list[dict[str, Any]],
    lookup_results: list[dict[str, Any]],
    pdf_name: str = "",
    parser: str = "",
) -> str:
    """
    Return a Markdown string summarising reference quality.
    """
    major, moderate, venue, not_found, verified = _classify(refs, lookup_results)

    n_total   = len(refs)
    n_major   = len(major)
    n_mod     = len(moderate)
    n_venue   = len(venue)
    n_missing = len(not_found)
    n_ok      = len(verified)

    # Overall quality rating
    error_rate   = (n_major + n_mod) / n_total if n_total else 0
    missing_rate = n_missing / n_total if n_total else 0
    if error_rate == 0 and missing_rate <= 0.10:
        quality = "**Excellent** — references appear accurate and well-formed."
    elif error_rate <= 0.05 and missing_rate <= 0.20:
        quality = "**Good** — a small number of issues detected."
    elif error_rate <= 0.15 and missing_rate <= 0.35:
        quality = "**Fair** — several issues need attention."
    else:
        quality = "**Poor** — significant reference quality problems detected."

    lines: list[str] = []

    # --- Header ---
    title_line = f"# Reference Quality Report"
    if pdf_name:
        title_line += f" — {pdf_name}"
    lines.append(title_line)
    lines.append(f"_Generated {date.today().isoformat()}"
                 + (f" · Parser: {parser}" if parser else "") + "_\n")

    # --- Summary table ---
    lines.append("## Summary\n")
    lines.append(f"| | |")
    lines.append(f"|---|---|")
    lines.append(f"| Total references | {n_total} |")
    lines.append(f"| ✅ Verified | {n_ok} |")
    lines.append(f"| ❌ Major errors (wrong paper cited) | {n_major} |")
    lines.append(f"| ⚠️ Moderate errors (wrong DOI) | {n_mod} |")
    lines.append(f"| 🟡 Minor concerns (venue mismatch) | {n_venue} |")
    lines.append(f"| 🟡 Did not find in databases | {n_missing} |\n")
    lines.append(f"**Overall quality:** {quality}\n")

    # Narrative
    if n_major:
        lines.append(
            f"> ⚠️ **{n_major} reference{'s' if n_major > 1 else ''} appear to cite the wrong paper.** "
            "The title found in academic databases differs substantially from what is listed in the document. "
            "These should be checked carefully.\n"
        )
    if n_mod:
        lines.append(
            f"> ⚠️ **{n_mod} reference{'s' if n_mod > 1 else ''} contain a DOI that resolves to a different paper.** "
            "The DOI may have been copy-pasted from a nearby reference.\n"
        )
    if n_venue:
        lines.append(
            f"> 🟡 **{n_venue} reference{'s' if n_venue > 1 else ''} have a venue mismatch.** "
            "The paper was found and the title is correct, but the conference or journal name "
            "does not match what the databases record.\n"
        )
    if n_missing:
        lines.append(
            f"> 🟡 **{n_missing} reference{'s' if n_missing > 1 else ''} could not be located in any database** "
            "in Semantic Scholar, DBLP, OpenAlex, Crossref, or Open Library. "
            "This is normal for books, theses, and very recent preprints, "
            "but warrants a manual check if the reference is a conference or journal paper.\n"
        )

    # --- Major errors ---
    if major:
        lines.append("---\n")
        lines.append("## ❌ Major Errors — Wrong Paper Cited\n")
        lines.append(
            "The following references were found in academic databases, but the title "
            "in the document does not match the paper the citation data describes. "
            "This strongly suggests a wrong or hallucinated citation.\n"
        )
        for n, ref, lr in major:
            lines.append(_ref_header(n, ref))
            lines.append(_ref_raw(ref))
            best = _best_found(lr)
            if best:
                sim   = best.get("similarity")
                url   = best.get("url", "")
                ft    = best.get("found_title", "")
                sim_s = f" ({sim:.0%} title similarity)" if sim is not None else ""
                link  = f"[{ft}]({url})" if url else ft
                lines.append(f"- **Best database match:** {link}{sim_s}")
                fv = best.get("found_venue", "")
                if fv:
                    lines.append(f"- **Found venue:** {fv}")
            lines.append("")

    # --- Moderate errors ---
    if moderate:
        lines.append("---\n")
        lines.append("## ⚠️ Moderate Errors — Wrong DOI\n")
        lines.append(
            "The following references have a title that matches a real paper, "
            "but the DOI listed in the document resolves to a *different* paper. "
            "The DOI may have been accidentally copied from a neighbouring reference.\n"
        )
        for n, ref, lr in moderate:
            lines.append(_ref_header(n, ref))
            lines.append(_ref_raw(ref))
            wrong = _has_wrong_doi(lr)
            lines.append(f"- **DOI in document:** `{wrong}`")
            best = _best_found(lr)
            if best and best.get("url"):
                lines.append(f"- **Correct paper:** [{best.get('found_title', '')}]({best['url']})")
            lines.append("")

    # --- Venue concerns ---
    if venue:
        lines.append("---\n")
        lines.append("## 🟡 Minor Concerns — Venue Mismatch\n")
        lines.append(
            "The following references have a confirmed title match, but the "
            "conference or journal name in the document differs from what academic "
            "databases record. This may indicate a preprint citation (e.g., arXiv) "
            "used in place of the published version, or a transcription error.\n"
        )
        for n, ref, lr in venue:
            lines.append(_ref_header(n, ref))
            lines.append(_ref_raw(ref))
            ev = ref.get("venue", "")
            if ev:
                lines.append(f"- **Cited venue:** {ev}")
            for src in _VALIDATION_SOURCES:
                r = lr.get(src, {})
                if (r.get("status") == "found"
                        and r.get("label") == "match"
                        and r.get("venue_label") == "mismatch"):
                    fv  = r.get("found_venue", "")
                    vsim = r.get("venue_sim")
                    vsim_s = f" ({vsim:.0%})" if vsim is not None else ""
                    src_label = src.replace("_", " ").title()
                    lines.append(f"- **{src_label} venue:** {fv}{vsim_s}")
                    break
            lines.append("")

    # --- Not found ---
    if not_found:
        lines.append("---\n")
        lines.append("## 🟡 Did Not Find in Databases\n")
        lines.append(
            "These references could not be matched in any of the queried databases. "
            "Books, dissertations, technical reports, and very recent preprints are "
            "commonly absent. Manual verification is recommended for conference and "
            "journal papers.\n"
        )
        for n, ref, lr in not_found:
            title = ref.get("title") or ref.get("raw", "")[:100]
            year  = ref.get("year", "")
            venue_s = ref.get("venue", "")
            meta  = ", ".join(p for p in [year, venue_s] if p)
            lines.append(f"- **[{n}]** {title}" + (f" ({meta})" if meta else ""))
        lines.append("")

    # --- Verified ---
    if verified:
        lines.append("---\n")
        lines.append(f"## ✅ Verified ({n_ok})\n")
        lines.append(
            "The following references were confirmed correct — title and venue "
            "match academic database records.\n"
        )
        for n, ref, lr in verified:
            title = ref.get("title") or ref.get("raw", "")[:80]
            best  = _best_found(lr)
            url   = best.get("url", "") if best else ""
            year  = ref.get("year", "")
            venue_s = ref.get("venue", "")
            meta  = ", ".join(p for p in [year, venue_s] if p)
            entry = f"[{title}]({url})" if url else title
            lines.append(f"- **[{n}]** {entry}" + (f" ({meta})" if meta else ""))
        lines.append("")

    # --- Detailed per-reference audit ---
    lines.append("---\n")
    lines.append("## 🔎 Detailed Per-Reference Audit\n")
    lines.append(
        "One card per reference: extracted fields, the conclusion the "
        "pipeline reached, and what each consulted source returned.\n"
    )
    for i, (ref, lr) in enumerate(zip(refs, lookup_results)):
        lines.append(_per_ref_audit(i + 1, ref, lr))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-reference audit card
# ---------------------------------------------------------------------------

_SOURCE_LABELS: list[tuple[str, str]] = [
    ("semantic_scholar", "Semantic Scholar"),
    ("crossref",         "Crossref"),
    ("openalex",         "OpenAlex"),
    ("acl_anthology",    "ACL Anthology"),
    ("dblp",             "DBLP"),
    ("arxiv",            "arXiv"),
    ("openlibrary",      "Open Library"),
    ("scholarly",        "Google Scholar"),
]

_STATUS_ICONS = {
    "found":            "✅",
    "not_found":        "❓",
    "error":            "⚠️",
    "skipped":          "—",
    "not_in_anthology": "—",
    "no_index":         "—",
}

_LABEL_ICONS = {
    "match":    "✅",
    "fuzzy":    "🟡",
    "mismatch": "❌",
}


def _pct(x: float | None) -> str:
    return f"{x:.0%}" if isinstance(x, (int, float)) else "—"


def _conclusion(ref: dict, lr: dict) -> tuple[str, str, str]:
    """Return (icon, headline, explanation) for the overall verdict."""
    status   = _overall_status(lr)
    best     = _best_found(lr)
    wrong    = _has_wrong_doi(lr)
    found_in = [
        name
        for src, name in _SOURCE_LABELS
        if lr.get(src, {}).get("status") == "found"
        and lr.get(src, {}).get("label") == "match"
    ]

    if status == "match" and wrong:
        # Title matches a real paper somewhere, but the DOI in the ref points
        # to a different one — copy-paste smell.
        return (
            "⚠️",
            "Wrong DOI",
            f"Title matches a real paper, but the DOI `{wrong}` resolves to a "
            f"different paper on Semantic Scholar. The DOI may have been "
            f"copy-pasted from a neighbouring reference.",
        )
    if status == "match":
        if found_in:
            return (
                "✅",
                "Verified",
                f"Title confirmed in {', '.join(found_in)}.",
            )
        return ("✅", "Verified", "Title confirmed by at least one source.")

    if status == "mismatch":
        if best:
            ft  = best.get("found_title") or ""
            sim = best.get("similarity")
            return (
                "❌",
                "Wrong paper cited",
                f"A paper was found in the databases but its title "
                f'(*"{ft}"*, {_pct(sim)} similar) is too far from the cited '
                f"title to be the same paper. The citation likely points to a "
                f"different paper than what the bibliographic data identifies.",
            )
        return ("❌", "Wrong paper cited", "Closest database match does not align with the cited title.")

    # status == "missing"
    queried = [
        name
        for src, name in _SOURCE_LABELS
        if lr.get(src, {}).get("status") in ("found", "not_found", "error")
    ]
    errored = [
        name
        for src, name in _SOURCE_LABELS
        if lr.get(src, {}).get("status") == "error"
    ]
    if lr.get("_router_path") == "junk_filter":
        if ref.get("_corruption") == "name_fragment_title":
            return (
                "🚫",
                "Parser corruption (needs LLM repair)",
                "The regex parser put a fragment of the author list into the "
                "title field (e.g. \"Li, P\"). The real title is somewhere "
                "in the raw text but only LLM repair can recover it. Excluded "
                "from the match-rate denominator until LLM repair is enabled.",
            )
        if ref.get("_not_a_citation"):
            return (
                "🚫",
                "Not a citation",
                "The LLM classifier flagged this as body text, an equation, "
                "or a caption that the PDF parser misidentified as a "
                "bibliography entry. Excluded from the match-rate denominator.",
            )
        return (
            "🚫",
            "Skipped (parser garbage)",
            "The extracted fields were too damaged for a meaningful lookup "
            "(e.g. year token as title, missing authors). Re-parse with a "
            "different backend or enable LLM repair to recover this ref.",
        )
    if errored and not queried:
        return (
            "⚠️",
            "Lookup failed",
            f"All sources returned errors ({', '.join(errored)}). Try again "
            f"after the rate limit window or check network connectivity.",
        )
    if queried:
        return (
            "🟡",
            "Did not find this reference",
            f"We searched all {len(queried)} indexed sources "
            f"({', '.join(queried)}) and couldn't locate this paper. "
            f"That does **not** necessarily mean the citation is wrong — "
            f"common reasons a real reference lands here: "
            f"blog-style publications (Transformer Circuits Thread, Distill, "
            f"company tech reports), books or theses not in academic DBs, "
            f"very recent preprints, niche workshop papers, or a garbled "
            f"extraction we couldn't repair. Verify manually if it's important.",
        )
    return ("❓", "Pending", "No lookup has run for this reference yet.")


def _per_ref_audit(n: int, ref: dict, lr: dict | None) -> str:
    title = ref.get("title") or "(no title extracted)"
    page  = ref.get("page")
    loc   = f" · p. {page}" if page else ""
    badge = " ✏️ *user-added*" if ref.get("_user_added") else ""

    icon, headline, explanation = _conclusion(ref, lr or {})

    out: list[str] = []
    out.append(f"### {icon} [{n}] {title}{loc}{badge}\n")
    raw = (ref.get("raw") or "").strip()
    if raw:
        out.append(f"> {raw}\n")

    # Extracted fields table
    field_rows = []
    for field, label in [
        ("authors", "Authors"),
        ("year",    "Year"),
        ("venue",   "Venue"),
        ("doi",     "DOI"),
        ("url",     "URL"),
    ]:
        val = (ref.get(field) or "").strip()
        if val:
            display = f"`{val}`" if field in ("doi", "url") else val
            field_rows.append(f"| {label} | {display} |")
    if field_rows:
        out.append("**Extracted fields**\n")
        out.append("| Field | Value |")
        out.append("|---|---|")
        out.extend(field_rows)
        out.append("")
    else:
        out.append("**Extracted fields:** _(none recovered from raw)_\n")

    out.append(f"**Conclusion:** {icon} **{headline}** — {explanation}\n")

    # Per-source table — skip rows that just say "skipped" so the verdict
    # block stays focused on sources that actually returned a signal.
    lr = lr or {}
    interesting = ("found", "not_found", "error")
    rows: list[str] = []
    for src, label in _SOURCE_LABELS:
        r = lr.get(src) or {}
        status = r.get("status", "absent")
        if status not in interesting:
            continue
        icon_s = _STATUS_ICONS.get(status, "·")
        sim    = _pct(r.get("similarity"))
        asim   = _pct(r.get("authors_sim"))
        vsim   = r.get("venue_sim")
        vlbl   = r.get("venue_label", "")
        if r.get("found_venue"):
            vcell = r["found_venue"]
            if vlbl in ("fuzzy", "mismatch") and vsim is not None:
                vcell += f" ({vsim:.0%}, {vlbl})"
        else:
            vcell = "—"

        if status == "found":
            mlbl = r.get("label")
            mark = _LABEL_ICONS.get(mlbl, "")
            ft   = r.get("found_title") or ""
            url  = r.get("url") or ""
            result_cell = f"{mark} [{ft}]({url})" if url and ft else (ft or "(found)")
        elif status == "not_found":
            result_cell = "not found"
        else:   # error
            err = (r.get("error") or "")[:80].replace("|", "\\|")
            result_cell = f"error: `{err}`"

        rows.append(f"| {label} | {icon_s} {status} | {sim} | {asim} | {vcell} | {result_cell} |")

    if rows:
        out.append("**Sources consulted**\n")
        out.append("| Source | Status | Title sim | Author sim | Venue | Result |")
        out.append("|---|:-:|:-:|:-:|---|---|")
        out.extend(rows)
        out.append("")
    else:
        out.append("**Sources consulted:** _(none — all sources skipped)_\n")

    route = lr.get("_router_path")
    if route:
        out.append(f"<sub>Routing path: `{route}`</sub>\n")

    return "\n".join(out)
