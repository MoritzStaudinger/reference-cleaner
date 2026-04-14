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

_VALIDATION_SOURCES = ("semantic_scholar", "acl_anthology", "dblp", "openalex")


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
        match_results = [r for r in found if r.get("label") == "match"]
        venue_labels  = [r.get("venue_label") for r in match_results if r.get("venue_label")]
        if venue_labels and all(vl == "mismatch" for vl in venue_labels):
            return "mismatch"
        return "match"
    if "fuzzy" in labels:
        return "missing"
    return "mismatch"


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
    lines.append(f"| 🔍 Not found / unverifiable | {n_missing} |\n")
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
            f"> 🔍 **{n_missing} reference{'s' if n_missing > 1 else ''} could not be verified** "
            "in Semantic Scholar, DBLP, OpenAlex, or ACL Anthology. "
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
        lines.append("## 🔍 Not Found / Unverifiable\n")
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

    return "\n".join(lines)
