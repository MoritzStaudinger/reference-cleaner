"""
PDF parsing backends.

Each function receives a path to a PDF file and returns a list of reference
dicts with (at least) these optional keys:
    raw     – the raw reference string as it appeared in the PDF
    title   – extracted title (if parseable)
    authors – extracted author string (if parseable)
    year    – extracted year (if parseable)
    venue   – journal / conference name (if parseable)
    doi     – DOI string (if parseable)
    url     – URL (if parseable)
"""

from __future__ import annotations
import os
import re
import xml.etree.ElementTree as ET
from typing import Any

import requests as _requests

GROBID_URL: str = os.getenv("GROBID_URL", "http://localhost:8070")
_TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


# ---------------------------------------------------------------------------
# Regex constants
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
_DOI_RE = re.compile(r"10\.\d{4,}/\S+")
_URL_RE = re.compile(r"https?://\S+")

# Matches a reference entry marker at the start of a line:
# - [1], * [1], [1], (1), 1., 1)
_REF_MARKER_RE = re.compile(r"^\s*(?:[-*]\s+)?[\[\(]?\d+[\]\)\.]\s+")

# Heading patterns for the reference section (markdown or plain)
_REF_SECTION_RE = re.compile(
    r"(?:^|\n)"                          # start of string or after newline
    r"\s*#{0,4}\s*"                      # optional markdown heading hashes
    r"\*{0,2}"                           # optional bold markers
    r"(?:References?|Bibliography|Works\s+Cited|Literature)"
    r"\*{0,2}"
    r"\s*\n",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _locate_reference_section(full_text: str) -> str | None:
    """Return the portion of *full_text* that starts after the reference heading."""
    match = _REF_SECTION_RE.search(full_text)
    if match:
        return full_text[match.end():]
    return None


def _split_into_references(text: str) -> list[str]:
    """
    Split a block of text into individual reference strings.

    Strategy:
    1. Try numbered markers ([1], 1., etc.) — join continuation lines first.
    2. Fall back to blank-line separation.
    """
    lines = text.splitlines()

    # --- Strategy 1: numbered markers ---
    # Check whether most lines that look like ref starts are numbered.
    marker_lines = [i for i, l in enumerate(lines) if _REF_MARKER_RE.match(l)]
    if len(marker_lines) >= 2:
        # Group lines: a new reference starts at each marker line.
        groups: list[list[str]] = []
        for i, line in enumerate(lines):
            if _REF_MARKER_RE.match(line):
                groups.append([line])
            elif groups:
                # Continuation line: strip leading whitespace and append.
                stripped = line.strip()
                if stripped:
                    groups[-1].append(stripped)
        return [" ".join(g).strip() for g in groups if g]

    # --- Strategy 2: blank-line separated ---
    paragraphs = re.split(r"\n\s*\n", text)
    refs = []
    for p in paragraphs:
        # Collapse internal newlines (continuation lines).
        collapsed = " ".join(line.strip() for line in p.splitlines() if line.strip())
        if collapsed:
            refs.append(collapsed)
    return refs


def _parse_raw_reference(raw: str) -> dict[str, Any]:
    """
    Best-effort field extraction from a raw reference string.

    Handles the most common academic citation styles:
      IEEE:  [1] Authors, "Title," Venue, year.
      APA:   Authors (year). Title. Venue.
      ACM:   Authors. year. Title. Venue.
      plain: Authors. Title. Venue, year.  ← year at end
    """
    ref: dict[str, Any] = {"raw": raw}

    # Strip leading marker (- [1], [1], 1., etc.) and markdown bold/italic
    body = _REF_MARKER_RE.sub("", raw).strip()
    body = re.sub(r"\*+", "", body)
    body = re.sub(r"_+", " ", body)
    body = re.sub(r"\s+", " ", body).strip()

    # --- Year ---
    year_match = _YEAR_RE.search(body)
    if year_match:
        ref["year"] = year_match.group(0)

    # --- DOI ---
    doi_match = _DOI_RE.search(body)
    if doi_match:
        ref["doi"] = doi_match.group(0).rstrip(".,)")

    # --- URL ---
    url_match = _URL_RE.search(body)
    if url_match:
        ref["url"] = url_match.group(0).rstrip(".,)")

    # --- IEEE style: title in double quotes ---
    ieee_title = re.search(r'"([^"]{10,})"', body)
    if ieee_title:
        ref["title"] = ieee_title.group(1).strip()
        before_quote = body[: ieee_title.start()].strip().rstrip(",")
        if before_quote:
            ref["authors"] = before_quote
        after_quote = body[ieee_title.end():].strip().lstrip(",").strip()
        if after_quote:
            venue_candidate = _YEAR_RE.split(after_quote)[0].strip().strip(".,")
            if venue_candidate:
                ref["venue"] = venue_candidate
        return ref

    # --- APA style: Authors (year). Title. Venue. ---
    # Distinguishing feature: year is in parentheses right after authors.
    apa_match = re.search(r"\((" + _YEAR_RE.pattern + r")\)\s*\.", body)
    if apa_match:
        ref["authors"] = body[: apa_match.start()].strip().rstrip(".,")
        after_year = body[apa_match.end():].strip()
        parts = re.split(r"\.\s+", after_year, maxsplit=2)
        if parts:
            ref["title"] = parts[0].strip().rstrip(".,")
        if len(parts) > 1:
            ref["venue"] = parts[1].strip().rstrip(".,")
        return ref

    # --- Numbered / plain style: Authors. Title. Venue, year. ---
    # Year is NOT in parens (it appears at the end). Split on ". " first.
    sentences = re.split(r"\.\s+", body)
    # Drop trailing empty/very-short segments and URL/DOI fragments
    sentences = [s.strip() for s in sentences if len(s.strip()) > 4]

    if len(sentences) >= 3:
        ref["authors"] = sentences[0]
        ref["title"] = sentences[1]
        # Venue is the rest up to (but not including) a trailing year
        venue_raw = ". ".join(sentences[2:])
        venue_clean = _YEAR_RE.sub("", venue_raw).strip(" .,;:")
        if venue_clean:
            ref["venue"] = venue_clean
    elif len(sentences) == 2:
        ref["authors"] = sentences[0]
        ref["title"] = sentences[1].rstrip(".,")
    elif sentences:
        ref["title"] = sentences[0].rstrip(".,")

    return ref


def _is_plausible_reference(ref: dict[str, Any]) -> bool:
    """Filter out obvious noise: too short, no year, no recognisable structure."""
    raw = ref.get("raw", "")
    if len(raw) < 30:
        return False
    # Must contain a plausible year or at least some typical reference punctuation
    if not ref.get("year") and not ref.get("doi") and not ref.get("url"):
        # Require at least a comma and a period to look like a citation
        if "," not in raw or "." not in raw:
            return False
    return True


# ---------------------------------------------------------------------------
# docling backend
# ---------------------------------------------------------------------------

def extract_references_docling(
    pdf_path: str, return_markdown: bool = False
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], str]:
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    doc = result.document

    md_text = doc.export_to_markdown()
    ref_section = _locate_reference_section(md_text)
    source = ref_section if ref_section else md_text

    raw_refs = _split_into_references(source)
    references = [_parse_raw_reference(r) for r in raw_refs]
    filtered = [r for r in references if _is_plausible_reference(r)]

    if return_markdown:
        return filtered, md_text
    return filtered


# ---------------------------------------------------------------------------
# pymupdf4llm backend
# ---------------------------------------------------------------------------

def extract_references_pymupdf4llm(
    pdf_path: str, return_markdown: bool = False
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], str]:
    import pymupdf4llm

    md_text: str = pymupdf4llm.to_markdown(pdf_path, use_ocr=False)
    ref_section = _locate_reference_section(md_text)
    source = ref_section if ref_section else md_text

    raw_refs = _split_into_references(source)
    references = [_parse_raw_reference(r) for r in raw_refs]
    filtered = [r for r in references if _is_plausible_reference(r)]

    if return_markdown:
        return filtered, md_text
    return filtered


# ---------------------------------------------------------------------------
# GROBID backend
# ---------------------------------------------------------------------------

def grobid_is_available() -> bool:
    """Return True if the GROBID service is reachable."""
    try:
        r = _requests.get(f"{GROBID_URL}/api/isalive", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _parse_tei_references(tei_xml: str) -> list[dict[str, Any]]:
    """Parse GROBID TEI XML and return a list of reference dicts."""
    try:
        root = ET.fromstring(tei_xml)
    except ET.ParseError:
        return []

    refs = []
    for bib in root.findall(".//tei:listBibl/tei:biblStruct", _TEI_NS):
        ref: dict[str, Any] = {}

        # Raw reference string (GROBID preserves this)
        note = bib.find("tei:note[@type='raw_reference']", _TEI_NS)
        if note is not None and note.text:
            ref["raw"] = note.text.strip()

        # Title — analytic (article) takes priority over monogr (book/chapter)
        for xpath in (
            "tei:analytic/tei:title[@type='main']",
            "tei:analytic/tei:title",
            "tei:monogr/tei:title[@level='m']",
            "tei:monogr/tei:title",
        ):
            el = bib.find(xpath, _TEI_NS)
            if el is not None and el.text:
                ref["title"] = el.text.strip()
                break

        # Authors
        authors = []
        for author in bib.findall(".//tei:author", _TEI_NS):
            forename = author.findtext("tei:persName/tei:forename", "", _TEI_NS).strip()
            surname  = author.findtext("tei:persName/tei:surname",  "", _TEI_NS).strip()
            name = f"{forename} {surname}".strip() if forename else surname
            if name:
                authors.append(name)
        if authors:
            ref["authors"] = ", ".join(authors)

        # Year
        date_el = bib.find(".//tei:imprint/tei:date[@type='published']", _TEI_NS)
        if date_el is not None:
            when = date_el.get("when", "")
            m = re.search(r"\b(1[89]\d{2}|20\d{2})\b", when)
            if m:
                ref["year"] = m.group(0)

        # Venue — journal title preferred over series/book title
        for xpath in (
            "tei:monogr/tei:title[@level='j']",   # journal
            "tei:monogr/tei:title[@level='m']",   # book / proceedings
            "tei:series/tei:title",
        ):
            el = bib.find(xpath, _TEI_NS)
            if el is not None and el.text:
                ref["venue"] = el.text.strip()
                break

        # DOI
        doi_el = bib.find("tei:idno[@type='DOI']", _TEI_NS)
        if doi_el is not None and doi_el.text:
            ref["doi"] = doi_el.text.strip()

        # arXiv → URL
        arxiv_el = bib.find("tei:idno[@type='arXiv']", _TEI_NS)
        if arxiv_el is not None and arxiv_el.text:
            ref["url"] = f"https://arxiv.org/abs/{arxiv_el.text.strip()}"

        # Web pointer fallback
        if not ref.get("url"):
            ptr = bib.find("tei:ptr[@type='web']", _TEI_NS)
            if ptr is not None:
                ref["url"] = ptr.get("target", "")

        # Reconstruct raw if GROBID didn't provide one
        if "raw" not in ref:
            parts = [p for p in (ref.get("authors"), ref.get("title"),
                                  ref.get("venue"), ref.get("year")) if p]
            ref["raw"] = ". ".join(parts)

        if ref.get("raw") or ref.get("title"):
            refs.append(ref)

    return refs


def extract_references_grobid(
    pdf_path: str, return_markdown: bool = False
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], str]:
    """
    Extract references using the GROBID service (must be running on GROBID_URL).
    ``return_markdown`` returns the raw TEI XML for debugging (not actual markdown).
    """
    with open(pdf_path, "rb") as fh:
        response = _requests.post(
            f"{GROBID_URL}/api/processFulltextDocument",
            files={"input": fh},
            data={"consolidateReferences": "0",
                  "includeRawReferences": "1"},
            timeout=120,
        )
    response.raise_for_status()
    tei_xml = response.text

    references = _parse_tei_references(tei_xml)
    filtered   = [ref for ref in references if _is_plausible_reference(ref)]

    if return_markdown:
        return filtered, tei_xml
    return filtered
