"""
Locate extracted references inside a PDF, add highlight annotations,
and return the annotated PDF bytes together with per-reference metadata.
"""

from __future__ import annotations
import re
from typing import Any

import fitz  # PyMuPDF


Color = tuple[float, float, float]

# Status → RGB highlight colour
#   match    = green   → confirmed by an academic source
#   mismatch = red     → wrong paper at this title (likely fabrication)
#   missing  = yellow  → couldn't find in any database (NOT proof of wrong;
#                        common for blogs/tech-reports/uncommon venues)
#   pending  = pale yellow → lookup not yet completed
COLORS: dict[str, Color] = {
    "match":    (0.18, 0.78, 0.35),   # green
    "mismatch": (0.93, 0.27, 0.27),   # red
    "missing":  (1.00, 0.85, 0.20),   # clearer yellow (was orange-amber)
    "pending":  (0.98, 0.96, 0.55),   # pale yellow (before lookup)
}


def _clean_for_search(text: str) -> str:
    """Normalise a reference string so it can be matched against PDF text."""
    # Remove markdown italic underscores and bold/italic asterisks
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"_+", " ", text)
    # Strip leading markdown list bullet: "- " or "* "
    text = re.sub(r"^\s*[-*]\s+", "", text)
    # Strip leading numbered marker: [1], (1), 1., 1)
    text = re.sub(r"^\s*[\[\(]?\d+[\]\)\.]\s*", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _search_snippets(raw: str, title: str = "") -> list[str]:
    """
    Return candidate search snippets in **longest-first** order so the highlighter
    covers the full reference whenever the raw text matches the PDF cleanly.

    Order:
      1. Full raw (capped at 500 chars — PyMuPDF handles multi-line / hyphenated
         line breaks for searches up to a few hundred chars).
      2. Progressively shorter raw prefixes — used as fallbacks when GROBID has
         reformatted the raw (e.g. initials' periods stripped) and the full
         string doesn't match byte-for-byte.
      3. Title.  Strong final fallback because titles usually appear verbatim
         in the PDF even when the surrounding author / venue text differs.
    """
    snippets: list[str] = []

    cleaned = _clean_for_search(raw)

    # 1. Full raw first — gives the whole-reference bounding box when matched
    if 30 <= len(cleaned) <= 500:
        snippets.append(cleaned)
    elif len(cleaned) > 500:
        cut = cleaned[:500].rsplit(" ", 1)[0].strip(" .,;")
        if cut:
            snippets.append(cut)

    # 2. Progressively shorter raw prefixes — fragment fallbacks
    for target_len in (180, 120, 70, 35):
        if len(cleaned) <= target_len:
            s = cleaned
        else:
            cut = cleaned[:target_len].rsplit(" ", 1)[0]
            s = cut.strip(" .,;")
        if len(s) >= 15 and s not in snippets:
            snippets.append(s)

    # 3. Title fallback
    title_clean = _clean_for_search(title)
    if title_clean and title_clean not in snippets and len(title_clean) >= 15:
        snippets.append(title_clean)
        if len(title_clean) > 40:
            cut = title_clean[:40].rsplit(" ", 1)[0].strip(" .,;")
            if cut and cut not in snippets and len(cut) >= 15:
                snippets.append(cut)

    return snippets


def highlight_references(
    pdf_bytes: bytes,
    references: list[dict[str, Any]],
    statuses: list[str] | None = None,
) -> tuple[bytes, list[dict[str, Any]]]:
    """
    Add highlights for each reference found in the PDF.

    Parameters
    ----------
    pdf_bytes : bytes
        Original PDF.
    references : list[dict]
        Extracted reference dicts (must have ``raw`` key).
    statuses : list[str] | None
        Per-reference status strings (``"match"``, ``"mismatch"``,
        ``"missing"``, ``"pending"``).  Defaults to ``"pending"`` for all.

    Returns
    -------
    annotated_pdf : bytes
    enriched_refs : list[dict]
        Each entry gains ``page`` (1-based int or None) and ``found`` (bool).
    """
    if statuses is None:
        statuses = ["pending"] * len(references)

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    enriched: list[dict[str, Any]] = []

    # Block-based highlighting:  PyMuPDF gives us a paragraph-level list of
    # text "blocks" per page, each with a bounding box.  A reference is
    # almost always exactly one block, so we find the block whose text
    # contains the ref (fuzzy match), then highlight the whole block rect.
    # This produces clean "one rectangle per ref" highlights instead of
    # the fragmented per-line search-for results.
    #
    # Each block: dict gathered up front so we can fuzzy-match each ref
    # against all of them once.
    from rapidfuzz import fuzz as _fuzz

    blocks: list[tuple[int, fitz.Rect, str, str]] = []   # (page_idx, rect, raw_text, norm_text)
    for page in doc:
        for b in page.get_text("blocks"):
            # b = (x0, y0, x1, y1, text, block_no, block_type)
            if len(b) < 5:
                continue
            text = (b[4] or "").strip()
            if len(text) < 30:
                continue
            rect = fitz.Rect(b[0], b[1], b[2], b[3])
            norm = _clean_for_search(text).lower()
            blocks.append((page.number, rect, text, norm))

    used_block_idx: set[int] = set()  # blocks already claimed by an earlier ref

    for ref, status in zip(references, statuses):
        entry = dict(ref)
        entry["found"] = False
        entry["page"] = None
        color = COLORS.get(status, COLORS["pending"])

        ref_raw  = _clean_for_search(ref.get("raw", "")).lower()
        ref_title = _clean_for_search(ref.get("title", "")).lower()
        if not ref_raw and not ref_title:
            enriched.append(entry)
            continue

        # Score every block by best fuzzy match on raw OR title.
        # Use partial_ratio so a short ref title still scores high against
        # a longer block (block can contain title + venue etc.).
        best_score = 0
        best_idx = -1
        for i, (pn, rect, text, norm) in enumerate(blocks):
            if i in used_block_idx:
                continue
            s = 0
            if ref_raw:
                s = max(s, _fuzz.partial_ratio(ref_raw, norm))
            if ref_title and len(ref_title) >= 15:
                s = max(s, _fuzz.partial_ratio(ref_title, norm))
            if s > best_score:
                best_score = s
                best_idx = i

        if best_idx >= 0 and best_score >= 80:
            pn, rect, _, _ = blocks[best_idx]
            page = doc[pn]
            # add_rect_annot with semi-transparent fill renders as a clean
            # straight rectangle in every PDF viewer.  PyMuPDF's
            # add_highlight_annot uses Quad-based highlights that some
            # viewers render with curved / marker-pen edges.
            annot = page.add_rect_annot(rect)
            annot.set_colors(stroke=color, fill=color)
            annot.set_border(width=0)
            annot.set_opacity(0.30)
            annot.update()
            used_block_idx.add(best_idx)
            entry["found"] = True
            entry["page"] = pn + 1

        enriched.append(entry)

    annotated_bytes = doc.tobytes(garbage=2, deflate=True)
    doc.close()
    return annotated_bytes, enriched
