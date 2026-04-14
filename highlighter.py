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
COLORS: dict[str, Color] = {
    "match":    (0.18, 0.78, 0.35),   # green
    "mismatch": (0.93, 0.27, 0.27),   # red
    "missing":  (1.00, 0.78, 0.08),   # amber
    "pending":  (1.00, 0.95, 0.00),   # yellow (before lookup)
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


def _search_snippets(raw: str) -> list[str]:
    """Return candidate search snippets of decreasing length, avoiding mid-word cuts."""
    cleaned = _clean_for_search(raw)
    snippets = []
    for target_len in (70, 50, 35):
        if len(cleaned) <= target_len:
            s = cleaned
        else:
            # Cut at last space before target_len to avoid splitting words
            cut = cleaned[:target_len].rsplit(" ", 1)[0]
            s = cut.strip(" .,;")
        if len(s) >= 15 and s not in snippets:
            snippets.append(s)
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

    for ref, status in zip(references, statuses):
        entry = dict(ref)
        entry["found"] = False
        entry["page"] = None

        color = COLORS.get(status, COLORS["pending"])
        snippets = _search_snippets(ref.get("raw", ""))

        for snippet in snippets:
            if len(snippet) < 15:
                continue
            for page in doc:
                quads = page.search_for(snippet, quads=True)
                if quads:
                    for quad in quads:
                        annot = page.add_highlight_annot(quad)
                        annot.set_colors(stroke=color)
                        annot.update()
                    if not entry["found"]:
                        entry["page"] = page.number + 1
                        entry["found"] = True
                    break
            if entry["found"]:
                break

        enriched.append(entry)

    annotated_bytes = doc.tobytes(garbage=2, deflate=True)
    doc.close()
    return annotated_bytes, enriched
