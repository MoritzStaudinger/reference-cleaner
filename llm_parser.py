"""
LLM-based reference field extraction using Claude.

Sends batches of raw reference strings to Claude and asks it to parse
each one into structured fields (title, authors, year, venue, doi, url).
This is more accurate than regex-based heuristics for tricky citation styles.
"""

from __future__ import annotations

import json
import os
from typing import Any

import anthropic
from pydantic import BaseModel

_BATCH_SIZE = 15
_MODEL = "claude-haiku-4-5-20251001"

_SYSTEM_PROMPT = """\
You are an expert bibliographer. You will receive a JSON array of raw reference strings
extracted from a PDF. For each reference, extract structured fields as accurately as possible.

Return a JSON array (same length, same order) where each element is an object with these fields:
  - "title":   the paper or book title (string, required — never null or empty if any title is present)
  - "authors": author names as a single string (string or null)
  - "year":    4-digit year as a string (string or null)
  - "venue":   journal or conference name (string or null)
  - "doi":     DOI string without URL prefix (string or null)
  - "url":     any URL present (string or null)

Rules:
- Return ONLY the JSON array. No explanation, no markdown fences.
- If a field cannot be determined, use null.
- For title: prefer the actual paper title over the first sentence or authors line.
  In IEEE style, titles appear in double quotes. In APA style, the title comes after
  the year parenthetical. In numbered style, the title is usually the second "sentence".
- Keep the array length identical to the input array.
"""


class _ParsedRef(BaseModel):
    title: str | None = None
    authors: str | None = None
    year: str | None = None
    venue: str | None = None
    doi: str | None = None
    url: str | None = None


def _call_llm(client: anthropic.Anthropic, raw_refs: list[str]) -> list[_ParsedRef]:
    """Send one batch to Claude and parse the JSON response."""
    user_content = json.dumps(raw_refs, ensure_ascii=False)
    message = client.messages.create(
        model=_MODEL,
        max_tokens=4096,
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_content}],
    )
    text = message.content[0].text.strip()
    # Strip markdown code fences if the model wraps output anyway
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0].strip()

    parsed_list = json.loads(text)
    results = []
    for item in parsed_list:
        if isinstance(item, dict):
            results.append(_ParsedRef(**{k: v for k, v in item.items() if k in _ParsedRef.model_fields}))
        else:
            results.append(_ParsedRef())
    return results


def parse_references_with_llm(
    references: list[dict[str, Any]],
    progress_cb=None,
) -> list[dict[str, Any]]:
    """
    Enhance a list of reference dicts with LLM-extracted fields.

    For each reference, if the LLM finds a field that the regex parser
    left empty (or finds a better title), the LLM value wins.

    Parameters
    ----------
    references : list[dict]
        Original reference dicts with at least a ``raw`` key.
    progress_cb : callable(done, total) | None
        Called after each batch completes.

    Returns
    -------
    list[dict]
        Same length; each dict is a copy enriched with LLM fields.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)

    total = len(references)
    enhanced: list[dict[str, Any]] = []
    done = 0

    for batch_start in range(0, total, _BATCH_SIZE):
        batch = references[batch_start : batch_start + _BATCH_SIZE]
        raw_texts = [r.get("raw", "") for r in batch]

        try:
            parsed = _call_llm(client, raw_texts)
        except Exception:
            # On failure, keep originals for this batch
            parsed = [_ParsedRef() for _ in batch]

        for ref, p in zip(batch, parsed):
            merged = dict(ref)
            for field in ("title", "authors", "year", "venue", "doi", "url"):
                llm_val = getattr(p, field, None)
                if llm_val:
                    merged[field] = llm_val
            enhanced.append(merged)

        done += len(batch)
        if progress_cb:
            progress_cb(done, total)

    return enhanced
