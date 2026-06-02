"""
LLM-based reference field extraction / repair.

Used as a post-pass over regex-parsed refs.  The regex parser
(`parsers._parse_raw_reference`) handles clean IEEE/APA/ACM styles well
but fails on edge cases — title and authors mashed together, year tokens
("2024b") promoted to title, DOI fragments dragged into venue.  The LLM
sees both the raw citation text AND the regex parse, and is asked to
*repair* fields that look wrong instead of re-extracting from scratch.

Backends
--------
- ``anthropic`` (default) — Claude Haiku 4.5, uses prompt caching on
  the system prompt.  Paid; see Anthropic pricing.
- ``aqueduct`` — free OpenAI-compatible endpoint hosted at
  https://aqueduct.ai.datalab.tuwien.ac.at/ (TU Wien DataLab).
  Supports Qwen 3.6 35B and others.  No prompt caching.

Switch via env:
    LLM_BACKEND=aqueduct LLM_MODEL=qwen-3.6-35b python ...
    LLM_BACKEND=anthropic LLM_MODEL=claude-haiku-4-5 python ...   # default
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from pydantic import BaseModel

_BATCH_SIZE = int(os.getenv("LLM_BATCH_SIZE") or "20")

# Parallel LLM batches.  Aqueduct/Qwen 3.6 35B is rate-limited around 120
# RPM (~2 RPS).  With 8 workers and batch latency ~3-5 s, we sit at well
# under that limit while collapsing N batches into ceil(N/8) wall-clock
# rounds — most papers finish in one round.  Anthropic Haiku tolerates
# higher concurrency too, so the same default works there.
_MAX_PARALLEL_BATCHES = int(os.getenv("LLM_PARALLEL", "8"))

_BACKEND = (os.getenv("LLM_BACKEND") or "anthropic").lower()
_MODEL = os.getenv("LLM_MODEL") or (
    "qwen-3.6-35b" if _BACKEND == "aqueduct" else "claude-haiku-4-5"
)
_AQUEDUCT_BASE_URL = os.getenv(
    "AQUEDUCT_BASE_URL", "https://aqueduct.ai.datalab.tuwien.ac.at/v1"
)

_SYSTEM_PROMPT = """\
You are an expert bibliographer.  For each input you'll receive a raw citation
string plus an initial regex parse of its fields (some of which may be wrong).

Your job is to return the *corrected* fields PLUS a classification of whether
the raw text is actually a bibliographic reference at all.

Common failure modes to watch for:
  - **Body text mistaken for a reference**: paragraphs from the paper's methods
    or appendix that got dragged into the bibliography (e.g. starts with
    "We", "Our", "Representing", "Similarly", "VSAs support…").  These have
    no author-year structure.  Set is_citation=false for these.
  - **Table/equation captions**: short notation strings like
    "Notation. n: sequence length…" — also is_citation=false.
  - **Title ↔ authors swap**: authors_field="Amir R", title_field="Kachooei
    and Mohammad H" — the real title is in the venue or after the swap.
    Fix the swap.
  - **Title ↔ venue swap**: title="Manning", venue="A structural probe for
    finding syntax in word representations" — the real title is the long
    descriptive string in venue.  Swap them.
  - A year token like "2024b" promoted to title
  - A DOI dragged into the venue field
  - Authors truncated at a period inside an initial ("J.")
  - Missing fields the regex couldn't find but the raw text contains

Return a JSON array (same length, same order) where each element has:
  - "is_citation": boolean.  True if the raw text is a bibliographic reference
                   (has author + year + title, even if scrambled).  False if
                   it's body text, an equation, a table caption, or otherwise
                   not actually a citation.
  - "title":   the paper or book title.  Must be the real title — never a year
               token, never an author name.  Null if is_citation=false.
  - "authors": author names, comma- or "and"-separated.  Null if not a citation.
  - "year":    4-digit year as a string, or null.
  - "venue":   journal / conference name with no DOI or page-range tail, or null.
  - "doi":     DOI string without URL prefix, or null.
  - "url":     any explicit URL in the raw string, or null.

Rules:
- Return ONLY the JSON array.  No prose, no markdown fences.
- Keep array length identical to input.
- Prefer the LONGER, more complete version of a field when the regex got a
  truncated piece of it.
- When swapping fields, swap them — don't just blank out the wrong one.
"""


class _ParsedRef(BaseModel):
    is_citation: bool = True
    title: str | None = None
    authors: str | None = None
    year: str | None = None
    venue: str | None = None
    doi: str | None = None
    url: str | None = None


import re as _re_lp


_LP_YEAR_RE        = _re_lp.compile(r"\b(?:19|20)\d{2}\.")
_LP_AUTHOR_OPENER  = _re_lp.compile(
    r"^[A-Z][a-zA-Z\-']+(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-zA-Z\-']+)?[,\s]+"
    r"(?:[A-Z]\.|[A-Z][a-zA-Z\-']+|and\s+\d+\s+others)"
)
_LP_AND_N_OTHERS_RE = _re_lp.compile(r"\band\s+\d+\s+others\b", _re_lp.IGNORECASE)


def _looks_garbled(ref: dict[str, Any]) -> bool:
    """
    Heuristic: does this regex- or GROBID-parsed ref need LLM repair?

    Used to skip the LLM for refs that already look clean, cutting cost
    by 70-80% on well-formatted bibs.  False positives are cheaper than
    false negatives here — better to send a borderline ref to the LLM
    than to skip a garbled one.
    """
    title = (ref.get("title") or "").strip()
    authors = (ref.get("authors") or "").strip()
    year = (str(ref.get("year") or "")).strip()

    if not title:
        return True
    if len(title) < 8:                       # e.g. "2024b", "ibid"
        return True
    if title.lower() in {"et al.", "et al", "ibid.", "ibid"}:
        return True
    # Year-only title
    if title[:4].isdigit() and len(title) <= 6:
        return True
    # Title contains "YYYY." somewhere — likely the ref's year got swept
    # into the title (GROBID failure mode on "and 1 others" refs).
    if _LP_YEAR_RE.search(title):
        return True
    # Title contains "and N others." — same author-list-as-title pattern.
    if _LP_AND_N_OTHERS_RE.search(title):
        return True
    # Title opens with an author-list pattern ("Lastname, F." style).
    if _LP_AUTHOR_OPENER.match(title):
        return True
    # Authors-shaped title ("X and Y", "X, Y, Z")
    if (" and " in title.lower() and len(title.split()) <= 6):
        return True
    # Suspiciously short authors that look like a truncated name initial
    if authors and len(authors) < 6 and authors.endswith(("R", "H", "M", "J", "A")):
        return True
    # Missing authors AND missing year — likely a non-citation entry
    if not authors and not year:
        return True
    return False


def _build_user_payload(batch: list[dict[str, Any]]) -> str:
    inputs = [
        {
            "raw": r.get("raw", "")[:1500],
            "regex_parse": {
                k: r.get(k)
                for k in ("title", "authors", "year", "venue", "doi", "url")
                if r.get(k)
            },
        }
        for r in batch
    ]
    return json.dumps(inputs, ensure_ascii=False)


def _parse_response_text(text: str | None, batch_len: int) -> list[_ParsedRef]:
    # Qwen reasoning models occasionally return None content when their
    # <think> tokens consume the entire generation budget.  Treat that
    # like an empty response — caller will retry the batch.
    if not text:
        raise ValueError("empty response (likely token budget exhausted)")
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0].strip()
    # Aqueduct/Qwen sometimes wrap the JSON in <think>...</think> reasoning
    # blocks; strip them before JSON parsing.
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    if not text:
        raise ValueError("response was only a <think>...</think> block")
    parsed_list = json.loads(text)
    out: list[_ParsedRef] = []
    for item in parsed_list:
        if isinstance(item, dict):
            out.append(_ParsedRef(**{k: v for k, v in item.items() if k in _ParsedRef.model_fields}))
        else:
            out.append(_ParsedRef())
    # If model returned fewer entries than asked, pad with defaults so the
    # caller's zip with idx_batch doesn't silently drop refs.
    while len(out) < batch_len:
        out.append(_ParsedRef())
    return out[:batch_len]


def _call_anthropic(client, batch: list[dict[str, Any]]) -> list[_ParsedRef]:
    msg = client.messages.create(
        model=_MODEL,
        max_tokens=4096,
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": _build_user_payload(batch)}],
    )
    return _parse_response_text(msg.content[0].text, len(batch))


def _call_aqueduct(client, batch: list[dict[str, Any]]) -> list[_ParsedRef]:
    # Qwen 3.x has a `<think>` reasoning mode that's enabled by default
    # and consumes a LOT of tokens (and wall time) before producing the
    # actual JSON.  Disabling it via vLLM's chat_template_kwargs makes
    # each call ~10× faster (0.6 s vs 6 s in direct tests) — and the
    # task (structured citation parsing) doesn't need chain-of-thought.
    #
    # `timeout=60` caps per-batch wall time so a hung Aqueduct request
    # doesn't freeze a Streamlit session forever — falls back to the
    # caller's batch-fail handler.
    resp = client.chat.completions.create(
        model=_MODEL,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _build_user_payload(batch)},
        ],
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        timeout=60.0,
    )
    return _parse_response_text(resp.choices[0].message.content, len(batch))


def _call_llm(client, batch: list[dict[str, Any]]) -> list[_ParsedRef]:
    """Dispatch to the active backend.  Raises so the caller can log."""
    if _BACKEND == "aqueduct":
        return _call_aqueduct(client, batch)
    return _call_anthropic(client, batch)


def _make_client():
    """Construct the SDK client for the active backend, or None on missing key."""
    if _BACKEND == "aqueduct":
        api_key = os.getenv("AQUEDUCT_API_KEY")
        if not api_key:
            return None
        from openai import OpenAI
        return OpenAI(base_url=_AQUEDUCT_BASE_URL, api_key=api_key)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    import anthropic
    return anthropic.Anthropic(api_key=api_key)


def parse_references_with_llm(
    references: list[dict[str, Any]],
    progress_cb=None,
    repair_only: bool = False,
) -> list[dict[str, Any]]:
    """
    Repair regex-parsed references with Claude.

    Parameters
    ----------
    references : list[dict]
        Refs with at least ``raw`` (and usually some regex-parsed fields).
    progress_cb : callable(done, total) | None
    repair_only : bool
        When True (default), only refs that look garbled go to the LLM —
        clean refs pass through unchanged.  When False, every ref is sent.

    Returns
    -------
    list[dict]
        Same length; suspect refs have LLM-repaired fields merged in.
    """
    if not references:
        return references

    client = _make_client()
    if client is None:
        return references

    targets_idx = (
        [i for i, r in enumerate(references) if _looks_garbled(r)]
        if repair_only
        else list(range(len(references)))
    )
    if not targets_idx:
        return references
    enhanced = [dict(r) for r in references]
    total = len(targets_idx)
    done = 0

    llm_errors: list[str] = []

    # Build all batches up front so we can dispatch them in parallel.
    batches: list[tuple[list[int], list[dict[str, Any]]]] = []
    for batch_start in range(0, total, _BATCH_SIZE):
        idx_batch = targets_idx[batch_start : batch_start + _BATCH_SIZE]
        ref_batch = [references[i] for i in idx_batch]
        batches.append((idx_batch, ref_batch))

    def _merge(idx_batch: list[int], parsed: list[_ParsedRef]) -> None:
        for i, p in zip(idx_batch, parsed):
            if not p.is_citation:
                enhanced[i]["_not_a_citation"] = True
                enhanced[i]["title"] = ""
                enhanced[i]["doi"] = ""
                enhanced[i]["url"] = ""
                continue
            for field in ("title", "authors", "year", "venue", "doi", "url"):
                val = getattr(p, field, None)
                if val:
                    enhanced[i][field] = val

    max_workers = min(_MAX_PARALLEL_BATCHES, len(batches))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_call_llm, client, ref_batch): (idx_batch, ref_batch)
            for idx_batch, ref_batch in batches
        }
        for f in as_completed(futures):
            idx_batch, ref_batch = futures[f]
            try:
                parsed = f.result()
            except Exception as exc:
                llm_errors.append(str(exc)[:200])
                parsed = [_ParsedRef() for _ in ref_batch]
            _merge(idx_batch, parsed)
            done += len(idx_batch)
            if progress_cb:
                progress_cb(done, total)

    if llm_errors:
        # Surface a single warning (first error message is usually enough
        # — the rest are identical, e.g. all "credit balance too low").
        import sys as _sys
        print(
            f"\n[llm_parser] WARNING: {len(llm_errors)}/{(total + _BATCH_SIZE - 1) // _BATCH_SIZE} "
            f"LLM batches failed. First error: {llm_errors[0]}",
            file=_sys.stderr,
            flush=True,
        )
    return enhanced
