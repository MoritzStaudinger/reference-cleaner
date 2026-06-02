"""
Reference lookup against Semantic Scholar, DBLP, OpenAlex, arXiv, ACL Anthology, and Crossref.

Pipeline
--------
1. **S2 batch pre-warm** — for every ref that carries a DOI or arXiv ID,
   collapse those into a single ``POST /paper/batch`` request to S2 and
   pre-populate the per-source cache.  Subsequent per-ref lookups become
   cache hits, eliminating dozens of sequential round trips.
2. **Router** — per-ref classification picks the most likely source(s):
   arXiv ID → arxiv only, DOI → Crossref only, CS-conf venue → DBLP only,
   book signal → OpenLibrary only.  Identifier-based routing is treated
   as authoritative — if the routed source returns ``not_found`` we do
   NOT escalate (a fabricated DOI/arXiv ID stays fabricated).
3. **Escalation** — refs that the router couldn't classify, or that the
   routed (non-identifier) source missed, fan out to the remaining APIs.
4. **ACL backfill** — derive aclanthology.org URLs from any source that
   returned an ACL DOI / S2 external ACL ID.
5. **OpenLibrary** — runs only when ``_VENUE_BOOK_RE`` matches the raw or
   venue text.  The previous "fall through if all academic sources
   missed" rule was burning ~100 calls/paper for zero hits.

All HTTP traffic goes through a module-level ``requests.Session()`` with
a 20-connection keep-alive pool, eliminating per-request TCP+TLS setup.

Validation sources (title + venue → card colour):
    semantic_scholar, acl_anthology, dblp, openalex, crossref

Identifier source (link only):
    arxiv
"""

from __future__ import annotations

import os
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from rapidfuzz import fuzz as _fuzz

import requests as _requests
from requests.adapters import HTTPAdapter
from dotenv import load_dotenv

from cache import cache_key as _cache_key, cached as _cached, get_cache as _get_cache

load_dotenv()

# Connection-pooled HTTP session — every source uses this so TCP+TLS
# handshakes amortise across the ~200 calls per paper.
_SESSION = _requests.Session()
_adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=0)
_SESSION.mount("https://", _adapter)
_SESSION.mount("http://", _adapter)

# Transient HTTP statuses worth retrying — public APIs hit these under load.
_RETRY_STATUSES = {429, 500, 502, 503, 504}


_MAX_RETRY_AFTER = 5.0   # never sleep longer than this between retries


def _http_get_retry(
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: int = 10,
    attempts: int = 3,
) -> _requests.Response:
    """
    GET with bounded exponential backoff on 429/5xx + transient network errors.

    Sleep schedule (seconds): 0.5, 1.5 between attempts (max 2 sleeps).
    `Retry-After` is honoured but CAPPED at _MAX_RETRY_AFTER seconds — some
    APIs send 30+ s values that would freeze the pipeline.  When a server is
    that overloaded, giving up and reporting `error` is better than hanging.
    """
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            r = _SESSION.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code in _RETRY_STATUSES and i < attempts - 1:
                ra = r.headers.get("Retry-After")
                retry_after = 0.0
                if ra:
                    try:
                        retry_after = float(ra)
                    except ValueError:
                        retry_after = 0.0
                sleep_for = min(_MAX_RETRY_AFTER, max(retry_after, 0.5 + i))
                time.sleep(sleep_for)
                continue
            return r
        except (_requests.ConnectionError, _requests.Timeout) as exc:
            last_exc = exc
            if i == attempts - 1:
                raise
            time.sleep(0.5 + i)
    assert last_exc is not None
    raise last_exc

S2_API_KEY: str | None = os.getenv("S2_API_KEY") or None

_S2_BASE   = "https://api.semanticscholar.org/graph/v1"
_S2_FIELDS = "title,authors,year,externalIds,venue"

_CR_BASE    = "https://api.crossref.org/works"
_CR_HEADERS = {"User-Agent": "ReferenceCleaner/1.0 (mailto:contact@example.com)"}

_DBLP_BASE  = "https://dblp.org/search/publ/api"
_OA_BASE    = "https://api.openalex.org/works"
# OpenAlex grants ~10× higher rate limits ("polite pool") to identifiable
# users via the mailto query parameter.  Set OPENALEX_MAILTO in .env to
# your real address; we fall back to a generic one only if not set.
_OA_MAILTO  = os.getenv("OPENALEX_MAILTO", "moritz.staudinger@tuwien.ac.at")
_OA_HEADERS = {"User-Agent": f"ReferenceCleaner/1.0 (mailto:{_OA_MAILTO})"}

# Title similarity thresholds
MATCH_THRESHOLD          = 0.90
FUZZY_THRESHOLD          = 0.70
AUTHOR_MATCH_THRESHOLD   = 0.35   # last-name Jaccard to confirm identity
VENUE_MISMATCH_THRESHOLD = 0.40

_ARXIV_ID_RE = re.compile(
    r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})"
    r"|arXiv:(\d{4}\.\d{4,5})",
    re.IGNORECASE,
)
# DOI embedded in a doi.org / dx.doi.org URL
_DOI_FROM_URL_RE = re.compile(
    r"(?:(?:dx\.)?doi\.org/|doi:)\s*(10\.\d{4,}/[^\s,>\"'<]+)",
    re.IGNORECASE,
)
_DOI_RE = re.compile(r"10\.\d{4,}/\S+")

# ACL Anthology DOI prefix → used to derive aclanthology.org links without S2
_ACL_DOI_RE = re.compile(r"10\.18653/v1/(.+)", re.IGNORECASE)

# --- Source-routing patterns ---
# DBLP indexes most CS conferences cleanly — when we see one of these venue
# names, query DBLP only instead of fanning out to all five sources.
_VENUE_DBLP_RE = re.compile(
    r"\b(?:NeurIPS|NIPS|ICML|ICLR|AAAI|IJCAI|CVPR|ICCV|ECCV|"
    r"KDD|SIGIR|UAI|AISTATS|COLT|FOCS|STOC|SODA|"
    r"VLDB|SIGMOD|OSDI|SOSP|ASPLOS|ISCA|MICRO|"
    r"USENIX\s+Security|CCS|NDSS|"
    r"IROS|ICRA|RSS|"
    r"InterSpeech|ICASSP|WWW|WSDM|RecSys|CIKM|ICDM|ECML|"
    r"CHI|UIST|CSCW)\b",
    re.IGNORECASE,
)
# Strong book/monograph signals — use Open Library only.
_VENUE_BOOK_RE = re.compile(
    r"\bISBN[- ]?(?:13|10)?:?\s*[\d\-Xx]{10,17}|"
    r"\b(?:O'Reilly|MIT Press|Cambridge University Press|"
    r"Springer-Verlag|Wiley|McGraw-Hill|Pearson|Manning Publications|"
    r"Oxford University Press|Princeton University Press)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Normalisation & similarity
# ---------------------------------------------------------------------------

_LEADING_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+")


def _ascii_fold(text: str) -> str:
    """Strip diacritics: Müller → Muller, naïve → naive."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def _normalize(text: str) -> str:
    text = _ascii_fold(text.lower())
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _LEADING_ARTICLE_RE.sub("", text)
    return text


def title_similarity(t1: str, t2: str) -> float:
    """
    Robust title similarity.

    Combines three scorers and adds a subtitle-aware pass so common parser
    failure modes ("Foo" vs "Foo: A Subtitle", "The Foo" vs "Foo") don't
    score below threshold for the same paper.
    """
    if not t1 or not t2:
        return 0.0
    n1, n2 = _normalize(t1), _normalize(t2)

    # Use token_sort_ratio only.  partial_ratio and WRatio look generous
    # in isolation but they over-match plausible-sounding fabrications by
    # treating shared substrings like "large language models" or "survey
    # of X" as near-identity (verified: "A survey of causal discovery and
    # causal inference" vs "A Survey on Causal Inference" scores 0.86 on
    # partial_ratio and 0.85 on WRatio — both above our fuzzy floor —
    # whereas token_sort_ratio correctly says 0.68).
    score = _fuzz.token_sort_ratio(n1, n2) / 100.0

    # Subtitle-aware pass: if either title has a colon, also score the
    # pre-colon prefix.  Catches "Foo" vs "Foo: A Detailed Survey" without
    # the false-positive risk of plain partial_ratio.
    p1 = t1.split(":", 1)[0]
    p2 = t2.split(":", 1)[0]
    if (p1 != t1 or p2 != t2) and len(p1.strip()) >= 4 and len(p2.strip()) >= 4:
        np1, np2 = _normalize(p1), _normalize(p2)
        score = max(score, _fuzz.token_sort_ratio(np1, np2) / 100.0)

    # Existing prefix-substring boost (kept for full-prefix matches like
    # "Foo" inside "Foo extended").
    shorter, longer = (n1, n2) if len(n1) <= len(n2) else (n2, n1)
    if longer.startswith(shorter) and len(shorter) / len(longer) >= 0.6:
        score = max(score, 1.0)
    return score


def venue_similarity(v1: str, v2: str) -> float:
    """Fuzzy venue match. Handles 'EMNLP' vs 'Proceedings of EMNLP 2020' etc."""
    if not v1 or not v2:
        return 0.0
    n1, n2 = _normalize(v1), _normalize(v2)
    return max(
        _fuzz.token_sort_ratio(n1, n2) / 100.0,
        _fuzz.partial_ratio(n1, n2)    / 100.0,
    )


def _extract_lastnames(authors: str) -> set[str]:
    """
    Extract normalised, ASCII-folded last names from a free-form author string.

    Handles:
      "J. Smith"          → {smith}
      "Yan Wang"          → {wang}      (was buggy: previously returned {yan})
      "Smith, J."         → {smith}     (the " J." sub-part is dropped as noise)
      "Smith and Jones"   → {smith, jones}
      "Müller"            → {muller}    (ASCII-folded via _normalize)
    """
    parts = re.split(r"[,;]|\band\b", authors, flags=re.IGNORECASE)
    lastnames: set[str] = set()
    for part in parts:
        tokens = part.strip().split()
        if not tokens:
            continue
        # Multi-token: assume the last token is the family name.  Covers
        # both "J. Smith" (initial-first) and "Yan Wang" (full First Last).
        # Single-token: assume it IS a lastname (or noise that won't match
        # anything anyway).
        lastnames.add(_normalize(tokens[-1] if len(tokens) >= 2 else tokens[0]))
    return lastnames - {""}


def author_similarity(a1: str, a2: str) -> float:
    """
    Fuzzy-Jaccard over normalised, ASCII-folded last names.

    A lastname in one set matches a lastname in the other when their
    `fuzz.ratio` is >= 85 — handles Müller/Mueller, Saint-Exupéry vs
    Saint Exupery, OCR-style one-letter typos, etc.  Each match consumes
    its pair so duplicates aren't double-counted.
    """
    if not a1 or not a2:
        return 0.0
    ln1 = {n for n in _extract_lastnames(a1) if n}
    ln2 = {n for n in _extract_lastnames(a2) if n}
    if not ln1 or not ln2:
        return 0.0

    smaller, larger = (ln1, ln2) if len(ln1) <= len(ln2) else (ln2, ln1)
    consumed: set[str] = set()
    intersection = 0
    for name in smaller:
        # Exact match (after _normalize) — cheap path
        if name in larger and name not in consumed:
            consumed.add(name)
            intersection += 1
            continue
        # Fuzzy match against unmatched names in the larger set
        best = None
        best_score = 0.0
        for other in larger:
            if other in consumed:
                continue
            s = _fuzz.ratio(name, other) / 100.0
            if s > best_score:
                best, best_score = other, s
        if best is not None and best_score >= 0.85:
            consumed.add(best)
            intersection += 1

    union = len(ln1) + len(ln2) - intersection
    return intersection / union if union else 0.0


def similarity_label(sim: float | None) -> str:
    if sim is None:
        return "unknown"
    if sim >= MATCH_THRESHOLD:
        return "match"
    if sim >= FUZZY_THRESHOLD:
        return "fuzzy"
    return "mismatch"


def _venue_label(vsim: float | None, found_venue: str = "") -> str:
    """
    Venue label.  arXiv is treated as a prepublisher — a found venue of 'arXiv'
    never counts as a mismatch against a conference/journal name.
    """
    if found_venue and "arxiv" in found_venue.lower():
        return "unknown"
    if vsim is None:
        return "unknown"
    if vsim >= 0.70:
        return "match"
    if vsim >= VENUE_MISMATCH_THRESHOLD:
        return "fuzzy"
    return "mismatch"


def _composite_label(
    tsim: float | None,
    asim: float | None,
    found_venue: str = "",
    vsim: float | None = None,
) -> str:
    """
    Composite match label requiring BOTH title and author agreement.

    Green (match) – title ≥ 0.90 AND authors ≥ 0.35
                    (or title ≥ 0.90 with no author data available)
                    OR title ∈ [0.70, 0.90) AND authors ≥ 0.35
    Fuzzy        – any one signal confirms but the other doesn't
    Red          – title < 0.70 AND no author rescue

    Requiring author confirmation alongside a strong title catches the
    fabrication pattern where a coincidentally-similar real paper exists
    by *different* authors.  When authors are missing from extraction we
    fall back to title-only at the green threshold.
    """
    is_arxiv = "arxiv" in (found_venue or "").lower()

    # Tier 1: strong title (≥ 0.90).
    if tsim is not None and tsim >= MATCH_THRESHOLD:
        # Require author confirmation when we have authors on both sides;
        # otherwise (asim is None) trust the strong title.
        if asim is None or asim >= AUTHOR_MATCH_THRESHOLD:
            # Venue contradiction (non-arXiv) downgrades to fuzzy
            if vsim is not None and vsim < VENUE_MISMATCH_THRESHOLD and not is_arxiv:
                return "fuzzy"
            return "match"
        # Title matches but authors clearly disagree → likely a different
        # paper with a similar title (classic fabrication / wrong-DOI tell).
        return "fuzzy"

    # Tier 2: medium title (0.70 — 0.90) needs an explicit author confirm.
    if tsim is not None and tsim >= FUZZY_THRESHOLD:
        if asim is not None and asim >= AUTHOR_MATCH_THRESHOLD:
            if vsim is not None and vsim < VENUE_MISMATCH_THRESHOLD and not is_arxiv:
                return "fuzzy"
            return "match"
        return "fuzzy"

    # Tier 3: weak title — authors or non-arXiv venue can salvage it to fuzzy.
    author_confirms = asim is not None and asim >= AUTHOR_MATCH_THRESHOLD
    venue_confirms  = vsim is not None and vsim >= 0.70 and not is_arxiv
    if author_confirms or venue_confirms:
        return "fuzzy"
    return "mismatch"


# ---------------------------------------------------------------------------
# Identifier extraction helpers
# ---------------------------------------------------------------------------

def _extract_doi(ref: dict[str, Any]) -> str | None:
    """Return DOI from the doi field or from a doi.org URL in url/raw fields."""
    doi = ref.get("doi", "") or ""
    if doi:
        return doi.rstrip(".,)")
    for field in ("url", "raw"):
        val = ref.get(field, "") or ""
        m = _DOI_FROM_URL_RE.search(val)
        if m:
            return m.group(1).rstrip(".,)")
    return None


def _extract_arxiv_id(ref: dict[str, Any]) -> str | None:
    for field in ("url", "raw", "doi"):
        val = ref.get(field, "") or ""
        m   = _ARXIV_ID_RE.search(val)
        if m:
            return m.group(1) or m.group(2)
    return None


_URL_FOLLOW_HEADERS = {"User-Agent": "ReferenceCleaner/1.0 (mailto:contact@example.com)"}


def _resolve_url_doi(url: str) -> str | None:
    """
    Follow a URL and try to extract a DOI from:
      1. The final redirect URL (e.g. publisher → doi.org/…)
      2. HTML <meta> citation tags (citation_doi, dc.identifier)
    Returns a DOI string or None.  Silently swallows all errors.
    """
    try:
        r = _SESSION.get(url, allow_redirects=True, timeout=5,
                          headers=_URL_FOLLOW_HEADERS)
        # Redirect landed on a doi.org URL
        m = _DOI_FROM_URL_RE.search(r.url)
        if m:
            return m.group(1).rstrip(".,)")
        # Scan HTML meta tags (first 30 KB is enough)
        if "html" in r.headers.get("content-type", "").lower():
            chunk = r.text[:30_000]
            for pattern in (
                r'<meta[^>]+name=["\']citation_doi["\'][^>]+content=["\']([^"\']+)["\']',
                r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']citation_doi["\']',
                r'<meta[^>]+name=["\']dc\.identifier["\'][^>]+'
                r'content=["\'](?:doi:)?(10\.\d{4,}/[^"\']+)["\']',
            ):
                m2 = re.search(pattern, chunk, re.IGNORECASE)
                if m2:
                    doi_val = m2.group(1).strip().lstrip("doi:").strip()
                    if re.match(r"10\.\d{4,}/", doi_val):
                        return doi_val
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------

def _s2_headers(api_key: str | None) -> dict:
    return {"x-api-key": api_key} if api_key else {}


def _s2_get(path: str, params: dict, api_key: str | None) -> dict | None:
    url = f"{_S2_BASE}/{path}"
    for attempt in range(3):
        r = _SESSION.get(url, params=params, headers=_s2_headers(api_key), timeout=8)
        if r.status_code == 403:
            raise PermissionError("S2 403 Forbidden")
        if r.status_code == 404:
            return None
        if r.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("S2 rate limit persists after retries")


def _s2_paper_to_result(
    data: dict,
    extracted_title: str,
    extracted_venue: str,
    extracted_authors: str,
) -> dict[str, Any]:
    found_title   = data.get("title") or ""
    found_venue   = data.get("venue") or ""
    found_authors = ", ".join(
        a.get("name", "") for a in (data.get("authors") or [])
    )
    tsim = title_similarity(extracted_title, found_title)           if extracted_title  else None
    asim = author_similarity(extracted_authors, found_authors)      if (extracted_authors and found_authors) else None
    vsim = venue_similarity(extracted_venue, found_venue)           if (extracted_venue and found_venue) else None
    ext  = data.get("externalIds") or {}

    result: dict[str, Any] = {
        "status":        "found",
        "found_title":   found_title,
        "found_authors": found_authors,
        "found_venue":   found_venue,
        "similarity":    tsim,
        "authors_sim":   asim,
        "label":         _composite_label(tsim, asim, found_venue, vsim),
        "venue_sim":     vsim,
        "venue_label":   _venue_label(vsim, found_venue),
        "year":          data.get("year"),
        "url":           f"https://www.semanticscholar.org/paper/{data['paperId']}",
    }
    if ext.get("ACL"):
        acl_id = ext["ACL"]
        result["acl_url"] = f"https://aclanthology.org/{acl_id}"
        result["acl_id"]  = acl_id
    if ext.get("ArXiv"):
        result["arxiv_id"]  = ext["ArXiv"]
        result["arxiv_url"] = f"https://arxiv.org/abs/{ext['ArXiv']}"
    if ext.get("DOI"):
        result["doi"] = ext["DOI"]
    return result


@_cached("semantic_scholar")
def _lookup_semantic_scholar(ref: dict[str, Any]) -> dict[str, Any]:
    extracted_title   = ref.get("title", "")
    extracted_venue   = ref.get("venue", "")
    extracted_authors = ref.get("authors", "")
    doi = _extract_doi(ref)
    keys_to_try: list[str | None] = [S2_API_KEY, None] if S2_API_KEY else [None]

    for api_key in keys_to_try:
        try:
            data      = None
            wrong_doi = None

            if doi:
                doi_data = _s2_get(f"paper/DOI:{doi}", {"fields": _S2_FIELDS}, api_key)
                if doi_data:
                    if not extracted_title or title_similarity(extracted_title, doi_data.get("title", "")) >= 0.5:
                        data = doi_data
                    else:
                        wrong_doi = doi

            if data is None:
                arxiv_id = _extract_arxiv_id(ref)
                if arxiv_id:
                    data = _s2_get(f"paper/ARXIV:{arxiv_id}", {"fields": _S2_FIELDS}, api_key)

            if data is None and extracted_title:
                resp   = _s2_get("paper/search", {"query": extracted_title, "limit": 3, "fields": _S2_FIELDS}, api_key)
                papers = (resp or {}).get("data", [])
                data   = papers[0] if papers else None

            if data is None:
                return {"status": "not_found"}

            result = _s2_paper_to_result(data, extracted_title, extracted_venue, extracted_authors)
            if wrong_doi:
                result["wrong_doi"] = wrong_doi
            return result

        except PermissionError:
            continue
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    return {"status": "error", "error": "S2 API key rejected (403)"}


# --- Batch S2 lookup for refs carrying a DOI or arXiv ID -------------------

_S2_BATCH_SIZE = 500   # S2 batch endpoint cap


def _s2_batch_get(ids: list[str], api_key: str | None) -> list[dict | None]:
    """
    POST {ids: [...]} to /paper/batch. Returns a list aligned with the
    input where each entry is either a paper dict or None (not found).
    Raises on transport errors (caller decides whether to fall back).
    """
    if not ids:
        return []
    url = f"{_S2_BASE}/paper/batch"
    headers = {**_s2_headers(api_key), "Content-Type": "application/json"}
    for attempt in range(3):
        r = _SESSION.post(
            url,
            params={"fields": _S2_FIELDS},
            headers=headers,
            json={"ids": ids},
            timeout=15,
        )
        if r.status_code == 403:
            raise PermissionError("S2 403 Forbidden")
        if r.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("S2 batch rate limit persists after retries")


def _s2_batch_prewarm(refs: list[dict[str, Any]]) -> int:
    """
    Pre-populate the S2 cache for every ref that carries a DOI or arXiv ID.
    One HTTP request replaces up to 500 sequential lookups.

    Returns the number of refs pre-warmed.  Silently no-ops on transport
    errors — the per-ref pipeline will retry normally.
    """
    if not refs:
        return 0

    cache = _get_cache()
    pending: list[tuple[str, str, dict[str, Any]]] = []
    seen_keys: set[str] = set()

    for ref in refs:
        key = _cache_key(ref)
        if not key or key in seen_keys:
            continue
        if cache.get("semantic_scholar", key) is not None:
            continue
        doi   = _extract_doi(ref)
        arxiv = _extract_arxiv_id(ref)
        if doi:
            pending.append((key, f"DOI:{doi}", ref))
            seen_keys.add(key)
        elif arxiv:
            pending.append((key, f"ARXIV:{arxiv}", ref))
            seen_keys.add(key)

    if not pending:
        return 0

    keys_to_try: list[str | None] = [S2_API_KEY, None] if S2_API_KEY else [None]
    papers: list[dict | None] | None = None
    for api_key in keys_to_try:
        try:
            papers = _s2_batch_get([s2id for _, s2id, _ in pending], api_key)
            break
        except PermissionError:
            continue
        except Exception:
            return 0   # fall back silently

    if papers is None:
        return 0

    for (key, _s2id, ref), paper in zip(pending, papers):
        if paper is None:
            cache.set("semantic_scholar", key, {"status": "not_found"})
        else:
            result = _s2_paper_to_result(
                paper,
                ref.get("title", ""),
                ref.get("venue", ""),
                ref.get("authors", ""),
            )
            cache.set("semantic_scholar", key, result)

    return len(pending)


# ---------------------------------------------------------------------------
# DBLP
# ---------------------------------------------------------------------------

@_cached("dblp")
def _lookup_dblp(ref: dict[str, Any]) -> dict[str, Any]:
    extracted_title   = ref.get("title", "")
    extracted_venue   = ref.get("venue", "")
    extracted_authors = ref.get("authors", "")
    if not extracted_title:
        return {"status": "skipped"}

    try:
        r = _http_get_retry(
            _DBLP_BASE,
            params={"q": extracted_title, "format": "json", "h": 10},
            timeout=8,
        )
        r.raise_for_status()
        hits = r.json().get("result", {}).get("hits", {}).get("hit", [])
        if not hits:
            return {"status": "not_found"}

        best = max(
            hits,
            key=lambda h: title_similarity(extracted_title, h.get("info", {}).get("title", "")),
        )
        info        = best.get("info", {})
        found_title = info.get("title", "")
        if title_similarity(extracted_title, found_title) < FUZZY_THRESHOLD:
            return {"status": "not_found"}

        found_venue = info.get("venue", "")
        raw_authors = info.get("authors", {}).get("author", [])
        if isinstance(raw_authors, dict):
            raw_authors = [raw_authors]
        found_authors = ", ".join(
            a.get("text", "") if isinstance(a, dict) else str(a)
            for a in raw_authors
        )

        tsim = title_similarity(extracted_title, found_title)
        asim = author_similarity(extracted_authors, found_authors) if (extracted_authors and found_authors) else None
        vsim = venue_similarity(extracted_venue, found_venue)      if (extracted_venue and found_venue) else None
        doi  = info.get("doi", "")
        url  = info.get("url", "") or (f"https://doi.org/{doi}" if doi else "")

        return {
            "status":        "found",
            "found_title":   found_title,
            "found_authors": found_authors,
            "found_venue":   found_venue,
            "similarity":    tsim,
            "authors_sim":   asim,
            "label":         _composite_label(tsim, asim, found_venue, vsim),
            "venue_sim":     vsim,
            "venue_label":   _venue_label(vsim, found_venue),
            "year":          info.get("year"),
            "doi":           doi,
            "url":           url,
        }

    except Exception as exc:
        return {"status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# OpenAlex
# ---------------------------------------------------------------------------

@_cached("openalex")
def _lookup_openalex(ref: dict[str, Any]) -> dict[str, Any]:
    extracted_title   = ref.get("title", "")
    extracted_venue   = ref.get("venue", "")
    extracted_authors = ref.get("authors", "")
    doi = _extract_doi(ref)

    try:
        data = None

        if doi:
            r = _http_get_retry(
                f"{_OA_BASE}/https://doi.org/{doi}",
                params={"mailto": _OA_MAILTO},
                headers=_OA_HEADERS,
                timeout=8,
            )
            if r.status_code == 200:
                data = r.json()

        if data is None and extracted_title:
            r = _http_get_retry(
                _OA_BASE,
                params={
                    "search":   extracted_title,
                    "per_page": 3,
                    "select":   "id,title,authorships,publication_year,primary_location,doi,open_access",
                    "mailto":   _OA_MAILTO,
                },
                headers=_OA_HEADERS,
                timeout=8,
            )
            r.raise_for_status()
            results = r.json().get("results", [])
            if results:
                data = max(
                    results,
                    key=lambda w: title_similarity(extracted_title, w.get("title", "") or ""),
                )

        if data is None:
            return {"status": "not_found"}

        return _oa_work_to_result(data, extracted_title, extracted_venue, extracted_authors)

    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def _oa_work_to_result(
    data: dict,
    extracted_title: str,
    extracted_venue: str,
    extracted_authors: str,
) -> dict[str, Any]:
    found_title   = data.get("title") or ""
    doi_found     = (data.get("doi", "") or "").replace("https://doi.org/", "")
    loc           = data.get("primary_location") or {}
    source        = loc.get("source") or {}
    found_venue   = source.get("display_name") or ""
    found_authors = ", ".join(
        (a.get("author") or {}).get("display_name", "")
        for a in (data.get("authorships") or [])
    )
    oa_url = (data.get("open_access") or {}).get("oa_url") or ""
    url    = oa_url or (f"https://doi.org/{doi_found}" if doi_found else data.get("id", ""))

    tsim = title_similarity(extracted_title, found_title) if extracted_title else None
    if extracted_title and tsim is not None and tsim < FUZZY_THRESHOLD:
        return {"status": "not_found"}
    asim = author_similarity(extracted_authors, found_authors) if (extracted_authors and found_authors) else None
    vsim = venue_similarity(extracted_venue, found_venue)      if (extracted_venue and found_venue) else None

    return {
        "status":        "found",
        "found_title":   found_title,
        "found_authors": found_authors,
        "found_venue":   found_venue,
        "similarity":    tsim,
        "authors_sim":   asim,
        "label":         _composite_label(tsim, asim, found_venue, vsim),
        "venue_sim":     vsim,
        "venue_label":   _venue_label(vsim, found_venue),
        "year":          data.get("publication_year"),
        "doi":           doi_found,
        "url":           url,
    }


def _oa_batch_prewarm(refs: list[dict[str, Any]]) -> int:
    """
    Pre-populate the OpenAlex cache by querying its filter API once per
    chunk of up to 50 DOIs.  Mirrors `_s2_batch_prewarm` and lets a DOI
    ref check two independent sources without two sequential round trips.
    """
    if not refs:
        return 0
    cache = _get_cache()

    pending: list[tuple[str, str, dict[str, Any]]] = []   # (cache_key, doi_lower, ref)
    seen: set[str] = set()
    for ref in refs:
        key = _cache_key(ref)
        if not key or key in seen:
            continue
        if cache.get("openalex", key) is not None:
            continue
        doi = _extract_doi(ref)
        if not doi:
            continue
        pending.append((key, doi.lower(), ref))
        seen.add(key)
    if not pending:
        return 0

    works_by_doi: dict[str, dict] = {}
    for i in range(0, len(pending), 50):
        chunk = pending[i : i + 50]
        doi_filter = "|".join(d for _, d, _ in chunk)
        params = {
            "filter":   f"doi:{doi_filter}",
            "per_page": 50,
            "select":   "id,title,authorships,publication_year,primary_location,doi,open_access",
            "mailto":   _OA_MAILTO,
        }
        try:
            r = _http_get_retry(_OA_BASE, params=params, headers=_OA_HEADERS, timeout=15)
            r.raise_for_status()
            for work in r.json().get("results", []):
                wdoi = (work.get("doi") or "").replace("https://doi.org/", "").lower()
                if wdoi:
                    works_by_doi[wdoi] = work
        except Exception:
            continue   # leave this chunk to the per-ref fallback

    for key, doi, ref in pending:
        work = works_by_doi.get(doi)
        if work is None:
            cache.set("openalex", key, {"status": "not_found"})
        else:
            result = _oa_work_to_result(
                work,
                ref.get("title", ""),
                ref.get("venue", ""),
                ref.get("authors", ""),
            )
            cache.set("openalex", key, result)
    return len(pending)


# ---------------------------------------------------------------------------
# arXiv  (identifier only — link + ID, no title/venue validation)
# ---------------------------------------------------------------------------

def _arxiv_batch_prewarm(refs: list[dict[str, Any]]) -> int:
    """
    Pre-populate the arXiv cache by querying its API with `id_list=`
    (max 100 IDs per request).  arXiv-bearing refs become cache hits
    instead of N sequential calls — and we hit the flaky export API
    only once per chunk.
    """
    if not refs:
        return 0
    cache = _get_cache()

    pending: list[tuple[str, str, dict[str, Any]]] = []
    seen: set[str] = set()
    for ref in refs:
        key = _cache_key(ref)
        if not key or key in seen:
            continue
        if cache.get("arxiv", key) is not None:
            continue
        arxiv_id = _extract_arxiv_id(ref)
        if not arxiv_id:
            continue
        pending.append((key, arxiv_id, ref))
        seen.add(key)
    if not pending:
        return 0

    import arxiv
    papers_by_id: dict[str, Any] = {}
    try:
        client = arxiv.Client(num_retries=5, delay_seconds=2, page_size=100)
        for i in range(0, len(pending), 100):
            chunk = pending[i : i + 100]
            search = arxiv.Search(id_list=[p[1] for p in chunk])
            for paper in client.results(search):
                full = paper.entry_id.split("/abs/")[-1]
                base = full.split("v")[0]     # strip version (1234.5678v3 -> 1234.5678)
                papers_by_id[base] = paper
    except Exception:
        return 0

    for key, arxiv_id, ref in pending:
        paper = papers_by_id.get(arxiv_id)
        if paper is None:
            cache.set("arxiv", key, {"status": "not_found"})
        else:
            full_id        = paper.entry_id.split("/abs/")[-1]
            found_title    = (paper.title or "").strip()
            found_authors  = ", ".join(a.name for a in (paper.authors or []) if a.name)
            extracted_title   = ref.get("title", "")
            extracted_authors = ref.get("authors", "")
            tsim = title_similarity(extracted_title, found_title) if extracted_title else None
            asim = (author_similarity(extracted_authors, found_authors)
                    if (extracted_authors and found_authors) else None)
            cache.set("arxiv", key, {
                "status":        "found",
                "found_title":   found_title,
                "found_authors": found_authors,
                "similarity":    tsim,
                "authors_sim":   asim,
                "label":         _composite_label(tsim, asim) if tsim is not None else None,
                "venue_sim":     None,
                "venue_label":   "unknown",
                "arxiv_id":      full_id,
                "url":           paper.entry_id,
            })
    return len(pending)


@_cached("arxiv")
def _lookup_arxiv(ref: dict[str, Any]) -> dict[str, Any]:
    """
    Resolve an arXiv ID and compare the actual paper's title against the
    cited title.  This catches the "valid arXiv ID but pointing to a
    different paper" failure mode — a common pattern in fabricated
    citations that pick a plausible-looking ID at random.

    Without title comparison, `arxiv: found` would falsely confirm a
    fabricated ref whose ID happens to resolve.
    """
    import arxiv

    arxiv_id = _extract_arxiv_id(ref)
    if not arxiv_id:
        return {"status": "skipped"}

    try:
        client = arxiv.Client(num_retries=5, delay_seconds=2)
        search = arxiv.Search(id_list=[arxiv_id])
        results = list(client.results(search))
        if not results:
            return {"status": "not_found"}

        paper          = results[0]
        arxiv_id_found = paper.entry_id.split("/abs/")[-1]
        found_title    = (paper.title or "").strip()
        found_authors  = ", ".join(a.name for a in (paper.authors or []) if a.name)

        extracted_title   = ref.get("title", "")
        extracted_authors = ref.get("authors", "")

        tsim = title_similarity(extracted_title, found_title) if extracted_title else None
        asim = (author_similarity(extracted_authors, found_authors)
                if (extracted_authors and found_authors) else None)

        return {
            "status":        "found",
            "found_title":   found_title,
            "found_authors": found_authors,
            "similarity":    tsim,
            "authors_sim":   asim,
            "label":         _composite_label(tsim, asim) if tsim is not None else None,
            "venue_sim":     None,
            "venue_label":   "unknown",
            "arxiv_id":      arxiv_id_found,
            "url":           paper.entry_id,
        }

    except Exception as exc:
        return {"status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# Crossref  (full validation when DOI present; title-search fallback)
# ---------------------------------------------------------------------------

@_cached("crossref")
def _lookup_crossref(ref: dict[str, Any]) -> dict[str, Any]:
    extracted_title   = ref.get("title", "")
    extracted_venue   = ref.get("venue", "")
    extracted_authors = ref.get("authors", "")
    doi = _extract_doi(ref)

    def _build(msg: dict, doi_val: str) -> dict[str, Any]:
        found_title = (msg.get("title") or [""])[0]
        ct          = msg.get("container-title") or []
        found_venue = ct[0] if ct else (msg.get("event") or {}).get("name", "")
        found_authors = "; ".join(
            f"{a.get('given', '')} {a.get('family', '')}".strip()
            for a in (msg.get("author") or [])
        )
        year_parts = (msg.get("published", {}).get("date-parts") or [[None]])[0]
        year = str(year_parts[0]) if year_parts and year_parts[0] else ""

        tsim = title_similarity(extracted_title, found_title)      if (extracted_title and found_title) else None
        asim = author_similarity(extracted_authors, found_authors)  if (extracted_authors and found_authors) else None
        vsim = venue_similarity(extracted_venue, found_venue)       if (extracted_venue and found_venue) else None

        return {
            "status":        "found",
            "found_title":   found_title,
            "found_authors": found_authors,
            "found_venue":   found_venue,
            "similarity":    tsim,
            "authors_sim":   asim,
            "label":         _composite_label(tsim, asim, found_venue, vsim),
            "venue_sim":     vsim,
            "venue_label":   _venue_label(vsim, found_venue),
            "year":          year,
            "doi":           doi_val,
            "url":           f"https://doi.org/{doi_val}",
        }

    try:
        if doi:
            r = _http_get_retry(f"{_CR_BASE}/{doi}", headers=_CR_HEADERS, timeout=8)
            if r.status_code == 200:
                msg = r.json().get("message", {})
                return _build(msg, msg.get("DOI", doi))
            if r.status_code == 404:
                # Crossref says this DOI doesn't exist → likely fabricated.
                # Don't fall back to title search; a title-search hit on a
                # DIFFERENT paper would mask the bad DOI.
                return {"status": "not_found"}

        if extracted_title:
            r = _http_get_retry(
                _CR_BASE,
                params={"query.title": extracted_title, "rows": 1},
                headers=_CR_HEADERS,
                timeout=8,
            )
            r.raise_for_status()
            items = r.json().get("message", {}).get("items", [])
            if items:
                result = _build(items[0], items[0].get("DOI", ""))
                if result.get("similarity") is not None and result["similarity"] < FUZZY_THRESHOLD:
                    return {"status": "not_found"}
                return result

        return {"status": "not_found"}

    except Exception as exc:
        return {"status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# Open Library  (fallback for books, proceedings, and government reports)
# ---------------------------------------------------------------------------

_OL_BASE    = "https://openlibrary.org/search.json"
_OL_HEADERS = {"User-Agent": "ReferenceCleaner/1.0 (mailto:contact@example.com)"}


@_cached("openlibrary")
def _lookup_openlibrary(ref: dict[str, Any]) -> dict[str, Any]:
    """
    Search Open Library by title (+ first-author hint when available).
    Used only as a fallback when all academic databases fail — it covers
    books, proceedings volumes, government reports, and edited collections
    that are absent from S2 / DBLP / OpenAlex / Crossref.
    Venue comparison is skipped: OL returns publishers, not conference names.
    """
    extracted_title   = ref.get("title", "")
    extracted_authors = ref.get("authors", "")
    if not extracted_title:
        return {"status": "skipped"}

    try:
        params: dict[str, Any] = {
            "title":  extracted_title,
            "limit":  5,
            "fields": "key,title,author_name,first_publish_year,publisher",
        }
        # Add first author's last name as an optional refinement
        lastnames = _extract_lastnames(extracted_authors) if extracted_authors else set()
        if lastnames:
            params["author"] = next(iter(lastnames))

        r = _SESSION.get(_OL_BASE, params=params, headers=_OL_HEADERS, timeout=8)
        r.raise_for_status()
        docs = r.json().get("docs", [])

        if not docs:
            # Retry without author constraint if the first attempt was empty
            if "author" in params:
                del params["author"]
                r = _SESSION.get(_OL_BASE, params=params, headers=_OL_HEADERS, timeout=8)
                r.raise_for_status()
                docs = r.json().get("docs", [])

        if not docs:
            return {"status": "not_found"}

        best = max(
            docs,
            key=lambda d: title_similarity(extracted_title, d.get("title", "") or ""),
        )
        found_title = best.get("title", "")
        tsim = title_similarity(extracted_title, found_title)

        if tsim < FUZZY_THRESHOLD:
            return {"status": "not_found"}

        found_authors = ", ".join(best.get("author_name") or [])
        asim = author_similarity(extracted_authors, found_authors) if (extracted_authors and found_authors) else None
        year = str(best.get("first_publish_year")) if best.get("first_publish_year") else ""
        key  = best.get("key", "")
        url  = f"https://openlibrary.org{key}" if key else ""

        return {
            "status":        "found",
            "found_title":   found_title,
            "found_authors": found_authors,
            "found_venue":   "",          # OL has publishers, not venues
            "similarity":    tsim,
            "authors_sim":   asim,
            "label":         _composite_label(tsim, asim),
            "venue_sim":     None,
            "venue_label":   "unknown",   # don't compare publisher to conference
            "year":          year,
            "url":           url,
        }

    except Exception as exc:
        return {"status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# Google Scholar  (via the `scholarly` library — last-resort title verifier)
# ---------------------------------------------------------------------------
#
# Scholar has by far the broadest coverage (theses, workshop papers, books,
# non-English work) but no API, aggressive CAPTCHA, and no batch endpoint.
# We therefore:
#   - call it only as Phase 5, when no academic source matched
#   - rate-limit to one request every _SCHOLARLY_MIN_DELAY seconds
#   - set a session-wide `_SCHOLARLY_BLOCKED` flag the moment Google
#     CAPTCHAs us, so we don't waste minutes hammering a wall

_SCHOLARLY_LAST_CALL: float = 0.0
_SCHOLARLY_LOCK = threading.Lock()
_SCHOLARLY_BLOCKED = False
_SCHOLARLY_MIN_DELAY = 2.5
_SCHOLARLY_BLOCK_HINTS = ("captcha", "blocked", "429", "too many requests",
                          "maxtriesexceeded")

# Per-lookup_all() budget: Scholar is rate-limited and serialised, so on
# papers with many unmatched refs it would otherwise dominate wall time.
# Reset by `lookup_all` at the start of each invocation.
_SCHOLARLY_BUDGET_DEFAULT  = 30
_SCHOLARLY_BUDGET_LOCK     = threading.Lock()
_SCHOLARLY_BUDGET_REMAINING: int = 0


def _scholarly_take_budget() -> bool:
    """Atomically claim one Scholar call.  Returns False once depleted."""
    global _SCHOLARLY_BUDGET_REMAINING
    with _SCHOLARLY_BUDGET_LOCK:
        if _SCHOLARLY_BUDGET_REMAINING <= 0:
            return False
        _SCHOLARLY_BUDGET_REMAINING -= 1
        return True


def _scholarly_reset_budget(n: int = _SCHOLARLY_BUDGET_DEFAULT) -> None:
    global _SCHOLARLY_BUDGET_REMAINING
    with _SCHOLARLY_BUDGET_LOCK:
        _SCHOLARLY_BUDGET_REMAINING = max(0, n)


@_cached("scholarly")
def _lookup_scholarly(ref: dict[str, Any]) -> dict[str, Any]:
    global _SCHOLARLY_LAST_CALL, _SCHOLARLY_BLOCKED

    if _SCHOLARLY_BLOCKED:
        return {"status": "skipped"}

    extracted_title   = (ref.get("title") or "").strip()
    extracted_venue   = ref.get("venue", "") or ""
    extracted_authors = ref.get("authors", "") or ""
    if len(extracted_title) < 10:
        return {"status": "skipped"}

    # Per-lookup_all() budget: ensures Scholar doesn't dominate wall time
    # on papers with many unmatched refs.  Refuses extra calls quietly.
    if not _scholarly_take_budget():
        return {"status": "skipped"}

    try:
        from scholarly import scholarly as _sch
    except Exception:
        return {"status": "skipped"}

    # Serialise calls and pace at ~one every 2.5 s.
    with _SCHOLARLY_LOCK:
        wait = _SCHOLARLY_MIN_DELAY - (time.time() - _SCHOLARLY_LAST_CALL)
        if wait > 0:
            time.sleep(wait)
        _SCHOLARLY_LAST_CALL = time.time()

        try:
            pub = _sch.search_single_pub(extracted_title)
        except Exception as exc:
            msg = str(exc).lower()
            if any(h in msg for h in _SCHOLARLY_BLOCK_HINTS):
                _SCHOLARLY_BLOCKED = True
            return {"status": "error", "error": str(exc)[:120]}

    if not pub:
        return {"status": "not_found"}

    bib            = pub.get("bib", {}) or {}
    found_title    = (bib.get("title") or "").strip()
    if not found_title:
        return {"status": "not_found"}

    raw_authors    = bib.get("author") or []
    found_authors  = ", ".join(raw_authors) if isinstance(raw_authors, list) else str(raw_authors)
    found_venue    = (bib.get("venue") or "").rstrip("…").strip()
    year           = bib.get("pub_year") or ""
    pub_url        = pub.get("pub_url") or ""

    tsim = title_similarity(extracted_title, found_title)
    if tsim < FUZZY_THRESHOLD:
        return {"status": "not_found"}

    asim = author_similarity(extracted_authors, found_authors) if (extracted_authors and found_authors) else None
    vsim = venue_similarity(extracted_venue, found_venue)      if (extracted_venue and found_venue) else None

    return {
        "status":        "found",
        "found_title":   found_title,
        "found_authors": found_authors,
        "found_venue":   found_venue,
        "similarity":    tsim,
        "authors_sim":   asim,
        "label":         _composite_label(tsim, asim, found_venue, vsim),
        "venue_sim":     vsim,
        "venue_label":   _venue_label(vsim, found_venue),
        "year":          year,
        "url":           pub_url,
    }


# ---------------------------------------------------------------------------
# Source routing
# ---------------------------------------------------------------------------

_LOOKUP_FNS = {
    "semantic_scholar": _lookup_semantic_scholar,
    "dblp":             _lookup_dblp,
    "openalex":         _lookup_openalex,
    "arxiv":            _lookup_arxiv,
    "crossref":         _lookup_crossref,
}
_ALL_API_SOURCES = list(_LOOKUP_FNS.keys())


def _classify_ref(ref: dict[str, Any]) -> list[str]:
    """
    Pick which API source(s) to query based on identifiers and venue cues.

    Identifier-bearing refs query multiple sources in parallel and vote —
    any single 503/missing entry should not mark a real paper as fabricated.
    Non-identifier refs return a single best-guess source and escalate
    if it misses.
    """
    if _extract_arxiv_id(ref):
        # arXiv + S2 (S2 indexes preprints via externalIds.ArXiv)
        return ["arxiv", "semantic_scholar"]
    if _extract_doi(ref):
        # Three independent DOI registries — vote prevents a single
        # source's miss from flagging a real paper as fabricated.
        return ["crossref", "semantic_scholar", "openalex"]

    text = " ".join([
        ref.get("raw", "") or "",
        ref.get("venue", "") or "",
    ])
    if _VENUE_DBLP_RE.search(text):
        return ["dblp"]
    if _VENUE_BOOK_RE.search(text):
        return ["openlibrary"]

    return []   # unclear — fan out


def _run_sources(
    ref: dict[str, Any],
    sources: list[str],
    early_exit_on_match: bool = True,
) -> dict[str, Any]:
    """
    Run named sources in parallel and return their results.

    With ``early_exit_on_match=True`` (the default), as soon as one source
    returns label="match" we stop waiting on the others and return.  The
    still-running threads finish in the background (they keep populating
    the cache for the next paper), but the caller no longer pays their
    tail latency.

    The slowest source on a title-only fanout would otherwise gate the
    whole ref — early-exit collapses that to "fastest match wins".
    """
    if not sources:
        return {}
    out: dict[str, Any] = {}
    fns = {s: _LOOKUP_FNS[s] for s in sources if s in _LOOKUP_FNS}
    if not fns:
        return out

    pool = ThreadPoolExecutor(max_workers=min(5, len(fns)))
    try:
        futures = {pool.submit(fn, ref): name for name, fn in fns.items()}
        for f in as_completed(futures):
            name = futures[f]
            try:
                out[name] = f.result()
            except Exception as exc:
                out[name] = {"status": "error", "error": str(exc)}

            if early_exit_on_match and out[name].get("label") == "match":
                # Mark remaining sources as skipped in the result so the
                # audit shows the short-circuit; the background threads
                # finish on their own.
                for other_name in fns:
                    out.setdefault(other_name, {"status": "skipped"})
                break
    finally:
        # wait=False returns immediately; in-flight threads continue but
        # don't block this ref's return path.
        pool.shutdown(wait=False, cancel_futures=True)
    return out


def _has_match(results: dict[str, Any], sources: list[str]) -> bool:
    return any(results.get(s, {}).get("label") == "match" for s in sources)


# ---------------------------------------------------------------------------
# Junk filter — skip parser garbage before any lookup
# ---------------------------------------------------------------------------

# Title is "2024", "2024b", possibly with trailing punctuation
_YEAR_TITLE_RE = re.compile(r"^\d{4}[a-z]?[\.\,]?$")
# Title is only digits / punctuation / whitespace
_NONALPHA_TITLE_RE = re.compile(r"^[\d\s\.\,\-\:;\(\)]+$")
# Common parser-junk strings that aren't real titles
_JUNK_TITLE_TOKENS = {"et al.", "et al", "ibid.", "ibid", "op. cit.", "n.d.", "n/a"}


def _is_junk_ref(ref: dict[str, Any]) -> bool:
    """
    Detect parser-garbage refs that aren't worth looking up.

    Catches:
      - LLM repair flagging the raw text as body-text / not a citation
      - The common pymupdf4llm failure where a year token like "2024b"
        becomes the title
      - Empty / numeric-only / near-empty titles
    These would otherwise burn a full 5-source escalation each AND poison
    the match-rate denominator.
    """
    if ref.get("_not_a_citation"):
        return True
    title = (ref.get("title") or "").strip()
    if not title:
        # No title at all — only useful if we have a DOI or arXiv ID
        return not (_extract_doi(ref) or _extract_arxiv_id(ref))

    low = title.lower().strip(".,;:")
    if low in _JUNK_TITLE_TOKENS:
        return True
    if _YEAR_TITLE_RE.match(title):
        return True
    if _NONALPHA_TITLE_RE.match(title):
        return True
    # Very short title with no authors is almost always parser noise
    if len(title) < 5 and not (ref.get("authors") or "").strip():
        return True
    return False


def _junk_result() -> dict[str, Any]:
    """Stub lookup result for a ref that was filtered as parser garbage."""
    out: dict[str, Any] = {
        src: {"status": "skipped"}
        for src in _ALL_API_SOURCES + ["openlibrary", "acl_anthology", "scholarly"]
    }
    out["_router_path"] = "junk_filter"
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _book_text(ref: dict[str, Any]) -> str:
    return " ".join([ref.get("raw") or "", ref.get("venue") or ""])


def _all_not_found(results: dict[str, Any], sources: list[str]) -> bool:
    """True only if every named source returned a clean not_found (no errors)."""
    return bool(sources) and all(
        results.get(s, {}).get("status") == "not_found" for s in sources
    )


def lookup_reference(ref: dict[str, Any]) -> dict[str, Any]:
    """
    Look up a single reference through a tiered pipeline:

      Phase 1  Source router — pick 1-2 APIs based on identifiers/venue.
               Identifier-based routes (arxiv / crossref) are treated as
               authoritative: a `not_found` answer there stops the pipeline.
      Phase 2  Escalate to the remaining APIs when the router missed.
      Phase 3  ACL anthology derivation from S2 externalIds / ACL DOIs.
      Phase 4  Open Library — only when `_VENUE_BOOK_RE` matches.  Earlier
               implementations ran it as a universal fallback and burned
               ~100 calls/paper for zero hits.
    """
    if _is_junk_ref(ref):
        return _junk_result()

    results: dict[str, Any] = {}

    # If the reference has a URL that isn't doi.org / arXiv, try to resolve it
    # to a DOI by following redirects and checking HTML meta tags.
    ref_url = ref.get("url", "") or ""
    if (ref_url
            and not _DOI_FROM_URL_RE.search(ref_url)
            and not _ARXIV_ID_RE.search(ref_url)):
        fetched_doi = _resolve_url_doi(ref_url)
        if fetched_doi:
            ref = {**ref, "doi": fetched_doi}

    if ref_url:
        results["_source_url"] = ref_url

    # --- Phase 1: Source router (with multi-source vote for ID refs) ---
    chosen = _classify_ref(ref)
    if chosen:
        is_id_ref = bool(_extract_arxiv_id(ref) or _extract_doi(ref))

        # Fast path: for ID refs, peek at the batch-prewarmed caches first
        # (S2, then OpenAlex).  Both are ~ms lookups when cached.  If either
        # confirms a match by ID, the remaining sources can only corroborate
        # — skip their per-ref API calls.
        chosen_to_run: list[str] = list(chosen)
        if is_id_ref:
            for fast_src in ("semantic_scholar", "openalex"):
                if fast_src in chosen_to_run:
                    results[fast_src] = _LOOKUP_FNS[fast_src](ref)
                    chosen_to_run = [s for s in chosen_to_run if s != fast_src]
                    if results[fast_src].get("label") == "match":
                        chosen_to_run = []
                        break

        if chosen_to_run:
            results.update(_run_sources(ref, chosen_to_run))
        # Verified: any chosen source returned a title match.
        # Fabricated (id_ref): every chosen source confirmed not_found.
        if _has_match(results, chosen) or (is_id_ref and _all_not_found(results, chosen)):
            results["_router_path"] = f"routed:{','.join(chosen)}"
            for src in _ALL_API_SOURCES:
                results.setdefault(src, {"status": "skipped"})
            _backfill_acl_from_doi(results)
            if "acl_anthology" not in results:
                results["acl_anthology"] = {"status": "not_in_anthology"}
            results["openlibrary"] = {"status": "skipped"}
            results["scholarly"]   = {"status": "skipped"}
            return results

    # --- Phase 2: Escalation — query whatever the router didn't ---
    leftover = [s for s in _ALL_API_SOURCES if s not in results]
    if leftover:
        results.update(_run_sources(ref, leftover))
    results["_router_path"] = "escalated" if chosen else "fanout"

    # --- Phase 3: ACL anthology backfill (S2 externalIds path) ---
    if results.get("acl_anthology", {}).get("status") != "found":
        s2 = results.get("semantic_scholar", {})
        if s2.get("acl_url"):
            results["acl_anthology"] = {
                "status":      "found",
                "found_title": s2["found_title"],
                "found_venue": s2.get("found_venue", ""),
                "similarity":  s2["similarity"],
                "label":       s2["label"],
                "venue_sim":   s2.get("venue_sim"),
                "venue_label": s2.get("venue_label", "unknown"),
                "url":         s2["acl_url"],
                "acl_id":      s2.get("acl_id"),
            }
        else:
            _backfill_acl_from_doi(results)
            if "acl_anthology" not in results:
                results["acl_anthology"] = {"status": "not_in_anthology"}

    # --- Phase 4: Open Library — only if the ref looks like a book ---
    academic = ("semantic_scholar", "dblp", "openalex", "crossref", "acl_anthology")
    no_academic_match = not any(
        results.get(s, {}).get("status") == "found"
        and results.get(s, {}).get("label") == "match"
        for s in academic
    )
    if _VENUE_BOOK_RE.search(_book_text(ref)) and no_academic_match:
        results["openlibrary"] = _lookup_openlibrary(ref)
    else:
        results["openlibrary"] = {"status": "skipped"}

    # --- Phase 5: Google Scholar — last-resort title verifier ---
    # Heavy rate-limit + CAPTCHA risk → call it only when nothing else
    # confirmed the paper.  Skip when the ref is so short or junk-like that
    # a Scholar query would just burn quota.
    if no_academic_match and results["openlibrary"].get("label") != "match":
        results["scholarly"] = _lookup_scholarly(ref)
    else:
        results["scholarly"] = {"status": "skipped"}

    return results


def _backfill_acl_from_doi(results: dict[str, Any]) -> None:
    """If any source returned an ACL DOI, populate results['acl_anthology']."""
    if results.get("acl_anthology", {}).get("status") == "found":
        return
    for src in ("crossref", "openalex", "dblp"):
        d = (results.get(src) or {}).get("doi", "") or ""
        m = _ACL_DOI_RE.match(d)
        if m:
            r = results[src]
            results["acl_anthology"] = {
                "status":      "found",
                "found_title": r.get("found_title", ""),
                "found_venue": r.get("found_venue", ""),
                "similarity":  r.get("similarity"),
                "label":       r.get("label", "match"),
                "venue_sim":   r.get("venue_sim"),
                "venue_label": r.get("venue_label", "unknown"),
                "url":         f"https://aclanthology.org/{m.group(1)}",
                "acl_id":      m.group(1),
            }
            return


def lookup_all(
    references: list[dict[str, Any]],
    delay: float = 0.0,
    progress_cb=None,
    max_concurrent: int = 5,
    scholarly_budget: int = _SCHOLARLY_BUDGET_DEFAULT,
) -> list[dict[str, Any]]:
    """
    Look up every reference with limited concurrency.

    Begins with a single S2 batch pre-warm for every ref carrying a DOI
    or arXiv ID — that collapses up to 500 sequential S2 lookups into
    one HTTP request and the per-ref S2 calls become cache hits.

    Each ref's internal API fanout uses up to 5 workers, so peak parallel
    HTTP requests = max_concurrent * 5.  We capped this at 5 (down from
    an earlier experiment at 8) because S2 starts returning 429s past
    ~25 parallel calls and its exponential backoff (`2**attempt`) erases
    the parallelism win on cold caches.

    Set ``scholarly_budget=0`` to disable Phase 5 Google Scholar lookups
    entirely (recommended for batch / benchmark runs — Scholar's 2.5 s
    rate-limit lock dominates wall time).  The UI default keeps Scholar on.
    """
    # Scholar budget — capped per lookup_all() call so a single paper with
    # many unmatched refs doesn't burn minutes on serialised Scholar queries.
    _scholarly_reset_budget(scholarly_budget)

    # Identifier-batch pre-warm — one round trip per source, populates the
    # cache so per-ref S2 / OpenAlex / arXiv calls become local lookups.
    with ThreadPoolExecutor(max_workers=3) as _pool:
        for _fn in (_s2_batch_prewarm, _oa_batch_prewarm, _arxiv_batch_prewarm):
            _pool.submit(_fn, references)
        # implicit wait on context-manager exit

    results: list[Any] = [None] * len(references)
    _lock = threading.Lock()

    def _worker(i: int, ref: dict) -> int:
        res = lookup_reference(ref)
        with _lock:
            results[i] = res
        return i

    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        futures = {pool.submit(_worker, i, ref): i for i, ref in enumerate(references)}
        done = 0
        for f in as_completed(futures):
            f.result()  # propagate exceptions
            done += 1
            if progress_cb:
                # Called on the main thread — safe for Streamlit UI updates
                progress_cb(done, len(references))

    return results
