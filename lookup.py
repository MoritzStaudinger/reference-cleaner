"""
Reference lookup against Semantic Scholar, DBLP, OpenAlex, arXiv, ACL Anthology, and Crossref.

Validation sources (title + venue matching → determines card colour):
    semantic_scholar, acl_anthology, dblp, openalex

Identifier sources (just provide links/IDs):
    arxiv, crossref
"""

from __future__ import annotations

import os
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from rapidfuzz import fuzz as _fuzz

import requests as _requests
from dotenv import load_dotenv

load_dotenv()

S2_API_KEY: str | None = os.getenv("S2_API_KEY") or None

_S2_BASE  = "https://api.semanticscholar.org/graph/v1"
_S2_FIELDS = "title,authors,year,externalIds,venue"

_CR_BASE     = "https://api.crossref.org/works"
_CR_HEADERS  = {"User-Agent": "ReferenceCleaner/1.0 (mailto:contact@example.com)"}

_DBLP_BASE   = "https://dblp.org/search/publ/api"
_OA_BASE     = "https://api.openalex.org/works"
_OA_HEADERS  = {"User-Agent": "ReferenceCleaner/1.0 (mailto:contact@example.com)"}

# Title similarity thresholds
MATCH_THRESHOLD = 0.90
FUZZY_THRESHOLD = 0.70

# Venue similarity threshold below which we flag a venue mismatch
VENUE_MISMATCH_THRESHOLD = 0.40

_ARXIV_ID_RE = re.compile(
    r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})"
    r"|arXiv:(\d{4}\.\d{4,5})",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Normalisation & similarity
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def title_similarity(t1: str, t2: str) -> float:
    if not t1 or not t2:
        return 0.0
    n1, n2 = _normalize(t1), _normalize(t2)
    # token_sort_ratio handles word-order differences (e.g. "BERT: Pre-training..."
    # vs "Pre-training of Deep Bidirectional Transformers (BERT)")
    ratio = _fuzz.token_sort_ratio(n1, n2) / 100.0
    # One title is a prefix of the other (subtitle added/dropped) → full match
    shorter, longer = (n1, n2) if len(n1) <= len(n2) else (n2, n1)
    if longer.startswith(shorter) and len(shorter) / len(longer) >= 0.6:
        ratio = max(ratio, 1.0)
    return ratio


def venue_similarity(v1: str, v2: str) -> float:
    """Fuzzy venue match. Handles 'EMNLP' vs 'Proceedings of EMNLP 2020' etc."""
    if not v1 or not v2:
        return 0.0
    n1, n2 = _normalize(v1), _normalize(v2)
    # partial_ratio handles abbreviation-in-full-name cases
    base = max(
        _fuzz.token_sort_ratio(n1, n2) / 100.0,
        _fuzz.partial_ratio(n1, n2)   / 100.0,
    )
    return base


def similarity_label(sim: float | None) -> str:
    if sim is None:
        return "unknown"
    if sim >= MATCH_THRESHOLD:
        return "match"
    if sim >= FUZZY_THRESHOLD:
        return "fuzzy"
    return "mismatch"


def _venue_label(vsim: float | None) -> str:
    if vsim is None:
        return "unknown"
    if vsim >= 0.70:
        return "match"
    if vsim >= VENUE_MISMATCH_THRESHOLD:
        return "fuzzy"
    return "mismatch"


# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------

def _s2_headers(api_key: str | None) -> dict:
    return {"x-api-key": api_key} if api_key else {}


def _s2_get(path: str, params: dict, api_key: str | None) -> dict | None:
    url = f"{_S2_BASE}/{path}"
    for attempt in range(3):
        r = _requests.get(url, params=params, headers=_s2_headers(api_key), timeout=15)
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


def _s2_paper_to_result(data: dict, extracted_title: str, extracted_venue: str) -> dict[str, Any]:
    found_title = data.get("title") or ""
    found_venue = data.get("venue") or ""
    tsim = title_similarity(extracted_title, found_title) if extracted_title else None
    vsim = venue_similarity(extracted_venue, found_venue) if (extracted_venue and found_venue) else None
    ext  = data.get("externalIds") or {}

    result: dict[str, Any] = {
        "status":       "found",
        "found_title":  found_title,
        "found_venue":  found_venue,
        "similarity":   tsim,
        "label":        similarity_label(tsim),
        "venue_sim":    vsim,
        "venue_label":  _venue_label(vsim),
        "year":         data.get("year"),
        "url":          f"https://www.semanticscholar.org/paper/{data['paperId']}",
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


def _lookup_semantic_scholar(ref: dict[str, Any]) -> dict[str, Any]:
    extracted_title = ref.get("title", "")
    extracted_venue = ref.get("venue", "")
    keys_to_try: list[str | None] = [S2_API_KEY, None] if S2_API_KEY else [None]

    for api_key in keys_to_try:
        try:
            data      = None
            wrong_doi = None

            if ref.get("doi"):
                doi_data = _s2_get(f"paper/DOI:{ref['doi']}", {"fields": _S2_FIELDS}, api_key)
                if doi_data:
                    if not extracted_title or title_similarity(extracted_title, doi_data.get("title", "")) >= 0.5:
                        data = doi_data
                    else:
                        wrong_doi = ref["doi"]

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

            result = _s2_paper_to_result(data, extracted_title, extracted_venue)
            if wrong_doi:
                result["wrong_doi"] = wrong_doi
            return result

        except PermissionError:
            continue
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    return {"status": "error", "error": "S2 API key rejected (403)"}


# ---------------------------------------------------------------------------
# DBLP
# ---------------------------------------------------------------------------

def _lookup_dblp(ref: dict[str, Any]) -> dict[str, Any]:
    extracted_title = ref.get("title", "")
    extracted_venue = ref.get("venue", "")
    if not extracted_title:
        return {"status": "skipped"}

    try:
        r = _requests.get(
            _DBLP_BASE,
            params={"q": extracted_title, "format": "json", "h": 10},
            timeout=15,
        )
        r.raise_for_status()
        hits = r.json().get("result", {}).get("hits", {}).get("hit", [])
        if not hits:
            return {"status": "not_found"}

        # Pick the hit with the highest title similarity, require a minimum score
        best = max(
            hits,
            key=lambda h: title_similarity(extracted_title, h.get("info", {}).get("title", "")),
        )
        if title_similarity(extracted_title, best.get("info", {}).get("title", "")) < FUZZY_THRESHOLD:
            return {"status": "not_found"}
        info        = best.get("info", {})
        found_title = info.get("title", "")
        found_venue = info.get("venue", "")
        tsim = title_similarity(extracted_title, found_title)
        vsim = venue_similarity(extracted_venue, found_venue) if (extracted_venue and found_venue) else None
        doi  = info.get("doi", "")
        url  = info.get("url", "") or (f"https://doi.org/{doi}" if doi else "")

        return {
            "status":      "found",
            "found_title": found_title,
            "found_venue": found_venue,
            "similarity":  tsim,
            "label":       similarity_label(tsim),
            "venue_sim":   vsim,
            "venue_label": _venue_label(vsim),
            "year":        info.get("year"),
            "doi":         doi,
            "url":         url,
        }

    except Exception as exc:
        return {"status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# OpenAlex
# ---------------------------------------------------------------------------

def _lookup_openalex(ref: dict[str, Any]) -> dict[str, Any]:
    extracted_title = ref.get("title", "")
    extracted_venue = ref.get("venue", "")

    try:
        data = None

        # DOI direct lookup
        if ref.get("doi"):
            r = _requests.get(
                f"{_OA_BASE}/https://doi.org/{ref['doi']}",
                headers=_OA_HEADERS,
                timeout=15,
            )
            if r.status_code == 200:
                data = r.json()

        # Title search
        if data is None and extracted_title:
            r = _requests.get(
                _OA_BASE,
                params={
                    "search":   extracted_title,
                    "per_page": 3,
                    "select":   "id,title,authorships,publication_year,primary_location,doi,open_access",
                },
                headers=_OA_HEADERS,
                timeout=15,
            )
            r.raise_for_status()
            results = r.json().get("results", [])
            if results:
                # Pick best title match
                data = max(
                    results,
                    key=lambda w: title_similarity(extracted_title, w.get("title", "") or ""),
                )

        if data is None:
            return {"status": "not_found"}

        found_title = data.get("title") or ""
        doi         = data.get("doi", "") or ""
        doi         = doi.replace("https://doi.org/", "")
        loc         = data.get("primary_location") or {}
        source      = loc.get("source") or {}
        found_venue = source.get("display_name") or ""
        oa_url      = (data.get("open_access") or {}).get("oa_url") or ""
        url         = oa_url or (f"https://doi.org/{doi}" if doi else data.get("id", ""))

        tsim = title_similarity(extracted_title, found_title) if extracted_title else None
        # Require minimum similarity for title-search results
        if extracted_title and tsim is not None and tsim < FUZZY_THRESHOLD:
            return {"status": "not_found"}
        vsim = venue_similarity(extracted_venue, found_venue) if (extracted_venue and found_venue) else None

        return {
            "status":      "found",
            "found_title": found_title,
            "found_venue": found_venue,
            "similarity":  tsim,
            "label":       similarity_label(tsim),
            "venue_sim":   vsim,
            "venue_label": _venue_label(vsim),
            "year":        data.get("publication_year"),
            "doi":         doi,
            "url":         url,
        }

    except Exception as exc:
        return {"status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# arXiv  (identifier only)
# ---------------------------------------------------------------------------

def _extract_arxiv_id(ref: dict[str, Any]) -> str | None:
    for field in ("url", "raw", "doi"):
        val = ref.get(field, "") or ""
        m   = _ARXIV_ID_RE.search(val)
        if m:
            return m.group(1) or m.group(2)
    return None


def _lookup_arxiv(ref: dict[str, Any]) -> dict[str, Any]:
    import arxiv

    extracted = ref.get("title", "")
    arxiv_id  = _extract_arxiv_id(ref)

    try:
        client = arxiv.Client()
        if arxiv_id:
            search = arxiv.Search(id_list=[arxiv_id])
        elif extracted:
            search = arxiv.Search(
                query=f'ti:"{extracted}"',
                max_results=1,
                sort_by=arxiv.SortCriterion.Relevance,
            )
        else:
            return {"status": "skipped"}

        results = list(client.results(search))
        if not results:
            return {"status": "not_found"}

        paper       = results[0]
        arxiv_id_found = paper.entry_id.split("/abs/")[-1]
        return {"status": "found", "arxiv_id": arxiv_id_found, "url": paper.entry_id}

    except Exception as exc:
        return {"status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# Crossref  (identifier only)
# ---------------------------------------------------------------------------

def _lookup_crossref(ref: dict[str, Any]) -> dict[str, Any]:
    extracted = ref.get("title", "")
    try:
        doi = None

        if ref.get("doi"):
            r = _requests.get(f"{_CR_BASE}/{ref['doi']}", headers=_CR_HEADERS, timeout=15)
            if r.status_code == 200:
                doi = r.json().get("message", {}).get("DOI", ref["doi"])

        if doi is None and extracted:
            r = _requests.get(
                _CR_BASE,
                params={"query.title": extracted, "rows": 1, "select": "DOI"},
                headers=_CR_HEADERS,
                timeout=15,
            )
            r.raise_for_status()
            items = r.json().get("message", {}).get("items", [])
            doi   = items[0].get("DOI") if items else None

        if not doi:
            return {"status": "not_found"}

        return {"status": "found", "doi": doi, "url": f"https://doi.org/{doi}"}

    except Exception as exc:
        return {"status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lookup_reference(ref: dict[str, Any]) -> dict[str, Any]:
    """
    Look up a single reference across all sources concurrently.

    Validation sources (title + venue → card colour):
        semantic_scholar, acl_anthology, dblp, openalex
    Identifier sources (links only):
        arxiv, crossref
    """
    results: dict[str, Any] = {}

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {
            pool.submit(_lookup_semantic_scholar, ref): "semantic_scholar",
            pool.submit(_lookup_dblp,             ref): "dblp",
            pool.submit(_lookup_openalex,          ref): "openalex",
            pool.submit(_lookup_arxiv,             ref): "arxiv",
            pool.submit(_lookup_crossref,          ref): "crossref",
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as exc:
                results[key] = {"status": "error", "error": str(exc)}

    # ACL Anthology: free, derived from S2's externalIds
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
        results["acl_anthology"] = {"status": "not_in_anthology"}

    return results


def lookup_all(
    references: list[dict[str, Any]],
    delay: float = 0.5,
    progress_cb=None,
) -> list[dict[str, Any]]:
    """Look up every reference sequentially. progress_cb(done, total) if provided."""
    all_results = []
    for i, ref in enumerate(references):
        result = lookup_reference(ref)
        all_results.append(result)
        if progress_cb:
            progress_cb(i + 1, len(references))
        if i < len(references) - 1:
            time.sleep(delay)
    return all_results
