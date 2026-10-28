"""One copy of every record, whichever search found it: identity keys for the merge, and the per-search log that compares the legacy and the routed searches."""
import json
import re
from urllib.parse import urlparse

MIN_TITLE_CHARS = 25      # a shorter title is too generic to identify a paper


def normalize_doi(doi) -> str:
    return re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", str(doi or "").strip().lower()).strip()


def normalize_title(title) -> str:
    """Lower case, markup and punctuation removed, one space between words: the same title written two ways compares equal."""
    text = re.sub(r"<[^>]+>", " ", str(title or "").lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()


def normalize_url(url) -> str:
    parts = urlparse(str(url or "").strip())
    return (parts.netloc.lower().removeprefix("www.") + parts.path.rstrip("/")).lower()


def paper_keys(rec: dict) -> list[tuple[str, str]]:
    """The identities of a paper in the order they are tried: PMID, then DOI, then normalized title."""
    keys = []
    if rec.get("pmid"):
        keys.append(("pmid", str(rec["pmid"])))
    if normalize_doi(rec.get("doi")):
        keys.append(("doi", normalize_doi(rec["doi"])))
    title = normalize_title(rec.get("title"))
    if len(title) >= MIN_TITLE_CHARS:
        keys.append(("title", title))
    return keys


def find_paper(papers: dict, rec: dict) -> str | None:
    """The key in `papers` of a record that is the same paper as `rec` (same PMID, else same DOI, else same normalized title), or None."""
    if rec.get("pmid") and str(rec["pmid"]) in papers:
        return str(rec["pmid"])
    doi, title = normalize_doi(rec.get("doi")), normalize_title(rec.get("title"))
    for key, known in papers.items():
        if doi and normalize_doi(known.get("doi")) == doi:
            return key
        if len(title) >= MIN_TITLE_CHARS and normalize_title(known.get("title")) == title:
            return key
    return None


# ------------------------------------------------------------------------------------------------ the log that compares the two modes
def identities(key: str, text) -> set[str]:
    """The records one search returned, as identity strings (pmid:, nct:, url:, fda:) so the same record found by two searches counts once."""
    try:
        data = json.loads(text) if isinstance(text, str) else text
    except ValueError:
        return set()
    if isinstance(data, dict):
        if data.get("error"):
            return set()
        out = set()
        for r in data.get("results", []) or []:
            if not isinstance(r, dict):
                continue
            if r.get("pmid"):
                out.add(f"pmid:{r['pmid']}")
            elif r.get("nct_id"):
                out.add(f"nct:{r['nct_id']}")
            elif r.get("generic"):
                out.add(f"fda:{str(r['generic']).lower()}")
            elif r.get("url"):
                out.add(f"url:{normalize_url(r['url'])}")
        return out
    if isinstance(data, list):
        return {f"url:{normalize_url(r.get('url'))}" for r in data if isinstance(r, dict) and r.get("url")}
    return set()


def search_log(raw: dict, routing_keys: set[str]) -> dict:
    """Per search: the mode that wrote it (routing for the model-written searches, legacy for everything else), its source, how many records came back and how many of them the other mode did not find;
    and the summary line. `raw` is {search key: the tool's JSON}; `routing_keys` the keys that belong to routed searches."""
    found = {key: identities(key, text) for key, text in raw.items()}
    mode_of = {key: ("routing" if key in routing_keys else "legacy") for key in raw}
    by_mode = {m: set().union(*[ids for k, ids in found.items() if mode_of[k] == m] or [set()]) for m in ("legacy", "routing")}
    rows = []
    for key, ids in found.items():
        other = by_mode["legacy" if mode_of[key] == "routing" else "routing"]
        source = re.sub(r"[_\d]+$", "", key)
        rows.append({"search": key, "mode": mode_of[key], "source": source, "returned": len(ids), "new_vs_other_mode": len(ids - other), "error": bool(isinstance(_safe(raw[key]), dict) and _safe(raw[key]).get("error"))})
    both = by_mode["legacy"] & by_mode["routing"]
    summary = {"legacy": len(by_mode["legacy"]), "legacy_only": len(by_mode["legacy"] - both), "routing": len(by_mode["routing"]), "routing_only": len(by_mode["routing"] - both), "overlap": len(both)}
    line = (f"legacy: {summary['legacy']} records ({summary['legacy_only']} only legacy), routing: {summary['routing']} records ({summary['routing_only']} only routing), overlap: {summary['overlap']}")
    return {"searches": rows, "summary": summary, "summary_line": line}


def _safe(text):
    try:
        return json.loads(text) if isinstance(text, str) else text
    except ValueError:
        return None
