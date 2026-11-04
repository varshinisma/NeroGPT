"""A per-conversation store of what the earlier questions retrieved: papers, trial records, FDA label records and web pages.

A follow-up starts from this store, so it searches only for what the store does not already hold (retrieval and ranking are the slowest steps of a run). Nothing here knows a disease or a drug:
the store is keyed by the conversation id and holds the research state's own records. A stored record is one that passed that run's checks, and it carries its own identifiers, so the citation
check treats it like any retrieved record."""
from __future__ import annotations

import json
import re
from pathlib import Path

STORE_DIR = Path(__file__).resolve().parent / "conversation_history" / "sources"
LIMITS = {"papers": 120, "trials_by_id": 60, "web": 40}      # records kept per kind (the newest stay)
KINDS = tuple(LIMITS)      # papers and trial records are keyed by their id; web pages are a list


def _path(conversation_id: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "", str(conversation_id or ""))[:64]
    return STORE_DIR / f"{safe}.json" if safe else STORE_DIR / "_none.json"


def load(conversation_id: str) -> dict:
    try:
        data = json.loads(_path(conversation_id).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _merge(old, new, limit: int):
    if isinstance(new, dict):
        merged = dict(old if isinstance(old, dict) else {})
        for key, value in new.items():
            merged.pop(key, None)      # a record seen again moves to the newest position
            merged[key] = value
        return dict(list(merged.items())[-limit:])
    seen, out = set(), []
    for item in list(old if isinstance(old, list) else []) + list(new or []):
        key = json.dumps(item, sort_keys=True, default=str)[:300]
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out[-limit:]


def save(conversation_id: str, state: dict) -> None:
    """Add the sources of one finished run to the conversation's store."""
    if not re.sub(r"[^a-zA-Z0-9_-]", "", str(conversation_id or "")) or not isinstance(state, dict):
        return
    stored = load(conversation_id)
    for kind in KINDS:
        stored[kind] = _merge(stored.get(kind), state.get(kind) or {}, LIMITS[kind])
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    _path(conversation_id).write_text(json.dumps(stored, ensure_ascii=False, default=str), encoding="utf-8")


def seed(state: dict, stored: dict) -> dict:
    """Put the stored papers, trial records and web pages into a new run's state, in place (a record the run already has is kept as it is). Returns how many of each were reused.
    FDA label records are not copied: the label lookup is quick and decides which labels fit the question, so it always runs."""
    counts = {}
    for kind in ("papers", "trials_by_id"):
        have = state.setdefault(kind, {})
        counts[kind] = 0
        for key, value in (stored.get(kind) or {}).items():
            if key not in have and isinstance(value, dict):
                have[key] = value
                counts[kind] += 1
    web = state.setdefault("web", [])
    urls = {str(w.get("url")) for w in web if isinstance(w, dict)}
    counts["web"] = 0
    for page in stored.get("web") or []:
        if isinstance(page, dict) and str(page.get("url")) not in urls:
            web.append(page)
            urls.add(str(page.get("url")))
            counts["web"] += 1
    return counts


def covered(state: dict, drug: str, topic_words: tuple[str, ...], minimum: int = 2) -> bool:
    """Do the stored papers already give at least `minimum` papers that name this drug AND speak of the topic asked (the topic words are those the search itself would use)? Then that search is not repeated."""
    name = " ".join(re.findall(r"[a-z0-9]+", str(drug).lower()))
    if not name:
        return False
    hits = 0
    for paper in (state.get("papers") or {}).values():
        text = " ".join(re.findall(r"[a-z0-9]+", f"{paper.get('title') or ''} {paper.get('abstract') or ''}".lower()))
        if name in text and any(w in text for w in topic_words):
            hits += 1
    return hits >= minimum
