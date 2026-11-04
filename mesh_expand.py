"""Synonyms of a search term from the MeSH vocabulary (NCBI's MeSH database), so a paper that says the same thing in other words is also retrieved. A database lookup, not a model guess:
the term is mapped to its official MeSH heading and the heading's entry terms (the other names the vocabulary lists for it) are added to the search with OR."""
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

from clinical_tools import EUTILS, _get

OFF_WORDS = {"off", "0", "false", "no"}
CACHE_FILE = Path(__file__).parent / "cache" / "mesh_terms.json"
MAX_ENTRY_TERMS = 6          # entry terms added per heading: enough for the usual alternative names, few enough not to flood the search
LOOKUP_TIMEOUT_S = float(os.getenv("MESH_TIMEOUT_S", "3"))
ROUTING_WAIT_S = float(os.getenv("MESH_ROUTING_WAIT_S", "1.5"))      # a routed search waits this long for a word it has not looked up before; a late lookup finishes in the background and the next search finds it in the cache
MESH_TAG = re.compile(r'"([^"]+)"\[MeSH Terms\]')

_memory: dict[str, dict] = {}
_in_flight: dict[str, threading.Lock] = {}
_lock = threading.Lock()
_disk_loaded = False


def enabled() -> bool:
    return os.getenv("ENABLE_MESH_EXPANSION", "true").strip().lower() not in OFF_WORDS


def similar_enabled() -> bool:
    return os.getenv("ENABLE_SIMILAR_ARTICLES", "true").strip().lower() not in OFF_WORDS


def _key(term: str) -> str:
    return " ".join(str(term or "").lower().split())


def _load_disk() -> None:
    global _disk_loaded
    with _lock:
        if _disk_loaded:
            return
        _disk_loaded = True
        try:
            _memory.update(json.loads(CACHE_FILE.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass


def _save_disk() -> None:
    try:
        CACHE_FILE.parent.mkdir(exist_ok=True)
        with _lock:
            CACHE_FILE.write_text(json.dumps(_memory, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass          # the cache is a convenience: a read-only folder only means the next run looks the term up again


def entry_terms(names: list, heading: str) -> list[str]:
    """The heading's other names as plain phrases: the inverted forms (a comma in the name) and very long names are dropped."""
    out = []
    for name in names:
        text = " ".join(str(name).split())
        if "," in text or not 3 <= len(text) <= 60 or text.lower() == heading.lower() or text.lower() in {o.lower() for o in out}:
            continue
        out.append(text)
    return out[:MAX_ENTRY_TERMS]


def _fetch(term: str) -> dict:
    """{'heading', 'entry_terms'} for a term, or {'heading': None, ...} when MeSH has no heading for it. Two calls to NCBI (search, summary); raises when NCBI cannot be reached."""
    found = json.loads(_get(f"{EUTILS}/esearch.fcgi", {"db": "mesh", "term": term, "retmax": 5, "retmode": "json"}))["esearchresult"]
    tagged = MESH_TAG.search(found.get("querytranslation") or "")      # NCBI's own mapping of the term to a heading: the first heading is the best match, the id list order is not
    ids = found.get("idlist") or []
    if not tagged or not ids:
        return {"heading": None, "entry_terms": []}
    heading = tagged.group(1)
    summary = json.loads(_get(f"{EUTILS}/esummary.fcgi", {"db": "mesh", "id": ",".join(ids), "retmode": "json"}))["result"]
    for uid in ids:
        names = (summary.get(uid) or {}).get("ds_meshterms") or []
        if names and str(names[0]).lower() == heading.lower():
            return {"heading": names[0], "entry_terms": entry_terms(names[1:], names[0])}
    return {"heading": heading, "entry_terms": []}


def lookup(term: str) -> dict | None:
    """The MeSH heading and entry terms of a term (cached in memory and on disk), or None when the lookup failed (a failure is not cached)."""
    key = _key(term)
    if len(key) < 3 or key.isdigit():
        return {"heading": None, "entry_terms": []}
    _load_disk()
    if key in _memory:
        return _memory[key]
    with _lock:
        one_at_a_time = _in_flight.setdefault(key, threading.Lock())
    with one_at_a_time:      # the same term asked by several searches at once is looked up once; the others wait for that answer
        if key in _memory:
            return _memory[key]
        try:
            record = _fetch(key)
        except Exception:
            return None
        _memory[key] = record
    _save_disk()
    return record


def clauses(term: str, record: dict | None, target: str = "pubmed") -> list[str]:
    """The OR alternatives to add for a term: its MeSH heading (as the source's MeSH search) and the entry terms as exact phrases. Empty when MeSH has nothing for the term."""
    if not record or not record.get("heading"):
        return []
    tag = f'"{record["heading"]}"[MeSH Terms]' if target == "pubmed" else f'MESH:"{record["heading"]}"'
    own = {_key(term)}
    return [tag] + [f'"{t}"' for t in [record["heading"]] + record.get("entry_terms", []) if _key(t) not in own]


def records(terms: list[str], timeout: float | None = None) -> dict[str, dict]:
    """{term: MeSH record} for the terms looked up in parallel within the time limit; a term whose lookup is late or failed is left out (the search simply runs without it)."""
    terms = [t for t in dict.fromkeys(_key(t) for t in terms) if len(t) >= 3]
    if not terms:
        return {}
    pool = ThreadPoolExecutor(max_workers=min(6, len(terms)))
    futures = {t: pool.submit(lookup, t) for t in terms}
    wait(list(futures.values()), timeout=LOOKUP_TIMEOUT_S if timeout is None else timeout)
    pool.shutdown(wait=False, cancel_futures=True)
    return {t: f.result() for t, f in futures.items() if f.done() and not f.cancelled() and f.result() is not None}


def prefetch(terms: list[str]) -> None:
    """Start the lookups now and return at once: they finish in the background and are in the cache when a search asks for them."""
    for term in dict.fromkeys(_key(t) for t in terms if len(_key(t)) >= 3):
        threading.Thread(target=lookup, args=(term,), daemon=True).start()


def expand(terms: list[str], target: str = "pubmed", timeout: float | None = None) -> dict[str, list[str]]:
    """{term: OR alternatives} for the terms MeSH knows, in the syntax of the target source."""
    found = {t: clauses(t, record, target) for t, record in records(terms, timeout).items()}
    return {t: extra for t, extra in found.items() if extra}


def group(term: str, extra: list[str]) -> str:
    """One concept as a group: the term itself or any of its MeSH alternatives."""
    return "(" + " OR ".join([f'"{term}"'] + extra) + ")"


def clauses_for_terms(words: list[str], terms: list[str], target: str) -> dict[str, list[str]]:
    """For the routing adapters: the MeSH alternatives of the plan's condition terms that a routed search requires (every word of the term is one of its required words). Only these few terms are looked up
    (the legacy searches look up the same ones, so they are in the cache), never the question's other words. Words already looked up cost nothing; a new term is waited for only briefly, so a routed search
    is never held up."""
    have = {w.lower() for w in words}
    present = [t for t in terms if t.split() and all(w in have for w in t.split())]
    return expand(present, target, ROUTING_WAIT_S)
