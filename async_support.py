"""Async support steps for the NeuroGPT pipeline: MeSH synonym lookup, the embedding call, and the two ranking stages. Each is the async twin of a function in mesh_expand.py / hybrid_search.py and shares
its cache and its maths with it, so the two can not drift apart.

- MeSH: the two NCBI calls are awaited; the cache (memory and disk) is the one mesh_expand uses.
- Embeddings: the batches are requested at the same time with the async HTTP client and written to hybrid_search's own cache. The ranking itself then finds every vector in the cache.
- Ranking: BM25, the fusion and the local ONNX cross-encoder are CPU work, not waiting. They are handed to asyncio.to_thread so the event loop keeps serving the other tasks."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os

import async_core
import hybrid_search as hs
import mesh_expand as me
from clinical_tools import EUTILS

# ------------------------------------------------------------------ MeSH
_tasks: dict[str, asyncio.Future] = {}


async def _fetch(term: str) -> dict:
    found = (await async_core.aget_json(f"{EUTILS}/esearch.fcgi", {"db": "mesh", "term": term, "retmax": 5, "retmode": "json"}))["esearchresult"]
    tagged = me.MESH_TAG.search(found.get("querytranslation") or "")
    ids = found.get("idlist") or []
    if not tagged or not ids:
        return {"heading": None, "entry_terms": []}
    heading = tagged.group(1)
    summary = (await async_core.aget_json(f"{EUTILS}/esummary.fcgi", {"db": "mesh", "id": ",".join(ids), "retmode": "json"}))["result"]
    for uid in ids:
        names = (summary.get(uid) or {}).get("ds_meshterms") or []
        if names and str(names[0]).lower() == heading.lower():
            return {"heading": names[0], "entry_terms": me.entry_terms(names[1:], names[0])}
    return {"heading": heading, "entry_terms": []}


async def alookup(term: str) -> dict | None:
    """Async twin of mesh_expand.lookup: cached in memory and on disk; a failure is not cached. The same term asked by several searches at once is looked up once."""
    key = me._key(term)
    if len(key) < 3 or key.isdigit():
        return {"heading": None, "entry_terms": []}
    me._load_disk()
    if key in me._memory:
        return me._memory[key]
    running = _tasks.get(key)
    if running is None:
        async def work():
            try:
                record = await _fetch(key)
            except Exception:
                return None
            me._memory[key] = record
            await asyncio.to_thread(me._save_disk)      # a small file write
            return record
        running = _tasks[key] = asyncio.ensure_future(work())
        running.add_done_callback(lambda _f, k=key: _tasks.pop(k, None))
    return await asyncio.shield(running)      # a caller that stops waiting does not cancel the lookup for the others


async def arecords(terms: list[str], timeout: float | None = None) -> dict[str, dict]:
    """Async twin of mesh_expand.records: the terms are looked up at the same time within the time limit; a late or failed term is left out (the search runs without it)."""
    terms = [t for t in dict.fromkeys(me._key(t) for t in terms) if len(t) >= 3]
    if not terms:
        return {}
    tasks = {t: asyncio.ensure_future(alookup(t)) for t in terms}
    done, pending = await asyncio.wait(list(tasks.values()), timeout=me.LOOKUP_TIMEOUT_S if timeout is None else timeout)
    for task in pending:
        task.cancel()
    return {t: task.result() for t, task in tasks.items() if task in done and not task.cancelled() and task.exception() is None and task.result() is not None}


async def aexpand(terms: list[str], target: str = "pubmed", timeout: float | None = None) -> dict[str, list[str]]:
    """Async twin of mesh_expand.expand."""
    found = {t: me.clauses(t, record, target) for t, record in (await arecords(terms, timeout)).items()}
    return {t: extra for t, extra in found.items() if extra}


def aprefetch(terms: list[str]) -> None:
    """Async twin of mesh_expand.prefetch: start the lookups now and return at once (they are tasks of the running loop)."""
    for term in dict.fromkeys(me._key(t) for t in terms if len(me._key(t)) >= 3):
        asyncio.ensure_future(alookup(term))


# ------------------------------------------------------------------ embeddings and ranking
async def aembed(texts: list[str], timeout: float | None = None) -> list[list[float]] | None:
    """Async twin of hybrid_search.embed: the batches are requested at the same time and stored in hybrid_search's cache; None when the key is missing or a batch fails.
    EMBED_PROVIDER=local: the vectors come from the local model (local_embed), in a worker thread, within the same time limit (a late model keeps working and fills the cache for the next call)."""
    import local_embed
    if local_embed.enabled():
        wait_local = timeout if timeout is not None else float(os.getenv("EMBED_LOCAL_TIMEOUT_S", "20"))
        try:
            return await asyncio.wait_for(asyncio.shield(asyncio.ensure_future(asyncio.to_thread(local_embed.embed, texts))), timeout=wait_local)
        except Exception:
            return None
    use_openai = os.getenv("EMBED_PROVIDER", "").strip().lower() == "openai"
    key = os.getenv("OPENAI_API_KEY" if use_openai else "MISTRAL_API_KEY")
    if not key or not texts:
        return None
    url = "https://api.openai.com/v1/embeddings" if use_openai else hs.EMBED_URL
    model = os.getenv("EMBED_MODEL", "").strip() or ("text-embedding-3-small" if use_openai else hs.DEFAULT_EMBED_MODEL)
    wait = timeout if timeout is not None else float(os.getenv("EMBED_TIMEOUT_S", "6"))
    keys = [hashlib.sha1((model + "|" + t).encode("utf-8")).hexdigest() for t in texts]
    with hs._lock:
        missing = [(k, t) for k, t in zip(keys, texts) if k not in hs._cache]
    if missing:
        unique = list(dict.fromkeys(missing))
        batches = [unique[i:i + hs.BATCH] for i in range(0, len(unique), hs.BATCH)]

        async def one(batch) -> bool:
            for attempt in (1, 2):
                try:
                    response = await async_core.client().post(url, headers={"Authorization": f"Bearer {key}"}, json={"model": model, "input": [t for _, t in batch]}, timeout=wait)
                    if response.status_code in (429, 500, 502, 503) and attempt == 1:
                        await asyncio.sleep(1.0)
                        continue
                    response.raise_for_status()
                    data = sorted(response.json().get("data") or [], key=lambda d: d.get("index", 0))
                    if len(data) != len(batch):
                        return False
                    with hs._lock:
                        for (k, _), item in zip(batch, data):
                            hs._cache[k] = item["embedding"]
                    return True
                except Exception:
                    if attempt == 2:
                        return False
                    await asyncio.sleep(1.0)
            return False
        if not all(await asyncio.gather(*[one(b) for b in batches])):
            return None
    with hs._lock:
        return [hs._cache[k] for k in keys]


async def arerank(question: str, papers: dict, trace: dict | None = None, queries: list[str] | None = None) -> None:
    """Async twin of hybrid_search.rerank. The embedding request is awaited first and fills the cache; the ranking (BM25, the fusion) then runs in a worker and finds every vector in the cache."""
    if hs.enabled() and len(papers) >= 3:
        docs = [hs.paper_text(p) for p in papers.values()]
        asked = [question] + [q for q in dict.fromkeys(" ".join(str(x).split()) for x in (queries or [])) if q and q != question][:hs.MAX_QUERIES]
        await aembed(asked + docs)
    await asyncio.to_thread(hs.rerank, question, papers, trace, queries)


async def across_rerank(question: str, papers: dict, trace: dict | None = None, top: int = 36, wait: float = 3.0, scorer=None) -> None:
    """Async twin of hybrid_search.cross_rerank: the local cross-encoder is CPU work, so it runs in a worker while the event loop serves the other tasks."""
    await asyncio.to_thread(hs.cross_rerank, question, papers, trace, top, wait, scorer)
