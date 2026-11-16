"""Async web searches for the NeuroGPT pipeline: the Tavily searches (general, guideline organisations, regulators) and the guideline lookup built on them. Each function is the async twin of the function of
the same name in qa_stream.py / guidelines.py (without the leading 'a'): same arguments, same JSON text back. Tavily's own async client does the waiting. The DuckDuckGo fallback (server.search_live_web) is a
blocking library, so it is handed to asyncio.to_thread: the event loop is never blocked by it."""
from __future__ import annotations

import asyncio
import json
import os
import re
import time

import contextvars

import async_core
import guidelines as gl

_client = None


def _tavily():
    """One async Tavily client (created on first use, inside the running loop)."""
    global _client
    if _client is None:
        from tavily import AsyncTavilyClient
        _client = AsyncTavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    return _client


def _refused(error: Exception) -> bool:
    text, name = str(error).lower(), type(error).__name__
    return "usage limit" in text or "plan's set" in text or "quota" in text or any(k in name for k in ("Forbidden", "Unauthorized", "UsageLimit", "InvalidAPIKey"))


exhausted = False      # True once Tavily has said its plan limit is reached or the key is refused
_next_slot: dict[str, float] = {}


def _mark_exhausted(error: Exception) -> None:
    """Tavily refused the request: the quota is finished. The person running the backend is told once, in the terminal (never in the report); Exa and Brave take over the web searches."""
    global exhausted
    if not exhausted:
        exhausted = True
        others = [n for n, k in (("Exa", "EXA_API_KEY"), ("Brave", "BRAVE_API_KEY")) if os.getenv(k)]
        print(f"[tavily] QUOTA FINISHED or key refused ({type(error).__name__}: {str(error)[:120]}). "
              + (f"Web searches now use {' + '.join(others)}." if others else "No other web search key is set (EXA_API_KEY / BRAVE_API_KEY): web searches are OFF until a key or credit is added to .env."), flush=True)
    os.environ.pop("TAVILY_API_KEY", None)


async def _space(name: str, per_second: float) -> None:
    """Spaces the calls to one provider (Brave's free plan allows about one request a second)."""
    now = time.monotonic()
    slot = max(now, _next_slot.get(name, 0.0))
    _next_slot[name] = slot + 1.0 / per_second
    if slot > now:
        await asyncio.sleep(slot - now)


async def _exa(query: str, count: int, chars: int, domains: list | None) -> list:
    """Exa: neural search that returns the page text. POST /search with the key in x-api-key."""
    key = os.getenv("EXA_API_KEY")
    if not key:
        return []
    body = {"query": query, "numResults": count, "type": os.getenv("EXA_SEARCH_TYPE", "neural").strip() or "neural", "contents": {"text": {"maxCharacters": max(chars, 500)}}}
    if domains:
        body["includeDomains"] = list(domains)[:40]
    try:
        r = await async_core.client().post("https://api.exa.ai/search", json=body, headers={"x-api-key": key, "Content-Type": "application/json"})
        r.raise_for_status()
        return [{"title": x.get("title"), "url": x.get("url"), "snippet": (x.get("text") or "")[:chars]} for x in (r.json().get("results") or []) if x.get("url")]
    except Exception as error:
        print(f"[exa] search failed: {type(error).__name__}: {str(error)[:100]}", flush=True)
        return []


async def _brave(query: str, count: int, chars: int, domains: list | None) -> list:
    """Brave Search API: GET /res/v1/web/search with the key in X-Subscription-Token. Domains are limited with site: operators."""
    key = os.getenv("BRAVE_API_KEY")
    if not key:
        return []
    q = query if not domains else f"{query} (" + " OR ".join(f"site:{d}" for d in list(domains)[:8]) + ")"
    try:
        gap = 1.0 / float(os.getenv("BRAVE_PER_SECOND", "1"))
        if max(0.0, _next_slot.get("brave", 0.0) - time.monotonic()) > float(os.getenv("BRAVE_MAX_WAIT_S", "3")):
            return []      # many searches at once: this one skips Brave rather than waiting in its queue (Tavily and Exa still answer it)
        await _space("brave", 1.0 / gap)
        r = await async_core.client().get("https://api.search.brave.com/res/v1/web/search", params={"q": q[:390], "count": count, "extra_snippets": "true"},
                                           headers={"X-Subscription-Token": key, "Accept": "application/json"})
        r.raise_for_status()
        out = []
        for x in ((r.json().get("web") or {}).get("results") or []):
            if not x.get("url"):
                continue
            text = " ".join([re.sub(r"<[^>]+>", "", x.get("description") or "")] + [re.sub(r"<[^>]+>", "", e) for e in (x.get("extra_snippets") or [])])
            out.append({"title": re.sub(r"<[^>]+>", "", x.get("title") or ""), "url": x.get("url"), "snippet": text[:chars]})
        return out
    except Exception as error:
        print(f"[brave] search failed: {type(error).__name__}: {str(error)[:100]}", flush=True)
        return []


def _providers() -> set:
    """Which web search providers are used: WEB_PROVIDERS in .env (default: tavily and exa - every one that has a key; brave is supported but off). They all run at the same time on every web search."""
    return {x.strip().lower() for x in os.getenv("WEB_PROVIDERS", "tavily,exa").split(",") if x.strip()}


QUESTION = contextvars.ContextVar("web_question", default="")      # the person's question of this run: Tavily is sent it as a sentence beside the plan's keywords


def natural_query(query: str) -> str:
    """The question as a sentence plus the focus of this search, for a search engine that reads meaning. Off with TAVILY_NATURAL=off; unchanged when no question is set or the query is already long."""
    question = " ".join(QUESTION.get().split())
    if os.getenv("TAVILY_NATURAL", "on").lower() in ("off", "0", "false") or not question or len(query) > 120 or query.lower() in question.lower():
        return query
    return f"{question} Focus: {query}"[:390]


async def _tavily_items(query: str, count: int, chars: int, domains: list | None) -> list:
    if "tavily" not in _providers() or not os.getenv("TAVILY_API_KEY"):
        return []
    try:
        await async_core.limiter.acquire("tavily")
        extra = {"include_domains": list(domains)} if domains else {}
        found = (await _tavily().search(natural_query(query), max_results=count, search_depth=os.getenv("TAVILY_DEPTH", "advanced").strip() or "advanced", **extra)).get("results", [])
        return [{"title": r.get("title"), "url": r.get("url"), "snippet": (r.get("content") or "")[:chars]} for r in found if r.get("url")]
    except Exception as error:
        if _refused(error):
            _mark_exhausted(error)
        return []


async def _tavily_semantic(query: str, count: int, chars: int, domains: list | None) -> list:
    """Tavily asked for MORE pages than needed (a search costs the same credits whatever its max_results), then the `count` pages whose text is closest in MEANING to the question are kept: the question and each page
    (title + excerpt) are embedded and compared by cosine similarity. This is the semantic choice that Tavily itself does not offer. Off with WEB_SEMANTIC=off; when the embedding is unavailable or slow, Tavily's own order is kept."""
    if os.getenv("WEB_SEMANTIC", "on").strip().lower() in ("off", "0", "false", "no") or not QUESTION.get():
        return await _tavily_items(query, count, chars, domains)
    items = await _tavily_items(query, min(20, max(count, count * int(os.getenv("WEB_OVERFETCH", "3")))), max(chars, 700), domains)
    if len(items) <= count:
        return [{**x, "snippet": (x.get("snippet") or "")[:chars]} for x in items]
    try:
        import async_support
        import hybrid_search as hs
        vectors = await async_support.aembed([natural_query(query)] + [f"{x.get('title') or ''}. {(x.get('snippet') or '')[:700]}" for x in items], timeout=float(os.getenv("WEB_EMBED_TIMEOUT_S", "3")))
        if vectors:
            scored = sorted(range(len(items)), key=lambda i: -hs.cosine(vectors[0], vectors[i + 1]))
            items = [items[i] for i in scored]
    except Exception as error:
        print(f"[web] semantic selection skipped: {type(error).__name__}", flush=True)
    return [{**x, "snippet": (x.get("snippet") or "")[:chars]} for x in items[:count]]


async def _search_all(query: str, chars: int = 500, domains: list | None = None, count: int = 5, cap: int = 8) -> list:
    """The same search on Tavily, Exa and Brave at the same time; the pages are merged (each URL once, the providers' lists interleaved so every provider's best page comes first).
    DuckDuckGo is no longer used. A provider without a key, or that fails, simply adds nothing."""
    wanted = _providers()
    lists = await asyncio.gather(_tavily_semantic(query, count, chars, domains),
                                 _exa(query, count, chars, domains) if "exa" in wanted else asyncio.sleep(0, result=[]),
                                 _brave(query, count, chars, domains) if "brave" in wanted else asyncio.sleep(0, result=[]))
    seen, merged = set(), []
    for i in range(max(len(x) for x in lists)):
        for lst in lists:
            if i < len(lst) and lst[i]["url"] not in seen:
                seen.add(lst[i]["url"])
                merged.append(lst[i])
    return merged[:cap]


async def aweb_search(query: str, snippet_chars: int = 500) -> str:
    """Async twin of qa_stream.web_search: Tavily, Exa and Brave side by side, results merged."""
    items = await _search_all(query, snippet_chars)
    return json.dumps(items, ensure_ascii=False) if items else "Web search failed: No results found."


async def aguideline_search(query: str) -> str:
    """Async twin of qa_stream.guideline_search: the three providers limited to the guideline organisations' domains; results are re-checked against that list afterwards."""
    import qa_stream
    items = await _search_all(query, 1500, list(qa_stream.load_care_config()["authoritative_domains"]))
    return json.dumps(items, ensure_ascii=False) if items else "Web search failed: No results found."


async def aregulator_search(query: str) -> str:
    """Async twin of qa_stream.regulator_search: the query as written and the same words as a press announcement run side by side (each on all providers); documents are dropped; each page appears once."""
    import qa_stream
    domains = list(qa_stream.REGULATOR_DOMAINS)
    batches = await asyncio.gather(_search_all(query, 1500, domains, count=8, cap=10), _search_all("FDA press announcement approves OR safety communication " + query, 1500, domains, count=8, cap=10))
    seen, items = set(), []
    for batch in batches:
        for r in batch:
            if r.get("url") and r["url"] not in seen and not qa_stream.REGULATOR_DOCUMENT_URL.search(r["url"]):
                seen.add(r["url"])
                items.append(r)
    return json.dumps(items[:8], ensure_ascii=False) if items else "Web search failed: No results found."


async def aregulator_search_many(queries: list[str]) -> str:
    """Async twin of qa_stream.regulator_search_many: several regulator-site searches merged (each page once)."""
    import qa_stream
    seen, merged = set(), []
    for raw in await asyncio.gather(*[aregulator_search(q) for q in queries]):
        data = qa_stream._json(raw)
        for item in data if isinstance(data, list) else []:
            if item.get("url") and item["url"] not in seen:
                seen.add(item["url"])
                merged.append(item)
    return json.dumps(merged, ensure_ascii=False) if merged else "Web search failed: No results found."


async def afind_guidelines(question: str, condition_words: list[str], searcher=aguideline_search, config: dict | None = None) -> dict:
    """Async twin of guidelines.find_guidelines: one search per care topic, all topics at the same time. `searcher` is an async function (query -> JSON text)."""
    config = config or gl.load_config()
    topics = gl.topics_for(question, config)
    topics = [t for t in topics if t.get("drugs")] or topics
    stems = [w[:5] for w in condition_words if len(w) >= 4]
    condition = " ".join(condition_words)

    async def one(topic: dict) -> dict:
        query = f"{condition} {topic['query']} guideline recommendations"
        raw = await searcher(query)
        if not str(raw).lstrip().startswith("["):      # rate-limited or failed: ONE retry after a short pause
            await asyncio.sleep(1.0)
            raw = await searcher(query)
        try:
            items = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            return {"topic": topic["id"], "label": topic["label"], "error": str(raw)[:120]}
        if not isinstance(items, list):
            return {"topic": topic["id"], "label": topic["label"], "error": str(raw)[:120]}
        for item in items:
            url = item.get("url") or ""
            org = gl.organisation_for(url, config)
            excerpt = gl._excerpt(item.get("snippet") or "")
            title = re.sub(r"\s+", " ", item.get("title") or "").strip()
            blob = f"{title} {item.get('snippet') or ''}".lower()
            if org and excerpt and title and (not stems or sum(1 for s in stems if s in blob) >= min(2, len(stems))):
                return {"topic": topic["id"], "label": topic["label"], "title": title, "organisation": org, "year": gl._year(title, item.get("snippet"), url),
                        "url": url, "excerpt": excerpt, "text": re.sub(r"\s+", " ", item.get("snippet") or "")[:1500]}
        return {"topic": topic["id"], "label": topic["label"], "none": True}

    rows = await asyncio.gather(*[one(t) for t in topics])
    failed = sum(1 for r in rows if r.get("error"))
    if failed == len(rows):
        return {"error": f"guideline search failed for all {len(rows)} topics: {rows[0].get('error')}"}
    return {"results": list(rows), "failed": failed, "topics": len(rows)}
