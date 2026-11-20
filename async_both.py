"""SOURCES_MODE=both: the source's own API and Tavily searched AT THE SAME TIME, merged and de-duplicated (the direct record wins) before the ranking.

Latency: the wait is that of the direct search plus at most BOTH_TAVILY_GRACE_S (default 1 s). A Tavily answer that comes later is dropped. When the direct search gives nothing, Tavily is waited for in full.
One failing side never fails the other (asyncio.gather with return_exceptions=True)."""
from __future__ import annotations

import asyncio
import json
import os
import re

import async_tavily_sources as via


def _grace() -> float:
    try:
        return float(os.getenv("BOTH_TAVILY_GRACE_S", "1.0"))
    except ValueError:
        return 1.0


def key_of(r: dict) -> str | None:
    """The identity of a record: PMID, else NCT number, else the canonical URL (no scheme, www, query, fragment or trailing slash)."""
    pmid = re.sub(r"\D", "", str(r.get("pmid") or ""))
    if pmid:
        return "pmid:" + pmid
    nct = str(r.get("nct_id") or "").upper().strip()
    if re.fullmatch(r"NCT\d{8}", nct):
        return "nct:" + nct
    url = str(r.get("url") or "").lower().strip()
    url = re.sub(r"^https?://(www\.)?", "", url).split("#")[0].split("?")[0].rstrip("/")
    return "url:" + url if url else None


def _rows(raw) -> tuple[dict, list]:
    try:
        data = json.loads(raw) if isinstance(raw, str) else {}
    except ValueError:
        return {}, []
    rows = data.get("results") if isinstance(data, dict) else None
    return (data if isinstance(data, dict) else {}), (rows if isinstance(rows, list) else [])


def merge(direct_raw, tavily_raw, limit: int) -> str:
    """Direct records first (their abstract and metadata are the cleaner ones); a Tavily record is added only when its key is not already there."""
    d_data, d_rows = _rows(direct_raw)
    t_data, t_rows = _rows(tavily_raw)
    if not t_rows:
        return direct_raw if (d_rows or not t_data) and direct_raw is not None else (tavily_raw if tavily_raw is not None else direct_raw)
    if not d_rows:
        return tavily_raw
    seen = {k for k in map(key_of, d_rows) if k}
    extra = []
    for r in t_rows:
        k = key_of(r)
        if k and k in seen:
            continue
        if k:
            seen.add(k)
        extra.append(r)
    out = {**d_data, "results": (d_rows + extra)[:max(limit, len(d_rows))], "merged": {"direct": len(d_rows), "tavily_only": len(extra)}}
    return json.dumps(out, ensure_ascii=False)


async def both(name: str, direct_fn, tavily_fn, limit: int, *args, **kwargs) -> str:
    direct = asyncio.ensure_future(direct_fn(*args, **kwargs))

    async def tavily_leg():
        token = via.NO_FALLBACK.set(True)      # Tavily must not call the direct API itself: it is already running
        try:
            t = asyncio.ensure_future(tavily_fn(*args, **kwargs))      # the task copies the context now
        finally:
            via.NO_FALLBACK.reset(token)
        await asyncio.wait({t, direct}, return_when=asyncio.FIRST_COMPLETED)
        if not t.done():
            ok = direct.done() and not direct.exception() and bool(_rows(direct.result())[1])
            await asyncio.wait({t}, timeout=_grace() if ok else None)      # direct gave nothing (or is still running): Tavily is waited for in full
        if not t.done():
            t.cancel()
            raise asyncio.TimeoutError(f"{name}: Tavily was late")
        return t.result()

    d, t = await asyncio.gather(direct, tavily_leg(), return_exceptions=True)
    d = None if isinstance(d, BaseException) else d
    t = None if isinstance(t, BaseException) else t
    if d is None and t is None:
        return json.dumps({"error": f"{name}: both the direct API and Tavily failed"})
    return merge(d, t, limit)
