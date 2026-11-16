"""Warm-up of the ASYNC side (the twin of qa_stream.warm_up / clinical_tools.warm_connections): opens the data-source connections of the async HTTP client and the model connections of the async
model calls once, at start-up, so the first question does not pay the handshakes inside the fast search's 2-3 second limit. It runs on the async loop (async_runner.get_loop)."""
from __future__ import annotations

import asyncio

import async_core
import clinical_tools as ct
import qa_stream as qs


async def awarm() -> None:
    sources = ((f"{ct.EUTILS}/einfo.fcgi", {"db": "pubmed", "retmode": "json"}),
               ("https://www.ebi.ac.uk/europepmc/webservices/rest/search", {"query": "treatment", "format": "json", "pageSize": 1}),
               (ct.CTGOV, {"query.cond": "disease", "pageSize": 1}),
               ("https://api.fda.gov/drug/label.json", {"search": "openfda.product_type:\"HUMAN PRESCRIPTION DRUG\"", "limit": 1}),
               ("https://pmc-oa-opendata.s3.amazonaws.com/", {"list-type": "2", "prefix": "PMC1.", "max-keys": "1"}))

    async def source(url: str, params: dict) -> None:
        try:
            await async_core.aget(url, params)
        except Exception:
            pass

    async def model(size: int, temperature, purpose: str) -> None:
        try:
            async for _ in async_core.astream_llm("Reply: ok", "ok", size, attempts=1, temperature=temperature, purpose=purpose):
                pass
        except Exception:
            pass
    try:      # the Tavily client object (no search is made: a search costs a credit)
        import async_web
        async_web._tavily()
    except Exception:
        pass
    models = ((qs.FAST_TOKENS, None, "writer"), (qs.DEEP_TOKENS, None, "writer"), (qs.PLAN_TOKENS, 0.0, "planning"), (700, 0.0, "triage"), (900, 0.0, "triage"), (600, 0.0, "triage"))
    await asyncio.gather(*(source(u, p) for u, p in sources), *(model(*m) for m in models))
