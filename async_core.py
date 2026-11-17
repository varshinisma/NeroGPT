"""Async leaf layer for the NeuroGPT pipeline: one rate limiter per data source, one HTTP client per event loop, the streamed model call and the JSON model call, and a runner that starts
several jobs at the same time and gives each its own time limit. Nothing here uses a thread: every wait is an `await`. The threaded pipeline (qa_stream.py) does not import this file."""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import AsyncIterator, Awaitable, Callable

import httpx

import observability
from ratelimit import per_second, source_of_url

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
USER_AGENT = "NeuroGPT-agent/1.0"
HTTP_TIMEOUT_S = float(os.getenv("ASYNC_HTTP_TIMEOUT_S", "15"))
FIRST_CHUNK_S, IDLE_CHUNK_S = 20.0, 25.0      # same limits as the threaded model call: a call silent for this long has stalled
RATE_LIMIT_RETRIES = int(os.getenv("RATE_LIMIT_RETRIES", "4"))
RATE_LIMIT_MAX_WAIT_S = float(os.getenv("RATE_LIMIT_MAX_WAIT_S", "25"))


# ------------------------------------------------------------------ rate limit
class AsyncSourceLimiter:
    """Spaces the calls to each source at least 1/rate seconds apart (the same rates as ratelimit.py). A slot is reserved without awaiting, so two tasks can never take the same slot; the wait is an
    asyncio.sleep, so the other tasks keep running."""

    def __init__(self) -> None:
        self._next: dict[str, float] = {}

    async def acquire(self, source: str) -> float:
        gap = 1.0 / per_second(source)
        now = time.monotonic()
        slot = max(now, self._next.get(source, 0.0))
        self._next[source] = slot + gap
        wait = slot - now
        if wait > 0:
            await asyncio.sleep(wait)
        return max(0.0, wait)


limiter = AsyncSourceLimiter()


# ------------------------------------------------------------------ HTTP
_clients: dict[int, httpx.AsyncClient] = {}


def client() -> httpx.AsyncClient:
    """One keep-alive client for the running event loop (an AsyncClient must not be shared between loops)."""
    key = id(asyncio.get_running_loop())
    found = _clients.get(key)
    if found is None or found.is_closed:
        found = httpx.AsyncClient(timeout=HTTP_TIMEOUT_S, headers={"User-Agent": USER_AGENT}, follow_redirects=True)
        _clients[key] = found
    return found


async def aclose() -> None:
    for key, http in list(_clients.items()):
        if not http.is_closed:
            await http.aclose()
        _clients.pop(key, None)


async def aget(url: str, params: dict | None = None) -> bytes:
    """GET with the per-source limit and one retry on a rate-limit / temporary server error (same behaviour as clinical_tools._get). Raises httpx.HTTPError on failure."""
    params = dict(params or {})
    if url.startswith(EUTILS) and os.getenv("NCBI_API_KEY"):
        params["api_key"] = os.environ["NCBI_API_KEY"]
    source = source_of_url(url)
    for attempt in range(2):
        if source != "other":
            await limiter.acquire(source)
        response = await client().get(url, params=params)
        if response.status_code in (429, 500, 502, 503) and attempt < 1:
            await asyncio.sleep(1.0)
            continue
        response.raise_for_status()
        return response.content
    raise RuntimeError("unreachable")


async def aget_json(url: str, params: dict | None = None) -> dict:
    return json.loads(await aget(url, params))


# ------------------------------------------------------------------ model calls
def retry_after_seconds(error_text: str, default: float) -> float:
    """The wait a provider asks for in a rate-limit message ('try again in 7.2s', '460ms', '1m5.5s'), plus a little; never longer than RATE_LIMIT_MAX_WAIT_S."""
    found = re.search(r"try again in\s+(?:(\d+)m(?!s))?\s*(?:(\d+(?:\.\d+)?)s)?\s*(?:(\d+(?:\.\d+)?)ms)?", str(error_text), re.I)
    wait_s = default
    if found and any(found.groups()):
        wait_s = float(found.group(1) or 0) * 60 + float(found.group(2) or 0) + float(found.group(3) or 0) / 1000 + 0.5
    return max(1.0, min(wait_s, RATE_LIMIT_MAX_WAIT_S))


async def _with_deadline(source: AsyncIterator[str], first: float = FIRST_CHUNK_S, idle: float = IDLE_CHUNK_S) -> AsyncIterator[str]:
    """Yield what a (possibly stalling) async iterator produces; silent for longer than the limit -> TimeoutError, so one stalled model call can never hold the whole answer."""
    iterator, waited = source.__aiter__(), first
    while True:
        try:
            item = await asyncio.wait_for(iterator.__anext__(), timeout=waited)
        except StopAsyncIteration:
            return
        except asyncio.TimeoutError:
            raise TimeoutError(f"the model gave no answer for {waited:.0f} s") from None
        waited = idle
        yield item


@observability.amodel_call
async def astream_llm(instructions: str, prompt: str, max_tokens: int, attempts: int = 2, temperature: float | None = None, purpose: str = "writer") -> AsyncIterator[str]:
    """Async twin of qa_stream.stream_llm: text chunks from a FRESH agent, retried only before the first chunk, a provider's rate limit waited out (not counted as an attempt)."""
    import qa_stream      # imported when called: it loads the models and the skills
    if qa_stream.CLAUDE_MODE:
        raise RuntimeError("the async pipeline does not support CLAUDE_MODE")
    from agno.agent import Agent

    async def one_attempt() -> AsyncIterator[str]:
        agent = Agent(model=qa_stream.model_for(max_tokens, temperature, purpose), instructions=[instructions], markdown=True)
        got = False
        async for event in agent.arun(prompt, stream=True):
            kind, content = str(getattr(event, "event", "")), getattr(event, "content", None)
            if "Error" in kind or (not got and str(content or "").startswith(("API error", "Error"))):
                raise RuntimeError(str(content or kind)[:900])
            if kind == "RunContent" and content:
                got = True
                yield str(content)
    last, tries, limited = None, 0, 0
    while tries < attempts:
        tries += 1
        got = False
        try:
            async for chunk in _with_deadline(one_attempt()):
                got = True
                yield chunk
            return
        except Exception as error:
            last = error
            if got:
                raise
            limit_hit = "429" in str(error) or "rate limit" in str(error).lower()
            observability.count("rate_limit_retries" if limit_hit else "model_retries")
            if limit_hit and limited < RATE_LIMIT_RETRIES:
                limited += 1
                tries -= 1
                await asyncio.sleep(retry_after_seconds(str(error), 3.0 * limited))
    raise RuntimeError(f"The model provider returned an error: {last}")


async def acollect(instructions: str, prompt: str, max_tokens: int, **kwargs) -> str:
    return "".join([chunk async for chunk in astream_llm(instructions, prompt, max_tokens, **kwargs)])


def parse_json_answer(raw: str) -> dict:
    """The JSON object in a model answer, as parsed. An answer cut off by the token limit keeps the complete objects before the cut (same rule as qa_stream.ask_json)."""
    found = re.search(r"\{.*\}", raw, re.S)
    text = found.group(0) if found else ""
    try:
        data = json.loads(text) if text else {}
    except json.JSONDecodeError:
        data = {}
        for key in ("fda_labels", "drugs", "items", "verdicts", "papers", "trials", "queries"):
            part = re.search(rf'"{key}"\s*:\s*\[(.*)', raw, re.S)
            if part:
                objects, depth, start = [], 0, None
                for i, ch in enumerate(part.group(1)):
                    if ch == "{":
                        start, depth = (i if depth == 0 else start), depth + 1
                    elif ch == "}" and depth:
                        depth -= 1
                        if depth == 0 and start is not None:
                            try:
                                objects.append(json.loads(part.group(1)[start:i + 1]))
                            except json.JSONDecodeError:
                                pass
                data[key] = objects
    data = data if isinstance(data, dict) else {}
    data.setdefault("_raw", raw[:2500])
    return data


_json_slots: asyncio.Semaphore | None = None


def _slots() -> asyncio.Semaphore | None:
    """MODEL_JSON_CONCURRENCY decision calls at once (0 = no limit), created inside the running loop."""
    global _json_slots
    count = int(os.getenv("MODEL_JSON_CONCURRENCY", "0") or 0)
    if count > 0 and _json_slots is None:
        _json_slots = asyncio.Semaphore(count)
    return _json_slots


async def aask_json(question: str, task: str, payload: dict, max_tokens: int, skill: str = "drug-intelligence", purpose: str = "triage") -> dict:
    """Async twin of qa_stream.ask_json: one call that must answer with a JSON object, returned as parsed (nothing in it is trusted until the caller has verified it)."""
    import qa_stream
    prompt = (f"TASK: {task} Write the JSON compactly on a single line, with no indentation or line breaks.\n\nQUESTION: {question}\nRETRIEVED DATA (JSON):\n"
              + json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    slots = _slots()
    if slots is not None:
        async with slots:
            raw = await acollect(qa_stream.load_skill_text(skill), prompt, max_tokens, temperature=0.0, purpose=purpose)
    else:
        raw = await acollect(qa_stream.load_skill_text(skill), prompt, max_tokens, temperature=0.0, purpose=purpose)
    return parse_json_answer(raw)


# ------------------------------------------------------------------ running jobs side by side
async def run_jobs_async(jobs: dict[str, Callable[[], Awaitable]], timeout: float) -> tuple[dict, dict]:
    """Start every job at once; each result that arrives before `timeout` is returned as {name: result}. A job that raises is returned in the second dict as {name: error text}; a job still running at
    the time limit is cancelled and named 'timed_out'. (The async twin of qa_stream.run_jobs.)"""
    results: dict = {}
    status: dict = {}
    tasks = {name: asyncio.ensure_future(job()) for name, job in jobs.items()}
    done, pending = await asyncio.wait(tasks.values(), timeout=timeout) if tasks else (set(), set())
    for name, task in tasks.items():
        if task in done:
            try:
                results[name] = task.result()
                status[name] = "ok"
            except Exception as error:
                status[name] = f"{type(error).__name__}: {str(error)[:100]}"
        else:
            task.cancel()
            status[name] = "timed_out"
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)      # let the cancelled tasks finish cancelling
    return results, status
