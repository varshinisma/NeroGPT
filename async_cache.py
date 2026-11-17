"""A small time-limited cache for awaited lookups, with in-flight sharing: two callers that ask for the same thing at the same moment wait for ONE call, and a later caller within the time limit gets the
stored answer. Used for the openFDA label lookups (labels change rarely; the same drug is looked up many times in a run and across questions). Errors are never stored. Every caller gets its own copy,
so nobody can change what another caller reads. A time limit of 0 (the default) turns the cache off and every call goes straight to `fetch`."""
from __future__ import annotations

import asyncio
import copy
import time

MAX_ENTRIES = 600
_store: dict = {}
_inflight: dict = {}


async def cached_call(key: str, ttl: float, fetch):
    """`fetch` is a function with no arguments that returns an awaitable. Returns what it returns, from the cache when a stored answer is younger than `ttl` seconds."""
    if ttl <= 0:
        return await fetch()
    hit = _store.get(key)
    if hit is not None and time.monotonic() - hit[0] < ttl:
        return copy.deepcopy(hit[1])
    flight = (id(asyncio.get_running_loop()), key)      # a future belongs to the loop that made it
    pending = _inflight.get(flight)
    if pending is None:
        async def run():
            value = await fetch()
            if len(_store) >= MAX_ENTRIES:
                _store.clear()
            _store[key] = (time.monotonic(), value)
            return value
        pending = asyncio.ensure_future(run())
        _inflight[flight] = pending
        pending.add_done_callback(lambda f, k=flight: (_inflight.pop(k, None), f.cancelled() or f.exception()))
    return copy.deepcopy(await asyncio.shield(pending))


def clear() -> None:
    _store.clear()
    _inflight.clear()
