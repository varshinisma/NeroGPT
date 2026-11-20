"""The async pipeline (slice 6): the stage mixins combined into one class, the main async generator `astream_answer`, and the one adapter the synchronous web layer needs.

astream_answer(question, plan) yields exactly the events of qa_stream.stream_answer for a NEW question. A follow-up always runs the threaded stream_answer (api.py decides).
`stream_answer_async(question, plan)` is the synchronous-iterator view of the same generator: api.ask_events is still a synchronous generator (Starlette iterates it in its own worker), so
the async run lives on one long-lived event loop and each event is handed over with run_coroutine_threadsafe. Everything the pipeline does runs on that loop.
"""
from __future__ import annotations

import asyncio
import os
import threading
from typing import AsyncIterator, Iterator

import async_agent_run
import async_core
import observability
import qa_stream as qs
import stages as stages_mod
from async_pipeline import AsyncRun
from async_stages_deep import DeepStages
from async_stages_drugs import DrugStages
from async_stages_evidence import EvidenceStages
from async_stages_final import FinalStages
from async_stages_writers import WriterStages

OVERLAPPED = ("deep_search", "rank")      # run beside fast_answer (their events are held until it ends)


def overlap_enabled() -> bool:
    return os.getenv("OVERLAP_FAST_ANSWER", "on").strip().lower() not in {"off", "0", "false", "no"}


STAGE_NAMES = ("fast_search", "fast_answer", "deep_search", "rank", "read_evidence", "drugs", "escalate", "write_start", "figures", "write_stream", "assemble", "limitations", "verify", "finalize")


class AsyncPipeline(AsyncRun, DeepStages, EvidenceStages, DrugStages, WriterStages, FinalStages):
    """One run: the base state and the fast stages (AsyncRun) plus every other stage as a mixin."""

    def __init__(self, question: str, plan: dict | None = None):
        super().__init__(question, plan)
        self.prefetched: set[str] = set()      # stages that finished beside the fast answer
        self.init_deep()
        self.init_evidence()
        self.init_drugs()
        self.init_writers()
        self.init_final()

    async def with_heartbeat(self, stream: AsyncIterator[dict], stage_name: str) -> AsyncIterator[dict]:
        """The stage's own events, unchanged and in the same order, plus a 'heartbeat' event every HEARTBEAT_S while the stage is busy and sends nothing."""
        iterator = stream.__aiter__()
        pending = asyncio.ensure_future(iterator.__anext__())
        try:
            while True:
                done, _ = await asyncio.wait([pending], timeout=qs.HEARTBEAT_S)
                if not done:
                    yield self.event("heartbeat", stage_name=stage_name, papers=len(self.papers), trials=len(self.trials), drugs=len(self.state.get("regulatory") or []))
                    continue
                try:
                    event = pending.result()
                except StopAsyncIteration:
                    return
                yield event
                pending = asyncio.ensure_future(iterator.__anext__())
        finally:
            if not pending.done():
                pending.cancel()
            try:
                await iterator.aclose()
            except Exception:
                pass

    async def run_stage(self, name: str) -> AsyncIterator[dict]:
        self.ran.append(name)
        began, began_abs = self.clock.now(), observability.now()
        stream = getattr(self, f"stage_{name}")()
        async for event in (stream if self.focused else self.with_heartbeat(stream, name)):      # a follow-up streams without heartbeats, as before
            yield event
        observability.record("stage", name, began_abs, observability.now())
        seconds = self.state.setdefault("stage_seconds", {})
        seconds[name] = round(self.clock.now() - began, 1)
        if not self.halted and "finalize" not in self.ran:      # the final event stays the last event of the run
            yield self.event("progress", stage_name=name, took=seconds[name], papers=len(self.papers), trials=len(self.trials))

    async def _prefetch(self, held: list) -> None:
        """deep_search and rank only read the searches (not the fast answer's text), so they run WHILE the fast answer is written. Their events are held (heartbeats dropped) and sent after the fast answer's
        own events, so the page sees the same order as before. A stage that fails here is taken out of `ran` and left undone: the agent (or the guard) runs it normally afterwards."""
        for name in OVERLAPPED:
            if self.halted:
                return
            try:
                async for event in self.run_stage(name):
                    if event.get("type") != "heartbeat":
                        held.append(event)
                self.prefetched.add(name)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                if name in self.ran:
                    self.ran.remove(name)
                self.state.setdefault("overlap_errors", {})[name] = f"{type(error).__name__}: {str(error)[:100]}"
                return      # the later stage needs this one: nothing more is overlapped

    async def run(self) -> AsyncIterator[dict]:
        yield self.event("stage", stage="INITIAL")
        held: list = []
        overlap: asyncio.Future | None = None
        for name in stages_mod.FIRST:      # the fast path starts before any plan exists
            if name == "fast_answer" and overlap_enabled() and not self.focused:      # the deep searches are already running: merge and rank them beside the fast answer
                overlap = asyncio.ensure_future(self._prefetch(held))
            async for event in self.run_stage(name):
                yield event
            if self.halted:
                if overlap is not None and not overlap.done():
                    overlap.cancel()
                return
        if overlap is not None:
            try:
                await overlap
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
            for event in held:
                yield event
            if self.halted:
                return
        async for event in async_agent_run.drive(self):      # every run, new question or follow-up: the agent chooses the next stage (its tools are the stages); code enforces the rules and completes what it leaves
            yield event


async def astream_answer(question: str, plan: dict | None = None) -> AsyncIterator[dict]:
    """Async twin of qa_stream.stream_answer for a new question."""
    run = AsyncPipeline(question, plan)
    try:
        async for event in run.run():
            yield event
    finally:
        # whatever is still running (searches, writers) is stopped when the page is gone; the figures may still finish for the PDF
        for task in list(run._deep_tasks) + list(run._writer_tasks):
            if not task.done():
                task.cancel()


# ---------------------------------------------------------------- the synchronous view (for api.ask_events)
_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()


def get_loop() -> asyncio.AbstractEventLoop:
    """The one long-lived event loop all async runs share (created on first use). It is the loop the figures finish on after the report is final."""
    global _loop
    with _loop_lock:
        if _loop is None or _loop.is_closed():
            loop = asyncio.new_event_loop()
            threading.Thread(target=loop.run_forever, name="neurogpt-async-loop", daemon=True).start()
            _loop = loop
        return _loop


def stream_answer_async(question: str, plan: dict | None = None) -> Iterator[dict]:
    """The events of astream_answer as a normal iterator. Closing the iterator (the page was closed) closes the async generator on its loop."""
    loop = get_loop()
    stream = astream_answer(question, plan)
    async def next_event():
        return await stream.__anext__()
    try:
        while True:
            future = asyncio.run_coroutine_threadsafe(next_event(), loop)
            try:
                yield future.result()
            except StopAsyncIteration:
                return
    finally:
        try:
            asyncio.run_coroutine_threadsafe(stream.aclose(), loop).result(timeout=5)
        except Exception:
            pass


def stream_answer_auto(question: str, plan: dict | None = None) -> Iterator[dict]:
    """The async run; if it cannot even start (an error before its first event), the threaded stream_answer answers instead."""
    started = False
    try:
        for event in stream_answer_async(question, plan):
            started = True
            yield event
    except Exception:
        if started:
            raise
        yield from (qs.stream_answer(question, plan=plan) if plan else qs.stream_answer(question))
