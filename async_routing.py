"""Async twin of retrieval_routing.Execution: the routed searches run at the same time under a time budget, with no thread. A search that never got a slot before the budget ended is `not_run`, one that
was running is `timed_out` (it keeps running and is kept in `late` if it answers afterwards), and a transient failure is retried once. The adapters are async functions (op -> raw reply).

Must be created inside a running event loop (it starts tasks); `await execution.finish()` waits for the budget and records what finished (the threaded version did this in its constructor unless `hold` was set)."""
from __future__ import annotations

import asyncio
import json
import time

import observability
from retrieval_routing import Operation, Result, classify_raw, is_transient


class AsyncExecution:
    def __init__(self, ops: list[Operation], adapters: dict, budget_s: float, workers: int = 6, clock=time.monotonic, retry: bool = True, hold: bool = False):
        """`hold=True`: the searches start as soon as they are submitted (`submit`) and the budget starts when `finish()` is called."""
        self.ops, self.adapters, self.budget_s, self.clock, self.retry = list(ops), adapters, budget_s, clock, retry
        self.results: dict[str, Result] = {op.id: Result() for op in ops}
        self.late: dict[str, Result] = {}
        self._deadline = float("inf") if hold else clock() + budget_s
        self._closed = False
        self._slots = asyncio.Semaphore(max(1, workers if hold else min(workers, len(ops) or 1)))
        self._started: set[str] = set()
        self._tasks: dict[str, asyncio.Task] = {op.id: asyncio.ensure_future(self._run(op)) for op in ops}
        self._finish_task: asyncio.Future | None = None

    def submit(self, ops: list[Operation]) -> None:
        if self._closed:      # the budget has ended: a search written after that is not started
            return
        for op in ops:
            self.ops.append(op)
            self.results[op.id] = Result()
            self._tasks[op.id] = asyncio.ensure_future(self._run(op))

    async def finish(self) -> None:
        """Wait for the searches up to the budget (counted from now) and record what did not finish. Safe to call more than once: later calls wait for the first."""
        self._closed = True
        if self._finish_task is None:
            self._finish_task = asyncio.ensure_future(self._finish())
        await asyncio.shield(self._finish_task)

    async def _finish(self) -> None:
        self._deadline = self.clock() + self.budget_s
        tasks = list(self._tasks.values())
        done, pending = await asyncio.wait(tasks, timeout=self.budget_s) if tasks else (set(), set())
        by_task = {t: op_id for op_id, t in self._tasks.items()}
        for task in done:
            try:
                self.results[by_task[task]] = task.result()
            except Exception as error:
                self.results[by_task[task]] = Result(status="failed", error=f"{type(error).__name__}: {str(error)[:100]}")
        for task in pending:
            op_id = by_task[task]
            if op_id not in self._started:
                task.cancel()
                self.results[op_id] = Result(status="not_run", error="not started before the budget ended")
            else:
                self.results[op_id] = Result(status="timed_out", error=f"no answer within {self.budget_s:.0f} s")
                task.add_done_callback(lambda t, op_id=op_id: self._arrived_late(op_id, t))

    def _arrived_late(self, op_id: str, task) -> None:
        if task.cancelled():
            return
        try:
            result = task.result()
        except Exception as error:
            result = Result(status="failed", error=f"{type(error).__name__}: {str(error)[:100]}")
        result.late = True
        self.late[op_id] = result

    async def _call(self, op: Operation) -> Result:
        began = self.clock()
        try:
            with observability.tool(f"{op.target}"):
                raw = await self.adapters[op.target](op)
        except KeyError:
            return Result(status="failed", error=f"no adapter for target {op.target}", seconds=0.0, attempts=1)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            return Result(status="failed", error=f"{type(error).__name__}: {str(error)[:120]}", seconds=self.clock() - began, attempts=1)
        status, n, error = classify_raw(raw)
        return Result(status=status, n=n, error=error, seconds=self.clock() - began, attempts=1, raw=raw if isinstance(raw, str) else json.dumps(raw))

    async def _run(self, op: Operation) -> Result:
        async with self._slots:
            self._started.add(op.id)
            result = await self._call(op)
            if result.status == "failed" and self.retry and is_transient(result.error) and self.clock() < self._deadline:      # a transient failure is retried ONCE on the same source
                second = await self._call(op)
                second.attempts = 2
                return second
            return result

    def late_results(self) -> dict[str, Result]:
        return dict(self.late)
