"""Async evidence writers (slice 4): the twin of qa_stream.make_writing / launch_writer / drain / stage_write_start / stage_write_stream.

Each writing part (latest, approach, current, tail) is one asyncio task that puts its text into an asyncio.Queue as the model produces it; the Current evidence rows are written as two
concurrent tasks (CURRENT_SPLIT). The stream stage shows the parts in ready-order by waiting on the parts' `done` events, not by polling.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import AsyncIterator

import async_core
import qa_stream as qs


class WriterStages:
    """Methods added to AsyncRun (with DeepStages, EvidenceStages, DrugStages). `init_writers()` must be called once, inside the running loop."""

    def init_writers(self) -> None:
        self.writing: dict | None = None
        self.needs_workup: bool | None = None
        self.writers_started: set = set()
        self._writer_tasks: list[asyncio.Task] = []

    def make_writing(self) -> None:
        """The queues of the writing tasks (once): the Latest findings can start before the others, so they must exist before write_start."""
        if self.writing is not None:
            return
        self.needs_workup = qs.wants_workup(self.state.get("understanding") or {}, self.question)
        names = ("latest", "approach", "current", "tail") if self.needs_workup else ("latest", "current", "tail")
        self.writing = {name: {"queue": asyncio.Queue(), "text": "", "produced": [], "done": asyncio.Event()} for name in names}

    def launch_writer(self, name: str) -> None:
        """Start ONE writing task, once, from the evidence in the state at this moment."""
        self.make_writing()
        if name in self.writers_started or (name == "approach" and not self.needs_workup):
            return
        self.writers_started.add(name)
        state, question, clock, skipped = self.state, self.question, self.clock, self.skipped
        understanding = state.get("understanding") or {}
        task = {"latest": qs.LATEST_TASK, "approach": qs.APPROACH_TASK + " SECTIONS: " + qs.approach_sections(understanding, question) + ".", "current": qs.CURRENT_TASK, "tail": qs.TAIL_TASK}[name]
        context = qs.build_evidence_context(state, question)      # one synchronous read: nothing can change the papers while it runs
        writing = self.writing

        async def write_part() -> None:
            prompt = f"TASK: {task}\n\nQUESTION: {question}\nRETRIEVED DATA (JSON):\n{json.dumps(context, ensure_ascii=False, separators=(',', ':'))}"
            trace = writing[name].setdefault("trace", {"started": round(clock.now(), 1)})
            compact = "\n\nWrite the table compactly: ONE space on each side of every '|' and a plain '|---|---|' separator row. Never pad cells with spaces to align columns."
            held: list[str] = []      # the Current evidence part is released only when it is complete: a retry must not leave a half table on the screen

            def emit(text: str) -> None:
                if name == "current":
                    held.append(text)
                else:
                    writing[name]["produced"].append(text)
                    writing[name]["queue"].put_nowait(text)

            async def run(task_prompt: str, temperature: float | None, out: list | None = None) -> bool:
                """Stream one attempt, squeezing runs of spaces. Returns False when the model fell into endless padding and the attempt was abandoned."""
                blanks, last = 0, "\n"
                async for chunk in async_core.astream_llm(qs.load_skill_text("evidence-synthesis"), task_prompt, qs.DEEP_TOKENS, temperature=temperature, purpose=("writer_small" if name == "tail" else "writer")):
                    trace.setdefault("first_chunk", round(clock.now(), 1))
                    squeezed = re.sub(r"[ \t]{2,}", " ", chunk)
                    if last in " \t" and squeezed.startswith(" "):
                        squeezed = squeezed[1:]
                    blanks = blanks + len(chunk) if not chunk.strip(" \t") else 0
                    if blanks > 400:
                        return False
                    if squeezed:
                        last = squeezed[-1]
                        (out.append if out is not None else emit)(squeezed)
                return True

            async def run_split() -> bool:
                """The two halves of the Current evidence rows, written at the same time; the lines are joined, a repeated approach is kept once."""
                first_note = "\n\nYou are one of two writers working at the same time. Write the heading and AT MOST 4 lines: the 4 approaches with the strongest evidence."
                second_note = "\n\nYou are one of two writers working at the same time. Another writer covers the 4 approaches with the strongest evidence. Write NO heading and AT MOST 4 lines, for the NEXT strongest approaches (5th to 8th). If fewer than 5 approaches have a matching paper, write nothing."
                parts: list[list[str]] = [[], []]

                async def half(i: int, note: str) -> bool:
                    try:
                        return await run(prompt + note, None, parts[i])
                    except Exception:
                        return False
                results = await asyncio.gather(half(0, first_note), half(1, second_note))
                seen, lines = set(), []
                for line in "".join("".join(p) + chr(10) for p in parts).split(chr(10)):
                    if "||" in line:
                        key = re.sub(r"\W+", " ", line.split("||")[0]).strip().lower()
                        if key in seen:
                            continue
                        seen.add(key)
                    if line.strip():
                        lines.append(line)
                held.append(chr(10).join(lines))
                return results[0] and (results[1] or not "".join(parts[1]).strip())
            try:
                finished = await run_split() if (name == "current" and qs.CURRENT_SPLIT and not self.focused) else await run(prompt, None)
                if name == "current":
                    held[:] = [qs.rows_to_table("".join(held), state["papers"])]      # final table order and relevance labels come from the same screened records as all other evidence sections
                    if not finished or not re.search(r"(?m)^\|\s*-{3}", held[0]):
                        trace["retried"] = True      # endless padding, or no rows at all: the part is written again once, deterministically
                        held.clear()
                        await run(prompt + compact, 0.0)
                        held[:] = [qs.rows_to_table("".join(held), state["papers"])]
                    if not re.search(r"(?m)^\|\s*-{3}", "".join(held)):      # no row survived: the heading is not left alone on the screen
                        held[:] = ["## Current evidence\n\nNo retrieved paper reports a treatment result for this question; the papers found are reviews or background sources, which do not support a treatment row. See Key studies and the drug landscape for what was found."]
                        skipped.append("current evidence: the writer produced no rows (the retrieved papers are background or indirect)")
                    writing[name]["produced"][:] = ["".join(held)]
                    writing[name]["queue"].put_nowait("".join(held))
                elif not finished:
                    trace["retried"] = True
                    await run(prompt + compact, 0.0)
                trace["ended"] = round(clock.now(), 1)
            except Exception as error:
                skipped.append(f"evidence writing, part {name} ({str(error)[:100]})")
            finally:
                writing[name]["done"].set()
                writing[name]["queue"].put_nowait(None)
        self._writer_tasks.append(asyncio.ensure_future(write_part()))

    async def drain(self, name: str, live: bool = True) -> AsyncIterator[dict]:
        """Everything one writing task produces, in order: shown as it arrives (live) or collected silently. A part already read to its end is not read again."""
        part = self.writing[name]
        while not part.get("drained"):
            chunk = await part["queue"].get()
            if chunk is None:
                part["drained"] = True
                return
            part["text"] += chunk
            if live:
                yield self.event("token", stage="EVIDENCE_ENRICHING", text=chunk)

    # ---- stage: write_start
    async def stage_write_start(self) -> AsyncIterator[dict]:
        if self.focused and "evidence" not in self.focus_sections:      # nothing is written by the evidence writers: their parts are empty and already finished
            self.needs_workup = False
            self.writing = {name: {"queue": asyncio.Queue(), "text": "", "produced": [], "done": asyncio.Event()} for name in ("latest", "current", "tail")}
            for part in self.writing.values():
                part["queue"].put_nowait(None)
                part["done"].set()
            return
        self.make_writing()
        for name in ("latest", "current", "tail"):
            self.launch_writer(name)      # these parts do not use the guidelines: they start now (the Latest findings may already be running: EARLY_LATEST, never started twice)
        if self.needs_workup:      # only the clinical approach waits for the guideline retrieval; the other parts are being written meanwhile
            try:
                await asyncio.wait_for(self.guidelines_ready.wait(), timeout=max(0.0, self.T.deep_budget - self.clock.now()))
            except asyncio.TimeoutError:
                self.skipped.append("guideline retrieval did not finish before the report deadline; the clinical approach may be incomplete")
        self.launch_writer("approach")
        return
        yield

    # ---- stage: write_stream
    async def stage_write_stream(self) -> AsyncIterator[dict]:
        T, clock = self.T, self.clock
        if qs.SECTION_ORDER == "ready" and not self.focused:      # each section is shown as soon as it is finished; the standard order is kept in the saved report and the PDF
            async for e in self.drain("latest"):      # Latest findings first, live
                yield e
            yield self.event("token", stage="EVIDENCE_ENRICHING", text="\n\n")
            rest = (["approach"] if self.needs_workup else []) + ["current"]
            drugs_pending = self.landscape_task is not None
            while rest or drugs_pending:
                if drugs_pending and not self.landscape_running():      # the drug landscape and the trials are ready: shown now, whatever the writers are still doing
                    async for e in self.emit_landscape_and_trials():
                        yield e
                    drugs_pending = False
                    continue
                done_part = next((n for n in rest if self.writing[n]["done"].is_set()), None)
                if done_part:      # a finished part appears at once
                    rest.remove(done_part)
                    async for e in self.drain(done_part):
                        yield e
                    yield self.event("token", stage="EVIDENCE_ENRICHING", text="\n\n")
                    continue
                if rest:      # nothing is finished yet: the next part in the report's order is shown live as it is written
                    part_name = rest.pop(0)
                    async for e in self.drain(part_name):
                        yield e
                    yield self.event("token", stage="EVIDENCE_ENRICHING", text="\n\n")
                    continue
                await self.wait_landscape(0.3)      # only the drug stage is left: wait for it (the assemble stage forces it after its own limit)
                if clock.now() > T.deep_budget + T.landscape:
                    break
            clock.mark("evidence_done_s")
            return
        async for e in self.drain("latest"):
            yield e
        yield self.event("token", stage="EVIDENCE_ENRICHING", text="\n\n")
        if self.needs_workup:
            async for e in self.drain("approach"):
                yield e
            yield self.event("token", stage="EVIDENCE_ENRICHING", text="\n\n")
        async for e in self.drain("current"):
            yield e
        clock.mark("evidence_done_s")
