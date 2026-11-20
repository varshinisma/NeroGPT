"""The agent-driven run: an Agno agent decides which stage to run next (the stages are its tools). async_runner.AsyncPipeline.run hands every run to `drive` after the fast path.

What changes, and what does not
- The fast path (fast_search, fast_answer) still runs first, in code, so the first answer stays as quick as before. The agent starts after it.
- Each later stage of stages.py is offered to the agent as a TOOL (GROUPS below). A tool runs the real stage on the shared AsyncPipeline, pushes the stage's events (tokens, progress, heartbeats, final) into a queue
  that `drive` yields unchanged, and returns only a SHORT summary to the agent: counts, which sources failed, whether the cross-encoder ran, which tools it may call next. The papers, trials and drug state stay in
  code. api.py and the browser see the same events as before.
- One extra tool, `search_more`, lets the agent search again (PubMed / Europe PMC, trials, FDA labels, web) when the evidence is thin. The new papers are ranked like all others: the ranking is marked stale and
  `rank_evidence` (hybrid search, then the local cross-encoder; the very same code as the fixed run) must run again before anything reads the papers.

Guardrails that live in code, not in the prompt
- A tool refuses to run before the stages it reads (the prerequisite table of stages.py). finalize needs verify, verify needs limitations, and so on.
- Ranking is mandatory: read_evidence, analyze_drugs and write_report refuse while the paper pool is unranked or stale. search_more is refused once the papers have been read.
- A cross-encoder or embedding that was skipped is written to the report's limitations (run.skipped) and told to the agent.
- Every required stage runs. If the agent stops early, fails, runs out of steps or time, `drive` finishes the missing stages in the default order (the fixed run), so a report is always produced and verified.
- The verification code (verification.py, qa_stream.verify_and_repair) is untouched; the agent cannot skip it.

This is how EVERY run works (new question or follow-up): there is no off switch. Limits: AGENT_MAX_STEPS (20 tool calls), AGENT_BUDGET_S (300 s for the agent phase), AGENT_MAX_SEARCHES (3 extra searches)."""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from typing import AsyncIterator, Callable

import recovery
import stages as stages_mod

STAGES = stages_mod.STAGES
FIRST = stages_mod.FIRST

# tool name -> the stages it runs, in order (write_report always includes `figures`: they run beside the writing and never hold the answer)
GROUPS: dict[str, tuple[str, ...]] = {
    "collect_deep_evidence": ("deep_search",),
    "rank_evidence": ("rank",),
    "read_evidence": ("read_evidence",),
    "analyze_drugs": ("drugs",),
    "write_report": ("escalate", "write_start", "figures", "write_stream", "assemble"),
    "list_limitations": ("limitations",),
    "verify_report": ("verify",),
    "finalize_report": ("finalize",),
}
TOOL_OF = {stage: tool for tool, names in GROUPS.items() for stage in names}
EXTRA_NEEDS = {"escalate": ("drugs",)}      # the drug stage is started BEFORE the reading is finished, so it runs beside the reading and the writing instead of after them


def needs(name: str) -> tuple[str, ...]:
    return tuple(STAGES[name][1]) + EXTRA_NEEDS.get(name, ())


NEEDS_FRESH_RANK = ("read_evidence", "drugs", "write_start")      # stages that read the ranked papers
READ_STAGES = ("read_evidence", "drugs", "write_start")      # once any of these ran, the papers are being read: no new search
SEARCH_SOURCES = ("pubmed", "trials", "fda", "web")


def _num(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return float(default)


def max_steps() -> int:
    return int(_num("AGENT_MAX_STEPS", 20))


def budget_s() -> float:
    return _num("AGENT_BUDGET_S", 300.0)


def max_searches() -> int:
    return int(_num("AGENT_MAX_SEARCHES", 3))


INSTRUCTIONS = """You drive the research run for one clinical question. The quick first answer has already been shown to the person. Finish the report by calling tools; the tools do all the work and write all the clinical content. You write nothing yourself.

TOOLS (each one runs a stage and answers with counts and the tools you may call next):
- collect_deep_evidence: wait for the planned deep searches and merge the papers, trials, labels and web results.
- search_more(source, query): one extra search when the evidence is thin. source is pubmed, trials, fda or web. Use the question's own words. At most {searches} in all. Only before read_evidence.
- rank_evidence: rank the papers (keyword + meaning + a local cross-encoder). Required before the papers are read, and required AGAIN after any search_more that added papers.
- read_evidence: START the relevance reading of papers and trials and the full-text fetch (it returns at once; the reading finishes inside write_report).
- analyze_drugs: START the drug stage (labels, doses, landscape, safety, approvals, trials table); it runs beside the reading and the writing. Call it right after read_evidence.
- write_report: finishes the reading (and its follow-up searches), writes the sections and puts them in the report's order. Needs analyze_drugs first.
- list_limitations: lists what could not be retrieved, screened or verified.
- verify_report: checks every citation, number and date against the retrieved sources and repairs failed claims.
- finalize_report: sets the report final. Needs verify_report first.

USUAL ORDER: collect_deep_evidence, rank_evidence, read_evidence, analyze_drugs, write_report, list_limitations, verify_report, finalize_report.
After collect_deep_evidence read the counts. If the evidence is thin for what the question asks (for example fewer than 8 papers, no trial for a question about trials, no FDA label for a question about drugs) use search_more on the thin source, then rank_evidence again. If the evidence is enough, do not search.
A tool that refuses tells you what to call first: do that. If a tool reports an error, do not repeat it more than once; go on with the next tool, the program completes anything missing. Never skip verify_report. When finalize_report succeeds, answer with the single word: done."""


class AgentContext:
    """The state the agent's tools share for one run: the pipeline, the event queue, which stages are complete, whether the ranking is stale."""

    def __init__(self, run, queue: asyncio.Queue | None = None, now: Callable[[], float] = time.monotonic):
        self.run = run
        self.queue: asyncio.Queue = queue if queue is not None else asyncio.Queue()
        early = set(getattr(run, "prefetched", ()))      # deep_search / rank may have finished beside the fast answer; a stage that was started and did not finish is not complete
        self.completed: set[str] = {n for n in getattr(run, "ran", []) if n not in ("deep_search", "rank")} | early
        self.lock = asyncio.Lock()      # the agent may emit several tool calls at once: the stages run one at a time
        self.current: asyncio.Future | None = None      # the stage in flight (it finishes even when the agent is cancelled)
        self.rank_stale = False
        self.searches = 0
        self.now, self.started = now, now()
        self.finished_event = asyncio.Event()
        self.info: dict = {"mode": "full", "steps": [], "searches": [], "guard_ran": [], "error": None}

    # ------------------------------------------------------------ small helpers
    def elapsed(self) -> float:
        return self.now() - self.started

    def over_budget(self) -> bool:
        return self.elapsed() > budget_s()

    def note(self, tool: str, result: str, began: float) -> None:
        self.info["steps"].append({"tool": tool, "result": " ".join(result.split())[:160], "seconds": round(self.now() - began, 1), "papers": len(self.run.papers)})

    def snapshot(self) -> str:
        run, state = self.run, self.run.state
        counts = f"papers={len(run.papers)}, trials={len(run.trials)}, fda_labels={len(state.get('regulatory') or [])}, web={len(run.web)}"
        failed = sorted((state.get("retrieval_errors") or {}))[:6]
        done = [n for n in stages_mod.DEFAULT_ORDER if n in self.completed and n not in FIRST]
        parts = [f"STATE: {counts}", "stages done: " + (", ".join(done) or "none"), "ranking: " + ("STALE, call rank_evidence" if self.rank_stale else "fresh" if "rank" in self.completed else "not done")]
        if failed:
            parts.append("failed sources: " + ", ".join(failed))
        parts.append("you may call: " + (", ".join(self.available()) or "nothing, the run is complete"))
        return "\n".join(parts)

    def available(self) -> list[str]:
        out = []
        for tool, names in GROUPS.items():
            pending = [n for n in names if n not in self.completed and not (tool == "write_report" and n == "figures")]
            if tool == "rank_evidence":
                if "rank" in self.completed and not self.rank_stale:
                    continue
                if "deep_search" in self.completed:
                    out.append(tool)
                continue
            if not pending or not all(r in self.completed for r in needs(pending[0])):
                continue
            if self.rank_stale and pending[0] in NEEDS_FRESH_RANK:
                continue
            out.append(tool)
        if self.search_refusal() is None:
            out.append("search_more")
        return out

    def search_refusal(self) -> str | None:
        if self.run.halted:
            return "the run has ended"
        if "deep_search" not in self.completed:
            return "call collect_deep_evidence first"
        if any(s in self.completed for s in READ_STAGES):
            return "the papers are already being read; a new search would not be ranked or read"
        if self.searches >= max_searches():
            return f"the limit of {max_searches()} extra searches is reached"
        if self.over_budget():
            return "the time budget for extra searches is used up"
        return None

    # ------------------------------------------------------------ running a stage
    async def _stage(self, name: str) -> None:
        async def consume() -> None:
            async for event in self.run.run_stage(name):
                await self.queue.put(event)
            self.completed.add(name)
        self.current = asyncio.ensure_future(consume())
        await asyncio.shield(self.current)      # a cancelled agent does not stop the stage; `drive` waits for it

    async def run_group(self, tool: str, names: list[str]) -> str:
        run = self.run
        if run.halted:
            return "STOPPED: the run has ended."
        if all(n in self.completed for n in names):
            return f"{tool} was already done."
        for name in names:
            if name in self.completed:
                continue
            missing = [r for r in needs(name) if r not in self.completed]
            if missing:
                return "NOT RUN: " + name + " needs " + ", ".join(f"{m} (call {TOOL_OF.get(m, m)})" for m in missing) + " first."
            if self.rank_stale and name in NEEDS_FRESH_RANK:
                return "NOT RUN: new papers were added after the ranking. Call rank_evidence first."
            try:
                await self._stage(name)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                self.info["error"] = f"{name}: {type(error).__name__}: {str(error)[:120]}"
                return f"ERROR in {name}: {type(error).__name__}: {str(error)[:120]}. It is not repeated here; go on with the next tool."
            if run.halted:
                return "STOPPED: the run has ended."
        return "ok: " + ", ".join(n for n in names if n in self.completed)

    async def call(self, tool: str, names: list[str]) -> str:
        async with self.lock:
            began = self.now()
            result = await self.run_group(tool, names)
            if tool == "finalize_report" and "finalize" in self.completed:
                self.finished_event.set()
            self.note(tool, result, began)
            return result + "\n" + self.snapshot()

    # ------------------------------------------------------------ ranking
    def ranking_note(self) -> str:
        """Tell the agent how the ranking went, and write a skipped cross-encoder or embedding into the report's limitations."""
        trace = self.run.state.get("hybrid_trace") or {}
        texts = {k: v for k, v in trace.items() if isinstance(v, str)}
        status, error = str(texts.get("cross_encoder") or ""), trace.get("error")
        if error or status.startswith("skipped"):
            note = f"ranking: {error or status}"
            if note not in self.run.skipped:
                self.run.skipped.append(note)
        brief = "; ".join(f"{k}: {v}" for k, v in list(texts.items())[:4])
        return f"ranked {len(self.run.papers)} papers" + (f" ({brief[:200]})" if brief else "") + (f" ERROR: {error}" if error else "")

    async def rank(self) -> str:
        async with self.lock:
            began = self.now()
            if "deep_search" not in self.completed:
                result = "NOT RUN: rank needs deep_search (call collect_deep_evidence) first."
            elif "rank" in self.completed and not self.rank_stale:
                result = "already ranked; no new papers since."
            else:
                if "rank" in self.completed:
                    self.completed.discard("rank")      # a second ranking of the whole pool, new papers included
                result = await self.run_group("rank_evidence", ["rank"])
                if "rank" in self.completed:
                    self.rank_stale = False
                    result = self.ranking_note()
            self.note("rank_evidence", result, began)
            return result + "\n" + self.snapshot()

    # ------------------------------------------------------------ the extra search
    async def _do_search(self, source: str, query: str) -> None:
        """One extra search through the same functions and merge the recovery step uses."""
        import async_deep
        import async_stages_deep as deep
        import qa_stream as qs
        fn = deep.arecovery_searches()[source]
        raw, _ = await async_deep.arun_jobs({f"{source}_agent_{self.searches}": (lambda: fn(query))}, 12.0)
        state = self.run.state
        state.setdefault("_search_raw", {}).update(raw)
        state["retrieval_errors"].update(qs.ingest(self.run.papers, self.run.trials, self.run.web, raw))
        qs.ingest_regulatory(state, raw)
        qs.record_source_status(state, raw)

    async def search_more(self, source: str, query: str) -> str:
        async with self.lock:
            began = self.now()
            src, text = str(source or "").strip().lower(), " ".join(str(query or "").split())[:200]
            plan = self.run.plan_box.get("plan") or {}
            known = recovery.words(self.run.question) | recovery.words(plan.get("condition") or "") | recovery.words(" ".join(map(str, plan.get("drugs") or [])))
            refusal = self.search_refusal()
            if refusal:
                result = "NOT RUN: " + refusal + "."
            elif src not in SEARCH_SOURCES:
                result = "NOT RUN: source must be one of " + ", ".join(SEARCH_SOURCES) + "."
            elif len(text) < 3 or not (recovery.words(text) & known):
                result = "NOT RUN: the query must use words of the question or its condition."
            else:
                before = (len(self.run.papers), len(self.run.trials), len(self.run.state.get("regulatory") or []))
                try:
                    await self._do_search(src, text)
                    result = "ok"
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    result = f"ERROR: the search failed ({type(error).__name__}: {str(error)[:100]})."
                self.searches += 1
                after = (len(self.run.papers), len(self.run.trials), len(self.run.state.get("regulatory") or []))
                added = {"papers": after[0] - before[0], "trials": after[1] - before[1], "fda_labels": after[2] - before[2]}
                if added["papers"] > 0 and "rank" in self.completed:
                    self.rank_stale = True
                self.info["searches"].append({"source": src, "query": text, "added": added, "result": result[:60]})
                result += f": added {added['papers']} papers, {added['trials']} trials, {added['fda_labels']} FDA labels." + (" Call rank_evidence again." if self.rank_stale else "")
            self.note("search_more", result, began)
            return result + "\n" + self.snapshot()


# ---------------------------------------------------------------- the tools the agent sees
def build_tools(ctx: AgentContext) -> dict[str, Callable]:
    """name -> async function. The docstrings are what the agent reads."""

    async def collect_deep_evidence() -> str:
        """Wait for the planned deep searches (literature, trials, drug labels, web) to finish and merge their results into the evidence pools. Call this first."""
        return await ctx.call("collect_deep_evidence", list(GROUPS["collect_deep_evidence"]))

    async def search_more(source: str, query: str) -> str:
        """Run one extra search when the evidence is thin. source: 'pubmed', 'trials', 'fda' or 'web'. query: a short search that uses the question's own words (not new medical claims). Allowed only after collect_deep_evidence and before read_evidence; the new papers must then be ranked again with rank_evidence."""
        return await ctx.search_more(source, query)

    async def rank_evidence() -> str:
        """Rank all retrieved papers: keyword (BM25) + meaning (embeddings) + source order, then a local cross-encoder reads the question with each of the best papers. Required before read_evidence, analyze_drugs and write_report, and required again after search_more added papers."""
        return await ctx.rank()

    async def read_evidence() -> str:
        """Start reading the ranked papers and trials for relevance and fetch free full text. It returns at once; the reading is finished inside write_report. Call analyze_drugs next."""
        return await ctx.call("read_evidence", list(GROUPS["read_evidence"]))

    async def analyze_drugs() -> str:
        """Start the drug stage: FDA labels, doses, the treatment drug landscape, safety warnings, recent approvals and the trials table. It runs beside the reading and the writing. Needs read_evidence first; call it before write_report."""
        return await ctx.call("analyze_drugs", list(GROUPS["analyze_drugs"]))

    async def write_report() -> str:
        """Write the report sections (clinical-answer-writer and evidence-synthesis skills) and put them in the report's order. Needs read_evidence and analyze_drugs first. The figures of the relevant open-access papers are always looked for beside the writing (they never hold the answer)."""
        return await ctx.call("write_report", list(GROUPS["write_report"]))

    async def list_limitations() -> str:
        """List what could not be retrieved, screened or verified. Needs write_report first."""
        return await ctx.call("list_limitations", list(GROUPS["list_limitations"]))

    async def verify_report() -> str:
        """Check every citation, number, date and link of the report against the retrieved sources and repair failed claims. Needs list_limitations first. Never skip this."""
        return await ctx.call("verify_report", list(GROUPS["verify_report"]))

    async def finalize_report() -> str:
        """Set the report to FINAL, number the references and send it. Needs verify_report first."""
        return await ctx.call("finalize_report", list(GROUPS["finalize_report"]))

    return {f.__name__: f for f in (collect_deep_evidence, search_more, rank_evidence, read_evidence, analyze_drugs, write_report, list_limitations, verify_report, finalize_report)}


def brief(ctx: AgentContext) -> str:
    plan = ctx.run.plan_box.get("plan") or {}
    keys = ("intent", "condition", "drugs", "info_needed", "medicines_needed", "asked")
    plan_line = json.dumps({k: plan.get(k) for k in keys if plan.get(k) not in (None, "", [], {})}, ensure_ascii=False)[:600]
    focus = f"FOLLOW-UP: the person asks only for {', '.join(ctx.run.focus_sections)}; the stages research and write only that part.\n" if getattr(ctx.run, "focus_sections", None) else ""
    return f"QUESTION: {ctx.run.question}\nPLAN: {plan_line}\n{focus}The fast answer is done.\n" + ctx.snapshot()


def default_agent(ctx: AgentContext, tools: dict[str, Callable]):
    """The Agno agent with the stage tools. The model is the planning model of the orchestrator's own calls."""
    from agno.agent import Agent
    import orchestrator as orch
    from qa_stream import model_for
    return Agent(model=model_for(orch.PLAN_TOKENS, 0.0, "planning"), instructions=[INSTRUCTIONS.format(searches=max_searches())], tools=list(tools.values()), tool_call_limit=max_steps(), markdown=False)


async def _agent_phase(ctx: AgentContext, make_agent=None) -> None:
    """Run the agent until it finishes, runs out of time, or finalize succeeded. Never raises: any failure is recorded and `drive` completes the run."""
    try:
        tools = build_tools(ctx)
        agent = (make_agent or default_agent)(ctx, tools)
        prompt = brief(ctx)
        try:
            import observability
            span = observability.span("model", "agent_run", in_chars=len(prompt))
        except Exception:
            span = contextlib.nullcontext()
        with span:
            agent_task = asyncio.ensure_future(agent.arun(prompt))
            finish = asyncio.ensure_future(ctx.finished_event.wait())
            try:
                await asyncio.wait({agent_task, finish}, timeout=budget_s() + 60.0, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for task in (agent_task, finish):
                    if not task.done():
                        task.cancel()
            if agent_task.done() and not agent_task.cancelled() and agent_task.exception():
                raise agent_task.exception()
    except asyncio.CancelledError:
        raise
    except Exception as error:
        ctx.info["error"] = ctx.info.get("error") or f"agent: {type(error).__name__}: {str(error)[:120]}"


# ---------------------------------------------------------------- the run
def _quiet(task: asyncio.Future | None) -> None:
    if task is not None and task.done() and not task.cancelled():
        task.exception()      # marks it as retrieved


async def _pump(ctx: AgentContext, task: asyncio.Future) -> AsyncIterator[dict]:
    """The queued events, in order, until `task` is done; then whatever is still queued."""
    while True:
        getter = asyncio.ensure_future(ctx.queue.get())
        done, _ = await asyncio.wait({getter, task}, return_when=asyncio.FIRST_COMPLETED)
        if getter in done:
            yield getter.result()
            continue
        getter.cancel()
        if getter.done() and not getter.cancelled():      # an event arrived in the instant before the cancel
            yield getter.result()
        while not ctx.queue.empty():
            yield ctx.queue.get_nowait()
        return


async def _complete_missing(ctx: AgentContext) -> AsyncIterator[dict]:
    """The guard: every required stage the agent left undone runs now, in the default order (the fixed run), with a fresh ranking before anything reads the papers."""
    run = ctx.run
    for name in stages_mod.DEFAULT_ORDER:
        if name in FIRST or name in ctx.completed:
            continue
        if run.halted:
            return
        if name in NEEDS_FRESH_RANK and ctx.rank_stale:
            ctx.completed.discard("rank")
            ctx.info["guard_ran"].append("rank")
            async for event in run.run_stage("rank"):
                yield event
            ctx.completed.add("rank")
            ctx.rank_stale = False
            ctx.ranking_note()
        ctx.info["guard_ran"].append(name)
        async for event in run.run_stage(name):
            yield event
        ctx.completed.add(name)
        if name == "rank":
            ctx.ranking_note()


async def drive(run, make_agent=None) -> AsyncIterator[dict]:
    """Yields the events of the stages after the fast path, exactly as the fixed loop does, in the order the agent chose; then finishes whatever the agent left undone."""
    ctx = AgentContext(run)
    run.state["stage_order"] = {"chosen_by": "agent_loop", "requested": [], "order": run.ran, "ran": run.ran}      # the live list: it holds the order the stages actually ran in
    run.state["agent"] = ctx.info      # the live record of the agent's steps (saved with the run state)
    runner = asyncio.ensure_future(_agent_phase(ctx, make_agent))
    try:
        async for event in _pump(ctx, runner):
            yield event
        if ctx.current is not None and not ctx.current.done():      # the agent was cut off in the middle of a stage: let it end, with its events
            async for event in _pump(ctx, ctx.current):
                yield event
        _quiet(ctx.current)
        async for event in _complete_missing(ctx):
            yield event
    finally:
        if not runner.done():
            runner.cancel()
