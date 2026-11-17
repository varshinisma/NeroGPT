"""The async pipeline for a NEW question (a follow-up keeps the threaded stream_answer). This file holds the async SPINE: one run object, its events, and the stages as `async def`, with every wait an `await`
and every parallel step an asyncio task. Work that is not ported yet is handed to `asyncio.to_thread` (marked BRIDGE in the code); each bridge is replaced by a real async function one at a time.

Slice 1 (this file now): the plan task, the fast search and the fast answer (stages fast_search and fast_answer of stages.py)."""
from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime
from typing import AsyncIterator

import async_core
import async_orchestrator
import async_tools as at
import async_web
import async_misc
import observability
import qa_stream as qs


# ------------------------------------------------------------------ async twins of the small search helpers of qa_stream.py
async def acached(kind: str, key: str, ttl: float, afn):
    """Async twin of qa_stream.cached; the same cache store. Errors are never cached."""
    full = f"{kind}:{key}"
    with qs._cache_lock:
        hit = qs._cache.get(full)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
    value = await afn()
    if not str(value).startswith(("Web search failed", "No web results")) and '"error"' not in str(value)[:20]:
        with qs._cache_lock:
            qs._cache[full] = (time.time(), value)
    return value


def awith_retry(afn, retries: int = 2, pause: float = 0.4):
    """Async twin of qa_stream.with_retry: a result that is an error is retried, then returned as it is."""
    async def run():
        last = None
        for _ in range(retries + 1):
            last = await afn()
            if '"error"' not in str(last)[:40] and not str(last).startswith("Web search failed"):
                return last
            await asyncio.sleep(pause)
        return last
    return run


async def aprimary_with_fallback(primary, fallback, grace: float = 3.0) -> str:
    """Async twin of qa_stream.primary_with_fallback: both searches start at once; the primary's papers are used when it is back within `grace` seconds, the fallback's papers are added behind them."""
    f_primary, f_fallback = asyncio.ensure_future(primary()), asyncio.ensure_future(fallback())

    async def papers(task, wait):
        try:
            data = json.loads(await asyncio.wait_for(asyncio.shield(task), timeout=wait))
            return data if isinstance(data, dict) and data.get("results") else None
        except Exception:
            return None
    try:
        first = await papers(f_primary, grace)
        second = await papers(f_fallback, 0.5 if first else max(grace, 6.0))
        if first is None and second is None:
            return await asyncio.wait_for(asyncio.shield(f_primary), timeout=max(1.0, grace))
        merged = list((first or second)["results"])
        have = {r.get("pmid") for r in merged}
        if first and second:
            merged += [r for r in second["results"] if r.get("pmid") not in have]
        return json.dumps({**(first or second), "results": merged}, ensure_ascii=False)
    finally:
        for task in (f_primary, f_fallback):
            if not task.done():
                task.cancel()


def afast_jobs(question: str, grace: float = 3.0) -> dict:
    """Async twin of qa_stream.fast_jobs: {name: async function}, one retry with backoff for every source of the fast pass."""
    query, years = qs.pubmed_query_from_question(question)
    kw, _, _ = qs.keywords(question)
    cond = " ".join(kw) or question

    async def pubmed_fast():
        return await acached("pm", f"{query}|{years}", 900, lambda: aprimary_with_fallback(
            lambda: at.asearch_europepmc_quick(query, max_results=6, years_back=years, sort="relevance"), lambda: at.asearch_pubmed_quick(query, 4, years, "relevance"), grace=grace))

    async def pubmed_fast_q():
        return await acached("pm", f"{query}|{years}|q", 900, lambda: at.asearch_europepmc_quick(
            f"({query}) AND (PUB_TYPE:\"Systematic Review\" OR PUB_TYPE:\"Meta-Analysis\" OR PUB_TYPE:\"Randomized Controlled Trial\" OR PUB_TYPE:\"Practice Guideline\")",
            max_results=4, years_back=years + 1, sort="relevance"))

    async def trials_fast():
        return await acached("ct", cond, 300, lambda: at.asearch_clinical_trials(cond, "", "", 4))

    async def trials_drugs_fast():
        return await acached("ct", f"{cond}|drug-fast", 300, lambda: at.asearch_clinical_trials(cond, "", "", 4, drug_only=True))

    async def fda_fast():
        return await acached("fda", cond, 3600, lambda: at.asearch_fda_labels(kw, None, 4))

    async def web_fast():
        return await acached("web", cond, 300, lambda: async_web.aweb_search(f"{cond} FDA approved treatment drugs"))
    jobs = {"pubmed_fast": pubmed_fast, "pubmed_fast_q": pubmed_fast_q, "trials_fast": trials_fast, "trials_drugs_fast": trials_drugs_fast, "fda_fast": fda_fast, "web_fast": web_fast}
    return {name: awith_retry(fn, retries=1, pause=0.4) for name, fn in jobs.items()}


async def arun_jobs_late(jobs: dict, timeout: float, essential: list[str], extra_until: float = 0.6) -> tuple[dict, dict, dict]:
    """Async twin of qa_stream.run_jobs_late: a job that is not back in time is NOT cancelled; its task is returned in `late` so the caller can collect the result later."""
    started = time.time()

    async def timed(name, afn):
        t0 = time.time()
        try:
            with observability.tool(name):
                return await afn(), time.time() - t0
        except Exception as error:
            return json.dumps({"error": str(error)}), time.time() - t0
    tasks = {name: asyncio.ensure_future(timed(name, fn)) for name, fn in jobs.items()}
    must = [tasks[n] for n in essential if n in tasks]
    if must:
        await asyncio.wait(must, timeout=timeout)
    await asyncio.wait(list(tasks.values()), timeout=max(0.0, min(extra_until, timeout) - (time.time() - started)))
    raw, secs, late = {}, {}, {}
    for name, task in tasks.items():
        if task.done():
            raw[name], secs[name] = task.result()[0], round(task.result()[1], 1)
        else:
            raw[name], secs[name] = json.dumps({"error": f"timed out after {timeout:.0f}s (partial)"}), timeout
            late[name] = task
    return raw, secs, late


# ------------------------------------------------------------------ the run
class AsyncRun:
    """Everything one run shares: the scrubbed question, the clock, the research state and the plan task. The stages are methods; each yields the same event dicts as the threaded stream_answer."""

    def __init__(self, question: str, plan: dict | None = None):
        self.scrubbed = qs.phi.scrub(question)
        self.question = self.scrubbed.text or question
        import async_web; async_web.QUESTION.set(self.question)      # Tavily gets the question as a sentence too
        prior = None
        if isinstance(plan, dict) and "prior_sources" in plan:
            plan = dict(plan)
            prior = plan.pop("prior_sources", None)      # what the earlier questions of this conversation retrieved (source_store): a follow-up starts from it
        self.given_plan = plan if isinstance(plan, dict) else None
        self.clock, self.state = qs.Clock(), qs.new_state(self.question)
        self.state["pipeline"] = "async"      # the saved state file says which pipeline produced it
        observability.begin_run(self.question)
        self.state["privacy"] = {"removed": self.scrubbed.removed, "detector": self.scrubbed.detector}
        if isinstance(prior, dict) and prior:
            import source_store
            self.state["reused_sources"] = source_store.seed(self.state, prior)      # the records are added in place, so the containers below hold them
            if isinstance(plan, dict) and isinstance(plan.get("focus"), list):
                words_of = {**qs.COVER_WORDS, "drug_facts": tuple(w for a in plan.get("attributes") or [] for w in re.findall(r"[a-z]{4,}", str(a).lower()))[:8]}
                plan["stored_cover"] = {k: [d for d in plan.get("drugs") or [] if isinstance(d, str) and words_of[k] and source_store.covered(self.state, d, words_of[k])]
                                        for k in ("drugs", "safety", "drug_facts") if k in plan["focus"]}
                self.state["reused_sources"]["covered"] = plan["stored_cover"]
        self.setup_focus(plan)
        self.papers, self.trials, self.web = self.state["papers"], self.state["trials_by_id"], self.state["web"]
        self.skipped: list[str] = []
        self.halted = False
        self.header = self.opening = self.summary_head = ""
        self.stems: set = set()
        self.late_fda: dict[str, asyncio.Task] = {}
        self._plan_task: asyncio.Task | None = None
        self.plan_box: dict = {}
        self.plan_ready = asyncio.Event()

    def setup_focus(self, plan) -> None:
        """A follow-up that asks for one part (the drug tables, the trials, the evidence) researches and writes only that: same rules as qa_stream.stream_answer."""
        is_dict = isinstance(plan, dict)
        sections = [x for x in ([plan.get("focus")] if is_dict and isinstance(plan.get("focus"), str) else (plan.get("focus") if is_dict else None) or [])
                    if x in ("drugs", "safety", "drug_facts", "drug_discovery", "trials", "evidence")]
        for needs_drugs in ("drugs", "safety", "drug_facts"):
            if needs_drugs in sections and not plan.get("drugs"):
                sections.remove(needs_drugs)
        self.discovery = "drug_discovery" in sections      # the medicines of the condition are discovered afresh
        if self.discovery:
            sections = list(dict.fromkeys("drugs" if x == "drug_discovery" else x for x in sections))
        self.focus_sections = sections
        self.light_focus = bool(sections) and not ({"evidence", "trials"} & set(sections))      # about the earlier drugs only: no relevance reading, cross-encoder or full-text reading
        self.focused = bool(sections)
        self.focus_lead = ""
        self.state["focused"] = self.focused

    def event(self, kind: str, **data) -> dict:
        return {"type": kind, "t": self.clock.now(), **data}

    # ---- the plan: started once, as a task (it needs only the question)
    def start_plan(self) -> None:
        if self._plan_task is None:
            self._plan_task = asyncio.ensure_future(self._plan_worker())

    async def _plan_worker(self) -> None:
        try:
            if self.given_plan:
                self.plan_box["plan"] = qs.finish_plan(self.given_plan, "ok", self.question)
            else:
                import orchestrator
                raw = await async_orchestrator.aplan_only(self.question) if (orchestrator.AGENT_ENABLED and orchestrator.PLAN_BY_AGENT) else None
                self.plan_box["plan"] = qs.finish_plan(raw, "ok", self.question) if raw else await async_misc.aplan_with_skill(self.question)
        except Exception as error:
            self.plan_box["error"] = str(error)[:200]
        finally:
            self.plan_ready.set()

    async def wait_plan(self, timeout: float) -> None:
        try:
            await asyncio.wait_for(self.plan_ready.wait(), timeout=max(0.0, timeout))
        except asyncio.TimeoutError:
            pass

    # ---- stage: fast_search
    async def stage_fast_search(self) -> AsyncIterator[dict]:
        if hasattr(self, "start_deep"):
            self.start_deep()      # the deep retrieval groups start now, beside the fast pass
        if not self.focused:
            self.start_plan()
        if self.focused:      # a follow-up searches exactly as before
            import async_deep
            raw_fast, _ = await async_deep.arun_jobs(afast_jobs(self.question), qs.FAST_TIMEOUT_S, essential=["pubmed_fast", "trials_fast", "fda_fast"], extra_until=2.5)
            late = {}
        else:
            jobs = afast_jobs(self.question, qs.FAST_GRACE_S)
            raw_fast, _, late = await arun_jobs_late(jobs, qs.FAST_FIRST_S, essential=["pubmed_fast", "trials_fast"], extra_until=qs.FAST_EXTRA_S)
        if "pubmed_fast" in late:      # the papers are the first answer: give them the old time before writing without them
            await asyncio.wait([late["pubmed_fast"]], timeout=max(0.0, qs.FAST_TIMEOUT_S - qs.FAST_FIRST_S))
            if late["pubmed_fast"].done():
                raw_fast["pubmed_fast"] = late.pop("pubmed_fast").result()[0]
        for name in [n for n in late if n.startswith("fda")]:
            self.late_fda[name] = late[name]
            raw_fast.pop(name, None)      # collected in the fast answer's label decisions
        state = self.state
        state.setdefault("_search_raw", {}).update(raw_fast)
        state["retrieval_errors"].update(qs.ingest(self.papers, self.trials, self.web, raw_fast))
        qs.ingest_regulatory(state, raw_fast)
        qs.record_source_status(state, raw_fast)
        state["core_stems"] = sorted(qs.core_stems(self.question))
        state["patient_age_years"] = qs.patient_age_years(self.question)
        state["fast_evidence"] = list(self.papers.values())
        self.clock.mark("retrieval_fast_s")
        if not self.papers and not self.trials and not state["regulatory"] and not state["retrieval_errors"]:
            note = "I answer clinical research questions using PubMed, ClinicalTrials.gov and the web. I found no medical literature for this question - please rephrase it as a clinical question."
            yield self.event("token", stage="INITIAL", text=note)
            yield self.event("final", markdown=note, timings=dict(self.clock.marks), state=state, skipped=[])
            self.halted = True
            return
        self.stems = set(state["core_stems"])
        if hasattr(self, "start_early_evidence"):
            self.start_early_evidence()      # the deep PubMed searches and the routed searches start here

    # ---- stage: fast_answer
    async def _decide_after_late_labels(self) -> None:
        """The FDA label search that was still running when the text started: collected now, then the labels are decided."""
        if self.late_fda:
            await asyncio.wait(list(self.late_fda.values()), timeout=qs.FAST_LATE_FDA_S)
            raw_late = {n: (t.result()[0] if t.done() else json.dumps({"error": f"timed out after {qs.FAST_LATE_FDA_S:.0f}s (partial)"})) for n, t in self.late_fda.items()}
            self.state.setdefault("_search_raw", {}).update(raw_late)
            self.state["retrieval_errors"].update(qs.ingest(self.papers, self.trials, self.web, raw_late))
            qs.ingest_regulatory(self.state, raw_late)
            qs.record_source_status(self.state, raw_late)
        await async_misc.afast_drug_decisions(self.state, self.question)

    async def stage_fast_answer(self) -> AsyncIterator[dict]:
        state = self.state
        state["searched_on"] = f"{datetime.now():%Y-%m-%d}"
        self.header = f"**Evidence searched on {state['searched_on']}** (PubMed, ClinicalTrials.gov, FDA drug labels, web)\n\n"
        if self.focused:      # no summary is written: the answer is the drug table (and its figures), built from the drugs of the earlier answers
            self.summary_head, self.opening, self.focus_lead = "", "", ""
            self.clock.mark("first_token_s")
            self.start_plan()
            yield self.event("token", stage="INITIAL", text=self.header + self.focus_lead)
            self.clock.mark("first_useful_s")
            state["timestamps"]["first_response"] = self.clock.now()
            state["medicines_needed"] = bool({"drugs", "safety", "drug_facts"} & set(self.focus_sections))
            yield self.event("milestone", name="first_useful_answer", seconds=self.clock.marks["first_useful_s"], first_token=self.clock.marks.get("first_token_s"))
            await self.wait_plan(3.0)
            if self.plan_box.get("plan"):
                state["conditions"] = self.plan_box["plan"].get("conditions") or []
            return
        self.summary_head = "## Clinical summary\n\n"
        yield self.event("token", stage="INITIAL", text=self.header + self.summary_head)
        fast_prompt = (f"TASK: {qs.FAST_TASK}\n\nQUESTION: {self.question}\nRETRIEVED DATA (JSON):\n"
                       f"{json.dumps(qs.build_fast_context(state, self.question), ensure_ascii=False, separators=(',', ':'))}")
        decisions = asyncio.ensure_future(self._decide_after_late_labels())      # runs while the fast answer is written
        fast_text = self.header
        try:
            async for chunk in async_core.astream_llm(qs.load_skill_text("clinical-answer-writer"), fast_prompt, qs.FAST_TOKENS, purpose="writer"):
                if "first_token_s" not in self.clock.marks:
                    self.clock.mark("first_token_s")
                    self.start_plan()
                fast_text += chunk
                yield self.event("token", stage="INITIAL", text=chunk)
        except Exception as error:
            decisions.cancel()
            yield self.event("error", message=f"The fast answer could not be written: {error}")
            self.halted = True
            return
        model_text = qs.tidy_summary(fast_text[len(self.header):])
        self.clock.mark("first_useful_s")
        state["timestamps"]["first_response"] = self.clock.now()
        yield self.event("milestone", name="first_useful_answer", seconds=self.clock.marks["first_useful_s"], first_token=self.clock.marks.get("first_token_s"))
        try:
            await asyncio.wait_for(asyncio.shield(decisions), timeout=12.0)      # the table needs the model's decisions on the labels and trials
        except Exception:
            pass
        await self.wait_plan(8.0 - self.clock.now())
        if self.plan_box.get("plan"):
            understanding = self.plan_box["plan"].get("understanding") or {}
            state["medicines_needed"] = understanding.get("medicines_needed")
            state["medicines_about"] = understanding.get("medicines_about")
            state["conditions"] = self.plan_box["plan"].get("conditions") or []
        yield self.event("token", stage="INITIAL", text="\n\n" + qs.top_drugs_markdown(state))
        self.clock.mark("top_drugs_s")
        self.opening = model_text
