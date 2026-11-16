"""Slice 2 of the async pipeline: the deep retrieval (stage deep_search) and the ranking (stage rank). A mixin for async_pipeline.AsyncRun: the run's state lives on `self`, every thread of the threaded
version is an asyncio task, every threading.Event an asyncio.Event. The planning helpers, merge_group and the verification code are those of qa_stream.py, used unchanged."""
from __future__ import annotations

import asyncio
import os
import json
import time
from types import SimpleNamespace
from typing import AsyncIterator

import async_deep
import async_orchestrator
import async_pipeline
import async_route
import async_support as asu
import async_tools as at
import async_web
import mesh_expand
import qa_stream as qs
import recovery
import retrieval_routing as rr

GROUPS_ORDER = ("evidence", "drugs", "trials", "guidelines")
def legacy_subset(jobs: dict) -> dict:
    """The older deep search jobs that run. All of them when ENABLE_LEGACY_SEARCH is on. When it is off, the DRUG jobs (the drug literature searches, FDA labels, the web searches for medicines by name, for symptom drugs
    and for approvals, the regulator search) still run unless ENABLE_DRUG_SEARCH=off: the drug landscape needs them to find many drugs."""
    if qs.legacy_deep_enabled():
        return jobs
    if os.getenv("ENABLE_DRUG_SEARCH", "on").strip().lower() in qs.OFF_WORDS:
        return {}
    return {n: fn for n, fn in jobs.items() if n in qs.LEGACY_BY_NEED["drugs"]}


EARLY_EVIDENCE_JOBS = {"pubmed_recent", "pubmed_quality", "pubmed_negative", "pubmed_drugs", "pubmed_nondrug", "pubmed_bestmatch"}


def new_question_limits(focused: bool = False) -> SimpleNamespace:
    """The time limits of a NEW question (the short ones), as in the threaded run; a follow-up gets exactly the ordinary ones."""
    if focused:
        return SimpleNamespace(expansion=qs.EXPANSION_TOTAL_S, routing_budget=qs.ROUTING_BUDGET_S, deep_budget=qs.DEEP_BUDGET_S, deep_timeout=qs.DEEP_TIMEOUT_S, grace=qs.EVIDENCE_GRACE_S,
                               escalation=qs.ESCALATION_DEADLINE_S, landscape=qs.LANDSCAPE_WAIT_S)
    return SimpleNamespace(expansion=qs.NEW_EXPANSION_TOTAL_S, routing_budget=qs.NEW_ROUTING_BUDGET_S, deep_budget=min(qs.DEEP_BUDGET_S, qs.NEW_DEEP_BUDGET_S),
                           deep_timeout=qs.NEW_DEEP_TIMEOUT_S, grace=qs.NEW_EVIDENCE_GRACE_S, escalation=qs.NEW_ESCALATION_DEADLINE_S, landscape=qs.NEW_LANDSCAPE_WAIT_S)


def arecovery_searches() -> dict:
    """Twin of qa_stream.recovery_searches: the searches a retry may use, one per source."""
    return {
        "pubmed": lambda q: async_pipeline.aprimary_with_fallback(lambda: at.asearch_europepmc_quick(q, max_results=8, years_back=10, sort="relevance"),
                                                                    lambda: at.asearch_pubmed_quick(q, 8, 10, "relevance")),
        "trials": lambda q: at.asearch_clinical_trials(q, "", "", 8),
        "fda": lambda q: at.asearch_fda_labels(qs.keywords(q)[0], None, 8),
        "web": lambda q: async_web.aweb_search(f"{q} FDA approved treatment"),
    }


async def arecover_group(state: dict, group: str, question: str, plan: dict, skipped: list, now=time.time) -> None:
    """Twin of qa_stream.recover_group: an empty retrieval group goes to the orchestrator agent, which retries with a broader query (at most twice in all) or goes on and says what is missing."""
    try:
        found = recovery.problems(state, group)
        if not found:
            return
        began = now()
        choice = await async_orchestrator.arecover(recovery.report(state, group, question, plan, found), group, question, plan)
        available, jobs = arecovery_searches(), {}
        for i, retry in enumerate(choice["retries"]):
            jobs[f"{retry['source']}_recovery_{i}"] = (lambda fn=available[retry["source"]], q=retry["query"]: fn(q))
        if jobs:
            raw, _ = await async_deep.arun_jobs(jobs, 12.0)
            state.setdefault("_search_raw", {}).update(raw)
            qs.ingest(state["papers"], state["trials_by_id"], state["web"], raw)
            qs.ingest_regulatory(state, raw)
            qs.record_source_status(state, raw)
        after = recovery.problems(state, group)
        state.setdefault("recovery", []).append({"group": group, "problems": found, "retries": choice["retries"], "note": choice["note"], "still_empty": after, "seconds": round(now() - began, 1)})
        if after:
            skipped.append(f"{group}: " + (choice["note"] or "; ".join(after)))
    except Exception as error:
        state.setdefault("recovery", []).append({"group": group, "error": f"{type(error).__name__}: {str(error)[:80]}"})


class DeepStages:
    """Methods added to AsyncRun. `init_deep()` must be called once, inside the running loop, before the stages."""

    def init_deep(self) -> None:
        self.T = new_question_limits(self.focused)
        self.deep_done = asyncio.Event()
        self.evidence_ready = asyncio.Event()      # set when the escalation phase is over: the drug landscape reads its evidence only then
        self.papers_ranked = asyncio.Event()
        self.group_done = {g: asyncio.Event() for g in GROUPS_ORDER}
        self.recovered: set = set()
        self._deep_tasks: list[asyncio.Task] = []

    # ---- the deep searches that do not need the plan, then the routed searches
    def start_early_evidence(self) -> None:
        question, T, plan_box = self.question, self.T, self.plan_box
        provisional = {"drugs": [], "condition": " ".join(qs.keywords(question)[0]) or question, "trial_condition": "", "trial_intervention": "", "intent": "comprehensive_research"}
        early = {n: fn for n, fn in legacy_subset(async_deep.adeep_jobs(provisional, question)).items() if n in EARLY_EVIDENCE_JOBS}
        routing_box: dict = {}
        started = time.time()

        async def run_routing():
            """Once the quick plan call has chosen the skills, each loads its SKILL.md and writes its searches; they are routed to their sources and run beside the main searches."""
            try:
                if not qs.routing_enabled():
                    routing_box["trace"] = {"skipped": "deep routing is off"}
                else:
                    await self.wait_plan(T.expansion)
                    if not plan_box.get("plan"):
                        routing_box["trace"] = {"skipped": "the plan was not ready in time"}
                    else:
                        routing_box["t_plan_ready"] = round(time.time() - started, 1)
                        routing_box.update(await async_route.arun_routing_stage(question, plan_box["plan"], budget_s=T.routing_budget, box=routing_box))
                        routing_box["t_executed"] = round(time.time() - started, 1)
            except Exception as error:
                routing_box["trace"] = {"error": f"{type(error).__name__}: {str(error)[:80]}"}
        routing_task = asyncio.ensure_future(run_routing())

        async def run_similar(raw: dict, similar_box: dict) -> None:
            """PubMed's 'similar articles' of the first papers of the main searches."""
            try:
                seeds = qs.seed_pmids(raw)
                if seeds:
                    similar_box["raw"] = await async_pipeline.acached("pm", f"{','.join(seeds)}|sim", 900, lambda: at.asimilar_articles(seeds))
            except Exception as error:
                similar_box["error"] = f"{type(error).__name__}: {str(error)[:80]}"

        async def run():
            try:
                jobs_now = early
                if qs.legacy_deep_enabled():
                    await self.wait_plan(max(0.0, T.expansion - (time.time() - started)))
                    if plan_box.get("plan", {}).get("plan_status") in ("ok", "retried"):
                        kept, _ = qs.legacy_jobs_for_intent(plan_box["plan"], async_deep.adeep_jobs(plan_box["plan"], question))
                        jobs_now = {n: fn for n, fn in kept.items() if n in EARLY_EVIDENCE_JOBS or n.startswith("pubmed_plan_")}
                plan_box["raw_evidence"] = (await async_deep.arun_jobs(jobs_now, T.deep_timeout, essential=[n for n in ("pubmed_recent", "pubmed_quality") if n in jobs_now], extra_until=T.grace))[0] if jobs_now else {}
                similar_box: dict = {}
                similar_task = None
                if mesh_expand.similar_enabled() and plan_box["raw_evidence"]:      # runs beside the routed searches, which take longer
                    similar_task = asyncio.ensure_future(run_similar(dict(plan_box["raw_evidence"]), similar_box))
                await asyncio.wait([routing_task], timeout=max(0.0, T.expansion - (time.time() - started)))      # the routed searches are waited for a bounded time
                if not routing_task.done() and routing_box.get("execution") is not None:      # the deadline ended while the stage was still going: the searches already started are used, the rest is recorded as unfinished
                    execution = routing_box["execution"]
                    execution.budget_s = 0.0
                    await execution.finish()
                    submitted = list(execution.ops)
                    routing_box.update(ops=submitted, skills=[], rejected=[], raw=rr.raw_for_ingest(submitted, execution.results),
                                       trace={"partial": "the deadline ended while the searches were still being written; the searches already started were used"})
                plan_box["raw_evidence"].update(routing_box.get("raw") or {})
                if mesh_expand.similar_enabled():
                    if (similar_task is None or similar_task.done()) and not similar_box and not plan_box["raw_evidence"].get("pubmed_similar"):
                        similar_task = asyncio.ensure_future(run_similar(dict(plan_box["raw_evidence"]), similar_box))
                    if similar_task is not None:
                        await asyncio.wait([similar_task], timeout=4.0)
                    if similar_box.get("raw"):
                        plan_box["raw_evidence"]["pubmed_similar"] = similar_box["raw"]
                    elif similar_box.get("error"):
                        routing_box["similar_error"] = similar_box["error"]
                routing_box["unfinished"] = not routing_task.done()
                routing_box["waited_s"] = round(time.time() - started, 1)
                plan_box["routing"] = dict(routing_box)      # a snapshot: whatever finishes later is not silently merged into a finished state
            except Exception as error:
                plan_box["error_evidence"] = f"evidence retrieval crashed: {type(error).__name__}: {str(error)[:150]}"
            finally:
                self.group_done["evidence"].set()
        self._deep_tasks.append(asyncio.ensure_future(run()))

    # ---- the other three groups, once the plan is in
    async def deep_worker(self) -> None:
        question, T, plan_box, state = self.question, self.T, self.plan_box, self.state
        try:
            try:
                await asyncio.wait_for(self.plan_ready.wait(), timeout=60)
            except asyncio.TimeoutError:
                pass
            if "plan" not in plan_box:
                plan_box["error"] = plan_box.get("error", "planning did not finish")
                return
            jobs = legacy_subset(async_deep.adeep_jobs(plan_box["plan"], question))
            if jobs:
                jobs, legacy_trace = qs.legacy_jobs_for_intent(plan_box["plan"], jobs)
                state["legacy_selection"] = legacy_trace
            retry_pool = {**async_pipeline.afast_jobs(question), **async_deep.aretry_jobs(question, plan_box["plan"])}
            for name, st in list(state["source_status"].items()):      # a fast-pass source that still failed gets one more try here, with the longer timeout
                if st.get("retrieval_status") != "success" and name in retry_pool and name not in jobs:
                    jobs[name] = retry_pool[name]
                    plan_box.setdefault("retried", []).append(name)
            by_group: dict[str, dict] = {g: {} for g in GROUPS_ORDER}
            for name, fn in jobs.items():
                if name in EARLY_EVIDENCE_JOBS or name.startswith("pubmed_plan_"):
                    continue
                group = "evidence" if name.startswith("pubmed") else "trials" if name.startswith("trials") else "guidelines" if name.startswith("guideline") else "drugs"
                by_group["drugs" if name.startswith("pubmed") else group][name] = fn      # a retried fast PubMed job is merged with the drug group

            async def run_group(g: str) -> None:
                try:
                    if not by_group[g]:
                        plan_box["raw_" + g] = {}
                    elif g == "drugs" and "fda_deep" in by_group[g]:      # the FDA labels are essential; the web searches get a grace period
                        plan_box["raw_" + g], _ = await async_deep.arun_jobs(by_group[g], T.deep_timeout, essential=["fda_deep"], extra_until=12.0)
                    else:
                        plan_box["raw_" + g], _ = await async_deep.arun_jobs(by_group[g], T.deep_timeout)
                except Exception as error:
                    plan_box["error_" + g] = f"{g} retrieval crashed: {type(error).__name__}: {str(error)[:150]}"
                finally:
                    self.group_done[g].set()
            for g in GROUPS_ORDER:
                if g != "evidence":      # the evidence group was started earlier by start_early_evidence()
                    self._deep_tasks.append(asyncio.ensure_future(run_group(g)))
        except Exception as error:
            plan_box["error"] = f"deep retrieval crashed: {type(error).__name__}: {str(error)[:150]}"
        finally:
            if "plan" not in plan_box or plan_box.get("error"):
                for ev in self.group_done.values():
                    ev.set()
            self.deep_done.set()

    def start_deep(self) -> None:
        self._deep_tasks.append(asyncio.ensure_future(self.deep_worker()))

    # ---- absorb: wait for one group, merge it into the state
    async def absorb(self, group: str) -> None:
        T, state, plan_box = self.T, self.state, self.plan_box
        try:
            await asyncio.wait_for(self.group_done[group].wait(), timeout=max(3.0, T.deep_budget - self.clock.now()))
            finished = True
        except asyncio.TimeoutError:
            finished = False
        if group != "evidence" and not self.deep_done.is_set():
            try:
                await asyncio.wait_for(self.deep_done.wait(), timeout=max(1.0, T.deep_budget - self.clock.now()))
            except asyncio.TimeoutError:
                pass
        plan = plan_box.get("plan")
        if group != "evidence" and plan_box.get("error") and not state["source_status"].get("deep_retrieval"):
            state["retrieval_errors"]["deep_retrieval"] = plan_box["error"]
            state["source_status"]["deep_retrieval"] = {"retrieval_status": "failed", "evidence_status": "unavailable", "error": plan_box["error"]}
        if not finished:
            state["source_status"][f"deep_{group}"] = {"retrieval_status": "failed", "evidence_status": "unavailable",
                                                       "error": f"{group} retrieval did not finish within the {T.deep_budget:.0f} s budget"}
            return
        qs.merge_group(state, self.question, group, plan_box, self.skipped)
        await self.checkpoint(group)

    async def checkpoint(self, group: str) -> None:
        """A group that came back empty is put to the orchestrator agent; a group with content costs nothing."""
        if group in self.recovered or not recovery.enabled() or group not in recovery.GROUP_SOURCES:
            return
        self.recovered.add(group)
        await arecover_group(self.state, group, self.question, self.plan_box.get("plan") or {}, self.skipped, self.clock.now)

    def stage_on(self, name: str) -> bool:
        return name not in ((self.plan_box.get("plan") or {}).get("skip_stages") or [])

    # ---- stage: deep_search
    async def stage_deep_search(self) -> AsyncIterator[dict]:
        self.state["answer_status"] = "ENRICHING"
        yield self.event("stage", stage="EVIDENCE_ENRICHING")
        await self.absorb("evidence")

    # ---- stage: rank
    async def stage_rank(self) -> AsyncIterator[dict]:
        state, papers = self.state, self.papers
        try:
            began = self.clock.now()
            await asu.arerank(self.question, papers, state.setdefault("hybrid_trace", {}), queries=qs.intent_queries(state, self.plan_box.get("plan") or {})[:(11 if self.focused else 6)])
            state["hybrid_trace"]["embedding_s"] = round(self.clock.now() - began, 1)
            if not self.light_focus:
                await asu.across_rerank(self.question, papers, state["hybrid_trace"], top=(qs.TRIAGE_PAPERS if self.focused else min(qs.TRIAGE_PAPERS, qs.CROSS_TOP_NEW)))
            state["hybrid_trace"]["cross_encoder_s"] = round(self.clock.now() - began - state["hybrid_trace"]["embedding_s"], 1)
        except Exception as error:
            state["hybrid_trace"] = {"error": f"{type(error).__name__}: {str(error)[:80]}"}
        finally:
            self.papers_ranked.set()
        state["deep_evidence"] = list(papers.values())
        return
        yield
