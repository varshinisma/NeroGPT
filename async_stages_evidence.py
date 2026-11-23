"""Slice 3 of the async pipeline: reading the evidence (stage read_evidence) and the escalation (stage escalate). Async twins of triage_evidence, propose_alternates and run_escalation_phase of qa_stream.py,
and the two stages as methods of a mixin for async_pipeline.AsyncRun. Decisions are verified by the same code as before (landscape.verify_*_triage, retrieval_routing); only the waiting is different:
the model reads of the evidence parts run at the same time as tasks, a wave of alternate-source searches is an AsyncExecution task, and every `join(timeout)` is an `asyncio.wait(timeout=...)`."""
from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime
from typing import AsyncIterator

import async_core
import async_route
import async_tools as at
import chunking
import conditions as cond_groups
import fulltext
import hybrid_search
import plan_audit
import qa_stream as qs
import retrieval_routing as rr
import async_routing
from async_route import aask_queries


async def apropose_alternates(question: str, plan: dict, wanted: list, anchors, all_ops: list, ask=aask_queries) -> dict:
    """Twin of qa_stream.propose_alternates: ONE batched model call for the essential needs that have no second source yet."""
    if not wanted:
        return {}
    section = qs.skill_section("latest-evidence", "Query transformation by intent")
    sources = section[section.find("### Sources you may target"):section.find("### Output")] if "### Sources you may target" in section else section
    payload = {"question": question, "understanding": {k: v for k, v in (plan.get("understanding") or {}).items() if v},
               "needs": [{"need_id": n.need_id, "need": n.need, "tried": [{"target": o.target, "query": o.query[:120]} for o in all_ops if o.id in n.op_ids], "outcome": n.state} for n in wanted]}
    try:
        reply = await ask(question, qs.ALTERNATES_TASK + "\n\nSECTION:\n" + sources, payload, 700, skill="latest-evidence")
    except Exception:
        return {}
    names = {n.need_id: n.need for n in wanted}
    items = reply.get("queries") if isinstance(reply, dict) and isinstance(reply.get("queries"), list) else (reply.get("alternates") if isinstance(reply, dict) and isinstance(reply.get("alternates"), list) else [])
    items = [{**i, "need": i.get("need") or names.get(i.get("need_id"), "")} for i in items if isinstance(i, dict)]
    ops, _ = rr.parse_operations("alternates", {"queries": items}, question, anchors)
    tried = {n.need_id: {o.target for o in all_ops if o.id in n.op_ids} for n in wanted}
    out = {}
    for op in ops:
        if op.need_id in tried and op.target not in tried[op.need_id] and op.need_id not in out:
            out[op.need_id] = {"target": op.target, "query": op.query, "args": op.args}
    return out


async def atriage_evidence(state: dict, question: str, skipped: list[str], only: set[str] | None = None, head_done: asyncio.Event | None = None, head_n: int = 16) -> None:
    """Twin of qa_stream.triage_evidence: evidence-synthesis reads ALL retrieved papers and trials and decides relevance, role and population; code verifies every decision against the record's own text."""
    from landscape import numbered_sentences
    today = f"{datetime.now():%Y-%m-%d}"
    if only is None:
        papers = [p for p in state["papers"].values() if not p.get("retracted")][:qs.TRIAGE_PAPERS]
    else:
        papers = [state["papers"][i] for i in only if i in state["papers"] and not state["papers"][i].get("retracted") and not state["papers"][i].get("triage")][:qs.TRIAGE_PAPERS]
    offered_p = {p["pmid"]: {"sentences": [p["title"]] + numbered_sentences(p["abstract"], 700), "study_type": p.get("study_type") or p.get("type") or []} for p in papers}
    information_needs = list((state.get("understanding") or {}).get("info_needed") or [])
    clinical_profile = {"condition": str((state.get("entities") or {}).get("condition") or ""), "case_facts": list((state.get("understanding") or {}).get("case_facts") or []),
                        "information_needs": information_needs}
    if cond_groups.groups_of(state):
        clinical_profile["conditions"] = cond_groups.names(cond_groups.groups_of(state))
    if only is None:
        trials = [t for t in list(state["trials_by_id"].values())[:qs.TRIALS_SHOWN] if not t.get("triage")]
    else:
        trials = [state["trials_by_id"][i] for i in only if i in state["trials_by_id"] and not state["trials_by_id"][i].get("triage")][:qs.TRIALS_SHOWN]
    offered_t = {t["nct_id"]: {"sentences": [t["title"] or "", "Conditions: " + "; ".join(t.get("conditions") or []), "Interventions: " + "; ".join(t.get("interventions") or []),
                                             "Primary outcome: " + str(t.get("primary_endpoint") or "not stated")]} for t in trials}
    jobs = []
    keys = list(offered_p)
    for i in range(0, len(keys), 6):
        part = keys[i:i + 6]
        jobs.append(("papers", part, {"today": today, "clinical_profile": clinical_profile, "papers": [{"id": k, "year": state["papers"][k]["year"], "type": state["papers"][k]["type"],
                                                                   "sentences": [f"{n}. {t}" for n, t in enumerate(offered_p[k]["sentences"], 1)]} for k in part]}, qs.PAPER_TRIAGE_TASK))
    if offered_t:
        jobs.append(("trials", list(offered_t), {"clinical_profile": clinical_profile,
                                                   "trials": [{"id": k, "sentences": [f"{n}. {t}" for n, t in enumerate(v["sentences"], 1)]} for k, v in offered_t.items()]}, qs.TRIAL_TRIAGE_TASK))
    trace: list = []
    head_keys = set(keys[:-(-head_n // 6) * 6]) if head_done is not None else set()      # the best-ranked papers (whole parts of 6): the ones the writers read
    head_waiting = {"trials"} if (head_done is not None and offered_t) else set()
    judged: set = set()

    def check_head() -> None:
        """head_done is set as soon as the best-ranked papers (and the trials) are judged AND at least `head_n` of those are direct: the writers read the best `head_n` papers, so the rest can be read behind them.
        With fewer direct papers in the head the writers wait for the whole reading, as before."""
        if head_done is None or head_done.is_set() or head_waiting or not head_keys <= judged:
            return
        if sum(1 for k in head_keys if (state["papers"][k].get("triage") or {}).get("relevance") == "direct") >= head_n:
            head_done.set()

    def apply_decisions(kind, part, items):
        decisions = (qs.verify_paper_triage(items, offered_p, information_needs=information_needs, broad_question=len(information_needs) > 1) if kind == "papers"
                     else qs.verify_trial_triage(items, offered_t))
        for k in part:
            record = state["papers"][k] if kind == "papers" else state["trials_by_id"][k]
            record["triage"] = decisions.get(k) or ({"relevance": "uncertain", "role": None, "population": "", "evidence": ""} if kind == "papers" else {"relevance": "uncertain", "kind": "other", "evidence": ""})

    async def run(job):
        kind, part, payload, task = job
        began = time.time()
        try:
            answer = await async_core.aask_json(question, task, payload, 900, skill="evidence-synthesis")
        except Exception as error:
            trace.append({"kind": kind, "error": str(error)[:100], "seconds": round(time.time() - began, 1)})
            return kind, part, None
        trace.append({"kind": kind, "seconds": round(time.time() - began, 1), "sample": (answer.get(kind) or [])[:2], "count": len(answer.get(kind) or []), "raw_tail": answer.get("_raw", "")[-600:],
                      "sent": len(payload.get("papers") or payload.get("trials") or [])})
        items = answer.get(kind) or []
        apply_decisions(kind, part, items)      # applied as soon as this part is back: the writer is not held up by the slowest part
        if kind == "papers":
            judged.update(part)
        else:
            head_waiting.discard("trials")
        check_head()
        return kind, part, items
    results = await asyncio.gather(*[run(j) for j in jobs])
    state["triage_trace"] = trace if only is None else list(state.get("triage_trace") or []) + trace
    failed = sum(1 for _, _, items in results if items is None)
    if not failed and only is None:
        for p in state["papers"].values():
            if not p.get("triage") and not p.get("retracted"):
                p["triage"] = {"relevance": "uncertain", "role": None, "population": "", "evidence": ""}
    if only is not None:
        for p in papers:
            p.setdefault("triage", {"relevance": "uncertain", "role": None, "population": "", "evidence": ""})
        for t in trials:
            t.setdefault("triage", {"relevance": "uncertain", "kind": "other", "evidence": ""})
    if failed:
        skipped.append(f"the relevance reading of {failed} part(s) of the retrieved evidence did not complete; those records were offered to the writer as retrieved")


async def _wait(task: asyncio.Future, timeout: float) -> bool:
    """True when the task finished within the timeout (the async `join(timeout)` + `not is_alive()`)."""
    if task.done():
        return True
    await asyncio.wait([task], timeout=max(0.0, timeout))
    return task.done()


async def arun_escalation_phase(state: dict, question: str, plan: dict, stage: dict, triage_task: asyncio.Future, skipped: list[str], *, make_adapters=async_route.areal_adapters,
                                scorer=None, ask=aask_queries, deadline_s: float = qs.ESCALATION_DEADLINE_S, clock=time.monotonic) -> dict:
    """Twin of qa_stream.run_escalation_phase: before the evidence is written, essential needs that found nothing are tried once in an alternate source WHILE the relevance reading runs; everything is
    bounded by one hard deadline, and unfinished work is recorded and reported in the limitations, never waited for."""
    record = {"waves": [], "skipped": [], "unfinished": [], "needs": {}}
    began = clock()
    deadline = began + deadline_s

    def remaining() -> float:
        return max(0.0, deadline - clock())
    if not stage or not stage.get("execution"):
        if not await _wait(triage_task, deadline_s):
            skipped.append("the relevance reading did not finish before the deadline; papers it had not judged were not used as evidence")
        return record
    ops, first_results = stage["ops"], dict(stage["execution"].results)
    all_ops = list(ops)
    results = dict(first_results)
    adapters = make_adapters(stage["anchors"])
    executed = {o.key for o in ops}
    escalated: set[str] = set()

    def launch(wave: int, new_ops: list) -> dict:
        """Run one wave's searches as a task, so that the relevance reading can go on beside it."""
        job = {"wave": wave, "ops": new_ops, "execution": None}

        async def work():
            execution = async_routing.AsyncExecution(new_ops, adapters, max(0.05, remaining()))      # never longer than what is left of the hard deadline
            await execution.finish()
            job["execution"] = execution
        job["task"] = asyncio.ensure_future(work())
        return job

    async def plan_wave(needs: dict, wave: int) -> list:
        if os.getenv("ESCALATION_SEARCH", "off").strip().lower() not in ("on", "1", "true", "yes"):
            return []      # no extra gap-search rounds (no model call, no alternate-source searches): the coverage of each need is still recorded below, and the reading is still waited for
        room = rr.MAX_ESCALATIONS - len(escalated)
        new_ops, missing, skips = rr.plan_escalations(needs, all_ops, executed, already=escalated, max_escalations=max(0, room))
        record["skipped"] += [{**s, "wave": wave} for s in skips]
        if missing and remaining() > qs.ESCALATION_MIN_S:
            wanted = [needs[m] for m in missing]
            alternates = await apropose_alternates(question, plan, wanted, stage["anchors"], all_ops, ask=ask)
            for need_id, alt in alternates.items():
                st = needs[need_id]
                source = next((o for o in all_ops if o.id in st.op_ids), None)
                op = rr.Operation(id=f"escalation:{need_id}", need_id=need_id, need=st.need, target=alt["target"], query=alt["query"], args=alt["args"], priority="essential", origin="escalation",
                                  skill=source.skill if source else "", escalation={"from": source.id if source else "", "from_target": source.target if source else "", "reason": "failed" if st.state == "failed" else st.reason or "no_relevant", "round": 1})
                if op.key not in executed and len(new_ops) < room:
                    new_ops.append(op)
        new_ops = [o for o in new_ops if o.key not in executed][:max(0, room)]
        for op in new_ops:
            executed.add(op.key)
            escalated.add(op.need_id)
            all_ops.append(op)
        return new_ops

    async def finish(job: dict, wait_s: float) -> set[str]:
        await _wait(job["task"], wait_s)
        execution = job["execution"]
        if execution is None:
            record["unfinished"].append(f"wave {job['wave']}: the alternate-source searches did not finish before the deadline and were not used")
            return set()
        results.update(execution.results)
        unfinished = [o for o in job["ops"] if execution.results[o.id].status in ("timed_out", "not_run")]
        if unfinished:
            record["unfinished"].append(f"wave {job['wave']}: {len(unfinished)} alternate-source search(es) did not finish within the deadline and were not used")
        raw = rr.raw_for_ingest(job["ops"], execution.results, index_in=all_ops)
        new_ids = qs.ingest_routed(state, raw)
        record["waves"].append({"wave": job["wave"], "searches": [{"need_id": o.need_id, "need": o.need, "from": o.escalation, "target": o.target, "query": o.query[:160], "status": execution.results[o.id].status,
                                                                    "n": execution.results[o.id].n, "error": execution.results[o.id].error[:100]} for o in job["ops"]], "new_records": len(new_ids)})
        return new_ids

    async def read_new(new_ids: set[str]) -> None:
        if not new_ids or remaining() <= 0.5:
            for i in new_ids:
                for table in ("papers", "trials_by_id"):
                    if i in state[table]:
                        state[table][i].setdefault("triage", {"relevance": "uncertain", "role": None, "population": "", "evidence": "", "kind": "other"})
            if new_ids:
                record["unfinished"].append(f"the relevance reading of {len(new_ids)} record(s) found by the alternate-source searches did not finish before the deadline; they are not shown as relevant")
            return
        reading = asyncio.ensure_future(atriage_evidence(state, question, skipped, only=new_ids))
        if not await _wait(reading, remaining()):
            record["unfinished"].append("the relevance reading of the records found by the alternate-source searches did not finish before the deadline")

    # ---- wave 1: needs whose searches found nothing or failed; runs beside the relevance reading
    needs = rr.evaluate_needs(all_ops, results, qs.relevance_counter(state, question, optimistic=True))
    wave1 = await plan_wave(needs, 1)
    job1 = launch(1, wave1) if wave1 else None
    triage_done = await _wait(triage_task, remaining())
    if not triage_done:
        record["unfinished"].append("the relevance reading of the retrieved records did not finish before the deadline; records without a verified decision were offered as retrieved")
    new_ids = await finish(job1, remaining()) if job1 else set()
    await read_new(new_ids)
    # ---- wave 2: needs whose records were all judged irrelevant (needs the finished reading)
    if triage_done and remaining() > qs.ESCALATION_MIN_S and os.getenv("ESCALATION_SEARCH", "off").strip().lower() in ("on", "1", "true", "yes"):      # wave 2 only plans gap searches: with those off it would only load the local scoring model and score every need for nothing
        scorer = scorer if scorer is not None else await asyncio.to_thread(hybrid_search.cross_scorer)      # CPU: the local model
        needs = await asyncio.to_thread(rr.evaluate_needs, all_ops, results, qs.relevance_counter(state, question, scorer))      # CPU: the screen scores web results with the local model
        wave2 = await plan_wave(needs, 2)
        if wave2:
            job2 = launch(2, wave2)
            await read_new(await finish(job2, remaining()))
    final = await asyncio.to_thread(rr.evaluate_needs, all_ops, results, qs.relevance_counter(state, question, scorer if scorer is not None else None, optimistic=not triage_done))
    record["needs"] = {k: {"need": v.need, "priority": v.priority, "state": v.state, "reason": v.reason, "relevant": v.relevant} for k, v in final.items()}
    record["seconds"] = round(clock() - began, 1)
    record["all_ops"] = all_ops
    record["results"] = results
    for note in record["unfinished"]:
        skipped.append(note)
    return record


class EvidenceStages:
    """Methods added to AsyncRun (after DeepStages). Needs self.T, self.papers_ranked, self.evidence_ready from init_deep."""

    def init_evidence(self) -> None:
        self.fulltext_task: asyncio.Task | None = None
        self.triage_task: asyncio.Task | None = None
        self.reading_head = asyncio.Event()      # set when the papers the writers read are judged (or when the whole reading ends)
        self.reading_head_task: asyncio.Task | None = None
        self.early_latest: dict = {}

    async def _run_fulltext(self) -> None:
        state, papers = self.state, self.papers
        try:      # beside the triage and the escalation: the free full text of the best-ranked open-access papers for the writer; others stay abstract-only
            best = [p for p in list(papers.values())[:qs.TRIAGE_PAPERS] if p.get("pmcid") and not p.get("retracted")][:qs.FULL_TEXT_PAPERS]
            if fulltext.enabled():
                state["full_text"] = await at.aattach_fulltext(best, cap_s=qs.FULL_TEXT_WAIT_S)
            if chunking.enabled():      # then the passages of the best-ranked papers are scored against the question (local cross-encoder)
                scorer = await asyncio.to_thread(hybrid_search.cross_scorer, 2.0)
                state["passage_ranking"] = (await asyncio.to_thread(chunking.select, self.question, list(papers.values())[:qs.MAX_CONTEXT_PAPERS], scorer)) if scorer else {"skipped": "the cross-encoder was not ready"}
        except Exception:
            pass

    async def _run_triage(self) -> None:
        try:
            if self.light_focus:      # a follow-up about the earlier drugs: no relevance reading
                self.clock.mark("triage_s")
                return
            head = self.reading_head if os.getenv("READING_HEAD_START", "on").strip().lower() not in ("off", "0", "false", "no") and not self.focused else None
            await atriage_evidence(self.state, self.question, self.skipped, head_done=head, head_n=qs.MAX_CONTEXT_PAPERS)      # the prose waits for the best-ranked papers (reading_head), the drug landscape does not wait at all
            self.clock.mark("triage_s")
        finally:
            self.reading_head.set()

    async def stage_read_evidence(self) -> AsyncIterator[dict]:
        self.fulltext_task = asyncio.ensure_future(asyncio.sleep(0))      # a finished task stands in for a reading that is left out
        if self.light_focus:
            self.state["full_text_skipped"] = "a focused follow-up about the earlier drugs does not read full texts"
        elif (fulltext.enabled() or chunking.enabled()) and self.stage_on("full_text"):
            self.fulltext_task = asyncio.ensure_future(self._run_fulltext())
        elif not self.stage_on("full_text"):
            self.state["full_text_skipped"] = "left out by the orchestrator plan"
        self.clock.mark("retrieval_deep_s")
        yield self.event("milestone", name="retrieval_complete", seconds=self.clock.marks["retrieval_deep_s"])
        self.triage_task = asyncio.ensure_future(self._run_triage())
        self.reading_head_task = asyncio.ensure_future(self.reading_head.wait())
        if not self.focused and not self.light_focus and hasattr(self, "start_figures"):
            self.start_figures()      # the figure pages are fetched beside the relevance reading, not after the escalation
        if qs.EARLY_LATEST and not self.focused and hasattr(self, "launch_writer"):      # the Latest findings are written from the papers the relevance reading has judged, as soon as that reading is done, beside the escalation
            self.make_writing()
            self.early_latest["pending"] = True

            async def start_latest_when_judged():
                try:
                    if await _wait(self.reading_head_task, self.T.escalation + 5):
                        self.launch_writer("latest")
                        self.state["early_latest_started_s"] = round(self.clock.now(), 1)
                except Exception as error:
                    self.skipped.append(f"the early start of the Latest findings failed ({type(error).__name__}); they are written after the escalation")
            asyncio.ensure_future(start_latest_when_judged())

    async def stage_escalate(self) -> AsyncIterator[dict]:
        state, plan_box = self.state, self.plan_box
        if self.focused and "evidence" not in self.focus_sections:      # no literature sections are written, so no gaps are searched for
            await _wait(self.triage_task, 8.0)
            self.evidence_ready.set()
            return
        try:      # the relevance reading and the escalation finish BEFORE the evidence is written, under one hard deadline; unfinished work is reported in the limitations
            escalation_task = asyncio.ensure_future(arun_escalation_phase(state, self.question, plan_box.get("plan") or {}, plan_box.get("routing") or {}, self.reading_head_task, self.skipped,
                                                                         deadline_s=self.T.escalation))
            if self.early_latest.get("pending") and getattr(self, "writing", None) is not None:      # the Latest findings are shown as they are written, while the escalation runs
                part = self.writing["latest"]
                while not escalation_task.done() and not part.get("drained"):
                    getter = asyncio.ensure_future(part["queue"].get())
                    await asyncio.wait([getter, escalation_task], return_when=asyncio.FIRST_COMPLETED)
                    if not getter.done():
                        getter.cancel()
                        break
                    chunk = getter.result()
                    if chunk is None:
                        part["drained"] = True
                        break
                    part["text"] += chunk
                    yield self.event("token", stage="EVIDENCE_ENRICHING", text=chunk)
                state["latest_shown_during_escalation_s"] = round(self.clock.now(), 1)
            escalation = await escalation_task
            state["escalations"] = {k: v for k, v in escalation.items() if k not in ("all_ops", "results")}
            qs.record_plan_audit(state, "needs", plan_audit.audit_needs(escalation.get("needs")))
            if escalation.get("all_ops"):
                state["section_states"] = qs.compute_section_states(escalation)
                stage = plan_box.get("routing") or {}
                late = stage["execution"].late_results() if stage.get("execution") else {}
                state["routing"] = {**(state.get("routing") or {}), **rr.routing_record(escalation["all_ops"], escalation["results"], rejected=stage.get("rejected"), trace=stage.get("trace"), late=late)}
        except Exception as error:
            self.skipped.append(f"the escalation phase did not complete ({type(error).__name__}); the evidence is written from what was retrieved")
            await _wait(self.triage_task, 3.5)
        if (plan_box.get("routing") or {}).get("unfinished"):
            self.skipped.append("not all source-aware searches were written and finished before the deadline; those that had finished were used and the rest were not")
        self.evidence_ready.set()
        await _wait(self.fulltext_task, 4.0)      # normally finished long ago (it ran beside the triage); a late one is simply not used
