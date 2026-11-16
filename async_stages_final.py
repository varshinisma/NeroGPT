"""Async figures and final stages (slice 5): the twins of qa_stream.run_figures, stage_figures, stage_assemble, stage_limitations, stage_verify and stage_finalize (new questions only).

BRIDGE (worker thread, marked in the code): search_paper_figures and find_figures (page parsing, image download and measuring, and the sync figure model calls), verify_and_repair
(CPU only). Everything that waits for a model is awaited.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
from typing import AsyncIterator

import async_core
import async_figures
import medication_assessment as ma
import observability
import plan_audit
import qa_stream as qs
from async_stages_evidence import _wait
from presentation import closing_note, numberize


async def agrounded_call(build, question: str, max_tokens: int):
    """Runs a synchronous builder that makes ONE model call through `ask(task, payload)` (safety_answer.markdown / facts_markdown) with the call awaited: the builder is run once to learn the
    request, the request is awaited, and the builder is run again with the answer. The table-building code stays the one in safety_answer; nothing is copied."""
    seen: dict = {}

    def capture(task, payload):
        seen["request"] = (task, payload)
        return {}
    build(capture)
    answer, failure = None, None
    if "request" in seen:
        try:
            answer = await async_core.aask_json(question, seen["request"][0], seen["request"][1], max_tokens)
        except Exception as error:
            failure = error

    def replay(task, payload):
        if failure is not None:
            raise failure
        return answer
    return build(replay)


class FigureJob:
    """What qa_stream.collect_figures expects of a figure job (join / is_alive), backed by an asyncio task. It is the one bridge back to the synchronous PDF step."""

    def __init__(self, task: asyncio.Future):
        self._done = threading.Event()
        task.add_done_callback(lambda _t: self._done.set())

    def join(self, timeout: float | None = None) -> None:
        self._done.wait(timeout)

    def is_alive(self) -> bool:
        return not self._done.is_set()


async def aground_opening(question: str, report_text: str) -> str:
    """Async twin of qa_stream.write_grounded_opening."""
    try:
        text = (await async_core.acollect(qs.load_skill_text("clinical-answer-writer"), f"TASK: {qs.SUMMARY_REWRITE_TASK}\n\nQUESTION: {question}\nREPORT SECTIONS:\n{report_text[:9000]}", 450,
                                          temperature=0.0, purpose="writer")).strip()
    except Exception:
        return ""
    return qs.tidy_summary(text) if "Bottom line" in text and "PMID" in text else ""


class FinalStages:
    """Methods added to AsyncRun (with the other stage mixins). `init_final()` must be called once, inside the running loop."""

    def init_final(self) -> None:
        self.figure_box: dict = {"figures": []}
        self.figures_task: asyncio.Task | None = None
        self.grounded_opening = ""
        self.limits_md = ""
        self.paper_counts: dict = {}
        self.full = ""
        self.result: dict = {}
        self.ran: list[str] = []

    # ---- figures: BESIDE the writing
    async def _run_figures(self) -> None:
        state, papers, question = self.state, self.papers, self.question
        trace = self.figure_box.setdefault("trace", {})
        if not self.stage_on("figures"):
            trace["skipped"] = "left out by the orchestrator plan"
            return
        try:
            await _wait(self.triage_task, 25)      # the relevance judgement of every paper
            order = {"direct": 0, "indirect": 1, "uncertain": 2, "background": 3}
            ranked = sorted((p for p in papers.values() if p.get("pmcid") and not p.get("retracted") and (p.get("triage") or {}).get("relevance") in order),
                            key=lambda p: order.get((p.get("triage") or {}).get("relevance"), 2))
            views = [{"pmid": p["pmid"], "pmcid": p["pmcid"], "doi": p.get("doi"), "title": p.get("title"), "cite": p.get("cite"),
                      "design": ", ".join(p.get("type") or p.get("study_type") or []) or None} for p in ranked[:30]]
            fetched = json.loads(await async_figures.asearch_paper_figures(json.dumps(views), want=24, batches=4))
            trace.setdefault("reading", {})["page_status"] = fetched.get("page_status", [])
            found = fetched.get("figures") or []
            try:
                await asyncio.wait_for(self.writing["current"]["done"].wait(), timeout=30)      # the relevance test compares a figure with what the report already says
            except asyncio.TimeoutError:
                pass
            report = qs.cut_for_model(" ".join("".join(self.writing[n]["produced"]) for n in ("latest", "current")), 3500)
            self.figure_box["figures"] = await async_figures.afind_figures(question, papers, report, found, trace, (state.get("entities") or {}).get("condition") or "")
        except Exception as error:
            trace["error"] = f"{type(error).__name__}: {str(error)[:80]}"

    async def stage_figures(self) -> AsyncIterator[dict]:
        self.figures_task = asyncio.ensure_future(self._run_figures())      # the figures are found while the parts are written
        return
        yield

    # ---- stage: assemble
    async def stage_assemble(self) -> AsyncIterator[dict]:
        state, skipped, clock, papers, writing, parts = self.state, self.skipped, self.clock, self.papers, self.writing, self.parts
        T = self.T
        plan = self.given_plan or {}
        if self.focused:      # only the parts that were asked for
            if "drugs" in self.focus_sections:
                await self.wait_landscape(max(T.landscape, T.deep_budget - clock.now()))
                async for e in self.emit_landscape_and_trials(force=True):
                    yield e
            elif self.merge_task is not None:
                await asyncio.wait([self.merge_task], timeout=max(T.landscape, T.deep_budget - clock.now()))
            drugs_named = [d for d in plan.get("drugs") or [] if isinstance(d, str)]
            if "safety" in self.focus_sections:      # the side-effects table: each drug's label text and the papers and pages found for its side effects
                yield self.event("stage", stage="DRUGS_ENRICHING")
                import safety_answer
                asked_interactions = "drug_interaction" in {str(x) for x in plan.get("info_needed") or []}
                parts["safety"] = await agrounded_call(lambda ask: safety_answer.markdown(self.question, drugs_named, state, ask, want_interactions=asked_interactions), self.question, 2400)
                yield self.event("token", stage="DRUGS_ENRICHING", text="\n\n" + parts["safety"] + "\n\n")
                clock.mark("drugs_done_s")
            if "drug_facts" in self.focus_sections:      # a table of exactly the facts the agent named about the earlier drugs
                yield self.event("stage", stage="DRUGS_ENRICHING")
                import safety_answer
                parts["drug_facts"] = await agrounded_call(lambda ask: safety_answer.facts_markdown(self.question, drugs_named, state, ask, plan.get("attributes") or []), self.question, 2600)
                yield self.event("token", stage="DRUGS_ENRICHING", text="\n\n" + parts["drug_facts"] + "\n\n")
                clock.mark("drugs_done_s")
            if "trials" in self.focus_sections:
                yield self.event("stage", stage="TRIALS_ENRICHING")
                parts["trials"] = qs.trials_markdown(state)
                yield self.event("token", stage="TRIALS_ENRICHING", text="\n\n" + parts["trials"])
                clock.mark("trials_done_s")
            if "evidence" not in self.focus_sections:
                state["synthesis"], state["writing_raw"], state["writing_trace"] = "", {}, {}
                self.grounded_opening = ""
                clock.mark("evidence_done_s")
                self.sections[0] = self.header + self.focus_lead
                self.sections.extend(parts[k] for k in ("landscape", "medication", "safety", "drug_facts", "trials") if parts.get(k))
                return
        else:
            await self.wait_landscape(max(T.landscape, T.deep_budget - clock.now()))      # nothing else may read or change the papers while the drug stage works
            async for e in self.emit_landscape_and_trials(force=True):
                yield e
        latest_text, latest_cut = qs.trim_part(writing["latest"]["text"])
        if latest_cut:
            skipped.append("the Latest findings reached the model's output limit; its last, incomplete entry was removed")
        gen_text = "\n\n".join(x for x in (latest_text.strip(), qs.drop_uncited_steps(writing["approach"]["text"]).strip() if self.needs_workup else "", writing["current"]["text"].strip()) if x)
        async for _ in self.drain("tail", live=False):      # normally finished long ago; its cleaned text is shown below
            pass
        gen_text = gen_text + ("\n\n" + writing["tail"]["text"].strip() if writing["tail"]["text"].strip() else "")
        state["writing_raw"] = {k: v["text"] for k, v in writing.items()}
        state["writing_trace"] = {k: v.get("trace") for k, v in writing.items()}
        state["synthesis"], cut_off = qs.trim_incomplete(gen_text or "")
        state["synthesis"] = qs.tidy_model_sections(state["synthesis"])
        if cut_off:
            skipped.append("evidence section reached the model's output limit; its last, incomplete line was removed")
        synthesis = qs.clean_model_key_studies(state["synthesis"], qs.ordered_papers(papers, self.stems, include_background=True), self.stems)
        cut = qs.HOLD_FROM.search(synthesis)
        evidence_head, evidence_tail = (synthesis[:cut.start()].rstrip(), synthesis[cut.start():]) if cut else (synthesis, "")
        parts["head"] = evidence_head
        clock.mark("evidence_done_s")
        self.grounded_opening = await aground_opening(self.question, synthesis) if synthesis.strip() and not self.focused else ""      # a focused answer has no summary to rewrite
        if not self.grounded_opening and not self.focused:
            skipped.append("the opening summary could not be rewritten from the finished evidence; the first draft was kept")
        if evidence_tail.strip():      # Key studies, Conflicting evidence, What remains under investigation (held back until the drugs and trials were shown)
            parts["tail"] = evidence_tail.strip()
            yield self.event("token", stage="TRIALS_ENRICHING", text="\n\n" + parts["tail"])
        if qs.SHOW_MEDICATION_ASSESSMENT and not parts.get("medication"):
            state["medication_assessment"] = {"outcome": "insufficient", "label": ma.OUTCOMES["insufficient"], "failures": ["the drug stage did not reach the report"]}
            parts["medication"] = ma.markdown(ma.fallback("the drug stage did not finish in time"))
        self.sections.extend(parts[k] for k in ("head", "landscape", "medication", "trials", "tail") if parts.get(k))      # the document's order

    # ---- stage: limitations
    async def stage_limitations(self) -> AsyncIterator[dict]:
        state, papers, skipped, stems = self.state, self.papers, self.skipped, self.stems
        retracted = [p for p in papers.values() if p.get("retracted")]
        excluded_after_screen = [p for p in papers.values() if not p.get("retracted") and p.get("triage") and not qs.usable(p, stems)]
        unreviewed_papers = [p for p in papers.values() if not p.get("retracted") and not p.get("triage")]
        notes = [f"Excluded retracted publication: {p['title'][:90]} (PMID {p['pmid']}, {p['year']}). Retracted work is not used as evidence." for p in retracted]
        notes += qs.future_date_notes(papers, stems)
        if excluded_after_screen:
            notes.append(f"{len(excluded_after_screen)} screened record(s) were excluded because the relevance reading judged them not to answer the question.")
        if unreviewed_papers:
            notes.append(f"{len(unreviewed_papers)} retrieved record(s) were not screened before the deadline and were not used as evidence.")
        lifecycle = qs.evidence_lifecycle(state, "", "pending")
        self.paper_counts = lifecycle["papers"]
        pc = self.paper_counts
        duplicate_n = max(0, pc["retrieved_occurrences"] - pc["deduplicated"])
        notes.append(f"Evidence lifecycle: {pc['retrieved_occurrences']} retrieved record occurrence(s), {pc['deduplicated']} unique records after merging provenance ({duplicate_n} duplicate occurrence(s)), "
                     f"{pc['screened']} screened, {pc['included']} included for synthesis, {pc['unreviewed']} unreviewed.")
        if self.focused and "evidence" not in self.focus_sections:      # the literature counts are not part of this answer: only a real problem (a failed source, a retracted paper) is said
            notes = [n for n in notes if n.startswith("Excluded retracted")]
            real = [x for x in skipped if not str(x).lower().startswith(("trials", "the opening summary"))]
            self.limits_md = qs.limitations_markdown(state, real, notes) if (real or notes) else ""
            if self.limits_md:
                yield self.event("token", stage="TRIALS_ENRICHING", text="\n\n" + self.limits_md)
            return
        self.limits_md = qs.limitations_markdown(state, skipped, notes)
        yield self.event("token", stage="TRIALS_ENRICHING", text="\n\n" + self.limits_md)

    # ---- stage: verify
    async def stage_verify(self) -> AsyncIterator[dict]:
        state, clock, papers = self.state, self.clock, self.papers
        yield self.event("stage", stage="VERIFICATION")
        if self.grounded_opening and self.sections and self.opening and self.opening in self.sections[0]:
            self.sections[0] = self.sections[0].replace(self.opening, self.grounded_opening, 1)      # the saved report and PDF carry the summary written from the finished evidence
            state["opening_rewritten_from_evidence"] = True
        full = qs.strip_preliminary("\n\n".join(s for s in self.sections + [self.limits_md] if s))
        result = await asyncio.to_thread(qs.verify_and_repair, full, state)      # CPU only (no model call): kept off the event loop
        full = result["answer"]
        qs.record_plan_audit(state, "sections", plan_audit.section_decisions(self.plan_box.get("plan") or {}, full))
        state["verification"] = {"status": result["status"], "checked": result["checked"], "repairs": result["repairs"], "repair_count": result["repair_count"],
                                 "issues_before_repair": [i["message"] for i in result["issues_before"]]}
        state["evidence_lifecycle"] = qs.evidence_lifecycle(state, full, result["status"])
        cited_n = state["evidence_lifecycle"]["papers"]["citation_checked"]
        pc = self.paper_counts
        lifecycle_line = (f"Evidence lifecycle: {pc['retrieved_occurrences']} retrieved occurrence(s), {pc['deduplicated']} unique, {pc['screened']} screened, {pc['included']} included, "
                          f"{pc['unreviewed']} unreviewed; {cited_n} source identifier(s) cited after verification ({result['status']}).")
        full = re.sub(r"\n\n### Sources", "\n\n" + lifecycle_line + "\n\n### Sources", full, count=1)
        final_check = await asyncio.to_thread(qs.verify_and_repair, full, state, False)
        full = final_check["answer"]
        result = final_check
        state["verification"].update({"status": result["status"], "checked": result["checked"], "final_check": True})
        state["qc"] = {"status": result["status"], "checks": result["checks"], "remaining": [i["message"] for i in result["issues_after"]], "view": qs.build_qc_context(state)}
        clock.mark("verification_done_s")
        routing_keys = {row["key"] for row in (state.get("routing") or {}).get("operations", [])}
        state["search_log"] = qs.search_log(state.pop("_search_raw", {}), routing_keys)
        logging.getLogger("neurogpt.search").info("search summary: %s", state["search_log"]["summary_line"])
        self.full, self.result = full, result
        yield self.event("verification", status=result["status"], issues=[i["message"] for i in result["issues_after"]], repaired=result["repairs"],
                         checked=result["checked"], checks=result["checks"])

    # ---- stage: finalize
    async def stage_finalize(self) -> AsyncIterator[dict]:
        state, clock, papers, trials = self.state, self.clock, self.papers, self.trials
        state["answer_status"] = "FINAL"
        clock.mark("final_s")
        state["timestamps"].update(clock.marks)
        full = self.full
        state["citations"] = [papers[p]["cite"] for p in set(re.findall(r"PMID[:\s]*([0-9]{6,9})", full)) if p in papers]
        full, cited_order = numberize(full, papers)      # [1], [2] in the text and one reference list; the mapping is kept in the state
        state["reference_numbers"] = {str(i): pmid for i, pmid in enumerate(cited_order, 1)}
        state["figure_job"] = str(id(self.figure_box))      # the figures are collected when the PDF is made (collect_figures); the answer is not held for them
        task = self.figures_task if (self.figures_task is not None and "figures" in self.ran) else asyncio.ensure_future(asyncio.sleep(0))
        qs.FIGURE_JOBS[state["figure_job"]] = (FigureJob(task), self.figure_box)
        footer = closing_note(self.result["status"])
        state["observability"] = observability.end_run({"first_token_s": clock.marks.get("first_token_s"), "first_useful_s": clock.marks.get("first_useful_s"), "focused": state.get("focused"),
                                                         "papers": len(papers), "trials": len(trials)})
        yield self.event("final", markdown=full + footer, timings=dict(clock.marks), state=state, skipped=self.skipped)
