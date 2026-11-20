"""Async twin of the routing stage of qa_stream.py: each retrieval skill writes its searches (one model call each, all at the same time), every search starts the moment it is written, and the routed
searches run under a time budget (async_routing.AsyncExecution). The validation, merging and baseline rules are the code of retrieval_routing / qa_stream, used unchanged; only the waiting is different."""
from __future__ import annotations

import asyncio
import time

import async_core
import async_routing
import async_support as asu
import async_tools as at
import async_web
import conditions as cond_groups
import mesh_expand
import qa_stream as qs
import retrieval_routing as rr
from async_deep import arouted_pubmed, asearch_fda_labels_wide
from async_pipeline import acached

_MODE = "-" + at._sources_mode()      # the search mode is part of the cache key: a result from one mode is never served in another


async def aask_queries(question: str, task: str, payload: dict, max_tokens: int, skill: str = "drug-intelligence") -> dict:
    """Twin of qa_stream.ask_queries: the call that writes searches, with its own model setting."""
    return await async_core.aask_json(question, task, payload, max_tokens, skill=skill, purpose="queries")


async def _routed_fda(terms, drugs=None, limit: int = 8) -> str:
    return await asearch_fda_labels_wide([str(t) for t in (terms or [])], [str(d) for d in (drugs or [])], qs._ROUTE_QUESTION or " ".join(str(t) for t in (terms or [])))


def areal_adapters(anchors) -> dict:
    """Twin of qa_stream.real_adapters / retrieval_routing.make_adapters: target -> async function(operation) -> the search tool's own JSON text, wired to the async searches and the shared cache."""
    mesh_on = mesh_expand.enabled()

    async def expansion(target: str):
        """The MeSH alternatives of the plan's condition terms: looked up (and waited for only briefly) BEFORE the search form is built, because rr.boolean_form is a plain function."""
        if not mesh_on:
            return None
        records = await asu.arecords(list(getattr(anchors, "mesh_terms", [])), mesh_expand.ROUTING_WAIT_S)

        def expand(words):
            have = {w.lower() for w in words}
            found = {t: mesh_expand.clauses(t, record, target) for t, record in records.items() if t.split() and all(w in have for w in t.split())}
            return {t: extra for t, extra in found.items() if extra}
        return expand

    async def pubmed(op):
        form = rr.boolean_form(op.query, anchors, expand=await expansion("pubmed"))
        types = op.args.get("study_types") or []
        if types:
            form = f"({form}) AND (" + " OR ".join(rr.STUDY_FILTERS[t] for t in types) + ")"
        if op.args.get("open_access"):
            form = f"({form}) AND {rr.OPEN_ACCESS_FILTER}"
        sort, years = op.args.get("sort", "relevance"), op.args.get("years_back", 5)
        return await acached("pm" + _MODE, f"{form}|{sort}|{years}|routing", 900, lambda: arouted_pubmed(form, max_results=8 if sort == "date" else 6, years_back=years, sort=sort))

    async def europepmc(op):
        form = rr.boolean_form(op.query, anchors, expand=await expansion("europepmc"))
        sort, years = op.args.get("sort", "relevance"), op.args.get("years_back", 5)
        full = bool(op.args.get("fulltext"))
        return await acached("pm" + _MODE, f"{form}|{sort}|{years}|epmc|{full}", 900,
                             lambda: at.asearch_europepmc_quick(form, max_results=8 if full else 6, years_back=years, sort=sort, **({"fulltext": True} if full else {})))

    async def trials(op):
        condition, intervention = op.args.get("condition", ""), op.args.get("intervention", "")
        status, drug_only = rr.TRIAL_STATUS_FILTER.get(op.args.get("status", "any"), ""), bool(op.args.get("drug_only"))

        async def search(with_intervention: str):
            return await acached("ct" + _MODE, f"{condition}|{with_intervention}|{status}|{drug_only}|routing", 300,
                                 lambda: at.asearch_clinical_trials(condition, with_intervention, status, 12 if drug_only else 10, drug_only))
        raw = await search(intervention)
        if intervention and rr.classify_raw(raw)[0] == "completed_empty":
            wider = await search("")
            if rr.classify_raw(wider)[0] == "completed_found":
                return wider
        words = condition.split()
        if rr.classify_raw(raw)[0] == "completed_empty" and len(words) > 3 and " OR " not in condition:
            chunks = [" ".join(words[i:i + 3]) for i in range(0, len(words), 3)]
            alternatives = await acached("ct" + _MODE, f"{condition}|any-concept|{status}|{drug_only}|routing", 300,
                                         lambda: at.asearch_clinical_trials(" OR ".join(chunks), "", status, 10, drug_only))
            if rr.classify_raw(alternatives)[0] == "completed_found":
                return alternatives
        return raw

    async def fda_label(op):
        terms, drugs = op.args.get("condition_terms") or [], op.args.get("drug_names") or []
        return await acached("fda", f"{' '.join(terms)}|{','.join(drugs)}|routing|wide", 3600, lambda: _routed_fda(terms, drugs or None, 8))

    async def guideline_sites(op):
        raw = await acached("web", f"{op.query}|guideline", 300, lambda: async_web.aguideline_search(op.query))
        return qs.filter_authoritative(raw)      # pure

    async def regulator(op):
        return await acached("web", f"{op.query}|regulator", 300, lambda: async_web.aregulator_search(op.query))

    async def web(op):
        return await acached("web", f"{op.query}|web", 300, lambda: async_web.aweb_search(op.query))
    return {"pubmed": pubmed, "europepmc": europepmc, "trials": trials, "fda_label": fda_label, "guideline_sites": guideline_sites, "regulator": regulator, "web": web}


async def awrite_queries(question: str, plan: dict, skill: str, anchors, ask, attempts: int = 2) -> dict:
    """Twin of qa_stream._write_queries: one retrieval skill writes its searches. Returns {'ops', 'rejected', 'outcome', 'seconds'}."""
    began = time.time()
    rules = qs.skill_section(skill, "Query transformation by intent")
    if not rules:
        return {"ops": [], "rejected": [], "outcome": "error", "seconds": 0.0, "error": "the skill has no query transformation section"}
    payload = {"question": question, "intent": plan.get("intent"), "understanding": {k: v for k, v in (plan.get("understanding") or {}).items() if v}}
    if plan.get("conditions"):
        payload["conditions"] = cond_groups.names(plan["conditions"])
    task = (qs.EXPAND_TASK.format(skill=skill) + ("\n\nThe question names several conditions (`conditions`): write the searches for EACH condition separately, each search naming only one of them." if plan.get("conditions") else "")
            + "\n\nSECTION (the " + skill + " skill):\n" + rules)
    outcome, rejected, error = "error", [], ""
    for attempt in range(attempts):
        try:
            reply = await ask(question, task, payload, 1500, skill=skill)
        except Exception as exc:
            outcome, error = "error", f"{type(exc).__name__}: {str(exc)[:80]}"
            continue
        ops, rejected = rr.parse_operations(skill, reply, question, anchors)
        if ops:
            return {"ops": ops, "rejected": rejected, "outcome": "wrote", "seconds": round(time.time() - began, 1)}
        if skill == "latest-evidence" and attempt < attempts - 1 and (rr.declined(reply) or (isinstance(reply, dict) and reply.get("queries") == [])):
            outcome = "declined"
            continue
        if rr.declined(reply) or (isinstance(reply, dict) and reply.get("queries") == []):
            return {"ops": [], "rejected": [], "outcome": "declined", "seconds": round(time.time() - began, 1)}
        outcome = "all_rejected" if isinstance(reply, dict) and isinstance(reply.get("queries"), list) else "malformed"
    return {"ops": [], "rejected": rejected, "outcome": outcome, "seconds": round(time.time() - began, 1), "error": error}


async def afill_coverage(question: str, plan: dict, gaps: list[dict], ops: list, anchors, ask=aask_queries, call_s: float | None = None) -> dict:
    """Twin of qa_stream.fill_coverage: ONE bounded model call that writes the missing searches for the gaps found before retrieval."""
    began = time.time()
    limit = qs.COVERAGE_CALL_S if call_s is None else call_s
    section = qs.skill_section("latest-evidence", "Query transformation by intent")
    sources = section[section.find("### Sources you may target"):section.find("### Output")] if "### Sources you may target" in section else section
    discovery = any(g["kind"] == "discovery_shortfall" for g in gaps)
    payload = {"question": question, "understanding": {k: v for k, v in (plan.get("understanding") or {}).items() if v}, "gaps": gaps,
               "already_written": [{"need": o.need, "serves": o.serves, "target": o.target, "query": o.query[:110]} for o in ops]}
    try:
        reply = await asyncio.wait_for(ask(question, qs.COVERAGE_TASK.format(n=qs.MAX_COVERAGE_ADDITIONS) + "\n\nSECTION:\n" + sources, payload, 900,
                                           skill="drug-intelligence" if discovery else "latest-evidence"), timeout=limit)
    except asyncio.TimeoutError:
        return {"ops": [], "outcome": "timed_out", "seconds": round(time.time() - began, 1), "rejected": 0}
    except Exception as error:
        return {"ops": [], "outcome": "error", "seconds": round(time.time() - began, 1), "rejected": 0, "error": f"{type(error).__name__}: {str(error)[:80]}"}
    seconds = round(time.time() - began, 1)
    new, rejected = rr.parse_operations("coverage", reply, question, anchors, max_ops=qs.MAX_COVERAGE_ADDITIONS)
    kept = []
    for op in new:
        if any(rr.near_duplicate(op, existing) or op.key == existing.key for existing in list(ops) + kept):
            rejected.append({"item": None, "reason": "duplicate of a search already written"})
            continue
        op.origin = "coverage"
        if discovery and op.target == "web":
            op.skill = "drug-intelligence"
        kept.append(op)
    return {"ops": kept, "outcome": "added" if kept else "nothing_added", "seconds": seconds, "rejected": len(rejected)}


async def aroute_deep(question: str, plan: dict, ask=aask_queries, submit=None) -> dict:
    """Twin of qa_stream.route_deep: validate the plan's skills, let each skill write its searches AT THE SAME TIME (each search starts when its skill has answered), add the baselines and the coverage."""
    qs._ROUTE_QUESTION = question
    understanding = plan.get("understanding") or {}
    skills, select_trace = rr.select_skills(plan.get("skills"), understanding.get("info_needed"), qs.skill_catalog())
    for mandatory, why in (("drug-intelligence", "mandatory medication assessment"), ("clinical-trials", "mandatory trial search for the condition")):
        if mandatory in qs.skill_catalog() and mandatory not in skills:
            skills = skills + [mandatory]
            select_trace["added"] = list(select_trace.get("added") or []) + [{"skill": mandatory, "because": why}]
    anchors = qs.routing_anchors(question, plan)
    literature = rr.baseline_operations(question, skills, fallback=bool(select_trace.get("fallback")))
    ops, duplicates = rr.merge_operations(literature)
    if submit:
        submit(ops)

    def add(new: list, cap: int = rr.MAX_OPERATIONS) -> None:
        nonlocal ops, duplicates
        merged, more = rr.merge_operations(ops, new, cap=cap)
        fresh = merged[len(ops):]
        ops, duplicates = merged, duplicates + more
        if submit and fresh:
            submit(fresh)
    written: dict = {}
    pending = {asyncio.ensure_future(awrite_queries(question, plan, s, anchors, ask)): s for s in skills}
    while pending:
        done, _ = await asyncio.wait(list(pending), return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            skill = pending.pop(task)
            written[skill] = task.result()
            add(written[skill]["ops"])      # this skill's searches start now, while the others are still being written
    written = {s: written[s] for s in skills}
    baselines = []
    for skill, result in written.items():
        if result["outcome"] in ("malformed", "error", "all_rejected"):
            baselines += rr.baseline_for(skill, question, anchors, plan.get("condition", ""), plan.get("drugs"))
    add(baselines)
    add(qs.mandatory_case_operations(question, plan, ops, anchors), cap=rr.MAX_OPERATIONS + 4)
    add(qs.named_drug_operations(question, plan, ops), cap=max(rr.MAX_OPERATIONS + 4, len(ops) + 3))
    add(qs.dosing_paper_operations(question, plan, ops), cap=max(rr.MAX_OPERATIONS + 8, len(ops) + 4))
    add(qs.symptom_drug_operations(question, plan, ops), cap=max(rr.MAX_OPERATIONS + 5, len(ops) + 1))
    add(qs.condition_operations(plan, ops), cap=max(rr.MAX_OPERATIONS + 4, len(ops) + 2 * len(plan.get("conditions") or [])))
    coverage: dict = {"gaps": [], "outcome": "not_needed"}
    if qs.coverage_check_enabled():
        gaps = rr.coverage_gaps(ops, understanding.get("info_needed"), skills, qs.source_capabilities())
        coverage = {"gaps": gaps, "outcome": "no_gaps" if not gaps else "pending"}
        if gaps:
            filled = await afill_coverage(question, plan, gaps, ops, anchors, ask=ask)
            before = len(ops)
            add(filled["ops"], cap=max(rr.MAX_OPERATIONS + qs.MAX_COVERAGE_ADDITIONS, len(ops) + qs.MAX_COVERAGE_ADDITIONS))
            coverage.update({"outcome": filled["outcome"], "added": len(ops) - before, "seconds": filled["seconds"], "rejected": filled["rejected"]})
    return {"ops": ops, "anchors": anchors, "skills": skills, "rejected": [x for r in written.values() for x in r["rejected"]],
            "trace": {"select_skills": select_trace, "coverage": coverage, "skills": {s: {k: v for k, v in r.items() if k in ("outcome", "seconds", "error")} | {"operations": len(r["ops"])} for s, r in written.items()},
                      "duplicates_removed": len(duplicates), "baselines": [o.id for o in literature + baselines]}}


async def arun_routing_stage(question: str, plan: dict, make_adapters=areal_adapters, budget_s: float = qs.ROUTING_BUDGET_S, ask=aask_queries, box: dict | None = None) -> dict:
    """Twin of qa_stream.run_routing_stage: write the routed searches and run them at the same time under the budget (each starts as soon as it is written)."""
    anchors = qs.routing_anchors(question, plan)
    execution = async_routing.AsyncExecution([], make_adapters(anchors), budget_s, workers=qs.ROUTING_WORKERS, hold=True)
    if box is not None:
        box.update(execution=execution, anchors=anchors)
    routed = await aroute_deep(question, plan, ask=ask, submit=execution.submit)
    await execution.finish()
    return {**routed, "execution": execution, "raw": rr.raw_for_ingest(routed["ops"], execution.results)}
