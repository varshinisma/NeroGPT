"""Async twins of the last two threaded helpers of the fast stages: qa_stream.plan_with_skill and qa_stream.fast_drug_decisions (with triage_trials)."""
from __future__ import annotations

import asyncio
import json
import re

import async_core
import qa_stream as qs
from async_stages_drugs import ainterpret_fda_labels


async def aplan_with_skill(question: str) -> dict:
    """Async twin of qa_stream.plan_with_skill: the old JSON planner (only used when the plan agent made no plan); asked once more when the reply is missing or not JSON."""
    plan, status = {}, "ok"
    for attempt in (1, 2):
        try:
            text = await async_core.acollect(qs.load_skill_text("clinical-research-orchestrator"),
                                             f"TASK: {qs.PLAN_TASK}\nSKILL CATALOG (name: what it is for):\n{qs.retrieval_catalog()}\nQuestion: {question}", qs.PLAN_TOKENS, temperature=0.0,
                                             purpose="planning" if attempt == 1 or not qs.planner_choice()[0] else "planning_backup")
            plan = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
            status = "ok" if attempt == 1 else "retried"
            break
        except Exception:
            plan, status = {}, "fallback"
    return qs.finish_plan(plan, status, question)


async def atriage_trials(state: dict, question: str, trials: list[dict]) -> None:
    """Async twin of qa_stream.triage_trials."""
    offered_t = {t["nct_id"]: {"sentences": [t["title"] or "", "Conditions: " + "; ".join(t.get("conditions") or []), "Interventions: " + "; ".join(t.get("interventions") or []),
                                             "Primary outcome: " + str(t.get("primary_endpoint") or "not stated")]} for t in trials if not t.get("triage")}
    if not offered_t:
        return
    answer = await async_core.aask_json(question, qs.TRIAL_TRIAGE_TASK, {"trials": [{"id": k, "sentences": [f"{n}. {x}" for n, x in enumerate(v["sentences"], 1)]} for k, v in offered_t.items()]},
                                        700, skill="evidence-synthesis")
    decisions = qs.verify_trial_triage(answer.get("trials") or [], offered_t)
    for k in offered_t:
        state["trials_by_id"][k]["triage"] = decisions.get(k) or {"relevance": "uncertain", "kind": "other", "evidence": ""}


async def afast_drug_decisions(state: dict, question: str) -> None:
    """Async twin of qa_stream.fast_drug_decisions: the two reads (FDA labels, late-phase registry trials) run at the same time."""
    from landscape import is_late_phase
    state["fast_decision_trace"] = []
    candidates = sorted((t for t in state["trials_by_id"].values() if t.get("drug_interventions") and is_late_phase(t)), key=lambda t: t.get("last_update") or "", reverse=True)[:6]
    results = await asyncio.gather(ainterpret_fda_labels(question, state["regulatory"]), atriage_trials(state, question, candidates), return_exceptions=True)
    for r in results:
        state["fast_decision_trace"].append({"error": f"{type(r).__name__}: {str(r)[:100]}"} if isinstance(r, BaseException) else r)
