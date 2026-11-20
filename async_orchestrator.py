"""Async twin of the orchestrator's agent calls (orchestrator.py): the routing decision, the plan, and the recovery after an empty stage. The agent run is awaited with Agno's own `arun` and bounded in time
with asyncio.wait_for, so no worker thread is needed. All the checking of the agent's choice (to_reply, follow_up_plan, compose, clean_retries) is the code of orchestrator.py, used unchanged."""
from __future__ import annotations

import asyncio

import conversation_memory as memory_mod
import observability
import orchestrator as orch


async def adefault_run_agent(system: str, prompt: str, tools: list, limit: int = 1) -> None:
    """Async twin of orchestrator.default_run_agent: one fresh Agno agent run with the tools, one tool call, bounded in time (a late agent is cancelled)."""
    from agno.agent import Agent
    from qa_stream import model_for
    agent = Agent(model=model_for(orch.PLAN_TOKENS, 0.0, "planning"), instructions=[system], tools=tools, tool_call_limit=limit, markdown=False)
    with observability.span("model", "orchestrator_decision", in_chars=len(system) + len(prompt)):
        await asyncio.wait_for(agent.arun(prompt), timeout=orch.DECISION_TIMEOUT_S)


async def arun_once(prompt: str, memory: list[dict], run_agent=None) -> dict:
    record: dict = {}
    try:
        slim = orch.TRIM and not memory      # a new question: the shorter prompt and the plan-only tool
        await (run_agent or adefault_run_agent)(orch.instructions(slim), prompt, orch.make_tools(record, with_conversation=bool(memory), slim=slim))
    except Exception:
        record = {}
    return record


async def adecide(question: str, memory: list[dict], run_agent=None, fallback_ask=None) -> dict:
    """Async twin of orchestrator.decide: same result shape. `fallback_ask` stays a sync function (the old JSON router) and is only used when the agent made no usable choice."""
    unchanged = {"follow_up": False, "mode": "new", "question": question, "turns": []}
    if not memory or not memory_mod.worth_routing(question):
        return unchanged
    if memory_mod.clearly_new(question, memory):      # it names its own subject, shares none of the earlier subject words and points at nothing: a new question, decided by code at once (no routing call, no wait; its plan is written beside the fast answer)
        return {**unchanged, "routing_note": "standalone: new question decided by code, no routing call"}
    try:
        extra = orch.catalog()
    except Exception:
        extra = ""
    payload = memory_mod.resolve_payload(question, memory) + extra
    record = await arun_once(payload, memory, run_agent)
    if orch.to_reply(record) == {"mode": "new"} and memory_mod.refers_back(question):      # the message points back but was called a new question: the agent looks once more
        again = await arun_once(payload + orch.BACK_REF_HINT, memory, run_agent)
        if orch.to_reply(again) not in (None, {"mode": "new"}) or again.get("tool") == "ask_for_clarification":
            record = again
    if record.get("tool") == "ask_for_clarification" and str(record.get("question") or "").strip():
        return {"follow_up": False, "mode": "clarify", "question": question, "turns": [], "clarification": " ".join(str(record["question"]).split())[:300]}
    reply = orch.to_reply(record)
    if reply and reply.get("mode") == "research" and memory_mod.standalone(question, memory):      # called a follow-up (research), but it names its own subject and shares nothing with the earlier answers: a new question
        return {**unchanged, "routing_note": "standalone: the message names its own subject and shares none of the earlier subject words, so it is researched as a new question"}
    if reply and reply.get("mode") == "research" and not 3 <= len(" ".join(str(reply.get("ask") or "").split()).strip(" .?")) <= 200:
        reply["ask"] = " ".join(question.split()).strip(" .?")[:200]
    if reply is None:
        return await asyncio.to_thread(memory_mod.resolve, question, memory, fallback_ask) if fallback_ask else unchanged
    resolved = memory_mod.compose(question, memory, reply)
    plan = record.get("plan") if orch.PLAN_BY_AGENT else None
    if plan and resolved.get("mode") in ("research", "new"):
        resolved["plan"] = orch.follow_up_plan(plan, memory, resolved, record) if resolved.get("follow_up") else plan
    return resolved


async def aplan_only(question: str, run_agent=None) -> dict | None:
    """Async twin of orchestrator.plan_only: the plan for a message that needed no routing decision."""
    record = await arun_once(f"NEW QUESTION: {question}" + orch.catalog(), [], run_agent)
    plan = record.get("plan") if record.get("tool") == "research" else None
    return plan or None


async def arecover(report_text: str, group: str, question: str, plan: dict, run_agent=None) -> dict:
    """Async twin of orchestrator.recover: the agent's answer to an empty stage: {'retries': [validated searches], 'note': text}. Any failure means no retries."""
    import recovery
    record: dict = {}
    runner = run_agent or (lambda system, prompt, tools: adefault_run_agent(system, prompt, tools, recovery.MAX_RETRIES + 1))
    try:
        await runner("=== SKILL: clinical-research-orchestrator ===\n" + orch.skill_excerpt() + "\n" + orch.RECOVERY_GUIDE, report_text, orch.recovery_tools(record))
    except Exception:
        pass
    return {"retries": recovery.clean_retries(record.get("retries"), group, question, plan), "note": " ".join(str(record.get("note") or "").split())[:200]}
