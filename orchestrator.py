"""The orchestrator agent: ONE decision-maker for every message. It reads the clinical-research-orchestrator SKILL.md as its script and, in a single tool call, decides how the message is handled
(answer from the conversation, research it, ask back) AND, when it chooses research, writes the plan (intent, condition, conditions, medicines, information needed, answer format, skills). The planner call
and the follow-up router of old are folded into this one call; both remain only as fallbacks when the agent fails.

The agent only CHOOSES and describes: it never writes drugs, evidence or numbers. Code then runs the chosen path (the same research stages, the conversational answer, the quick check), copies the drugs and
conditions of a follow-up from the stored answers, and finishes and checks the plan. Any failure, timeout or missing decision falls back to the JSON router and the planner call, so a message is never lost.
Nothing here names a disease or a drug."""
from __future__ import annotations

import observability

import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import conversation_memory as memory_mod

SKILL_FILE = Path(__file__).resolve().parent / "skills" / "clinical-research-orchestrator" / "SKILL.md"
AGENT_ENABLED = os.getenv("ORCHESTRATOR_AGENT", "on").strip().lower() not in {"off", "0", "false", "no"}
PLAN_BY_AGENT = os.getenv("ORCHESTRATOR_PLANS", "on").strip().lower() not in {"off", "0", "false", "no"}      # the agent writes the plan (off: the planner call of old)
DECISION_TIMEOUT_S = float(os.getenv("ORCHESTRATOR_TIMEOUT_S", "20"))
PLAN_TOKENS = 1400
FOCUS_OF = {"dosing": "drugs", "safety": "safety", "drug_interaction": "safety", "regulatory_status": "drugs", "trial_status": "trials",
            "efficacy": "evidence", "guideline_recommendation": "evidence", "mechanism": "evidence", "prognosis": "evidence", "diagnosis": "evidence",
            "treatment_options": "evidence", "comparison": "evidence", "alternatives": "evidence", "management": "evidence", "monitoring": "evidence", "contraindications": "safety"}      # what a follow-up asks for (the agent's info_needed) -> the part of the report that answers it
SKILL_OF = {"drugs": "drug-intelligence", "safety": "drug-intelligence", "drug_facts": "drug-intelligence", "drug_discovery": "drug-intelligence", "trials": "clinical-trials"}
DRUG_ASKS = {"dosing", "safety", "drug_interaction", "regulatory_status"}      # asks about the earlier drugs that `attributes` (the facts the agent names) replaces


def clean_attributes(value) -> list[str]:
    """The facts the agent says the follow-up wants about the earlier drugs, as plain short phrases (a list, or one comma-separated text)."""
    items = value if isinstance(value, (list, tuple)) else re.split(r"[;,\n]", str(value or ""))
    out: list[str] = []
    for item in items:
        text = " ".join(str(item or "").split()).strip(" .")
        if 3 <= len(text) <= 50 and text.lower() not in {o.lower() for o in out}:
            out.append(text)
    return out[:6]


def focus_sections(plan: dict, record: dict, has_drugs: bool, changed_topic: bool) -> list[str]:
    """The parts of the report a follow-up needs: ['drugs'] (the drug tables), ['trials'], ['evidence'] (the literature sections) or a mix; [] means the whole report.
    The whole report is given when the agent says scope=full, when the person changes the condition or population, or when it is a new question; any other follow-up gets only the parts it asks for (an ask without its own part is answered from the literature)."""
    scope = str(record.get("scope") or "").lower()
    asks_medicines = str(record.get("kind") or "").lower() == "follow_up" and plan.get("medicines_needed") is True and not has_drugs and not changed_topic
    if asks_medicines and not clean_attributes(record.get("attributes")):      # "give me more drugs for the above question": the condition's medicines are found afresh and only they are shown, never the whole report (even when the agent wrote scope=full)
        return ["drug_discovery"]
    if scope == "full" or changed_topic:
        return []
    sections: list[str] = []
    attributes = clean_attributes(record.get("attributes"))
    if attributes and has_drugs:      # the agent named the facts it wants about the earlier drugs (contraindications, use in pregnancy, ...): one table of exactly those; the asks it replaces are not also answered
        sections.append("drug_facts")
    for ask in plan.get("info_needed") or []:
        if attributes and has_drugs and str(ask) in DRUG_ASKS:
            continue
        part = FOCUS_OF.get(str(ask), "evidence")      # an ask no part is named for (alternatives, options, anything else) is answered from a literature search on that one question
        if part not in sections:
            sections.append(part)
    if not sections and scope == "focused":
        sections = ["evidence"]
    for needs_drugs in ("drugs", "safety", "drug_facts"):      # these work on the earlier answers' drugs: without any, there is nothing to look up
        if needs_drugs in sections and not has_drugs:
            sections.remove(needs_drugs)
    if not has_drugs and plan.get("medicines_needed") is True and not ({"drugs", "safety", "drug_facts"} & set(sections)):
        sections.append("drug_discovery")      # the message asks about medicines but is NOT about the earlier drugs (for example "the drugs for the above question"): the medicines of the earlier condition are discovered afresh, the earlier drugs are not reused
    return sections

TOOL_GUIDE = """
YOU ARE THE ORCHESTRATOR. The skill above is your script. The research stages (fast answer, deep search, drug and trial stages, writing, verification) are run by the program after you choose; you decide how the
message is handled and what the research is about. Make ONE decision by calling EXACTLY ONE tool:
- answer_from_conversation: (only when EARLIER QUESTIONS exist) the answer is already in the earlier answers: explaining, summarising, comparing the listed drugs, picking the safest or first-line one among them, ranking, simplifying, putting it in a table, drafting a handout or note from it, or a thank-you. It is answered at once, in a fixed structure, with no search.
- research: the message needs evidence. Fill in the plan as the skill's Steps 2 to 4 describe (classify, understand, choose skills, extract entities). For a follow-up that depends on the earlier answers (kind=follow_up: for example
  their dosage, side effects or trials, or whether those drugs are used for another condition) write `ask` WITHOUT naming any drug the earlier answers listed, and leave `drugs` empty; the program copies the drugs and conditions from the stored answers.
  A follow-up of ANY kind about the same topic that needs new evidence is kind=follow_up: dosage, side effects, interactions, monitoring, the drugs in another population (children, adults, the elderly, pregnancy), trials, newer drugs, or another condition. A message that points back at the earlier answers ("these drugs", "the above", "they", "for her") is a follow-up even when it names another condition: ask whether the earlier drugs treat that condition, put the condition in own_condition, set needs_drugs=true and scope=focused. When the message changes the population or condition AND does not point back at anything earlier (it is about a new subject), put the changed topic, as written, in own_condition (for example '<the condition> in adults'), so the question is built for it.
  For a message that can be understood alone (kind=new) fill `drugs` only with medicines the message itself names.
  scope: a follow-up asks for something about the earlier answers and can ask ANYTHING: work out what it asks and how it relates to them (the skill's Step 2d) instead of choosing from a fixed list. Facts about the earlier drugs (contraindications, use in pregnancy, monitoring, dosage, side effects, mechanism, ...) are named in `attributes`; a follow-up that asks for ONE thing that needs research (those facts; the trials; the evidence on efficacy, guidelines, mechanism or prognosis) is scope=focused: classify what it asks in info_needed, and the program researches and writes only that part, not the whole report. A message that changes the condition or the population (own_condition), a new question, a case, or a broad one is scope=full, the whole report. Leave scope empty to let the program decide from info_needed.
- ask_for_clarification: the message cannot be understood even with the conversation. Use rarely; ask one short question.
Never name a drug or a condition that the question and the earlier answers did not state. Never answer the question yourself.

THE STAGES OF A RESEARCH RUN (the skill's Steps 5 to 8). With `research` you also choose their ORDER (the `stages` list). Code starts fast_search and fast_answer itself (the skill's fast path comes before any plan),
then runs the rest in YOUR order, moving a stage later only when it must wait for the stages it reads. Stages that start background work (read_evidence, drugs, figures) are best started early so they run beside the
writing; follow the order the skill describes unless the question gives a reason to change it. Every stage is required, `figures` included: the doctor wants the figures of the relevant papers for every question.
{STAGES}"""


def skill_excerpt() -> str:
    """The whole orchestrator skill (the agent's script)."""
    try:
        return SKILL_FILE.read_text(encoding="utf-8").split("---", 2)[-1].strip()
    except OSError:
        return ""


def instructions() -> str:
    import stages
    return "=== SKILL: clinical-research-orchestrator ===\n" + skill_excerpt() + "\n" + TOOL_GUIDE.replace("{STAGES}", stages.describe())


PLAN_KEYS = ("is_clinical", "intent", "secondary_intents", "condition", "conditions", "drugs", "pubmed_query", "main_problem", "associated_findings", "related_topics", "trial_condition",
             "trial_intervention", "asked", "purpose", "info_needed", "medicines_needed", "answer_format", "case_facts", "semantic_variants", "skills", "skip_stages", "stages")


def make_tools(record: dict, with_conversation: bool = True) -> list:
    """The agent's choices, as tools. Each only records what the agent chose (the program does the work afterwards), and stops the agent after the call."""
    try:
        from agno.tools import tool
        stop = tool(stop_after_tool_call=True)
    except Exception:
        stop = lambda f: f

    @stop
    def answer_from_conversation(turns: str = "all", ask: str = "") -> str:
        """Answer from the earlier answers of this conversation, with no new search. turns: the numbers of the earlier questions it refers to, such as '1,2', or 'all'. ask: what the person wants, in their own words and without naming any drug or condition (used only if the earlier answers turn out not to contain it)."""
        record.update(tool="answer_from_conversation", turns=turns, ask=ask)
        return "chosen"

    @stop
    def research(kind: str = "new", ask: str = "", turns: str = "all", needs_drugs: bool = False, own_condition: str = "", scope: str = "",
                 is_clinical: bool = True, attributes: list[str] | None = None, relation: str = "", intent: str = "", secondary_intents: list[str] | None = None, condition: str = "", conditions: list[dict] | None = None,
                 drugs: list[str] | None = None, pubmed_query: str = "", main_problem: str = "", associated_findings: list[str] | None = None, related_topics: list[str] | None = None,
                 trial_condition: str = "", trial_intervention: str = "", asked: str = "", purpose: str = "", info_needed: list[str] | None = None, medicines_needed: bool | None = None,
                 answer_format: str = "", case_facts: list[str] | None = None, semantic_variants: list[str] | None = None, skills: list[str] | None = None, skip_stages: list[str] | None = None, stages: list[str] | None = None) -> str:
        """Research the message and give the plan. kind: 'new' (understood alone) or 'follow_up' (depends on the earlier answers). For a follow-up: ask = what the person asks for, in their own words, WITHOUT naming a drug the earlier answers listed (e.g. 'Give the dosage information'); turns = the numbers of the earlier questions it refers to, or 'all'; needs_drugs = true ONLY when the message is about the drugs the earlier answers listed (it asks something about them); false when it asks which drugs or for the drugs of an earlier question's condition (then set medicines_needed=true and leave drugs empty: the program discovers the condition's medicines afresh); own_condition = a condition the message itself names that differs from the earlier answers (as written), else empty; scope = 'focused' (only what the follow-up asks), 'full' (the whole report) or empty (the program decides from info_needed). relation = ONE short sentence of your own saying how the message relates to the earlier answers (for example the same patient, the same drugs, the same condition from a new angle, a different subject); attributes = for a follow-up about the earlier drugs, the specific facts it wants about each drug, in the person's own words (for example contraindications, use in pregnancy, monitoring, mechanism, cost, how it is given); leave empty when it does not ask for facts about the drugs. Never default to dosage or side effects: name only what the message asks. The plan follows the skill: is_clinical; intent (disease_research|latest_findings|drug_treatment|clinical_trial|comparison|mechanism|diagnosis|safety|case_evaluation|comprehensive_research); secondary_intents; condition; conditions (ONLY when the message asks about two or more separate conditions: objects {name, aliases}, each written in the message; else empty); drugs (medicines the message names); pubmed_query; main_problem; associated_findings (copied word for word); related_topics (3 to 5 phrases, only for a diagnosis, work-up, case or management question); trial_condition; trial_intervention; asked; purpose; info_needed (treatment_options|efficacy|safety|dosing|diagnosis|prognosis|mechanism|guideline_recommendation|trial_status|regulatory_status|drug_interaction); medicines_needed; answer_format (options_comparison|stepwise_plan|summary|table|yes_no_with_evidence); case_facts (word for word); semantic_variants (3 to 5 phrases); skills (exact names from the SKILL CATALOG); stages (the order in which the run's stages are done, written as the skill's Steps 5 to 8 describe, from the STAGES list in your instructions; code checks it, so a stage never runs before the stages it reads; leave it empty for the order the skill describes); skip_stages (normally empty; 'full_text' only when the question plainly has no use for the free full text of papers)."""
        given = dict(intent=intent, secondary_intents=secondary_intents, condition=condition, conditions=conditions, drugs=drugs, pubmed_query=pubmed_query, main_problem=main_problem,
                     associated_findings=associated_findings, related_topics=related_topics, trial_condition=trial_condition, trial_intervention=trial_intervention, asked=asked, purpose=purpose,
                     info_needed=info_needed, medicines_needed=medicines_needed, answer_format=answer_format, case_facts=case_facts, semantic_variants=semantic_variants, skills=skills, skip_stages=skip_stages, stages=stages, is_clinical=is_clinical)
        record.update(attributes=attributes, relation=" ".join(str(relation or "").split())[:200], tool="research", kind=str(kind or "new").lower(), turns=turns, ask=ask, needs_drugs=needs_drugs, own_condition=own_condition, scope=str(scope or "").lower(),
                      plan={k: v for k, v in given.items() if v not in (None, "", [], {})} | ({"medicines_needed": medicines_needed} if medicines_needed is not None else {}))
        return "chosen"

    @stop
    def ask_for_clarification(question: str) -> str:
        """Ask the person one short clarifying question because the message cannot be understood even with the conversation."""
        record.update(tool="ask_for_clarification", question=question)
        return "chosen"

    return [answer_from_conversation, research, ask_for_clarification] if with_conversation else [research]


def default_run_agent(system: str, prompt: str, tools: list, limit: int = 1) -> None:
    """One Agno agent run with the tools above (a fresh agent, one tool call, bounded in time)."""
    from agno.agent import Agent
    from qa_stream import model_for
    agent = Agent(model=model_for(PLAN_TOKENS, 0.0, "planning"), instructions=[system], tools=tools, tool_call_limit=limit, markdown=False)
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        with observability.span("model", "orchestrator_decision", in_chars=len(system) + len(prompt)):
            pool.submit(agent.run, prompt).result(timeout=DECISION_TIMEOUT_S)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _turns(value, count: int):
    if isinstance(value, str) and value.strip().lower() != "all":
        found = [int(n) for n in re.findall(r"\d+", value)]
        return found or "all"
    return "all"


def to_reply(record: dict) -> dict | None:
    """The agent's choice in the form conversation_memory.compose reads (None when no usable choice was made)."""
    tool = record.get("tool")
    if tool == "answer_from_conversation":
        return {"mode": "conversation", "turns": _turns(record.get("turns"), 0), "ask": record.get("ask") or "", "needs_drugs": False}
    if tool == "research" and record.get("kind") == "follow_up":
        own = " ".join(str(record.get("own_condition") or "").split())
        return {"mode": "research", "turns": _turns(record.get("turns"), 0), "ask": record.get("ask") or "", "needs_drugs": bool(record.get("needs_drugs")), "own_condition": own, "names_own_condition": bool(own)}
    if tool == "research":
        return {"mode": "new"}
    return None


def follow_up_plan(plan: dict, memory: list[dict], resolved: dict, record: dict) -> dict:
    """The agent's plan for a follow-up, with the drugs and conditions COPIED BY CODE from the stored answers it points at (the agent was told not to write them)."""
    plan = dict(plan)
    used = [memory[i - 1] for i in resolved.get("turns") or [] if isinstance(i, int) and 1 <= i <= len(memory)]
    drugs = memory_mod.unique([d for t in used for d in memory_mod.subject_drugs(t)], memory_mod.MAX_DRUGS) if record.get("needs_drugs") is True else []      # the drugs the earlier question NAMED when it named some, else the drugs its answer listed
    plan["drugs"] = drugs      # always code's list from the stored answers (or none): whatever drugs the agent wrote are dropped
    conditions = memory_mod.unique([t.get("condition") or "" for t in used if t.get("kind") != "chat"], 4)
    own = " ".join(str(record.get("own_condition") or "").split())
    if own and any(own.lower() in c.lower() or c.lower() in own.lower() for c in conditions if c):
        own = ""      # the condition the agent calls 'another' is the earlier one: nothing changed
    if own:      # the person asks about another condition: the earlier ones are not added
        plan["condition"] = own
    elif conditions:
        plan["condition"] = " and ".join(conditions)
    profile = memory_mod.case_profile(memory, resolved.get("turns") or None)
    if profile["facts"] and not own:      # the case of the earlier answers (its facts, from the question each came from) is what 'her' or 'the patient' means
        understanding = dict(plan.get("understanding") or {})
        understanding["case_facts"] = [f["text"] for f in profile["facts"]]
        plan["understanding"] = understanding
    for key in ("pubmed_query", "trial_condition", "trial_intervention", "conditions"):      # built from the complete question afterwards
        plan.pop(key, None)
    sections = focus_sections(plan, record, bool(drugs), bool(own) and not drugs)      # a message that points at the earlier drugs ("do these drugs also treat another condition?") is a follow-up even when it names another condition: only a message about a new subject, with no reference to the earlier drugs, is a new question      # one thing asked: only that part is researched and written (qa_stream, focused run)
    if sections:
        plan["focus"] = sections
        if "drug_facts" in sections:
            plan["attributes"] = clean_attributes(record.get("attributes"))
        kept = [s for s in plan.get("skills") or [] if s not in SKILL_OF.values()] if "evidence" in sections else []
        plan["skills"] = list(dict.fromkeys(SKILL_OF[s] for s in sections if s in SKILL_OF)) + (kept or (["latest-evidence"] if "evidence" in sections else []))
        plan["medicines_needed"] = "drugs" in sections or "safety" in sections or "drug_facts" in sections or "drug_discovery" in sections
    return plan


def run_once(prompt: str, memory: list[dict], run_agent=None) -> dict:
    record: dict = {}
    try:
        (run_agent or default_run_agent)(instructions(), prompt, make_tools(record, with_conversation=bool(memory)))
    except Exception:
        record = {}
    return record


def catalog() -> str:
    from qa_stream import retrieval_catalog
    return "\nSKILL CATALOG (name: what it is for):\n" + retrieval_catalog()


def decide(question: str, memory: list[dict], run_agent=None, fallback_ask=None) -> dict:
    """The same result shape as conversation_memory.resolve (plus mode 'clarify' with its text, and 'plan' when the agent wrote one in this same decision). On any problem the JSON router decides instead."""
    unchanged = {"follow_up": False, "mode": "new", "question": question, "turns": []}
    if not memory or not memory_mod.worth_routing(question):
        return unchanged      # nothing to decide: the agent writes the plan in its own call, beside the fast answer
    try:
        extra = catalog()
    except Exception:
        extra = ""
    record = run_once(memory_mod.resolve_payload(question, memory) + extra, memory, run_agent)
    if record.get("tool") == "ask_for_clarification" and str(record.get("question") or "").strip():
        return {"follow_up": False, "mode": "clarify", "question": question, "turns": [], "clarification": " ".join(str(record["question"]).split())[:300]}
    reply = to_reply(record)
    if reply and reply.get("mode") == "research" and not 3 <= len(" ".join(str(reply.get("ask") or "").split()).strip(" .?")) <= 200:
        reply["ask"] = " ".join(question.split()).strip(" .?")[:200]      # the agent left out what is asked: the person's own message says it (the drugs and conditions are added by code from the stored answers)
    if reply is None:
        return memory_mod.resolve(question, memory, fallback_ask) if fallback_ask else unchanged
    resolved = memory_mod.compose(question, memory, reply)
    plan = record.get("plan") if PLAN_BY_AGENT else None
    if plan and resolved.get("mode") in ("research", "new"):
        resolved["plan"] = follow_up_plan(plan, memory, resolved, record) if resolved.get("follow_up") else plan
    return resolved


def plan_only(question: str, run_agent=None) -> dict | None:
    """The plan for a message that needed no routing decision (no conversation yet, or a long standalone question): the same agent, the same skill, the research tool only. None when it made no plan."""
    record = run_once(f"NEW QUESTION: {question}" + catalog(), [], run_agent)
    plan = record.get("plan") if record.get("tool") == "research" else None
    return plan or None


RECOVERY_GUIDE = """
A STAGE HAS COME BACK EMPTY. Follow the skill's rules for failures and its retry limit (at most 2 research retries in all). You are shown the stage, the problem and the searches already run. Choose:
- retry_search(source, query): run ONE more search from an allowed source with a query you write. Broaden the earlier wording (drop detail such as the patient group, use the condition's common name or a synonym) and use only
  words of the question; never add a drug or condition the question did not state. You may call it at most twice.
- proceed(note): go on without more searching. note: one short sentence for the doctor on what is missing (for example 'no clinical trial was registered for this condition').
Finish by calling proceed. Never answer the question yourself."""


def recovery_tools(record: dict) -> list:
    try:
        from agno.tools import tool
        stop = tool(stop_after_tool_call=True)
    except Exception:
        stop = lambda f: f

    def retry_search(source: str, query: str) -> str:
        """Run one more search. source: one of the allowed retry sources. query: the broadened search, in the question's own words."""
        record.setdefault("retries", []).append({"source": source, "query": query})
        return "queued: it runs after you call proceed"

    @stop
    def proceed(note: str = "") -> str:
        """Go on with what has been found. note: one short sentence on what is missing, or empty."""
        record["note"] = note
        return "done"

    return [retry_search, proceed]


def recover(report_text: str, group: str, question: str, plan: dict, run_agent=None) -> dict:
    """The agent's answer to an empty stage: {'retries': [validated searches], 'note': text}. Any failure means no retries (the report then says what is missing)."""
    import recovery
    record: dict = {}
    runner = run_agent or (lambda system, prompt, tools: default_run_agent(system, prompt, tools, recovery.MAX_RETRIES + 1))
    try:
        runner("=== SKILL: clinical-research-orchestrator ===\n" + skill_excerpt() + "\n" + RECOVERY_GUIDE, report_text, recovery_tools(record))
    except Exception:
        pass
    return {"retries": recovery.clean_retries(record.get("retries"), group, question, plan), "note": " ".join(str(record.get("note") or "").split())[:200]}
