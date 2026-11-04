"""What the assistant remembers inside one conversation, so a follow-up such as 'give the dosage of all the above drugs' can be answered.

After each answer the backend stores, for that conversation: the question, the condition(s) the plan named, the drugs the report listed and a short summary. A later question that refers back
('above', 'those', 'them', ...) is rewritten into one complete question built from that memory: the model only says what the person asks and which earlier answers it points at; the drugs and
conditions in the rewritten question are copied by code from the stored answers, never written by the model. Nothing here names a disease or a drug."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

MEMORY_DIR = Path(__file__).resolve().parent / "conversation_history" / "memory"
MAX_TURNS = 12
MAX_DRUGS = 14
MAX_REPORT_CHARS = 60000      # the full text of one answer kept in memory
CONTEXT_CHARS = int(__import__("os").getenv("CHAT_CONTEXT_CHARS", "40000"))      # how much of the earlier answers a conversational answer may read
SHORT_QUESTION_WORDS = 8
CHATTY = re.compile(r"(?i)^\s*(?:explain|summari[sz]e|simplify|compare|rank|which|why|how come|what do you mean|can you|could you|tell me more|elaborate|clarify|thanks?|thank you|ok(?:ay)?|and |also |what about|so )")

REFERS_BACK = re.compile(r"(?i)\b(?:above|previous(?:ly)?|earlier|before|former|latter|those|these|them|that one|this one|the same|same (?:drugs?|medicines?|medications?|conditions?|patients?|children|child)|"
                         r"mentioned|you (?:listed|gave|said|mentioned|showed)|the (?:first|second|third|last) (?:one|question|drug|medicine|answer)|both (?:of )?(?:them|conditions?|drugs?)|"
                         r"all (?:of )?the (?:drugs?|medicines?|medications?)|it|its|they|their)\b")


def looks_like_follow_up(question: str) -> bool:
    """A cheap check before any model call: does the question point back at something said earlier?"""
    return bool(REFERS_BACK.search(question or ""))


def worth_routing(question: str) -> bool:
    """Is it worth asking the model how to answer? A question that points back, is very short or opens like a conversational request might be answered from the conversation; a long question that
    stands alone is a new piece of research and costs no extra call."""
    return bool((question or "").strip())      # no word list or length rule decides this: with an earlier conversation, the orchestrator agent reads every message and decides whether it is a follow-up


def _path(conversation_id: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "", str(conversation_id or ""))[:64]
    return MEMORY_DIR / f"{safe}.json" if safe else MEMORY_DIR / "_none.json"


def load(conversation_id: str) -> list[dict]:
    try:
        data = json.loads(_path(conversation_id).read_text(encoding="utf-8"))
        return [t for t in data if isinstance(t, dict)][-MAX_TURNS:]
    except (OSError, ValueError):
        return []


def save(conversation_id: str, turns: list[dict]) -> None:
    if not re.sub(r"[^a-zA-Z0-9_-]", "", str(conversation_id or "")):
        return
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    _path(conversation_id).write_text(json.dumps(turns[-MAX_TURNS:], ensure_ascii=False, indent=1), encoding="utf-8")


def clean_drug(name) -> str:
    text = re.sub(r"\([^)]*\)", " ", str(name or ""))
    text = " ".join(text.replace("*", " ").split()).strip(" ,;:.")
    generic = re.search(r"(?i)^(?:none|not |no |the |a |an |this |these |those |patients?\b|children\b|child\b|adults?\b)|\bpatients?\b", text)      # 'the patient' is a person, not a drug
    return text if 2 < len(text) <= 60 and not generic else ""


def unique(items, limit: int) -> list[str]:
    out: list[str] = []
    for item in items:
        name = clean_drug(item)
        if name and name.lower() not in {x.lower() for x in out}:
            out.append(name)
    return out[:limit]


def asked_drugs_of(question: str, state: dict) -> list[str]:
    """The drugs the question ITSELF names (the plan's drugs that are written in the question). A comparison of two named drugs is about those two, not about every drug the report went on to list."""
    squashed = " ".join(re.findall(r"[a-z0-9]+", (question or "").lower()))
    plan = state.get("entities") or {}
    named = [d for d in plan.get("drugs") or [] if " ".join(re.findall(r"[a-z0-9]+", str(d).lower())) and " ".join(re.findall(r"[a-z0-9]+", str(d).lower())) in squashed]
    return unique(named, MAX_DRUGS)


def subject_drugs(turn: dict) -> list[str]:
    """The drugs a follow-up that points at this answer is about: the ones the question named, or (when it named none, such as a question about a condition) the drugs the answer listed."""
    return list(turn.get("asked_drugs") or turn.get("drugs") or [])


def case_of(state: dict) -> dict:
    """The case this answer was about, as fields: what the agent understood of the patient or population (facts, age, main problem, purpose). Nothing is guessed here; an empty field stays empty."""
    entities, understanding = state.get("entities") or {}, state.get("understanding") or {}
    facts: list[str] = []
    for item in list(understanding.get("case_facts") or []) + list(entities.get("associated_findings") or []):
        text = " ".join(str(item or "").split())[:120]
        if 3 <= len(text) and text.lower() not in {f.lower() for f in facts}:
            facts.append(text)
    age = state.get("patient_age_years")
    return {"facts": facts[:10], "age_years": age if isinstance(age, (int, float)) else None, "population": " ".join(str(entities.get("population") or "").split())[:80],
            "main_problem": " ".join(str(entities.get("main_problem") or "").split())[:100], "purpose": " ".join(str(understanding.get("purpose") or "").split())[:220]}


def conclusions_of(markdown: str) -> list[str]:
    """The key conclusions an answer stated: the sentences of its summary (the bottom line), at most four."""
    summary = re.search(r"(?s)## Clinical summary\s*(.*?)(?:\n## |\Z)", markdown or "")
    text = re.sub(r"\*\*[^*]+\*\*", " ", summary.group(1)) if summary else ""
    sentences = [" ".join(s.split()) for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 5]
    return [s[:240] for s in sentences[:4]]


def case_profile(memory: list[dict], chosen: list[int] | None = None) -> dict:
    """The facts of the conversation's cases, merged over the earlier answers (a follow-up adds none of its own), each with the question it came from, so the agent can point at the right earlier answer."""
    facts: list[dict] = []
    age, population = None, ""
    for i, t in enumerate(memory, 1):
        if chosen is not None and i not in chosen:
            continue
        case = t.get("case") or {}
        if t.get("follow_up") or not isinstance(case, dict):
            continue
        for text in case.get("facts") or []:
            if str(text).lower() not in {f["text"].lower() for f in facts}:
                facts.append({"text": str(text), "from": i})
        age = case.get("age_years") if case.get("age_years") is not None else age
        population = case.get("population") or population
    return {"facts": facts, "age_years": age, "population": population}


def profile_line(profile: dict) -> str:
    facts = "; ".join(f"{f['text']} (question {f['from']})" for f in profile.get("facts") or [])
    return ("CASE FACTS: " + facts if facts else "") + (f" AGE: {profile['age_years']:g}" if profile.get("age_years") is not None else "") + (f" POPULATION: {profile['population']}" if profile.get("population") else "")


def turn_from_final(question: str, asked_as: str, final: dict, follow_up: bool = False) -> dict:
    """One remembered turn, built from the finished report's research state."""
    state = final.get("state") or {}
    plan = state.get("entities") or {}
    drugs = unique(list(state.get("listed_drugs") or []) + [r.get("generic") for r in state.get("regulatory", []) if r.get("model_relevant")], MAX_DRUGS)
    summary = re.search(r"(?s)## Clinical summary\s*(.*?)(?:\n## |\Z)", final.get("markdown") or "")
    return {"question": question, "asked_as": asked_as, "condition": " ".join(str(plan.get("condition") or "").split()), "conditions": [g.get("name") for g in plan.get("conditions") or [] if isinstance(g, dict)],
            "drugs": drugs, "intent": plan.get("intent"), "summary": " ".join((summary.group(1) if summary else "").split())[:500], "markdown": (final.get("markdown") or "")[:MAX_REPORT_CHARS],
            "asked_drugs": asked_drugs_of(question, state), "case": case_of(state), "conclusions": conclusions_of(final.get("markdown") or ""), "follow_up": bool(follow_up), "kind": "report", "at": datetime.now().isoformat(timespec="seconds")}


def chat_turn(question: str, answer: str) -> dict:
    """A remembered conversational answer (no new research)."""
    return {"question": question, "asked_as": question, "condition": "", "conditions": [], "drugs": [], "intent": None, "summary": " ".join(answer.split())[:500], "markdown": answer[:MAX_REPORT_CHARS],
            "kind": "chat", "at": datetime.now().isoformat(timespec="seconds")}


def turns_from_history(history: list[dict]) -> list[dict]:
    """A conversation that has no stored memory (asked before this feature existed): the questions, and the drugs read back from each report's drug table."""
    turns, pending = [], None
    for message in history or []:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "user":
            pending = str(message.get("content") or "")
        elif message.get("role") == "assistant" and pending is not None:
            turns.append({"question": pending, "asked_as": pending, "condition": pending, "conditions": [], "drugs": drugs_in_report(str(message.get("content") or "")), "intent": None, "summary": "",
                          "markdown": str(message.get("content") or "")[:MAX_REPORT_CHARS], "kind": "report"})
            pending = None
    return turns[-MAX_TURNS:]


def drugs_in_report(markdown: str) -> list[str]:
    section = re.search(r"(?s)## Treatment Drug Landscape(.*?)(?:\n## |\Z)", markdown or "")
    names: list[str] = []
    for table in re.finditer(r"(?m)^\| (?:Condition \| )?Drug \|.*\n\|[-| ]+\n((?:\|.*\n?)+)", section.group(1) if section else ""):
        header = table.group(0).splitlines()[0]
        column = 1 if header.startswith("| Condition |") else 0
        for row in table.group(1).splitlines():
            cells = [c.strip() for c in row.strip().strip("|").split("|")]
            if len(cells) > column:
                names.append(re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", cells[column]))
    return unique(names, MAX_DRUGS)


RESOLVE_INSTRUCTIONS = ("You decide how a clinical chat assistant should answer the NEW question, given the numbered earlier questions (their answers are stored). Reply with ONE JSON object only: "
                        "{\"mode\": \"conversation\" when the answer is already in the earlier answers and only needs explaining, summarising, comparing, ranking, simplifying, reformatting or a thank-you "
                        "(for example 'explain that simply', 'which of those is safest', 'summarise the above'); \"research\" when it depends on the earlier answers but needs NEW evidence "
                        "(for example the dosage of the above drugs, their side effects, a trial for one of them); \"new\" when it can be understood alone and is a new question, "
                        "\"turns\": the numbers of the earlier questions it refers to, or \"all\", \"ask\": what the person asks for, in the person's own words, WITHOUT naming a drug that the earlier answers listed "
                        "(for example 'Give the dosage information', 'Are they also used to treat <the condition the person names>'), \"needs_drugs\": true when the answer needs the drugs the earlier answers listed, "
                        "\"own_condition\": that condition or topic written as the person wrote it (empty when there is none), "
                        "\"names_own_condition\": true when the new message itself names a condition or topic different from the earlier answers' (for example the earlier answers were about one disease and the person asks if the same drugs treat another)}. "
                        "Never invent a drug or a condition.")


def resolve_payload(question: str, memory: list[dict]) -> str:
    lines = []
    for i, t in enumerate(memory, 1):
        line = f"{i}. {t.get('question')}"
        if t.get("asked_drugs"):
            line += f" (the drugs the question named: {', '.join(t['asked_drugs'])}; other drugs the answer listed: {', '.join(d for d in t.get('drugs') or [] if d not in t['asked_drugs'])})"
        elif t.get("drugs"):
            line += f" (drugs listed: {', '.join(t['drugs'])})"
        if t.get("conclusions"):
            line += " | conclusions: " + " ".join(t["conclusions"])[:400]
        lines.append(line)
    profile = profile_line(case_profile(memory))
    latest = f"\nTHE MOST RECENT EARLIER QUESTION IS NUMBER {len(memory)}: 'the above', 'the previous question', 'this', 'that condition' or 'the last answer' mean it, unless the new message names another earlier question by its own words." if memory else ""
    return "EARLIER QUESTIONS:\n" + "\n".join(lines) + (f"\n{profile}" if profile else "") + latest + f"\n\nNEW QUESTION: {question}"


def parse_reply(raw: str) -> dict:
    found = re.search(r"\{.*\}", raw or "", re.S)
    try:
        data = json.loads(found.group(0)) if found else {}
    except ValueError:
        data = {}
    return data if isinstance(data, dict) else {}


def compose(question: str, memory: list[dict], reply: dict) -> dict:
    """{'follow_up', 'question', 'turns'}: the complete question built from the model's reading and the stored turns. A reply that cannot be used leaves the question as asked."""
    unchanged = {"follow_up": False, "mode": "new", "question": question, "turns": []}
    mode = reply.get("mode") or ("research" if reply.get("follow_up") is True else "new")
    if not memory or mode not in ("research", "conversation"):
        return unchanged
    ask = " ".join(str(reply.get("ask") or "").split()).strip(" .?")
    if mode == "research" and not 3 <= len(ask) <= 200:
        return unchanged
    wanted = reply.get("turns")
    chosen = [i for i in (wanted if isinstance(wanted, list) else []) if isinstance(i, int) and 1 <= i <= len(memory)]
    chosen = chosen or list(range(1, len(memory) + 1))
    if mode == "conversation":
        return {"follow_up": False, "mode": "conversation", "question": question, "turns": chosen, "research_question": _research_question(ask, memory, chosen, reply, question)}
    return {"follow_up": True, "mode": "research", "question": _research_question(ask, memory, chosen, reply, question), "turns": chosen,
            "quick_check": quick_check_request(question, memory, chosen, reply)}


def _research_question(ask: str, memory: list[dict], chosen: list[int], reply: dict, question: str) -> str:
    """One complete question: what the person asks, the drugs of the chosen answers (copied by code) and their conditions."""
    if not 3 <= len(ask) <= 200:
        return question
    used = [memory[i - 1] for i in chosen]
    drugs = unique([d for t in used for d in subject_drugs(t)], MAX_DRUGS) if reply.get("needs_drugs") is True else []
    conditions = unique([t.get("condition") or "" for t in used if t.get("kind") != "chat"], 4)
    if reply.get("names_own_condition") is True:      # the person asks about a different condition: the earlier conditions are not added (they would pull the search back to them)
        return ask + (f" (the drugs: {', '.join(drugs)})" if drugs else "") + "?"

    text = ask.rstrip(" ?.")
    facts = [f["text"] for f in case_profile(memory, chosen)["facts"] if f["text"].lower() not in text.lower()][:5]      # the case of the earlier answers goes with the question: 'her' becomes the patient the answers were about
    if facts:
        text += f" (the case: {'; '.join(facts)})"
    named = any(d.lower() in text.lower() for d in drugs)      # the agent sometimes writes the drugs into `ask` itself: they are not added a second time
    if drugs and not named:
        text += " for " + ", ".join(drugs)
    missing = [c for c in conditions if c.lower() not in text.lower()]
    if missing:
        text += (" used for " if drugs and not named else " for ") + " and for ".join(missing)
    return text + "?"


def quick_check_request(question: str, memory: list[dict], chosen: list[int], reply: dict) -> dict | None:
    """When the person asks whether the earlier drugs are also used for ANOTHER condition: {'drugs', 'condition'} for a quick first answer. The condition must be written in the message itself."""
    if reply.get("names_own_condition") is not True or reply.get("needs_drugs") is not True:
        return None
    condition = " ".join(str(reply.get("own_condition") or "").split())[:80]
    import conditions as cond_groups
    words = set(cond_groups._words(question))
    if not condition or not any(cond_groups._matches(s, words) for s in cond_groups.stems_of(condition)):
        return None
    drugs = unique([d for i in chosen for d in subject_drugs(memory[i - 1])], MAX_DRUGS)
    return {"drugs": drugs, "condition": condition} if drugs else None


def resolve(question: str, memory: list[dict], ask_model) -> dict:
    """The question rewritten as one complete question when it refers back to the conversation. `ask_model(instructions, prompt) -> str`. Any failure leaves the question as asked."""
    if not memory or not worth_routing(question):
        return {"follow_up": False, "mode": "new", "question": question, "turns": []}
    try:
        reply = parse_reply(ask_model(RESOLVE_INSTRUCTIONS, resolve_payload(question, memory)))
    except Exception:
        return {"follow_up": False, "mode": "new", "question": question, "turns": []}
    return compose(question, memory, reply)


# ---------------------------------------------------------------- a conversational answer, from what was already said
CHAT_INSTRUCTIONS = ("You are NeuroGPT, a clinical research assistant, continuing a conversation. Answer the person's new message using ONLY the EARLIER ANSWERS below, which you wrote from retrieved evidence. "
                     "You can explain, summarise, compare, rank, pick out (for example the safest or the first-line drug among those listed), simplify, reformat, or draft something from them (a patient handout, a note), whatever the person asks. "
                     "Do not add facts, doses, drug names or numbers that are not in the earlier answers. Do not use bracketed reference numbers: say which earlier answer a point comes from (for example 'from the earlier answer on <topic>'). "
                     "ALWAYS ANSWER IN THIS STRUCTURE (skip it only for a greeting or a thank-you, which gets one friendly sentence): "
                     "(1) **Direct answer**: one or two sentences that answer exactly what was asked. "
                     "(2) The detail under short bold headings, in the form that fits: a table for comparisons and for anything per drug (dose, side effects, evidence), numbered steps for a how-to, bullets for lists. "
                     "(3) **Where this comes from**: which earlier answer(s) it is taken from. "
                     "(4) Only if part of the message is not in the earlier answers: **Not in the earlier answers**: what is missing, in one line, and that you can research it if asked. "
                     "Answer the part you can. This is research support, not a substitute for clinical judgement. "
                     "If the earlier answers contain almost nothing of what is needed, reply with exactly NEEDS_RESEARCH and nothing else.")
NEEDS_RESEARCH = "NEEDS_RESEARCH"


def build_context(memory: list[dict], chosen: list[int], budget: int = CONTEXT_CHARS) -> str:
    """The chosen earlier answers, newest first, as far as the budget allows (a long answer is cut at the end, never in the middle of the budget for the others)."""
    blocks, used = [], 0
    for i in sorted(chosen, reverse=True):
        t = memory[i - 1]
        text = str(t.get("markdown") or t.get("summary") or "")
        room = budget - used
        if room <= 500:
            break
        text = text[:room]
        blocks.append(f"EARLIER ANSWER {i} (question: {t.get('question')})\n{text}")
        used += len(text)
    return "\n\n".join(reversed(blocks))


def chat_prompt(question: str, memory: list[dict], chosen: list[int]) -> str:
    return build_context(memory, chosen) + f"\n\nNEW MESSAGE: {question}"


def unmatched_numbers(answer: str, context: str) -> list[str]:
    """Numbers in a conversational answer that the earlier answers do not contain (a dose or a percentage that was not there): reported, never silently kept."""
    known = set(re.findall(r"\d+(?:[.,]\d+)?", context))
    found = []
    for n in re.findall(r"\d+(?:[.,]\d+)?", answer):
        if (len(n) >= 2 or "." in n) and n not in known and n not in found:
            found.append(n)
    return found[:8]
