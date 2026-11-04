"""Deterministic checks of a research plan against the searches written from it and against the report written from the evidence, so a plan that drifts is visible (in the saved state and the log)
instead of showing up only as a poor answer. Nothing here changes the plan or the searches: the checks only observe. Every check reads the question's own words, the plan and the operations;
there are no lists of conditions, medicines or populations."""
import re

import retrieval_routing as rr

MEDICINE_NEEDS = {"dosing", "drug_interaction", "regulatory_status"}      # information needs that are about medicines themselves
MEDICINE_SKILL, TRIAL_SKILL = "drug-intelligence", "clinical-trials"
LITERATURE_TARGETS = {"pubmed", "europepmc"}


def _stems(text: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[a-z]{4,}", str(text).lower())}


def _finding(check: str, severity: str, detail: str) -> dict:
    return {"check": check, "severity": severity, "detail": detail[:240]}


def medicines_justified(plan: dict) -> bool:
    """Medicine searches are justified by a medicine the question names, by the plan saying medicines are needed, or by an information need about medicines."""
    understanding = plan.get("understanding") or {}
    return bool(plan.get("drugs")) or understanding.get("medicines_needed") is True or bool(MEDICINE_NEEDS & set(understanding.get("info_needed") or []))


def trials_justified(plan: dict) -> bool:
    understanding = plan.get("understanding") or {}
    return "trial_status" in (understanding.get("info_needed") or []) or plan.get("intent") == "clinical_trial" or "clinical_trial" in (plan.get("secondary_intents") or [])


def audit_plan(question: str, plan: dict, ops: list, skills: list[str] | None = None) -> list[dict]:
    """Findings about the plan and its searches before retrieval results are known. An empty list means every check passed."""
    findings: list[dict] = []
    understanding = plan.get("understanding") or {}
    chosen = list(skills if skills is not None else plan.get("skills") or [])
    asked = _stems(question)

    if plan.get("plan_status") == "fallback":
        findings.append(_finding("plan_failed", "high", "the question could not be analysed; the searches use the question's own words"))
    if not understanding.get("info_needed"):
        findings.append(_finding("no_information_needs", "medium", "the plan names no information need, so nothing can be checked for coverage"))

    drug_ops = [o for o in ops if o.skill == MEDICINE_SKILL or o.target == "fda_label"]
    if (MEDICINE_SKILL in chosen or drug_ops) and not medicines_justified(plan):
        findings.append(_finding("unneeded_medicine_search", "medium", f"{len(drug_ops)} medicine search(es) although the question names no medicine and the plan does not need medicines"))
    trial_ops = [o for o in ops if o.skill == TRIAL_SKILL or o.target == "trials"]
    if (TRIAL_SKILL in chosen or trial_ops) and not trials_justified(plan):
        findings.append(_finding("unneeded_trial_search", "medium", f"{len(trial_ops)} trial search(es) although no trial information is needed"))

    served = {o.serves for o in ops if o.serves} | {n for o in ops for n in getattr(o, "also_serves", [])}
    for need in understanding.get("info_needed") or []:
        if need not in served:
            findings.append(_finding("need_without_search", "medium" if ops else "low", f"no search says it serves the information need '{need}'"))
    for o in ops:
        if o.origin != "baseline" and not (o.need_id and o.need):
            findings.append(_finding("search_without_need", "medium", f"a {o.target} search is not tied to an information need"))

    seen: list = []
    for o in ops:
        twin = next((p for p in seen if p.target == o.target and rr.near_duplicate(p, o)), None)
        if twin:
            findings.append(_finding("duplicate_search", "low", f"two near-identical {o.target} searches: '{o.query[:70]}'"))
        else:
            seen.append(o)

    patient = [g for g in re.split(r"\)\s+AND\s+\(", str(plan.get("pubmed_query") or ""))[1:2]]      # the planner's patient-group part of the query
    patient_stems = _stems(patient[0]) if patient else set()
    if patient_stems:
        lost = [o for o in ops if o.target in LITERATURE_TARGETS and o.origin != "baseline" and not (_stems(o.query) & patient_stems)]
        if lost:
            findings.append(_finding("patient_context_lost", "medium", f"{len(lost)} literature search(es) leave out the patient group: '{lost[0].query[:70]}'"))

    condition = _stems(plan.get("condition"))
    own = asked | {s for v in (plan.get("semantic_variants") or []) + (plan.get("related_topics") or []) for s in _stems(v)}
    if condition and len(condition - own) / len(condition) > 1 / 3:
        findings.append(_finding("condition_not_in_question", "high", f"the planner's condition uses words the question does not: {sorted(condition - own)[:4]}"))
    return findings


def audit_needs(needs: dict) -> list[dict]:
    """Findings after retrieval: an essential need that no relevant evidence covered (the escalation record says why). The 'question as asked' search is a safety net, not an information need: it is
    reported only when no other need was covered either."""
    real = {k: v for k, v in (needs or {}).items() if k != "baseline"}
    if len(real) < len(needs or {}) and not any(v.get("state") == "covered" for v in real.values()):
        real = dict(needs)
    return [_finding("essential_need_uncovered", "high", f"'{v.get('need')}' ({k}): {v.get('state')}{' - ' + v['reason'] if v.get('reason') else ''}")
            for k, v in real.items() if v.get("priority") == "essential" and v.get("state") != "covered"]


def _section(markdown: str, heading: str) -> str:
    match = re.search(r"^## " + re.escape(heading) + r"\s*\n(.*?)(?=^## |\Z)", markdown or "", re.S | re.M)
    return match.group(1).strip() if match else ""


def section_decisions(plan: dict, markdown: str) -> dict:
    """What the report did with the two conditional sections, against what the plan called for. A section is 'filled' when it holds a table or list, 'not_applicable' when it says it does not apply."""
    def state(body: str) -> str:
        if not body:
            return "missing"
        return "filled" if re.search(r"^\s*(\||[-*]\s|\d+\.)", body, re.M) else ("not_applicable" if re.search(r"not applicable|not listed|does not (ask|call|apply)", body, re.I) else "text")
    out = {}
    for name, heading, justified in (("landscape", "Treatment Drug Landscape", medicines_justified(plan)), ("trials", "Clinical trials", trials_justified(plan))):
        body_state = state(_section(markdown, heading))
        out[name] = {"called_for": justified, "state": body_state,
                     "ok": body_state != "missing" and (justified or body_state != "filled") and (body_state != "not_applicable" or not justified)}
    return out
