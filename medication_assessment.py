"""The medication assessment: a stage that runs for EVERY clinical question, whatever its primary intent (diagnosis, work-up, management, treatment ...).

It does not decide what to prescribe. It reads the medicines the retrieved sources describe (official FDA labels, guideline / review / web pages, papers), copies their fields unchanged,
says what the sources do not state, and ends in exactly one of four explicit outcomes:

  identified      - the sources describe at least one medicine relevant to the question (shown with the fields below)
  none_indicated  - a retrieved source states that no medicine treats the presenting problem (the sentence is quoted); comorbidities and cause-specific treatment are still reported
  insufficient    - the medicine sources failed or gave nothing usable, so nothing can be concluded (a failed search is never read as "no medicine exists")
  need_info       - medicines were found, but safe patient-specific dosing needs facts the question does not give (weight, age ...)

Nothing here names a medicine or a dose: every name, dose and warning comes from a retrieved source, and a field the sources do not state says so."""
import re

OUTCOMES = {
    "identified": "Relevant medication identified in the retrieved evidence",
    "none_indicated": "No specific medication is established for the presenting problem",
    "insufficient": "Evidence insufficient to conclude",
    "need_info": "More patient information is required for safe dosing",
}
NOT_STATED = "Not stated in the retrieved sources"
FIELDS = [("indication", "Indication / target"), ("eligibility", "Patient eligibility and age"), ("dose", "Dose"), ("dose_basis", "Dosing basis"), ("max_dose", "Maximum dose"),
          ("route", "Route"), ("frequency", "Frequency and administration"), ("duration", "Duration and reassessment"), ("contraindications", "Contraindications"),
          ("precautions", "Precautions and boxed warning"), ("adverse_effects", "Adverse effects"), ("interactions", "Interactions"), ("monitoring", "Monitoring"),
          ("approval", "Approval status"), ("citation", "Source"), ("evidence", "Evidence strength and limits")]
WEIGHT = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:kg|kilograms?)\b", re.I)
PER_KG = re.compile(r"\bmg\s*/\s*kg\b|\bper\s+kg\b|\bmcg\s*/\s*kg\b|\bunits?\s*/\s*kg\b|\bbody weight\b", re.I)
FREQUENCY = re.compile(r"\b(?:once|twice|three times|four times|daily|every \d+|at bedtime|each (?:morning|evening)|with (?:or without )?food|divided)\b", re.I)
CORE_NEGATIVE = re.compile(r"\b(?:no|not|neither|nor|none|cannot|limited|lack\w*|insufficient|without|only)\b", re.I)
MEDICINE_WORD = re.compile(r"\b(?:medic\w+|drugs?|pharmac\w+|psychotropic\w*)\b", re.I)
ROLES = ("core_no_drug", "cause_specific", "comorbidity_drug")

STATEMENT_TASK = (
    "From the numbered sentences of the sources, pick the sentences that bear on medicines for the patient in the question. For each pick give the source id, the sentence number and one role: "
    "\"core_no_drug\" = the sentence says that medicines do not treat, or are not established for, the core presenting problem or its main features; "
    "\"cause_specific\" = the sentence says a specific treatment exists for a confirmed underlying cause; "
    "\"comorbidity_drug\" = the sentence says a medicine is used for an associated or co-occurring problem. "
    "Pick only a sentence that says this about the question's own condition and patient group, never one about another condition. Pick nothing when no sentence does. "
    "Answer with JSON only: {\"picks\": [{\"id\": \"<source id>\", \"n\": <sentence number>, \"role\": \"<role>\"}]}. Do not write any sentence yourself.")


def patient_facts(question: str, age_years: float | None) -> dict:
    """What the question itself states about the patient: age, weight. Whether the patient is a child follows from the stated age only."""
    m = WEIGHT.search(question or "")
    return {"age_years": age_years, "weight_kg": float(m.group(1)) if m else None, "child": age_years is not None and age_years < 18}


def _squash(text) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _key(name: str) -> str:
    return re.sub(r"\s*\(.*?\)", "", str(name or "")).strip().lower()


def _frequency(text: str) -> str:
    """The first whole sentence of a dosing text that states how often or how to take the medicine (copied unchanged)."""
    for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", _squash(text)):
        if FREQUENCY.search(s):
            return s
    return ""


def _clip(text, limit: int = 420) -> str:
    t = _squash(text)
    if len(t) <= limit:
        return t
    cut = t.rfind(". ", 0, limit)
    return t[:cut + 1] if cut > 80 else t[:limit].rsplit(" ", 1)[0] + " ..."


def source_pool(state: dict, max_sources: int = 16, per_source: int = 8) -> dict:
    """The retrieved sources the statement picker may read: id -> {label, kind, sentences}. Retracted papers are never offered."""
    from landscape import numbered_sentences
    pool = {}
    papers = [p for p in state.get("papers", {}).values() if not p.get("retracted") and p.get("abstract")]
    for p in papers[:10]:
        sentences = numbered_sentences(p["abstract"], 1400)[:per_source]
        if sentences:
            pool[str(p["pmid"])] = {"label": f"PMID {p['pmid']} ({str(p.get('cite', '')).split(' [')[0]})", "kind": "paper", "sentences": sentences}
    for g in state.get("guidelines", [])[:4]:
        sentences = numbered_sentences(g.get("text") or g.get("excerpt") or "", 1400, complete_only=True)[:per_source]
        if sentences and g.get("url"):
            pool[g["url"]] = {"label": f"[{g.get('organisation', 'guideline')}: {str(g.get('title', ''))[:60]}]({g['url']})", "kind": "guideline", "sentences": sentences}
    for w in state.get("web", [])[:10]:
        if w.get("url") and w.get("snippet") and w["url"] not in pool:
            sentences = numbered_sentences(w["snippet"], 900, complete_only=True)[:per_source]
            if sentences:
                pool[w["url"]] = {"label": f"[{str(w.get('title') or 'web page')[:70]}]({w['url']})", "kind": "web", "sentences": sentences}
    return dict(list(pool.items())[:max_sources])


def verify_picks(raw, pool: dict) -> list[dict]:
    """Keep a pick only when its source and sentence number exist, the role is known and the sentence itself fits the role. The sentence shown is the source's own."""
    out, seen = [], set()
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        source = pool.get(str(item.get("id", "")).strip())
        try:
            n = int(item.get("n"))
        except (TypeError, ValueError):
            continue
        role = item.get("role")
        if not source or role not in ROLES or not 1 <= n <= len(source["sentences"]) or (source["label"], n) in seen:
            continue
        sentence = source["sentences"][n - 1]
        if role == "core_no_drug" and not (CORE_NEGATIVE.search(sentence) and MEDICINE_WORD.search(sentence)):
            continue
        if role != "core_no_drug" and not MEDICINE_WORD.search(sentence) and not re.search(r"\b(?:treat\w*|therapy|therapies|indicated|used (?:for|to)|prescrib\w*|approved|recommended)\b", sentence, re.I):
            continue
        seen.add((source["label"], n))
        out.append({"role": role, "sentence": sentence, "source": source["label"]})
    return out


def pick_statements(question: str, state: dict, ask) -> tuple[list[dict], str | None]:
    """The model picks sentence numbers; the code verifies them. `ask(question, task, payload, max_tokens)` returns a parsed JSON object. Returns (verified picks, error)."""
    pool = source_pool(state)
    if not pool:
        return [], None
    payload = {"condition": (state.get("entities") or {}).get("condition") or question,
               "sources": [{"id": k, "title": v["label"][:90], "sentences": [f"{i}. {s}" for i, s in enumerate(v["sentences"], 1)]} for k, v in pool.items()]}
    try:
        answer = ask(question, STATEMENT_TASK, payload, 600)
    except Exception as error:
        return [], f"{type(error).__name__}: {str(error)[:80]}"
    return verify_picks((answer or {}).get("picks"), pool), None


def _approval(record: dict, kind: str) -> str:
    if kind == "label":
        when = record.get("label_date") or "date not retrieved"
        if record.get("model_relation") == "treats_associated_symptom_or_comorbidity":
            return f"FDA-labelled for an associated symptom or comorbidity, not for the core condition (label effective {when})"
        return f"FDA-labelled for the use shown (label effective {when})"
    return "No FDA-labelled indication for this condition in the retrieved labels: standard-care or off-label use as the source describes it"


def entries_from_state(state: dict, used_rows: list[dict], facts: dict) -> list[dict]:
    """One entry per medicine: official-label records first, then the medicines the retrieved sources describe as used. Each field is copied from a source or left empty."""
    from landscape import age_note, fda_relevant, label_age_range
    entries: dict[str, dict] = {}
    for r in state.get("regulatory", []):
        if not fda_relevant(r):
            continue
        extras = r.get("label_extras") or {}
        dose_text = r.get("dose_summary") or r.get("dose_statement") or r.get("dose_general") or ""
        span = label_age_range(" ".join(str(r.get(k) or "") for k in ("indication_excerpt", "indication_statement", "dose_statement")))
        age = age_note(r, facts.get("age_years")).strip(" ()") if facts.get("age_years") is not None else ""
        eligibility = age or (f"labelled for ages {span[0]:g} to {span[1]:g} years" if span and span[1] < 120 else (f"labelled for ages {span[0]:g} years and older" if span else ""))
        boxed = r.get("boxed_warning") or next((d["boxed_warning"] for d in state.get("drug_labels", []) if d.get("boxed_warning") and str(d.get("drug", "")).lower().startswith(r["generic"].lower())), "")
        precautions = " ".join(x for x in ((f"Boxed warning: {_clip(boxed, 300)}" if boxed else ""), extras.get("precautions") or "") if x)
        entries[_key(r["generic"])] = {
            "drug": r["generic"], "kind": "label", "indication": _clip(r.get("indication_statement") or r.get("model_evidence") or r.get("indication_excerpt"), 360),
            "eligibility": eligibility.strip("; "), "dose": _clip(dose_text, 520), "dose_basis": ("weight-based (per kg), as the label states" if PER_KG.search(dose_text) else ""),
            "max_dose": extras.get("max_dose") or "", "route": r.get("route") or "", "frequency": _frequency(dose_text),
            "duration": extras.get("duration") or "", "contraindications": extras.get("contraindications") or "", "precautions": precautions,
            "adverse_effects": extras.get("adverse_effects") or "", "interactions": extras.get("interactions") or "", "monitoring": extras.get("monitoring") or "",
            "approval": _approval(r, "label"), "citation": f"[DailyMed label]({r['url']})" if r.get("url") else "FDA label", "evidence": "Official FDA label: shows approval and labelled use, not how well the drug works compared with others",
            "weight_based": bool(PER_KG.search(dose_text))}
    for r in used_rows:
        key = _key(r.get("drug"))
        if key in entries:
            if r.get("label"):
                entries[key]["citation"] += f"; {r['label']}"
            continue
        dose = r.get("dose") or ""
        pop = "" if str(r.get("population", "")).startswith("not stated") else r.get("population", "")
        entries[key] = {
            "drug": re.sub(r"\s*\(.*?\)", "", str(r.get("drug") or "")).strip() or str(r.get("drug")), "kind": "source", "indication": _clip(r.get("purpose"), 300), "eligibility": pop,
            "dose": _clip(dose, 400), "dose_basis": ("weight-based (per kg), as the source states" if PER_KG.search(dose) else ""), "max_dose": "", "route": "", "frequency": _frequency(dose),
            "duration": "", "contraindications": "", "precautions": "", "adverse_effects": "", "interactions": "", "monitoring": "", "approval": _approval(r, "source"),
            "citation": r.get("label") or "retrieved source", "evidence": f"{r.get('category') or 'Described by a retrieved source'}: " + ("the source states it for the condition itself" if r.get("relevance") == "treats_condition" else "the source states it for an associated problem or does not tie it to the condition itself"),
            "weight_based": bool(PER_KG.search(dose))}
    return list(entries.values())


def drug_sources_incomplete(state: dict, skipped: list[str]) -> bool:
    from landscape import family_status
    if any(str(s).startswith(("drugs used in practice", "regulator announcements", "FDA labels", "drug landscape", "dose lookup")) for s in skipped):
        return True
    return any(family_status(state, fam).get("retrieval_status") in ("failed", "partial") for fam in ("fda_label", "drugs"))


def assess(state: dict, question: str, used_rows: list[dict], picks: list[dict], skipped: list[str], pick_error: str | None = None) -> dict:
    """Decide the outcome. A failed or partial source is never read as 'no medicine exists'."""
    from landscape import patient_age_years
    facts = patient_facts(question, patient_age_years(question))
    entries = entries_from_state(state, used_rows, facts)
    incomplete = drug_sources_incomplete(state, skipped) or bool(pick_error)
    core = [p for p in picks if p["role"] == "core_no_drug"]
    needs = []
    weight_based = any(e["weight_based"] for e in entries)
    if entries and facts["weight_kg"] is None and (weight_based or facts["child"]):
        needs.append("the patient's current weight in kg (" + ("the retrieved doses are weight-based" if weight_based else "paediatric dosing is individualised") + ")")
    if entries and facts["age_years"] is None:
        needs.append("the patient's age (the retrieved labels state age limits)")
    if entries:
        needs.append("kidney and liver function, current medicines and allergies (for interactions and contraindications)")
    if entries and weight_based and facts["weight_kg"] is None:
        outcome = "need_info"
    elif entries:
        outcome = "identified"
    elif incomplete:
        outcome = "insufficient"
    elif core:
        outcome = "none_indicated"
    else:
        outcome = "insufficient"
    return {"outcome": outcome, "label": OUTCOMES[outcome], "entries": entries, "picks": picks, "facts": facts, "needs": needs, "incomplete": incomplete,
            "failures": [s for s in skipped if re.match(r"(drugs used in practice|regulator announcements|FDA labels|drug landscape|dose lookup)", str(s))][:4] + ([f"statement review: {pick_error}"] if pick_error else []),
            "sources_read": len(source_pool(state))}


def fallback(reason: str) -> dict:
    """The assessment itself could not be completed: the report still carries the section, and says it proves nothing about whether a medicine exists."""
    return {"outcome": "insufficient", "label": OUTCOMES["insufficient"], "entries": [], "picks": [], "facts": {}, "needs": [], "incomplete": True, "failures": [reason], "sources_read": 0}


def summary_line(a: dict) -> str:
    names = ", ".join(e["drug"] for e in a["entries"][:5])
    return f"**Medication assessment:** {a['label']}" + (f" ({names})." if names and a["outcome"] in ("identified", "need_info") else ".")


def _entry_block(e: dict) -> list[str]:
    lines = [f"#### {e['drug']}", "", "| Field | Information |", "|---|---|"]
    missing = []
    for key, label in FIELDS:
        value = _squash(e.get(key))
        if value:
            lines.append(f"| {label} | {value.replace('|', '/')} |")
        elif key not in ("citation", "evidence", "approval"):
            missing.append(label.lower())
    if missing:
        lines.append(f"\n*{NOT_STATED}: {', '.join(missing)}.*")
    return lines


def markdown(a: dict, primary: bool = False, compact: bool = False) -> str:
    """The section printed in every report. `primary` is True when medicines are what the question asks about (then the Treatment Drug Landscape already comes first)."""
    out = ["## Medication assessment", "", f"**Outcome: {a['label']}.**"]
    explanation = {
        "identified": "The retrieved sources describe the medicines below. Each field is copied from its source; a field no source states is listed as not stated, and nothing is filled in from memory.",
        "need_info": "The retrieved sources describe the medicines below, but their dosing depends on patient facts that the question does not give. No dose is recommended for this patient here.",
        "none_indicated": "A retrieved source states that medicines do not treat this problem (quoted below). Treatment of associated conditions or of a confirmed cause is a separate question and is covered below where sources describe it.",
        "insufficient": "The retrieved sources do not allow a conclusion about medicines for this problem. This is not evidence that no medicine exists.",
    }[a["outcome"]]
    out += ["", explanation]
    core = [p for p in a["picks"] if p["role"] == "core_no_drug"]
    cause = [p for p in a["picks"] if p["role"] == "cause_specific"]
    comorbid = [p for p in a["picks"] if p["role"] == "comorbidity_drug"]
    for title, items in (("Medicines and the core problem", core), ("Treatment of a confirmed cause", cause), ("Associated problems (if present in this patient)", comorbid)):
        if items:
            out += ["", f"### {title}", ""] + [f"- \"{p['sentence']}\" ({p['source']})" for p in items[:3]]
    if a["entries"]:
        out += ["", "### Medicines in the retrieved evidence", ""]
        if compact:      # medicines were not what the question asked: one line per medicine, one shared note
            for e in a["entries"][:5]:
                bits = [_squash(e.get(k)) for k in ("indication", "approval") if _squash(e.get(k))]
                out.append(f"- **{e['drug']}**: " + "; ".join(bits).replace("|", "/") + (f" ({_squash(e.get('citation'))})" if _squash(e.get("citation")) else ""))
            if len(a["entries"]) > 5:
                out.append(f"- {len(a['entries']) - 5} more medicine(s) are in the Treatment Drug Landscape.")
            out += ["", f"*{NOT_STATED} for these medicines: dose, route, contraindications, adverse effects and monitoring; see the Treatment Drug Landscape and the FDA label before any use.*"]
        else:
            for e in a["entries"][:8]:
                out += _entry_block(e) + [""]
            if len(a["entries"]) > 8:
                out.append(f"*{len(a['entries']) - 8} more medicine(s) are in the Treatment Drug Landscape.*")
    if a["needs"]:
        out += ["", "### Patient information needed for safe dosing", ""] + [f"- {n}" for n in a["needs"]]
    if a["incomplete"] or a["outcome"] == "insufficient":
        out += ["", "### What could not be verified", ""]
        out += [f"- {f}" for f in a["failures"]] or ["- No retrieved source stated which medicines are used for this problem."]
        out.append("- A search that returns nothing does not show that no medicine exists.")
    out += ["", "*A medicine is not recommended here merely because the child has these symptoms: prescribing needs a clinical assessment, the patient's weight and history, and the treating clinician's judgement.*"
            if a["facts"].get("child") else "*Prescribing needs a clinical assessment of the individual patient and the treating clinician's judgement.*"]
    return "\n".join(out).strip()


def has_section(markdown_text: str) -> bool:
    return bool(re.search(r"^##\s+Medication assessment\b", markdown_text or "", re.M | re.I))
