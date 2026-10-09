"""Deterministic sections built from the ResearchState (no model, nothing invented).

    record_source_status   retrieval_status (success|partial|failed) and evidence_status (found|not_found_after_search|unavailable) per source
    top_drugs_markdown     the Phase-1 'Top treatment drugs' table
    verify_*_decisions     the drug-intelligence model's classifications, accepted only when the quote is found in the cited source
    trials_markdown        treatment/intervention trials separated from peripheral registry records; status wording that never implies efficacy
    limitations_markdown   keeps 'source failed' apart from 'searched and nothing found'
    drug_intelligence_status
"""
from __future__ import annotations

import json
import re
from urllib.parse import quote

def _json(text):
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return {"error": str(text)[:200]}


# ------------------------------------------------------------------ retrieval status semantics
def record_source_status(state: dict, raw: dict) -> None:
    """success|partial|failed  x  found|not_found_after_search|unavailable. A timeout or failure is NEVER recorded as 'no evidence'."""
    for key, text in raw.items():
        data = _json(text)
        error = data.get("error") if isinstance(data, dict) else None
        if error:
            partial = "timed out" in str(error).lower()
            state["source_status"][key] = {"retrieval_status": "partial" if partial else "failed", "evidence_status": "unavailable", "error": str(error)[:160]}
            continue
        n = len(data) if isinstance(data, list) else len(data.get("results", []))
        state["source_status"][key] = {"retrieval_status": "success", "evidence_status": "found" if n else "not_found_after_search", "error": None}


def drug_intelligence_status(state: dict, drug_stage_ok: bool) -> str:
    status = state.get("source_status", {})
    reg_key = "fda_deep" if "fda_deep" in status else "fda_fast"
    reg_ok = status.get(reg_key, {}).get("retrieval_status") == "success"
    web = [v for k, v in status.items() if k.startswith("web")]
    web_all_failed = bool(web) and all(v["retrieval_status"] != "success" for v in web)
    if not drug_stage_ok and not reg_ok:
        return "failed"
    if not drug_stage_ok or not reg_ok or web_all_failed:
        return "partial"
    return "complete"


# ------------------------------------------------------------------ regulatory (FDA label) sections
def _cell(text: str, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", str(text)).replace("|", "/").strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " ..."


NUMBER_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10", "twelve": "12", "fourteen": "14"}
NO_META_CLAIM = re.compile(r"\b(?:the\s+)?(?:label|text|document|source)\s+(?:does not|doesn't|did not|has no)\b|\bnot specified\b|\bno (?:maximum|upper limit)\b|\bnot stated\b|"
                           r"\b(?:not\s+)?mentioned\b|\bin the (?:text|label)\b|\baccording to\b|\bno [\w-]+(?:\s[\w-]+)? (?:adjustments?|data|information) (?:are|is) (?:given|provided|available)\b", re.I)


def _numbers(text: str) -> set[str]:
    """Every number in a text (digits, and number words such as 'four'), normalised so '1.0' and '1' are the same number."""
    found = {str(float(n)).rstrip("0").rstrip(".") for n in re.findall(r"\d+(?:\.\d+)?", text or "")}
    return found | {digit for word, digit in NUMBER_WORDS.items() if re.search(rf"\b{word}\b", text or "", re.I)}


def dose_summary_ok(summary: str, source: str) -> bool:
    """A model-written dose summary is shown only if it is short (at most 4 sentences) and EVERY number in it appears in the label's own dosing text.
    A number the label does not contain means the model invented or altered a dose, so the summary is rejected."""
    from verification import _sentences
    text = re.sub(r"\s+", " ", summary or "").strip()
    sentences = [x for x in _sentences(text) if x.strip()]
    return bool(40 <= len(text) <= 700 and 1 <= len(sentences) <= 4 and _numbers(text) <= _numbers(source) and not re.search(r"\[|\]|\*\*|\|", text)
                and not NO_META_CLAIM.search(text))      # 'the label does not specify a maximum' is a claim about the label that nothing checks: not accepted


def dose_fallback(source: str, limit: int = 420) -> str:
    """When no faithful summary exists: the label's first complete sentences (copied unchanged) up to the limit."""
    from verification import _sentences
    out = ""
    for sentence in _sentences(re.sub(r"\s+", " ", source or "").strip()):
        if out and len(out) + len(sentence) + 1 > limit:
            break
        out = (out + " " + sentence).strip()
    return out


def product_name(r: dict) -> str:
    """The name of an FDA label's product. openFDA joins the active ingredients of a combination product with commas ('Avobenzone, Homosalate, Octisalate'): that is ONE product, shown as
    'Combination product (avobenzone + homosalate + octisalate)', with its brand name first when the label has one."""
    ingredients = [x.strip() for x in r["generic"].split(",") if x.strip()]
    brands = [b for b in r.get("brands", []) if b.lower() != r["generic"].lower()]
    if len(ingredients) < 2:
        return r["generic"] + (f" ({', '.join(brands)})" if brands else "")
    return f"{brands[0] if brands else 'Combination product'} (one product: {' + '.join(x.lower() for x in ingredients)})"


def drug_name(r: dict) -> str:
    """The drug's name, linked to that drug's own DailyMed label page when one was found."""
    return f"[{_cell(r['drug'])}]({r['fda_label']})" if r.get("fda_label") else _cell(r["drug"])


def label_link(r: dict, stems: list[str]) -> str:
    """The DailyMed label link, opened at the label's own wording about this condition: a browser text fragment (#:~:text=) highlights that passage when the page supports it;
    otherwise the label simply opens."""
    url = r.get("url") or ""
    statement = re.sub(r"[•●]+", " ", r.get("indication_statement") or r.get("indication_excerpt") or "")
    words = re.sub(r"[^\w\s'-]", " ", statement).split()
    for i, w in enumerate(words):
        if any(s and s in w.lower() for s in stems):
            phrase = " ".join(words[max(0, i - 3):i + 2])
            return url + "#:~:text=" + quote(phrase, safe="") if url and len(phrase) >= 8 else url
    return url


def _dose_cell(r: dict) -> str:
    """Route and the label's dosing text for THIS condition. Never inferred: when the label has no dosing section for the condition, the cell says so."""
    route = f"Route: {r['route']}. " if r.get("route") else ""
    if r.get("dose_summary"):      # condensed from the label's dosing section; the numbers were checked against the label text
        return route + _cell(r["dose_summary"], 700)
    if r.get("dose_statement"):
        return route + "\"" + _cell(r["dose_statement"], 1600) + "\""
    return route + "No dosing section for this condition was found in the retrieved label text; see the linked label."


CATEGORY_ORDER = ("Approved", "Guideline-recommended", "Off-label", "Emerging")


def landscape_markdown(state: dict, used_rows: list[dict], trial_rows: list[dict] | None = None) -> str:
    """The specification's Treatment Drug Landscape, as TWO tables: drugs used for treatment of this condition, and other drugs (symptom-directed or still in trials). Columns: drug, category, indication, dose / route (if sourced),
    approval status / date, key evidence. Investigational drugs are never listed as established treatment; doses come only from a label or source for the use shown."""
    head = "## Treatment Drug Landscape\n\n"
    status = state.get("source_status", {})
    st = status.get("fda_deep") or status.get("fda_fast") or {}
    reg = [r for r in state.get("regulatory", []) if fda_relevant(r)]
    intro = ""
    if st.get("retrieval_status") in ("failed", "partial") and not reg:
        intro = (f"**The FDA drug-label source could not be retrieved ({st.get('error')}).** The approved-drug part of this landscape is incomplete; "
                 "this does not mean that no approved drug exists.\n\n")
    elif not reg:
        terms = ", ".join(state.get("fda_terms") or []) or "the question's condition terms"
        intro = f"No FDA drug label with an indication matching the condition was found when searched (search terms: {terms}).\n\n"
    stems = [x for x in (state.get("core_stems") or []) if len(x) >= 4]
    entries: list[tuple[int, str, str]] = []
    for r in reg:
        brands = ", ".join(b for b in r.get("brands", []) if b.lower() != r["generic"].lower())
        statement = r.get("indication_statement") or r.get("model_evidence") or r["indication_excerpt"]
        associated = " [associated symptom or comorbidity, not the condition itself]" if r.get("model_relation") == "treats_associated_symptom_or_comorbidity" else ""
        efficacy = re.search(r"Efficacy was established[^.]*\.", statement or "")
        evidence = (f"FDA label: {efficacy.group(0)} " if efficacy else "FDA label. ") + (f"[DailyMed label]({label_link(r, stems)})" if r.get("url") else "")
        entries.append((0, "", f"| {product_name(r)} | \"{_cell(statement, 800)}\"{associated} | {_dose_cell(r)} | "
                           f"FDA-approved for the use shown; label effective {r.get('label_date') or 'date not retrieved'}; original approval date not retrieved | {_cell(evidence, 400)} |"))
    for r in used_rows:
        guideline = r.get("category", "").startswith("Guideline")
        tier = 1 if guideline and r.get("relevance") == "treats_condition" else 2      # a drug for an associated symptom is not a treatment of the condition itself, whoever recommends it
        said = f"\"{_cell(r['excerpt'], 420)}\" "
        pop = "" if r.get("population", "").startswith("not stated") else f" (population: {r['population']})"
        if r.get("relevance") == "treats_associated_symptom_or_comorbidity":
            pop += " [associated symptom or comorbidity, not the condition itself]"
        if guideline:
            pop += " [guideline-recommended]"
        entries.append((tier, f"| {drug_name(r)} | {_cell(r['purpose'], 200)}{pop} | {said}{_cell(r['label'], 500)} |", f"| {drug_name(r)} | {_cell(r['purpose'], 200)}{pop} | "
                              f"Not stated in the retrieved source for this use | {r.get('category') or 'Off-label'}; no FDA-labelled indication for this condition in the retrieved labels | {said}{_cell(r['label'], 500)} |"))
    for r in []:      # trial drugs are not listed here: the Clinical trials section is the only place trials appear
        link = f"[{r['nct']}]({r['url']})" if r.get("url") else r["nct"]
        entries.append((3, f"| {drug_name(r)} | Registered conditions: {_cell(r['conditions'], 160)}. Ages: {_cell(r['ages'], 80)} | Investigational, {r['phase']}; not approved. {r['status']}, last update {r['updated']}. {link}. Primary outcome: {_cell(r['primary'], 200)}. A registry entry does not show that the drug works. |", f"| {drug_name(r)} | Emerging: investigational, {r['phase']} | Registered conditions: {_cell(r['conditions'], 160)}. Ages: {_cell(r['ages'], 80)} | "
                           f"Investigational: trial-specific dosing is in the registry record | Not approved. {r['status']}, last update {r['updated']} | "
                           f"{link}. Primary outcome: {_cell(r['primary'], 200)}. A registry entry does not show that the drug works. |"))
    entries.sort(key=lambda e: e[0])
    columns = ["| Drug | Indication | Dose / route (if sourced) | Approval status / date | Key evidence |", "|---|---|---|---|---|"]
    short_columns = ["| Drug | Indication | Key evidence |", "|---|---|---|"]
    treatment = [e[2] for e in entries if e[0] <= 1]      # Approved, Guideline-recommended: used for the treatment of this condition
    other = [e[1] for e in entries if e[0] >= 2]      # three columns only: drug, indication, key evidence          # Off-label (symptom-directed) and Emerging (still in trials): not established treatments of this condition
    if not treatment:
        treatment = ["| None retrieved | | | | No FDA label or guideline page naming a drug for this condition was retrieved in this run. This does not mean none exists. |"]
    if not other:
        other = ["| None retrieved | | No symptom-directed drug for this condition was retrieved in this run. This does not mean none exists. |"]
    dose_note = ("Doses appear only where an FDA label or source gives them for the use shown" + ("; each dose entry is a short summary of the label's dosing section for this condition, "
                 "with every number checked against the label text (full text in the DailyMed link)" if any(r.get("dose_summary") for r in reg) else "") + "; individual dosing is a prescriber's decision.")
    first = ("### Drugs used for treatment of this condition\n\n"
             "*Approved: an FDA label names this condition. Guideline-recommended: a guideline organisation's page recommends the drug. " + dose_note + "*\n\n" + "\n".join(columns + treatment))
    second = ("### Other drugs: used for symptoms (not established treatments of this condition)\n\n"
              "*Drugs that a retrieved review, guideline or web page describes as used for the condition, its symptoms or co-occurring conditions, for which no FDA label for this use was retrieved. "
              "Drugs still being tested in trials are listed in the Clinical trials section below.*\n\n" + "\n".join(short_columns + other))
    return (head + intro + first + "\n\n" + second + "\n\n" + regulatory_updates_markdown(state) + "\n\n" + safety_alerts_markdown(state.get("drug_labels", []), state.get("listed_drugs")))


def is_late_phase(t: dict, min_phase: int = 2) -> bool:
    """Late-phase = every registered phase is phase 2 or later (so 'Phase 1/2', 'Phase 1' and 'not applicable' are not late-phase)."""
    ranks = [PHASE_RANK.get(p, 0) for p in (t.get("phase") or [])]
    return bool(ranks) and min(ranks) >= min_phase


def notable_latest_drug(state: dict) -> dict | None:
    """'Then the most notable latest drug': the most recently updated late-phase (phase 2 or later) registered drug trial that the model judged to be a treatment trial for the question
    (landscape.verify_trial_triage). Recency and phase are registry facts. Always investigational."""
    approved = {r["generic"].lower() for r in state.get("regulatory", []) if fda_relevant(r)}
    best, best_key = None, None
    for t in state.get("trials_by_id", {}).values():
        rank = max([PHASE_RANK.get(p, 0) for p in (t.get("phase") or [])] or [0])
        triage = t.get("triage") or {}
        if not is_late_phase(t) or triage.get("kind") != "treatment_trial" or triage.get("relevance") not in ("direct", "indirect"):
            continue
        for raw in t.get("drug_interventions") or []:
            name = _trial_drug_name(raw)
            key = (t.get("last_update") or "", rank)      # the most recently updated late-phase drug trial; the phase only breaks a tie
            if name and name.lower() not in approved and (best_key is None or key > best_key):
                best, best_key = {"drug": name, "nct": t["nct_id"], "phase": nice_phase(t.get("phase")), "status": nice_status(t.get("recruitment_status")), "url": t.get("url")}, key
    return best


def top_drugs_markdown(state: dict) -> str:
    """Phase 1 'top treatment drugs': the FDA-labelled drugs of this condition first (built by code from the label records), then the most notable latest drug (investigational)."""
    status = state.get("source_status", {}).get("fda_deep") or state.get("source_status", {}).get("fda_fast", {})
    where = "deep pass" if "fda_deep" in state.get("source_status", {}) else "fast pass"
    reg = [r for r in state.get("regulatory", []) if fda_relevant(r)][:4]
    latest = notable_latest_drug(state)
    rows = ["| Drug | Indication named in the FDA label | Source |", "|---|---|---|"]
    for r in reg:
        link = f"[DailyMed]({r['url']})" if r.get("url") else "FDA label"
        rows.append(f"| {product_name(r)} | \"{_cell(r.get('model_evidence') or r['indication_excerpt'], 420)}\" | {link} |")
    if latest:
        rows.append(f"| {latest['drug']} (latest, investigational) | Being tested in a registered {latest['phase']} trial, {latest['status']}; not an approved treatment | "
                    f"[{latest['nct']}]({latest['url']}) |" if latest.get("url") else f"| {latest['drug']} (latest, investigational) | Being tested in a registered {latest['phase']} trial; not an approved treatment | {latest['nct']} |")
    waiting = [r for r in state.get("regulatory", []) if r.get("model_relevant") is None and (r.get("indication_statement") or r.get("indication_excerpt"))]
    pending = f"\n\n*{len(waiting)} retrieved FDA label(s) were still being read when this table was shown; the drug landscape below lists every label once it has been read.*" if waiting else ""
    if reg or latest:
        return "**Top treatment drugs (established first, then the most notable latest drug)**\n\n" + "\n".join(rows) + pending
    if waiting:
        return f"**Top treatment drugs:** {len(waiting)} retrieved FDA label(s) were still being read when this was shown; the drug landscape below lists them once they have been read."
    if status.get("retrieval_status") in ("failed", "partial"):
        return ("**Top treatment drugs:** the FDA drug-label source was unavailable in the " + where + " (" + str(status.get("error")) +
                "); the full drug landscape follows. This does not mean that no drug exists.")
    return "**Top treatment drugs:** no retrieved FDA label was judged relevant to the question in the " + where + "; the full landscape follows."


LABEL_SPLIT = re.compile(r"\s*[•●]\s*|\s*\(\s*\d+(?:\.\d+)*\s*\)\s*|;\s+|\s+(?=\d+\.\d+\s+[A-Z])")


def condition_clause(excerpt: str, stems: list[str]) -> str:
    """The label's own COMPLETE clause that names the condition (for example 'Irritability Associated with Autistic Disorder'), copied unchanged from the
    indication text; if no short clause names it, the excerpt is cut at a word boundary and marked with '...'."""
    text = re.sub(r"\s+", " ", excerpt or "").strip().replace('"', "'")
    pieces = [p for p in (x.strip(" ,.-") for x in LABEL_SPLIT.split(text) if x) if p]
    named = [p for p in pieces if any(st in p.lower() for st in stems) and len(p) <= 180]
    if named:
        return min(named, key=len) if len(min(named, key=len)) >= 12 else named[0]
    return text if len(text) <= 200 else text[:200].rsplit(" ", 1)[0] + " ..."


def nice_status(status: str | None) -> str:
    """RECRUITING / NOT_YET_RECRUITING -> Recruiting / Not yet recruiting."""
    return re.sub(r"_+", " ", str(status or "status not stated")).strip().capitalize()


def nice_phase(phases) -> str:
    """['PHASE1', 'PHASE2'] -> Phase 1/2; ['NA'] or nothing -> Phase not applicable."""
    nums = [re.sub(r"\D", "", str(p)) for p in (phases or []) if re.search(r"\d", str(p))]
    return "Phase " + "/".join(nums) if nums else "Phase not applicable"


def _clause(text: str, limit: int = 110) -> str:
    """A reason/phrase cut at a clause boundary (never mid-word, no ellipsis)."""
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= limit:
        return text
    cut = max(text.rfind(", ", 0, limit), text.rfind("; ", 0, limit), text.rfind(" and ", 0, limit))
    return text[:cut].rstrip(" ,;") if cut > 30 else text[:limit].rsplit(" ", 1)[0]


def trial_reading(t: dict) -> str:
    """Status wording that never implies efficacy. Registry status is separate from results."""
    s, results = (t.get("recruitment_status") or "").upper(), bool(t.get("results_posted"))
    if s in ("RECRUITING", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION"):
        return "Under investigation; results not yet available; effectiveness cannot be inferred"
    if s == "COMPLETED":
        return "Completed; results posted (outcome not retrieved here)" if results else "Completed; results not posted/retrieved"
    if s == "TERMINATED":
        return f"Terminated early ({_clause(t.get('why_stopped') or 'reason not retrieved')}); " + ("results posted (not retrieved here)" if results else "no results retrieved")
    return f"Registry status {s or 'unknown'}; no effectiveness conclusion"


def is_treatment_trial(t: dict) -> bool:
    """Whether a registry record is shown as a treatment trial. The model's verified triage decides (landscape.verify_trial_triage); with no triage the registry's own metadata is used
    (an interventional study that names an intervention)."""
    triage = t.get("triage")
    if triage:
        return triage["relevance"] in ("direct", "indirect") and triage["kind"] == "treatment_trial"
    return (t.get("design") or "").upper() == "INTERVENTIONAL" and bool(t.get("interventions"))


def trials_markdown(state: dict) -> str:
    """Section 3: registered treatment trials (the table of other registry records is summarised in one line, not printed)."""
    head = "## Clinical trials\n\n"
    trials = list(state.get("trials_by_id", {}).values())[:12]
    if not trials:
        st = state.get("source_status", {}).get("trials_deep", {})
        if st.get("retrieval_status") in ("failed", "partial"):
            return head + "The trial registry could not be retrieved (" + str(st.get("error")) + "); trial status is unavailable. This does not mean that no trials exist."
        return head + "No trials were found in the registry search."
    direct = [t for t in trials if is_treatment_trial(t)]
    other = [t for t in trials if not is_treatment_trial(t)]
    if not direct:
        body = "No interventional treatment trial was identified among the retrieved records."
    else:
        out = ["| Trial | Intervention | Phase | Status (last update) | What the record shows |", "|---|---|---|---|---|"]
        for t in direct:
            note = " **Specific syndrome record; do not generalise to the whole condition.**" if re.search(r"syndrome", t.get("title") or "", re.I) else ""
            link = f"[{t['nct_id']}]({t['url']})" if t.get("url") else t["nct_id"]
            out.append(f"| {link} | {_cell('; '.join((t.get('interventions') or [])[:2]) or (t.get('title') or ''), 90)} | {nice_phase(t.get('phase'))} | "
                       f"{nice_status(t.get('recruitment_status'))} ({t.get('last_update')}) | {trial_reading(t)}.{note} |")
        body = "\n".join(out)
    extra = f"\n\n{len(other)} further registry record(s) were retrieved and are not shown (judged not to be treatment trials for this question)." if other else ""
    return head + body + extra + "\n\n*Registry status shows that research is being conducted, not that a treatment works. Registry status is separate from published results.*"


# ------------------------------------------------------------------ limitations
SOURCE_NAMES = (("pubmed", "PubMed / Europe PMC search"), ("trials", "ClinicalTrials.gov search"), ("fda", "FDA drug-label search"), ("web", "web search"),
                 ("guideline", "guideline-organisation search"), ("deep", "deep-retrieval stage"))


def source_name(key: str) -> str:
    return next((name for prefix, name in SOURCE_NAMES if key.startswith(prefix)), key)


def limitations_markdown(state: dict, skipped: list[str], notes: list[str] | None = None) -> str:
    """Plain-language limits of THIS report. Technical job names and retry details stay in the saved record."""
    items = ["Findings are drawn from abstracts, registry records and web pages, not full texts; study quality was not appraised from the full papers.",
             "Doses appear only where an FDA label gives them for this condition (section 2.1); for any other use, see a prescribing source. Approval dates appear only where a retrieved source states them."]
    items += notes or []
    partial: dict[str, str] = {}
    for key, st in state.get("source_status", {}).items():
        if st["retrieval_status"] != "success":
            partial.setdefault(source_name(key), "timed out" if "timed out" in str(st.get("error")).lower() else "did not respond fully")
    for name, why in partial.items():
        items.append(f"The {name} {why}, so it may be incomplete; the absence of evidence cannot be concluded from it.")
    if state.get("drug_intelligence_status") in ("partial", "failed"):
        items.append("**Drug landscape is incomplete because a regulatory/drug source failed or was only partly retrieved; this does not show that no drug exists.**")
    items += [f"Not completed within the time limit: {x}" for x in skipped]
    return "## Evidence limitations\n\n" + "\n".join(f"- {i}" for i in items)


# ------------------------------------------------------------------ drugs described as USED IN PRACTICE (beyond the FDA-labelled ones)
def _clean_source(text: str) -> str:
    """Whitespace tidied; the '[...]' that search services put between unrelated fragments becomes a sentence break, so fragments are never fused. Markdown heading marks
    ('### Title') that web pages leave in a snippet end a sentence instead of being part of it, and a doubled full stop is one."""
    text = re.sub(r"(?:^|\s)#{1,6}\s", ". ", re.sub(r"\s+", " ", (text or "").replace("[...]", ". "))).strip()
    return re.sub(r"(?<!\.)\.\.(?!\.)", ".", re.sub(r"^\.\s+", "", text))


def complete_sentence(sentence: str, limit: int = 700) -> str:
    """The WHOLE source sentence, from its first word. If the source text itself stops mid-sentence (a cut-off search snippet), it is ended at the last
    complete clause instead of being cut at a random word; no '...' is added."""
    s = sentence.strip()
    cut_in_source = bool(re.search(r"(?:\.\.\.|…)\s*$", s))      # the search service already cut this snippet with an ellipsis
    if cut_in_source:
        s = re.sub(r"\s*(?:\.\.\.|…)\s*$", "", s).rstrip(" ,;:")
    if not cut_in_source and len(s) <= limit and re.search(r"[.!?\"”)\]]\s*$", s):
        return s
    window = s[:limit]
    boundary = max(window.rfind(", "), window.rfind("; "), window.rfind(": "))
    if cut_in_source and len(s) <= limit:
        return s + " (source text ends here)"
    return (window[:boundary] if boundary > 60 else window.rsplit(" ", 1)[0]).rstrip(" ,;:") + ("" if len(s) <= limit and re.search(r"[.!?]\s*$", s) else " (source text ends here)")


# ------------------------------------------------------------------ the model INTERPRETS the retrieved drug evidence; code VERIFIES every answer against the source text
# The model (drug-intelligence skill) says which drugs a source describes as used for the condition, how relevant they are and what category they belong to, and quotes the source.
# Nothing the model writes is shown unless it is found, word for word, in the source it cites. The source stays the evidence.
RELEVANT = ("treats_condition", "treats_associated_symptom_or_comorbidity")
RELATIONS = ("treats_condition", "treats_associated_symptom_or_comorbidity")
PURPOSES = ("treat_or_manage", "prevent", "diagnose")
USED_CATEGORIES = {"guideline-recommended": "Guideline-recommended", "off-label": "Off-label (described as used in practice)"}
REGULATOR_CLASSES = {"approval": "Approval", "label_update": "Label update", "safety_communication": "Safety communication", "other_regulatory_action": "Other regulatory action"}
REGULATOR_RELATIONS = ("treats_condition", "treats_associated_symptom_or_comorbidity", "risk_or_safety_related", "other_indication", "unclear")
PAPER_RELEVANCE = ("direct", "indirect", "background", "not_relevant", "uncertain")
PAPER_ROLES = ("established", "emerging", "experimental", "insufficient_or_negative", "conflicting", "ongoing", "background")
TRIAL_KINDS = ("treatment_trial", "diagnostic_or_assessment", "other")
DRUG_NAME = re.compile(r"[A-Za-z][A-Za-z\-]{3,}(?: [A-Za-z\-]{3,}){0,2}")
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " ", "•": " ", "●": " "})


def fda_relevant(r: dict) -> bool:
    """Is this FDA label shown as a drug for the question? Only on the model's verified decision (landscape.verify_fda_decisions). A label with no verified decision is not shown:
    a retrieved label is not relevant merely because a search found it, and no keyword rule stands in for the decision."""
    return bool(r.get("model_relevant"))


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("[...]", " ").translate(_QUOTES)).strip()


def locate(source_text: str, quote_text: str) -> str | None:
    """The exact span of the source that the model's quote points to. Whitespace, letter case, quotation marks and a leading or trailing '...' are ignored;
    None when the quote (at least 20 characters) is not in the source."""
    source = _norm(source_text)
    whole = re.sub(r"^(?:\.\.\.|…)\s*|\s*(?:\.\.\.|…)$", "", _norm(quote_text).strip(" \"'")).strip()
    if len(whole) < 20:
        return None
    start = end = pos = None
    for piece in (p.strip(" \"',;") for p in re.split(r"\.\.\.|…", whole)):      # a quote may skip text with '...': each piece must be in the source, in order
        if len(piece) < 8:
            continue
        found = re.compile(r"\s+".join(re.escape(w) for w in piece.split(" ")), re.I).search(source, pos or 0)
        if not found:
            return None
        start = found.start() if start is None else start
        end = pos = found.end()
    return source[start:end] if start is not None else None


def _in(text: str, word: str) -> bool:
    return bool(word) and re.search(rf"(?<![A-Za-z]){re.escape(word)}(?![A-Za-z])", text, re.I) is not None


def numbered_sentences(text: str, limit_chars: int = 900, min_len: int = 25, complete_only: bool = False) -> list[str]:
    """The sentences of a source (list bullets count as sentences), up to limit_chars in all. The model is shown them numbered and answers with a number, so what is shown as the
    supporting quote is always the source's own sentence."""
    from verification import _sentences
    out, used = [], 0
    for sentence in _sentences(re.sub(r"\s+\.\s+", ". ", _clean_source((text or "").replace("\u2022", ". ").replace("\u25cf", ". ")))):      # ' . ' (page navigation text) ends a sentence
        sentence = sentence.strip()
        if len(sentence) < min_len:
            continue
        if complete_only and (not re.search(r"[.!?\"\u201d)\]]$", sentence) or sentence[0].islower()):
            continue      # a search snippet's cut-off end or a fragment that starts mid-sentence is not a quotable sentence
        if used + len(sentence) > limit_chars and out:
            break
        sentence = re.sub(r":\.$", ":", sentence)      # 'indicated for the treatment of:' + bullets: the colon stays, no stray full stop
        out.append(sentence)
        used += len(sentence)
    return out


def _number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return int(value) if isinstance(value, str) and value.strip().isdigit() else None


def evidence_span(src: dict, item: dict, keys=("sentence", "sentences"), max_sentences: int = 2) -> str | None:
    """The supporting text the model points to, taken from the source itself: the numbered sentence(s) it chose (checked to exist), or else a quote that is found word for word.
    None when neither is in the source."""
    sentences = src.get("sentences") or numbered_sentences(src["text"], 100000, 12)
    chosen = next((item[k] for k in keys if item.get(k) not in (None, "", [])), None)
    numbers = chosen if isinstance(chosen, list) else [chosen]
    picked = [_number(n) for n in numbers[:max_sentences]]
    if chosen is not None and picked and all(n is not None and 1 <= n <= len(sentences) for n in picked):
        return _norm(" ".join(sentences[n - 1] for n in picked))
    return locate(src["text"], str(item.get("quote", "")))


def _grounded(phrase: str, span: str) -> bool:
    """The phrase is in the words of the source text it was taken from: at least 3 in 4 of its words (inflection ignored: first 5 letters) occur in that text, and never fewer than two."""
    words = re.findall(r"[A-Za-z]{3,}", phrase)
    low = span.lower()
    hits = sum(1 for w in words if w[:5].lower() in low)
    return len(words) >= 1 and hits >= min(2, len(words)) and hits * 10 >= len(words) * 6


def with_context(sentences: list[str], n: int) -> str:
    """The sentence a decision points at, together with what makes it readable, all taken from the same source text: the lead-in of a bullet list ('X is indicated for the treatment of:') when the
    sentence is one of its bullets, and the next sentence when that is a full sentence (it usually gives the population or the evidence)."""
    span = sentences[n - 1]
    lead = next((sentences[i] for i in range(n - 2, max(-1, n - 8), -1) if sentences[i].endswith(":")), "") if not re.search(r"\bindicated\b|\buse[sd]?\b", span, re.I) else ""
    after = sentences[n] if n < len(sentences) and len(sentences[n]) >= 40 and sentences[n][0].isupper() and not sentences[n].endswith(":") else ""
    return " ".join(x for x in (lead + (" " if lead else "") + span, after) if x).strip()


def verify_fda_decisions(items, offered: dict, question_asks: str | None = None) -> dict:
    """The model's yes/no on each FDA label (does its indication cover the condition?). A decision counts only with a quote found in that label's indication text.
    Confirmed and excluded labels are marked on the record (model_relevant); a label with no verified decision is left as retrieved and reported as 'unreviewed'."""
    out = {"confirmed": [], "excluded": [], "unreviewed": []}
    decided = set()
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        rec = offered.get(str(item.get("id", "")))
        if rec is None or not isinstance(item.get("relevant"), bool):
            continue
        if item["relevant"] and question_asks == "treat_or_manage" and item.get("purpose") in ("prevent", "diagnose"):
            item = dict(item, relevant=False)      # a question about treating a condition: a label that is only for preventing or diagnosing it (sunscreen, a test) is not an answer
        if item["relevant"] and evidence_span({"text": rec["indication_text"], "sentences": rec.get("sentences")}, item) is None:
            continue      # a 'relevant' decision must point at a sentence of the label; a 'not relevant' decision needs no evidence
        rec["record"]["model_relevant"] = item["relevant"]
        span = (evidence_span({"text": rec["indication_text"], "sentences": rec.get("sentences")}, item) or "") if item["relevant"] else ""
        number = _number(item.get("sentence"))
        sentences = rec.get("sentences") or []
        rec["record"]["model_evidence"] = with_context(sentences, number) if span and number and 1 <= number <= len(sentences) and _norm(sentences[number - 1]) == span else span
        rec["record"]["model_relation"] = item.get("relation") if item.get("relation") in RELATIONS else None
        out["confirmed" if item["relevant"] else "excluded"].append(rec["record"]["generic"])
        decided.add(item["id"])
    out["unreviewed"] = [v["record"]["generic"] for k, v in offered.items() if k not in decided]
    return out


def verify_drug_decisions(items, offered: dict, state: dict, limit: int = 8) -> tuple[list[dict], list[tuple]]:
    """The model's drug classifications -> rows for the landscape, plus (drug, reason) for every rejection. Checks, all against the cited source:
    the source id was offered; the drug name is in the source AND in the quote; the quote is in the source word for word; the model judged the drug relevant;
    the category is an allowed one (Guideline-recommended needs an allow-listed guideline organisation's page, otherwise it is Off-label); the indication phrase is words of the quote."""
    from guidelines import load_config, organisation_for
    config = load_config()
    approved = {r["generic"].lower().split()[0] for r in state.get("regulatory", []) if fda_relevant(r)}
    rows, rejected, seen = [], [], set()
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        drug = re.sub(r"\s+", " ", re.sub(r"\s*\([^)]*\)", "", str(item.get("drug", ""))).strip())      # 'Methylphenidate (Ritalin)' -> the generic name
        why = None
        src = offered.get(str(item.get("source_id", item.get("source", ""))).strip())
        category = USED_CATEGORIES.get(str(item.get("category", "")).strip().lower())
        span = evidence_span(src, item) if src else None
        indication = _norm(str(item.get("indication", ""))).strip(" .,;:")
        if not DRUG_NAME.fullmatch(drug):
            why = "not a drug name"
        elif item.get("drug_kind") not in ("generic", "brand"):
            why = f"a {item.get('drug_kind') or 'drug group'} is not one specific drug"
        elif drug.lower() in seen or drug.lower().split()[0] in approved:
            why = "duplicate or already an FDA-labelled row"
        elif item.get("relevance") not in RELEVANT:
            why = f"model: {item.get('relevance') or 'no relevance given'}"
        elif item.get("statement_type") != "used_in_practice":
            why = f"the sentence is {item.get('statement_type') or 'not typed'}, not a statement of use in practice"
        elif src is None:
            why = "source id was not offered"
        elif category is None:
            why = "category not allowed"
        elif span is None:
            why = "quote not found in the source"
        elif not _in(span, drug):
            why = "drug name not in the source sentence"
        if why:
            rejected.append((drug, why))
            continue
        if category.startswith("Guideline") and not (src["kind"] == "guideline" or (src["url"] and organisation_for(src["url"], config))):
            category = USED_CATEGORIES["off-label"]      # 'guideline-recommended' needs a guideline organisation's page, not a review or a web page
        if not (4 <= len(indication) <= 140 and _grounded(indication, span)):
            indication = "as stated in the quoted source sentence"      # the model's own wording is not in the sentence: it is not shown, the sentence speaks for itself
        population = _norm(str(item.get("population", ""))).strip(" .,;:")
        seen.add(drug.lower())
        rows.append({"drug": drug, "relevance": item.get("relevance"), "source": str(item.get("source_id", item.get("source", ""))).strip(), "label": src["label"], "excerpt": span, "purpose": indication,
                     "population": population if population and _grounded(population, span) else "not stated in this sentence", "category": category})
        if len(rows) >= limit:
            break
    return rows, rejected


def verify_regulator_decisions(items, offered: dict, limit: int = 8) -> tuple[list[dict], list[tuple]]:
    """The model's reading of each regulator page: what kind of action it is (approval, label update, safety communication, other action, or not relevant / uncertain) and how it
    relates to the question (the condition itself, an associated symptom or comorbidity, a risk, another indication, unclear). An item is kept only when the sentences the model points at
    exist in the page, the action is relevant, and it does not concern another indication; the 'concerns' wording, the drug and the date are shown only if they are in the page."""
    kept, rejected = [], []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        page = offered.get(str(item.get("id", "")))
        kind, relation = item.get("classification"), item.get("relation")
        if page is None:
            rejected.append((str(item.get("id")), "id was not offered"))
            continue
        span = evidence_span(page, item)
        if kind not in REGULATOR_CLASSES:
            rejected.append((page["title"][:50], f"model: {kind}"))
        elif relation not in REGULATOR_RELATIONS or relation in ("other_indication", "unclear"):
            rejected.append((page["title"][:50], f"relation to the question: {relation}"))
        elif span is None:
            rejected.append((page["title"][:50], "supporting sentence not in the page text"))
        else:
            when = _norm(str(item.get("date", ""))).strip()
            drug = _norm(str(item.get("drug", ""))).strip()
            concerns = _norm(str(item.get("concerns", ""))).strip(" .,;:")
            kept.append({"classification": kind, "type": REGULATOR_CLASSES[kind], "relation": relation, "title": page["title"], "url": page["url"], "quote": span,
                         "concerns": concerns if 4 <= len(concerns) <= 160 and _grounded(concerns, span) else "",
                         "drug": drug if _in(span, drug) and DRUG_NAME.fullmatch(drug) else "",
                         "date": when if 6 <= len(when) <= 40 and re.search(r"(?:19|20)[0-9]{2}", when) and (when.lower() in _norm(page["text"]).lower() or when in page["url"]) else "date not stated"})
        if len(kept) >= limit:
            break
    return kept, rejected


def _from_line(item, names: tuple):
    """'id|relevance|role|3' -> the same dict the verifier reads; dicts pass through."""
    if isinstance(item, str):
        parts = [x.strip(" \"'") for x in item.split("|")]
        return dict(zip(names, parts)) if len(parts) >= len(names) else None
    return item


def verify_paper_triage(items, offered: dict) -> dict:
    """The model's reading of each retrieved paper: relevance to the question, evidence role, population. A paper counts as relevant (direct, indirect, background) only when the
    sentence number the model gives exists in that paper's text; otherwise it is 'uncertain'. The population wording is kept only if it is in the paper's text."""
    out = {}
    for item in items if isinstance(items, list) else []:
        item = _from_line(item, ("id", "relevance", "role", "sentence"))
        rec = offered.get(str(item.get("id", ""))) if isinstance(item, dict) else None
        if rec is None or item.get("relevance") not in PAPER_RELEVANCE:
            continue
        n = _number(item.get("sentence"))
        grounded = n is not None and 1 <= n <= len(rec["sentences"])
        relevance = item["relevance"] if (grounded or item["relevance"] in ("not_relevant", "uncertain")) else "uncertain"
        population = _norm(str(item.get("population", ""))).strip(" .,;:")
        out[str(item["id"])] = {"relevance": relevance, "role": item.get("role") if item.get("role") in PAPER_ROLES else None,
                                "population": population if 4 <= len(population) <= 120 and _grounded(population, " ".join(rec["sentences"])) else "",
                                "evidence": rec["sentences"][n - 1] if grounded else ""}
    return out


def verify_trial_triage(items, offered: dict) -> dict:
    """The model's reading of each registry record: relevance, and whether it is a treatment trial, a diagnostic / assessment study or something else. The sentence number must exist."""
    out = {}
    for item in items if isinstance(items, list) else []:
        item = _from_line(item, ("id", "relevance", "kind", "sentence"))
        rec = offered.get(str(item.get("id", ""))) if isinstance(item, dict) else None
        if rec is None or item.get("relevance") not in PAPER_RELEVANCE or item.get("kind") not in TRIAL_KINDS:
            continue
        n = _number(item.get("sentence"))
        grounded = n is not None and 1 <= n <= len(rec["sentences"])
        relevance = item["relevance"] if (grounded or item["relevance"] in ("not_relevant", "uncertain")) else "uncertain"
        out[str(item["id"])] = {"relevance": relevance, "kind": item["kind"], "evidence": rec["sentences"][n - 1] if grounded else ""}
    return out


PHASE_RANK = {"PHASE4": 4, "PHASE3": 3, "PHASE2": 2, "PHASE1": 1}
TRIAL_ARM_SKIP = re.compile(r"placebo|\bcontrol\b|standard|comparison|\bversus\b|\bvs\b|\bsaline\b|\bsham\b|\bmatching\b|\+", re.I)


def _trial_drug_name(raw: str) -> str | None:
    """The registered intervention name without dose labels ('Drugzol high dose' -> 'Drugzol'); None for placebo / comparator arms and long descriptions. Registry arm-name hygiene only."""
    name = re.sub(r"\b(?:high|low|medium)[- ]dose\b|\b\d+(?:\.\d+)?\s?(?:mg|mcg|g|ml)\b.*$", "", str(raw or ""), flags=re.I).strip(" -,;:")
    if not name or TRIAL_ARM_SKIP.search(name) or len(name.split()) > 3 or not re.fullmatch(r"[A-Za-z][A-Za-z0-9\-/ ]{2,40}", name):
        return None
    return name


def boxed_warning_text(text: str, limit: int = 450) -> str:
    """The label's own boxed-warning wording, complete sentences only, without the 'BOXED WARNING' heading."""
    body = re.sub(r"^\s*(?:BOXED WARNING|WARNING:?)\s*", "", re.sub(r"\s+", " ", text or "")).strip()
    if len(body) <= limit:
        return body
    cut = max(body.rfind(". ", 0, limit), body.rfind("; ", 0, limit))
    return body[:cut + 1] if cut > 80 else body[:limit].rsplit(" ", 1)[0] + " (the label continues)"


def _warning_key(drug: str, text: str) -> str:
    """The warning with the drug's own name and label wording removed, so labels that say the same thing group together."""
    t = boxed_warning_text(text).lower()
    for word in re.split(r"[\s/,\-]+", drug.lower()) + ["tablets", "extended-release", "oral", "injection", "caplyta"]:
        if len(word) >= 4:
            t = t.replace(word, " ")
    t = re.sub(r"\[\s*see[^\]]*\]", " ", t)
    return re.sub(r"[^a-z]+", " ", t).strip()[:260]


def safety_alerts_markdown(labels: list[dict], listed: list[str] | None = None) -> str:
    """2.5 FDA boxed warnings of the drugs, in the label's own words. Drugs whose boxed warning says the same thing share one row (each drug links to its own label).
    Side effects belong here, not in the 'used for' column."""
    rows = [x for x in labels if x.get("boxed_warning") and x.get("url")]
    head = "### Safety warnings (FDA boxed warnings)\n\n"
    with_label = {x["drug"].lower() for x in labels}
    no_box = [x["drug"] for x in labels if not x.get("boxed_warning")]
    no_label = [d for d in dict.fromkeys(listed or []) if d.lower() not in with_label]
    coverage = ""      # every drug listed in the landscape is accounted for: it has a boxed warning (above), has a label without one, or has no retrieved label
    if no_box:
        coverage += f" No boxed warning on the retrieved label of: {', '.join(dict.fromkeys(no_box))}."
    if no_label:
        coverage += f" No FDA label could be matched for: {', '.join(no_label)}."
    if not rows:
        return head + "*No boxed warning was found in the retrieved FDA labels of the drugs." + coverage + " This is not a complete safety review: read the full label.*"
    groups: dict[str, list[dict]] = {}
    for x in rows:
        groups.setdefault(_warning_key(x["drug"], x["boxed_warning"]), []).append(x)
    lines = ["| Drug(s) (FDA label) | Boxed warning, in the label's words |", "|---|---|"]
    for members in groups.values():
        drugs = ", ".join(f"[{_cell(m['drug'])}]({m['url']})" for m in members)
        lines.append(f"| {drugs} | \"{_cell(boxed_warning_text(members[0]['boxed_warning']), 900)}\" |")
    return head + "\n".join(lines) + "\n\n*These are the labels' own boxed warnings, not a complete safety review: read the full label." + coverage + "*"


REGULATOR_HOSTS = ("fda.gov", "ema.europa.eu", "cdsco.gov.in")
RECENT_YEARS = 3      # the section is 'recently approved or updated': an action counts as recent when its date, stated by the page itself, is within this many years of the search date


def parse_date(text: str):
    """'September 22, 2025', '09/22/2025', '2025-09-22' -> a date; None for anything else."""
    from datetime import datetime
    for fmt in ("%B %d, %Y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(text).strip(), fmt)
        except ValueError:
            continue
    return None


def _link_text(text: str, limit: int = 200) -> str:
    """Text for a Markdown link label: square brackets inside it would break the link."""
    return _cell(text, limit).replace("[", "(").replace("]", ")")


def regulatory_updates_markdown(state: dict, limit: int = 6) -> str:
    """Regulator actions from the regulators' own pages (fda.gov, ema.europa.eu, cdsco.gov.in): approvals, label updates, safety communications and other actions. What kind of action it
    is, and how it relates to the question, is the model's decision (state['regulator_items']) after the code verified it against the page text; the quote is the page's own sentence(s).
    Only actions with a date that the page itself states, within RECENT_YEARS of the search date, are listed as recent; the others are counted in one line."""
    from datetime import datetime, timedelta
    head = "### Recently approved or updated (FDA, EMA and CDSCO announcements)\n\n"
    try:
        today = datetime.strptime(str(state.get("searched_on") or ""), "%Y-%m-%d")
    except ValueError:
        today = datetime.now()
    recent, left_out = [], 0
    for x in state.get("regulator_items", []):
        when = parse_date(x["date"])
        if when is not None and today - timedelta(days=365 * RECENT_YEARS) <= when <= today:
            recent.append((when, x))
        else:
            left_out += 1
    recent.sort(key=lambda pair: pair[0], reverse=True)      # newest first
    note = (f"\n\n*{left_out} further regulator page(s) were read but are not listed: the page states no date, or a date more than {RECENT_YEARS} years old, so they cannot be shown as recent.*" if left_out else "")
    if not recent:
        return head + f"*No FDA, EMA or CDSCO announcement dated within the last {RECENT_YEARS} years was found for this question. This does not mean none exists.*" + note
    lines = ["| Date | Type | Announcement | What the regulator's page says |", "|---|---|---|---|"]
    for when, x in recent[:limit]:
        kind = x["type"] + (" (associated symptom or comorbidity)" if x["relation"] == "treats_associated_symptom_or_comorbidity" else "")
        lines.append(f"| {_cell(x['date'])} | {kind} | [{_link_text(x['title'])}]({x['url']}) | \"{_cell(x['quote'], 700)}\" |")
    return head + "\n".join(lines) + "\n\n*Page wording from the regulator's own site; read the page for the full announcement.*" + note
