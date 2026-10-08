"""Deterministic sections built from the ResearchState (no model, nothing invented).

    record_source_status   retrieval_status (success|partial|failed) and evidence_status (found|not_found_after_search|unavailable) per source
    regulatory_markdown    FDA-label facts quoted exactly; separates 'approved for the labelled indication' from everything else
    top_drugs_markdown     the Phase-1 'Top treatment drugs' table
    enforce_other_agents_table   dose and approval cells of model-written drug rows are fixed to 'Not retrieved' / 'Not verified ...'
    trials_markdown        treatment/intervention trials separated from peripheral registry records; status wording that never implies efficacy
    limitations_markdown   keeps 'source failed' apart from 'searched and nothing found'
    drug_intelligence_status
"""
from __future__ import annotations

import json
import re
from urllib.parse import quote

PERIPHERAL = re.compile(r"\b(imaging|mri|eeg|biomarker\w*|diagnos\w*|screen\w*|assessment|characteri[sz]\w*|natural history|qualitative|survey|dental|oral hygiene|"
                        r"habituation|dataset|machine learning|registry)\b", re.I)


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


def landscape_markdown(state: dict, used_rows: list[dict], trial_rows: list[dict]) -> str:
    """The specification's Treatment Drug Landscape, as TWO tables: drugs used for treatment of this condition, and other drugs (symptom-directed or still in trials). Columns: drug, category, indication, dose / route (if sourced),
    approval status / date, key evidence. Investigational drugs are never listed as established treatment; doses come only from a label or source for the use shown."""
    head = "## Treatment Drug Landscape\n\n"
    status = state.get("source_status", {})
    st = status.get("fda_deep") or status.get("fda_fast") or {}
    reg = [r for r in state.get("regulatory", []) if r.get("matches_condition")]
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
        statement = r.get("indication_statement") or r["indication_excerpt"]
        efficacy = re.search(r"Efficacy was established[^.]*\.", statement or "")
        evidence = (f"FDA label: {efficacy.group(0)} " if efficacy else "FDA label. ") + (f"[DailyMed label]({label_link(r, stems)})" if r.get("url") else "")
        entries.append((0, "", f"| {r['generic']}{' (' + brands + ')' if brands else ''} | \"{_cell(statement, 800)}\" | {_dose_cell(r)} | "
                           f"FDA-approved for the use shown; label effective {r.get('label_date') or 'date not retrieved'}; original approval date not retrieved | {_cell(evidence, 400)} |"))
    for r in used_rows:
        tier = 1 if r.get("category", "").startswith("Guideline") else 2
        said = f"\"{_cell(r['excerpt'], 260)}\" " if len(r.get("excerpt", "")) <= 260 else ""
        pop = "" if r.get("population", "").startswith("not stated") else f" (population: {r['population']})"
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
              "*Off-label: a retrieved review or page describes use for specific symptoms or co-occurring conditions; the drug has no FDA-labelled indication for this condition. "
              "Drugs still being tested in trials are listed in the Clinical trials section below.*\n\n" + "\n".join(short_columns + other))
    return (head + intro + first + "\n\n" + second + "\n\n" + regulatory_updates_markdown(state) + "\n\n" + safety_alerts_markdown(state.get("drug_labels", [])))


def is_late_phase(t: dict, min_phase: int = 2) -> bool:
    """Late-phase = every registered phase is phase 2 or later (so 'Phase 1/2', 'Phase 1' and 'not applicable' are not late-phase)."""
    ranks = [PHASE_RANK.get(p, 0) for p in (t.get("phase") or [])]
    return bool(ranks) and min(ranks) >= min_phase


def notable_latest_drug(state: dict) -> dict | None:
    """'Then the most notable latest drug': the most recently updated late-phase (phase 2 or later) registered drug trial of THIS condition. Always investigational."""
    stems = [x for x in (state.get("core_stems") or []) if len(x) >= 4]
    approved = {r["generic"].lower() for r in state.get("regulatory", []) if r.get("matches_condition")}
    best, best_key = None, None
    for t in state.get("trials_by_id", {}).values():
        about = (" ".join(t.get("conditions") or []) + " " + (t.get("title") or "")).lower()
        rank = max([PHASE_RANK.get(p, 0) for p in (t.get("phase") or [])] or [0])
        if not is_late_phase(t) or (stems and not any(x in about for x in stems)):
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
    reg = [r for r in state.get("regulatory", []) if r["matches_condition"]][:4]
    latest = notable_latest_drug(state)
    stems = [x for x in (state.get("core_stems") or []) if len(x) >= 4]
    rows = ["| Drug | Indication named in the FDA label | Source |", "|---|---|---|"]
    for r in reg:
        link = f"[DailyMed]({r['url']})" if r.get("url") else "FDA label"
        rows.append(f"| {r['generic']} | \"{condition_clause(r['indication_excerpt'], stems)}\" | {link} |")
    if latest:
        rows.append(f"| {latest['drug']} (latest, investigational) | Being tested in a registered {latest['phase']} trial, {latest['status']}; not an approved treatment | "
                    f"[{latest['nct']}]({latest['url']}) |" if latest.get("url") else f"| {latest['drug']} (latest, investigational) | Being tested in a registered {latest['phase']} trial; not an approved treatment | {latest['nct']} |")
    if reg or latest:
        return "**Top treatment drugs (established first, then the most notable latest drug)**\n\n" + "\n".join(rows)
    if status.get("retrieval_status") in ("failed", "partial"):
        return ("**Top treatment drugs:** the FDA drug-label source was unavailable in the " + where + " (" + str(status.get("error")) +
                "); the full drug landscape follows. This does not mean that no drug exists.")
    return "**Top treatment drugs:** no FDA label with an indication matching the condition was found in the " + where + "; the full landscape follows."


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


RISK_ONLY = re.compile(r"prenatal exposure|in utero|exposure cohort|\blinked to [^|]{0,40}\brisk\b|\bassociated with (?:a )?(?:higher|increased) risk\b|no treatment efficacy", re.I)
NON_SUBSTANCE = re.compile(r"\b(exercise|dance|music|diet|dietary|therapy|therapies|programme|program|training|behaviou?ral|behaviou?r|intervention|rehabilitation|"
                           r"occupational|device|stimulation|neurofeedback|yoga|massage)\b", re.I)


def _guideline_supported(evidence: str, state: dict) -> bool:
    """True only if the row cites a retrieved guideline: a paper typed/titled as a guideline, or a retrieved web page whose title or address says guideline."""
    for pmid in re.findall(r"PMID[:\s]*(\d{6,9})", evidence):
        p = state.get("papers", {}).get(pmid) or {}
        if any("guideline" in str(t).lower() for t in (p.get("type") or [])) or "guideline" in (p.get("title") or "").lower():
            return True
    for url in re.findall(r"https?://[^\s\]\)|]+", evidence):
        for w in state.get("web", []):
            if w.get("url", "").rstrip("/") == url.rstrip("/.,;") and "guideline" in f"{w.get('title', '')} {w.get('url', '')}".lower():
                return True
    return False


def enforce_other_agents_table(text: str, state: dict) -> str:
    """Model-written rows for agents outside the regulatory records. The system sets dose = 'Not retrieved' and approval =
    'Not verified in retrieved regulatory sources' in EVERY row, whatever the model wrote, and handles rows with any number of cells.
    The table is found anywhere inside its block (the model may write a stray line before it). Rows for non-substances (exercise, diet,
    therapy ...), 'none' placeholders and duplicates of regulatory drugs are dropped; if no row survives the table is replaced by 'None retrieved.'"""
    reg_names = {r["generic"].lower().split()[0] for r in state.get("regulatory", [])}
    lines, out, i = text.splitlines(), [], 0
    while i < len(lines):
        line = lines[i]
        if not (line.lstrip().startswith("####") and "other agents" in line.lower()):
            out.append(line)
            i += 1
            continue
        out.append(line)
        i += 1
        block = []
        while i < len(lines) and not (lines[i].startswith("#") or lines[i].strip() == "---" or lines[i].lstrip().startswith("**Safety")):
            block.append(lines[i])
            i += 1
        table = [b for b in block if b.lstrip().startswith("|")]
        notes = [b for b in block if b.strip() and not b.lstrip().startswith("|") and not b.strip().lower().startswith("none retrieved")]
        data = [tl for tl in table if not set(tl.strip()) <= set("|-: ")][1:]      # drop the header and the separator row
        rows = []
        for tl in data:
            cells = [c.strip() for c in tl.strip().strip("|").split("|")]
            name = cells[0].strip("* ").lower()
            if not name or name.startswith("none") or NON_SUBSTANCE.search(name) or name.split()[0] in reg_names:
                continue
            if RISK_ONLY.search(tl):   # a drug named only as an exposure / risk factor is not a treatment under evaluation
                continue
            evidence = cells[-1] if len(cells) >= 6 else " ".join(cells[4:]) if len(cells) > 4 else ""
            category = cells[1] if len(cells) > 1 else ""
            if re.search(r"guideline", category, re.I) and not _guideline_supported(evidence, state):
                category = "Category not verified (no retrieved guideline source is cited)"   # 'guideline-recommended' needs a guideline, not an approval or an article
            row = [cells[0], category, cells[2] if len(cells) > 2 else "", "Not retrieved", "Not verified in retrieved regulatory sources", evidence]
            rows.append("| " + " | ".join(row) + " |")
        if rows:
            out += ["", "| Drug (generic/brand) | Category | Population studied | Dose / route | Approval status | Key evidence |", "|---|---|---|---|---|---|"] + rows
        else:
            out.append("None retrieved.")
        out += notes + [""]
    return "\n".join(out)


# ------------------------------------------------------------------ trials
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
    text = f"{t.get('title') or ''} {' '.join(t.get('interventions') or [])} {t.get('primary_endpoint') or ''}"
    return (t.get("design") or "").upper() == "INTERVENTIONAL" and bool(t.get("interventions")) and not PERIPHERAL.search(text)


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
    extra = f"\n\n{len(other)} further registry record(s) about assessment, characterisation or peripheral research were retrieved and are not shown (they are not treatment trials)." if other else ""
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
# the sentence must say the drug is USED / PRESCRIBED / RECOMMENDED in practice; "improved symptoms compared with placebo" in a trial does not count
USE_WORDS = re.compile(r"\b(?:commonly|widely|frequently|often|routinely|typically)\s+(?:used|prescribed|given)\b|\b(?:used|prescribed)\s+(?:to|for|in|as|off-label)\b|\bfirst-line\b|\bsecond-line\b|"
                       r"\boff-label\b|\bstandard of care\b|\bclinical practice\b|\brecommended\s+(?:for|to|as)\b|\b(?:may|can|might)\s+be\s+(?:used|prescribed|considered)\b|"
                       r"\bmedications?\s+(?:such as|including|like)\b|\bdrugs?\s+(?:such as|including|like)\b", re.I)
NOT_USE = re.compile(r"\b(mice|mouse|rats?|rodents?|animal|in vitro|prenatal|in utero|exposure|exposed|teratogen\w*|protocol|will be|being (?:tested|evaluated|investigated)|"
                     r"investigational|phase [123]|under (?:study|investigation)|not (?:effective|recommended)|placebo|randomi[sz]ed|trial|RCT|SMD|effect sizes?|k\s?=|meta-analys\w+|"
                     r"compared (?:with|to)|versus|vs\.?)\b", re.I)


def _clean_source(text: str) -> str:
    """Whitespace tidied; the '[...]' that search services put between unrelated fragments becomes a sentence break, so fragments are never fused."""
    return re.sub(r"\s#{1,6}\s", ". ", re.sub(r"\s+", " ", (text or "").replace("[...]", ". "))).strip()


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


def excerpt_around(sentence: str, drug: str, limit: int = 300) -> str:
    """The part of the sentence that shows BOTH the drug name and the wording that says it is used (never a cut that hides that wording); copied unchanged."""
    if len(sentence) <= limit:
        return sentence
    d = re.search(rf"\b{re.escape(drug)}\b", sentence, re.I)
    u = USE_WORDS.search(sentence)
    points = [m for m in (d, u) if m]
    start = max(0, min(m.start() for m in points) - 60)
    end = min(len(sentence), max(m.end() for m in points) + 100)
    if end - start > limit:
        start, end = max(0, d.start() - 80), min(len(sentence), d.start() + limit - 80)
    start = sentence.rfind(" ", 0, start) + 1 if start else 0
    if end < len(sentence):
        end = sentence.rfind(" ", start, end) if sentence.rfind(" ", start, end) > start else end
    return ("... " if start else "") + sentence[start:end].strip() + (" ..." if end < len(sentence) else "")


SIDE_EFFECT = re.compile(r"\b(adverse|side[- ]effects?|weight gain|sedation|somnolence|toxicity|contraindicat\w*|withdrawn|discontinu\w*|tolerab\w*|limit(?:s|ed)? its use|safety)\b", re.I)
PURPOSE_CUT = re.compile(r",\s+and\s+\w+\s+for\b|\s+(?:although|while|whereas|but|which|because|and\s+(?:appears?|is|are|has|have|may|can|should|was|were)|as\s+described)\b|\s+\((?:e\.g|i\.e)", re.I)
PURPOSE_PATTERNS = [
    re.compile(r"\b(?:treatment|management|therapy)\s+of\s+(?P<p>[^.;()]{4,120})", re.I),
    re.compile(r"\bto\s+(?:treat|manage|address|reduce|improve|control|relieve)\s+(?P<p>[^.;()]{4,120})", re.I),
    re.compile(r"\bfirst-line\s+(?:medications?|drugs?|treatments?|agents?)\s+for\s+(?P<p>[^.;()]{4,120})", re.I),
    re.compile(r"\b(?:used|prescribed|given|recommended|indicated)\s+(?:in|for)\s+(?P<p>[^.;()]{4,120})", re.I),
    re.compile(r"(?<![A-Za-z])for\s+(?P<p>[a-z][^.;()]{3,80})", re.I),
]
POPULATION = re.compile(r"\b(?:in|among|for)\s+((?:children|adolescents|adults|youth|teenagers|infants|toddlers|preschool\w*|pediatric (?:patients|populations?)|autistic (?:children|adults|people|individuals|youth))"
                        r"(?:\s*(?:,|and|or|with)\s*(?:children|adolescents|adults|youth|ASD|autism\w*(?: spectrum disorders?)?|autistic \w+))*)", re.I)


def purpose_phrase(sentence: str, drug: str) -> str | None:
    """WHAT the source says the drug is used for, in its own words, taken from the text AFTER the drug name (so one drug in a list sentence never gets another drug's purpose).
    None when the sentence does not state a purpose for this drug."""
    m = re.search(rf"\b{re.escape(drug)}\b", sentence, re.I)
    if not m:
        return None
    after = sentence[m.end():]
    best = None
    for pattern in PURPOSE_PATTERNS:
        found = pattern.search(after)
        if found and (best is None or found.start() < best[0]):
            if pattern is PURPOSE_PATTERNS[-1] and found.start() > 90:      # a bare 'for ...' far from the drug probably belongs to something else
                continue
            best = (found.start(), found.group("p"))
        if best and pattern is PURPOSE_PATTERNS[2]:
            break
    if not best:
        return None
    phrase = PURPOSE_CUT.split(best[1])[0].strip(" ,;:")
    phrase = re.sub(r"^(?:the\s+)?(?:treatment|management)\s+of\s+", "", phrase, flags=re.I)
    return phrase if len(phrase.split()) >= 2 else None


def population_phrase(sentence: str) -> str:
    m = POPULATION.search(sentence)
    return re.sub(r"\s+", " ", m.group(1)).strip(" ,") if m else "not stated in this sentence"


def candidate_sentences(text: str) -> str:
    """Only the sentences of a source that say something is USED / PRESCRIBED / RECOMMENDED in practice (not trial comparisons, animal work, exposures or protocols).
    The model is shown just these, so it cannot pick a drug from a trial result."""
    from verification import _sentences
    keep = [s.strip() for s in _sentences(_clean_source(text)) if USE_WORDS.search(s) and not NOT_USE.search(s) and not RISK_ONLY.search(s) and not SIDE_EFFECT.search(s)]
    return " ".join(keep)


def used_in_practice_rows(items, sources: dict, state: dict, limit: int = 4) -> list[dict]:
    """The model only NAMES a drug and the source (a PMID or web address from the data). Code then requires that the drug name appears in that source,
    in a sentence that says it is used / prescribed / recommended / managed, and not as an exposure, an animal study, a protocol or an investigational agent.
    Drugs that are already FDA-labelled for the condition are left out (they have their own rows)."""
    from verification import _sentences
    approved = {r["generic"].lower() for r in state.get("regulatory", []) if r.get("matches_condition")}
    rows, seen = [], set()
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        drug = re.sub(r"\s+", " ", str(item.get("drug", "")).strip())
        source = str(item.get("source", "")).strip()
        key = drug.lower()
        if not re.fullmatch(r"[A-Za-z][A-Za-z\-]{3,}(?: [A-Za-z\-]{3,}){0,2}", drug) or NON_SUBSTANCE.search(drug) or key in seen or key.split()[0] in approved:
            continue
        found = sources.get(source)
        if not found:
            continue
        text, label = found
        sentence = next((s.strip() for s in _sentences(_clean_source(text)) if re.search(rf"\b{re.escape(drug)}\b", s, re.I)
                         and USE_WORDS.search(s) and not NOT_USE.search(s) and not RISK_ONLY.search(s) and not SIDE_EFFECT.search(s)), None)
        if not sentence:
            continue
        purpose = purpose_phrase(sentence, drug)
        if not purpose:      # no stated purpose for THIS drug: it is not listed (the reader must see what it is used for)
            continue
        seen.add(key)
        from guidelines import load_config, organisation_for
        guideline_page = source.startswith("http") and organisation_for(source, load_config()) is not None
        category = "Guideline-recommended" if guideline_page and re.search(r"recommend|first-line", sentence, re.I) else "Off-label (described as used in practice)"
        rows.append({"drug": drug, "source": source, "label": label, "excerpt": complete_sentence(sentence), "purpose": purpose, "population": population_phrase(sentence), "category": category})
        if len(rows) >= limit:
            break
    return rows


TRIAL_NAME_SKIP = re.compile(r"placebo|\bcontrol\b|standard|comparison|\bversus\b|\bvs\b|\bsaline\b|\bsham\b|\bmatching\b|\+", re.I)
PHASE_RANK = {"PHASE4": 4, "PHASE3": 3, "PHASE2": 2, "PHASE1": 1}


def _trial_drug_name(raw: str) -> str | None:
    """The registered drug name without dose labels ('Lumateperone high dose' -> 'Lumateperone'); None for placebo, controls, comparisons and long descriptions."""
    name = re.sub(r"\b(?:high|low|medium)[- ]dose\b|\b\d+(?:\.\d+)?\s?(?:mg|mcg|g|ml)\b.*$", "", str(raw or ""), flags=re.I).strip(" -,;:")
    if not name or TRIAL_NAME_SKIP.search(name) or NON_SUBSTANCE.search(name) or len(name.split()) > 3 or not re.fullmatch(r"[A-Za-z][A-Za-z0-9\-/ ]{2,40}", name):
        return None
    return name


def trial_drug_rows(state: dict, exclude: set, limit: int = 6, min_phase: int = 2) -> list[dict]:
    """Drugs that are being TESTED in registered clinical trials of this condition (registry intervention type DRUG or BIOLOGICAL). Built from the registry records only:
    name, trial ID, phase and status exactly as registered. Later phases first. Drugs that already have an FDA row or a 'used in practice' row are left out."""
    seen, rows = {e.lower() for e in exclude}, []
    stems = [s for s in (state.get("core_stems") or []) if len(s) >= 4]
    ordered = sorted(state.get("trials_by_id", {}).values(), key=lambda t: -max([PHASE_RANK.get(p, 0) for p in (t.get("phase") or [])] or [0]))
    for t in ordered:
        about = (" ".join(t.get("conditions") or []) + " " + (t.get("title") or "")).lower()
        if stems and not any(s in about for s in stems):
            continue      # the trial must be about this condition, not merely contain a matching word somewhere
        if not is_late_phase(t, min_phase):
            continue      # the specification lists LATE-phase emerging drugs; phase 1, phase 1/2 and not-applicable trials are left out
        for raw in t.get("drug_interventions") or []:
            name = _trial_drug_name(raw)
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            rows.append({"conditions": "; ".join(t.get("conditions") or []) or "not stated", "primary": (t.get("primary_endpoint") or "not stated"), "ages": t.get("ages") or "not stated",
                         "drug": name, "nct": t["nct_id"], "phase": nice_phase(t.get("phase")),
                         "status": nice_status(t.get("recruitment_status")), "updated": t.get("last_update") or "date not stated", "url": t.get("url")})
            if len(rows) >= limit:
                return rows
    return rows


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


def safety_alerts_markdown(labels: list[dict]) -> str:
    """2.5 FDA boxed warnings of the drugs, in the label's own words. Drugs whose boxed warning says the same thing share one row (each drug links to its own label).
    Side effects belong here, not in the 'used for' column."""
    rows = [x for x in labels if x.get("boxed_warning") and x.get("url")]
    head = "### Safety warnings (FDA boxed warnings)\n\n"
    if not rows:
        return head + "*No boxed warning was found in the retrieved FDA labels of the drugs. This is not a complete safety review: read the full label.*"
    groups: dict[str, list[dict]] = {}
    for x in rows:
        groups.setdefault(_warning_key(x["drug"], x["boxed_warning"]), []).append(x)
    lines = ["| Drug(s) (FDA label) | Boxed warning, in the label's words |", "|---|---|"]
    for members in groups.values():
        drugs = ", ".join(f"[{_cell(m['drug'])}]({m['url']})" for m in members)
        lines.append(f"| {drugs} | \"{_cell(boxed_warning_text(members[0]['boxed_warning']), 900)}\" |")
    return head + "\n".join(lines) + "\n\n*These are the labels' own boxed warnings, not a complete safety review: read the full label.*"


REGULATOR_HOSTS = ("fda.gov", "ema.europa.eu", "cdsco.gov.in")
DATE_RE = re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+(?:19|20)\d\d\b|\b(?:19|20)\d\d-\d\d-\d\d\b")


REG_VERB = re.compile(r"\b(approv\w*|label\w*|authori[sz]\w*|warn\w*|safety|announc\w*|action|expand\w*|broaden\w*|review\w*|clear\w*|recommend\w*|request\w*)\b", re.I)


def _regulatory_sentences(text: str, stems: list[str], limit: int = 2) -> str:
    """Up to two sentences of the page that state a regulatory action, copied unchanged (navigation text such as 'More Press Announcements' is skipped)."""
    from verification import _sentences
    candidates = []
    for sentence in _sentences(_clean_source(text)):
        sentence = sentence.strip()
        if 40 <= len(sentence) <= 420 and REG_VERB.search(sentence) and not re.search(r"press announcements|news release|skip to|menu|\bsearch\b", sentence, re.I):
            candidates.append(sentence)
    about = [s for s in candidates if any(x in s.lower() for x in stems)]      # sentences that name the condition come first, so the row shows why it is listed
    return " ".join((about + [s for s in candidates if s not in about])[:limit])


def regulatory_updates_markdown(state: dict, limit: int = 6) -> str:
    """2.4 Approvals, label changes and safety communications from the regulators' own pages (fda.gov, ema.europa.eu, cdsco.gov.in) that name the condition. Page wording only."""
    from urllib.parse import urlparse
    stems = [x for x in (state.get("core_stems") or []) if len(x) >= 4]
    head = "### Recently approved or updated (FDA, EMA and CDSCO announcements)\n\n"
    rows, seen, seen_titles = [], set(), set()
    for w in state.get("web", []):
        url = w.get("url") or ""
        page = url.split("#", 1)[0].split("?", 1)[0].rstrip("/").lower()      # the same page with another #anchor or ?query is one announcement
        host = (urlparse(url).hostname or "").lower()
        path = urlparse(url).path.lower()
        if w.get("source") != "web_regulatory" or not any(host == h or host.endswith("." + h) for h in REGULATOR_HOSTS) or page in seen:
            continue
        if host.endswith("fda.gov") and not any(m in path for m in ("press-announcements", "drug-safety-communications", "safety-announcements")):
            continue      # on fda.gov only press announcements and safety communications count: not committee meetings, Q&A pages, divisions or download files
        if path.rstrip("/") == "" or "/media/" in path:
            continue
        blob = f"{w.get('title') or ''} {(w.get('snippet') or '')[:500]}".lower()
        if stems and not any(x in blob for x in stems):
            continue      # the condition must be in the title or the opening of the page, not buried in a list further down
        title = re.sub(r"\s*/\s*FDA\s*$", "", re.sub(r"\s+", " ", w.get("title") or url).strip())
        text = re.sub(r"\s+", " ", w.get("snippet") or "")
        kind = ("Safety communication" if re.search(r"safety|warns?|warning", title, re.I)
                else "Approval or regulatory action" if re.search(r"approv|authori[sz]|clear", f"{title} {text[:300]}", re.I) else "Regulatory announcement")
        found = DATE_RE.search(text) or DATE_RE.search(url)
        if title.lower() in seen_titles:
            continue
        seen.add(page)
        seen_titles.add(title.lower())
        said = _regulatory_sentences(text, stems)
        rows.append((found.group(0) if found else "date not stated", kind, title, said or "No sentence stating the action was retrieved; open the page", url))
    rows.sort(key=lambda r: r[0] == "date not stated")      # dated announcements first
    rows = rows[:limit]
    if not rows:
        return head + "*No FDA, EMA or CDSCO announcement about this condition was retrieved. This does not mean none exists.*"
    lines = ["| Date | Type | Announcement | What the regulator's page says |", "|---|---|---|---|"]
    for date, kind, title, text, url in rows:
        lines.append(f"| {_cell(date)} | {kind} | [{_cell(title, 200)}]({url}) | {_cell(text, 700)} |")
    return head + "\n".join(lines) + "\n\n*Page wording from the regulator's own site; read the page for the full announcement.*"
