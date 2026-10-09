"""NeuroGPT citation-verification + quality-control + repair, working on the FINAL ANSWER STRING and the full ResearchState.

    WRITE -> normalise citations -> rebuild Sources from the answer text -> VERIFY
          -> NEEDS_REPAIR? -> ONE repair pass on the exact failed spans -> rebuild Sources -> VERIFY AGAIN -> QC -> PASS / NEEDS_REPAIR

PASS means the repaired final text itself has no remaining blocking issue. No model is used here, and nothing is added that
was not retrieved. General rules only: nothing in this file knows about any particular disease or drug.

ResearchState fields used (all plain dicts / lists):
    papers {pmid: record}, trials_by_id {nct: record}, web [record], regulatory [record],
    source_status {job: {retrieval_status, evidence_status, error}}, drug_intelligence_status
"""
from __future__ import annotations

import re
from collections import Counter

PMID_RE = re.compile(r"PMID[:\s]*([0-9]{6,9})")
NCT_RE = re.compile(r"NCT\s?(\d{8})")
DOI_RE = re.compile(r"10\.\d{4,9}/(?:\([^\s()]*\)|[^\s\]\)|,;>】])+")
URL_RE = re.compile(r"https?://[^\s\]\)>|,;]+")
# "Surname et al., [*Journal*,] Year" not already followed by [PMID n]
AY_RE = re.compile(r"([A-Z][A-Za-zÀ-ſ'\-]+) et al\.,?\s*(?:\*?[^*,;()\[\]\n]{2,80}\*?,\s*)?((?:19|20)\d\d)(?!\d)(?!\s*\[PMID)")

NUMBER_PATTERNS = [
    r"\d+(?:\.\d+)?\s*[–\-]\s*\d+(?:\.\d+)?\s*(?:mg|mcg|µg|mL|IU|g)(?:/(?:kg|day|m2))?",   # dose range
    r"\d+(?:\.\d+)?\s?(?:mg|mcg|µg|mL|IU|g)(?:/(?:kg|day|m2))?\b",                                  # dose
    r"\d+(?:\.\d+)?[\s\-]?%",                                                                              # percentage
    r"\bp\s?[<=>≤≥]\s?0?\.\d+",                                                                 # p-value
    r"\bn\*?\s?=\s?[\d,]+",                                                                                 # n = 1,250
    r"\b\d[\d,]{2,}\s+(?:participants|patients|subjects|children)",
    r"\b(?-i:HR|OR|RR)\s?[=:]?\s?\d+(?:\.\d+)?",                                                          # ratios
    r"\b(?:SMD|MD|g|d|r)\*?\s?=\s?-?\d+(?:\.\d+)?",                                                          # effect sizes (Hedges g, SMD, r)
]
OUTCOME_TERMS = ["social communication", "social interaction", "irritability", "aggression", "anxiety", "depression", "mania", "sleep", "motor",
                 "language", "repetitive", "seizure", "pain", "mortality", "survival", "relapse", "disability progression", "cognition", "quality of life"]
EFFICACY_WORDS = re.compile(r"\b(effective|effectiveness|efficacy|efficacious|improv\w+|reduc\w+|benefit\w*|superior|successful|works?|proven|proved)\b", re.I)
APPROVAL_POSITIVE = re.compile(r"\b(?:fda|ema|cdsco)[\s\-]approved\b|\bapproved (?:by|for|to treat)\b|\bnewly approved\b", re.I)
NEGATION = re.compile(r"\b(no|not|none|never|without|unapproved|un-approved|investigational)\b", re.I)
NO_DRUGS_CLAIM = re.compile(r"\bno (?:pharmacolog\w+|drug|medication)\w*\b[^.\n]{0,70}\b(?:identified|retrieved|found|exist\w*|available)\b", re.I)
NON_EXISTENCE = re.compile(r"\b(?:there (?:is|are) no|does not exist|do not exist|none exist|no [a-z\- ]{0,30}(?:exist|exists|available))\b", re.I)
SCOPED = re.compile(r"(timed out|unavailable|could not be retrieved|not retrieved|retrieval|failed|incomplete|partial|does not (?:mean|show)|cannot be concluded)", re.I)
OVERCLAIM = re.compile(r"\b(confirmed|confirms|proven|proved|proves|validated|definitive(?:ly)?|conclusively)\b", re.I)
DOSE_LANGUAGE = [(re.compile(r"dose[\s-]dependent", re.I), "dose-response (exploratory)"),
                 (re.compile(r"(?:with )?stronger effects? at higher doses", re.I), "with an exploratory dose-response relationship reported")]
SOFTER = {"confirmed": "reported", "confirms": "reports", "proven": "reported", "proved": "reported", "proves": "suggests", "validated": "supported", "definitive": "preliminary",
          "definitively": "preliminarily", "conclusively": "tentatively"}
ESTABLISHED_HEADING = re.compile(r"^\W*\*{0,2}\s*established\s*\*{0,2}\s*:?\s*$", re.I)
ESTABLISHED_INLINE = re.compile(r"\*\*established\*\*|\b(?:is|are) established\b|\bestablished (?:to|for|treatment)\b", re.I)
INDIRECT_LABEL = re.compile(r"indirect|related population|not (?:\w+-)?specific|adjacent", re.I)


# ------------------------------------------------------------------ helpers
def _without_foreign_urls(text: str) -> str:
    """A web address that merely CONTAINS something shaped like a DOI (e.g. a publisher path '10.1542/peo_document...') is a URL, not a DOI."""
    return re.sub(r"https?://(?!(?:dx\.)?doi\.org)[^\s\]\)>|,;]+", " ", text)


def _norm(text: str) -> str:
    return re.sub(r"[\s,\-*_‐-―−’']", "", str(text).lower().replace("≈", ""))


def _norm_url(url: str) -> str:
    return url.split("#:~:", 1)[0].rstrip(".,;:)/").lower()      # a text-fragment highlight (#:~:text=) does not change which page it is


def known_sources(state: dict) -> dict:
    papers, trials = state.get("papers", {}), state.get("trials_by_id", {})
    urls = set()
    for p in papers.values():
        urls |= {p.get("url"), f"https://pubmed.ncbi.nlm.nih.gov/{p['pmid']}/", f"https://doi.org/{p['doi']}" if p.get("doi") else None}
    urls |= {t.get("url") for t in trials.values()} | {w.get("url") for w in state.get("web", [])} | {r.get("url") for r in state.get("regulatory", [])} | {g.get("url") for g in state.get("guidelines", [])} | {d.get("url") for d in state.get("drug_labels", [])}
    return {"pmids": set(papers), "ncts": set(trials), "dois": {(p.get("doi") or "").lower() for p in papers.values()} - {""},
            "urls": {_norm_url(u) for u in urls if u}}


def evidence_blob(state: dict) -> str:
    """Text of everything RETRIEVED (never model output): the only place a number or dose may come from."""
    parts = []
    for p in state.get("papers", {}).values():
        parts += [p.get("title") or "", p.get("abstract") or "", str(p.get("year")), str(p.get("date"))]
    for t in state.get("trials_by_id", {}).values():
        parts += [t.get("title") or "", t.get("primary_endpoint") or "", " ".join(t.get("interventions") or []), str(t.get("last_update")), str(t.get("start_date")), str(t.get("estimated_completion"))]
    for w in state.get("web", []):
        parts += [w.get("title") or "", w.get("snippet") or ""]
    for r in state.get("regulatory", []):
        parts += [r.get("indication_excerpt") or "", r.get("indication_statement") or "", r.get("dose_statement") or "", str(r.get("label_date"))]
    for g in state.get("guidelines", []):
        parts += [g.get("title") or "", g.get("excerpt") or "", g.get("text") or "", str(g.get("year"))]
    for d in state.get("drug_labels", []):
        parts += [d.get("boxed_warning") or ""]
    return " ".join(parts)


def _is_row(line: str) -> bool:
    return line.lstrip().startswith("|")


PERSONAL_ADVICE = re.compile(r"\b(?:you|your (?:child|son|daughter|patient)|the patient) (?:should|must|need to|needs to|has been diagnosed|have been diagnosed|is diagnosed|are diagnosed)\b"
                             r"|\bI (?:recommend|advise|suggest|diagnose|prescribe)\b|\bwe recommend (?:that )?(?:you|the patient)\b"
                             r"|\b(?:start|stop|give|prescribe|increase|decrease) (?:the |your |this )?(?:patient|child|him|her)\b", re.I)
GENERAL_NOTE = "General information from the retrieved literature; it is not a diagnosis or a personal treatment decision."
LIST_MARK = re.compile(r"^(\s*(?:\d{1,3}[.)]|[-*•])\s+)")
NUMBERED = re.compile(r"^(\s*)(\d{1,3})([.)])(\s+)")
ABBREVIATION = re.compile(r"\b(?:et al|e\.g|i\.e|vs|cf|approx|Fig|Dr|Prof|St|Jr|Sr|ca)\.", re.I)


def is_list_item(line: str) -> bool:
    return bool(LIST_MARK.match(line))


def _sentences(line: str) -> list[str]:
    """Sentence split that does not cut at 'et al.', 'e.g.', 'vs.', 'Dr.' or at a list number such as '3.' (the list marker stays with the first sentence)."""
    marker = ""
    m = LIST_MARK.match(line)
    if m:
        marker, line = m.group(1), line[m.end():]
    line = ABBREVIATION.sub(lambda a: a.group(0).replace(".", "\x00"), line)
    parts = [p.replace("\x00", ".") for p in re.split(r"(?<=[.!?])\s+(?=[A-Z*(\[“\"])", line)]
    if marker:
        parts[0] = marker + parts[0]
    return parts


def _append_body(text: str, addition: str) -> str:
    """Add a paragraph to the report body, BEFORE the Sources list (rebuilding Sources cuts everything after it)."""
    cut = text.find("\n### Sources")
    return (text[:cut].rstrip() + "\n\n" + addition + "\n" + text[cut:]) if cut >= 0 else text.rstrip() + "\n\n" + addition


def renumber_lists(text: str) -> str:
    """Every numbered list counts 1, 2, 3 ... with no gap, whatever a repair removed. A list ends at a heading, rule or unindented paragraph."""
    out, count = [], 0
    for ln in text.splitlines():
        m = NUMBERED.match(ln)
        if m:
            count += 1
            ln = f"{m.group(1)}{count}{m.group(3)}{m.group(4)}" + ln[m.end():]
        elif ln.startswith("#") or ln.startswith("---") or (ln.strip() and not ln.startswith((" ", "\t")) and not is_list_item(ln)):
            count = 0
        out.append(ln)
    return "\n".join(out)


# ------------------------------------------------------------------ formatting lint (blocks PASS)
LINT_EXEMPT = re.compile(r"^\s*(?:#|\||---|>|\*{0,2}(?:Verification|Sources)\b)")


def lint_report(text: str) -> list[dict]:
    """Formatting defects a clinician would notice: a line cut off mid-sentence, an unbalanced ** marker, a numbered list that does not run 1, 2, 3."""
    issues: list[dict] = []
    expected, in_sources = 0, False
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("#"):
            in_sources = bool(re.match(r"#+\s*Sources", s, re.I))
        if not s or in_sources:
            continue
        m = NUMBERED.match(ln)
        if m:
            expected += 1
            if int(m.group(2)) != expected:
                issues.append(_issue("lint_numbering", "high", f"numbered list runs {m.group(2)} where {expected} was expected", s[:50], "lint_fix", line=ln))
                expected = int(m.group(2))
        elif s.startswith("#") or s.startswith("---") or (not ln.startswith((" ", "\t")) and not is_list_item(ln)):
            expected = 0
        if LINT_EXEMPT.match(ln):
            continue
        if ln.count("**") % 2:
            issues.append(_issue("lint_stray_bold", "high", "unbalanced ** marker (a line was cut off or edited mid-way)", s[:60], "lint_fix", line=ln))
        body = LIST_MARK.sub("", s)
        if len(body.split()) >= 5 and (re.search(r"\bet al\.?\**\s*$", body) or (not re.search(r"[.!?:)\]\"”*|]\**$", body) and not re.search(r"https?://\S+$", body) and body[-1:].isalnum() and len(body.split()) >= 8)):
            issues.append(_issue("lint_cut_off", "high", "a line ends in the middle of a sentence", s[:60], "lint_fix", line=ln))
    return issues


def lint_fix(text: str) -> tuple[str, list[str]]:
    """A cut-off or unbalanced list item is removed whole (never completed with invented words); other lines lose the stray marker. Lists are renumbered."""
    actions, out, in_sources = [], [], False
    for ln in text.splitlines():
        if ln.startswith("#"):
            in_sources = bool(re.match(r"#+\s*Sources", ln, re.I))
        if in_sources:                       # the Sources list is rebuilt from the text, never edited here
            out.append(ln)
            continue
        bad = [i for i in lint_report(ln) if i["kind"] in ("lint_cut_off", "lint_stray_bold")]
        if bad and is_list_item(ln):
            actions.append("removed a truncated list item")
            continue
        if any(i["kind"] == "lint_stray_bold" for i in bad):
            pos = ln.rfind("**")
            ln = ln[:pos] + ln[pos + 2:]
            actions.append("removed a stray ** marker")
        out.append(ln)
    # a label such as 'Effectiveness:' with nothing under it (the model's output stopped there) is removed
    kept: list[str] = []
    for i, ln in enumerate(out):
        nxt = next((x for x in out[i + 1:] if x.strip()), "")
        if re.fullmatch(r"\s*\*{0,2}[A-Za-z][A-Za-z /&-]{2,40}:\*{0,2}\s*", ln) and (not nxt or nxt.startswith(("#", "---"))):
            actions.append("removed an empty label")
            continue
        kept.append(ln)
    return renumber_lists("\n".join(kept)), actions


def _label_before(text: str, pos: int) -> str:
    start = max(text.rfind(c, 0, pos) for c in "(;]\n|")
    return text[start + 1:pos]


def _fda_relevant(r: dict) -> bool:
    from landscape import fda_relevant
    return fda_relevant(r)


def is_indirect(paper: dict, stems: set[str] | None = None) -> bool:
    """Indirect evidence (a related or different population, an associated condition, a related intervention) as decided by the evidence triage; False when there is no triage decision."""
    return (paper.get("triage") or {}).get("relevance") == "indirect"


MONTHS = {m: i for i, m in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}


def date_after(date_text: str, today) -> bool:
    """True when a publication date ('2026 Dec', '2026 Oct 20', '2026-12-01', '2027') is later than today. Day-level when the day is given, month-level when the month is, year-level otherwise."""
    text = str(date_text or "")
    year = re.search(r"\b(19|20)\d\d\b", text)
    if not year:
        return False
    y = int(year.group(0))
    low = text.lower()
    month = next((n for k, n in MONTHS.items() if re.search(rf"\b{k}", low)), None)
    day = None
    if month is not None:
        d = re.search(r"\b(?:" + "|".join(MONTHS) + r")[a-z]*\.?\s+(\d{1,2})\b", low)
        day = int(d.group(1)) if d else None
    else:
        m = re.search(r"\b(?:19|20)\d\d[-/](\d{1,2})(?:[-/](\d{1,2}))?\b", text)
        if m:
            month = int(m.group(1))
            day = int(m.group(2)) if m.group(2) else None
    if y != today.year:
        return y > today.year
    if month is None:
        return False
    if month != today.month:
        return month > today.month
    return day is not None and day > today.day


DATED_BULLET = re.compile(r"(?m)^(\s*[-*]\s*(?:\*\*)?)((?:19|20)\d\d(?:[ \-/][A-Za-z]{3,9}(?: \d{1,2})?|-\d{1,2}(?:-\d{1,2})?)?)(?=\s*[:*\)–—-])")      # '2026 Dec', '2027 Jan 15', '2026-12-01', '2027' at the start of a bullet


def fix_future_dates(text: str, state: dict) -> tuple[str, list[str]]:
    """A report that says the evidence was searched on date X cannot present a finding as dated after X: such a date is shown as an advance publication with its issue date. General: it only compares dates."""
    from datetime import datetime
    try:
        today = datetime.strptime(str(state.get("searched_on") or ""), "%Y-%m-%d")
    except ValueError:
        today = datetime.now()
    actions = []

    def fix(m):
        if date_after(m.group(2), today):
            actions.append("marked an advance-publication date (later than the search date)")
            return f"{m.group(1)}Advance publication (issue dated {m.group(2)})"
        return m.group(0)
    return DATED_BULLET.sub(fix, text), actions


def _failed_sources(state: dict) -> list[str]:
    return [k for k, v in state.get("source_status", {}).items() if v.get("retrieval_status") != "success"]


# ------------------------------------------------------------------ citation normalisation (adds retrieved identifiers only)
def normalize_citations(text: str, state: dict) -> tuple[str, list[str]]:
    """'(Surname et al., Journal, Year)' -> add [PMID n] when exactly one retrieved paper matches surname + year (the journal only breaks a tie).
    Grouped citations are handled one by one. Returns (text, unresolved citation strings)."""
    text = re.sub(r"PMID\s*\[(\d{6,9})\]\(https?://[^)\s]*\)", r"PMID \1", text)      # 'PMID [123](https://...)' is the same citation as 'PMID 123'
    text = re.sub(r"\[PMID\s*(\d{6,9})\]\(https?://[^)\s]*\)", r"[PMID \1]", text)
    papers = state.get("papers", {}).values()
    unresolved: list[str] = []

    def fix(m):
        surname, year = m.group(1).lower(), m.group(2)
        found = [p for p in papers if (p["authors"][0].split(" ")[0].lower() if p.get("authors") else "")[:5] == surname[:5] and p.get("year") == year]
        if len(found) > 1:
            label = m.group(0).lower()
            narrowed = [p for p in found if any(w in label for w in re.findall(r"[a-z]{4,}", (p.get("journal") or "").lower())[:2])]
            found = narrowed or found
        if len(found) == 1:
            return f"{m.group(0)} [PMID {found[0]['pmid']}]"
        unresolved.append(m.group(0))
        return m.group(0)

    fixed = AY_RE.sub(fix, text)
    # make sure the year is visible next to every PMID citation
    def add_year(m):
        p = state.get("papers", {}).get(m.group(1))
        label = _label_before(fixed_holder[0], m.start())
        if p and p.get("year") and not re.search(r"(?:19|20)\d\d", label):
            return f"{p['year']} {m.group(0)}"
        return m.group(0)
    fixed_holder = [fixed]
    fixed = re.sub(r"\[PMID[:\s]*([0-9]{6,9})\]", add_year, fixed)
    fixed = re.sub(r"(\[PMID (\d{6,9})\])[*_]*\s*(?:(?:19|20)\d\d\s*)?\[PMID \2\]", r"\1", fixed)   # a doubled citation collapses to one
    return fixed, unresolved


def citation_mismatches(text: str, state: dict) -> list[tuple[str, str]]:
    """A '[PMID n]' whose label contradicts the retrieved record (wrong year, wrong first author, or a different journal when no author is named)."""
    bad = []
    for m in re.finditer(r"\[PMID[:\s]*(\d{6,9})\]", text):
        p = state.get("papers", {}).get(m.group(1))
        if not p:
            continue
        label = _label_before(text, m.start()).lower()
        surname = (p["authors"][0].split(" ")[0].lower() if p.get("authors") else "")
        years = re.findall(r"(?:19|20)\d\d", label)
        has_author = "et al" in label
        wrong_year = bool(years) and p.get("year") not in years
        wrong_author = has_author and bool(surname) and surname[:5] not in label
        words = [w for w in re.findall(r"[a-z]{4,}", (p.get("journal") or "").lower()) if w not in {"journal", "international"}]
        acronym = "".join(w[0] for w in re.findall(r"[a-z]+", (p.get("journal") or "").lower()) if w not in {"the", "of", "and", "for", "in"})
        wrong_journal = (not has_author) and bool(words) and bool(re.search(r"[a-z]{3,}", label)) and not any(w in label for w in words) and acronym not in label
        if wrong_year or wrong_author or wrong_journal:
            bad.append((m.group(1), label.strip()[:60]))
    return bad


def unsupported_numbers(text: str, state: dict) -> list[str]:
    raw = evidence_blob(state)
    blob, found = _norm(raw), []
    for pattern in NUMBER_PATTERNS:
        for m in re.finditer(pattern, text, re.I):
            token = m.group(0).strip()
            size = re.fullmatch(r"n\*?\s?=\s?([\d,]+)", token, re.I)
            if size and re.search(rf"(?<![\d.]){re.escape(size.group(1).replace(',', ''))}(?![\d.])\s*(?:[A-Za-z\-]+\s+){{0,2}}"
                                  r"(?:participants|children|patients|subjects|people|individuals|youth|adults|infants|studies|trials|cases|controls|dyads|families)",
                                  raw.replace(",", ""), re.I):
                continue  # "n = 62" is supported by "62 children" in the source; a bare number that merely equals a year is not support
            if _norm(token) not in blob and token not in found:
                found.append(token)
    # a token wholly contained in a larger flagged token is covered by it
    return [t for t in found if not any(t != o and t in o for o in found)]


# ------------------------------------------------------------------ sources are ALWAYS rebuilt from the answer text
def rebuild_sources(text: str, state: dict) -> str:
    """Drop the old Sources list and rebuild it from what the final text actually cites, so no orphan reference can remain."""
    cut = text.find("\n### Sources")
    body = text[:cut].rstrip() if cut >= 0 else text.rstrip()
    papers, lines = state.get("papers", {}), []
    for pmid in sorted(set(PMID_RE.findall(body)), key=lambda x: -(int(papers[x]["year"] or 0) if x in papers and str(papers[x].get("year") or "").isdigit() else 0)):
        p = papers.get(pmid)
        if p and not p.get("retracted"):
            first = p["authors"][0] if p.get("authors") else "Unknown"
            lines.append(f"{first} et al.: {p['title']} *{p['journal']}*. {p['date']}. PMID {pmid}" + (f". DOI {p['doi']}" if p.get("doi") else ""))
    urls_in_body = {_norm_url(u) for u in URL_RE.findall(body)}
    for w in state.get("web", []):
        if w.get("url") and _norm_url(w["url"]) in urls_in_body:
            lines.append(f"{w['title']}. {w['url']}")
    for r in state.get("regulatory", []):
        if r.get("url") and _norm_url(r["url"]) in urls_in_body:
            lines.append(f"FDA drug label: {r['generic']} (label effective {r.get('label_date')}). {r['url']}")
    for g in state.get("guidelines", []):
        if g.get("url") and _norm_url(g["url"]) in urls_in_body:
            lines.append(f"{g['organisation']}: {g['title']}. {g['url']}")
    if not lines:
        return body
    return body + "\n\n### Sources\n" + "\n".join(f"{i}. {l}" for i, l in enumerate(lines, 1))


# ------------------------------------------------------------------ find issues (read-only)
def _issue(kind: str, severity: str, message: str, needle: str, action: str, **extra) -> dict:
    return {"kind": kind, "severity": severity, "message": message, "needle": needle, "action": action, **extra}


def find_issues(text: str, state: dict) -> list[dict]:
    issues: list[dict] = []
    K = known_sources(state)
    papers, trials = state.get("papers", {}), state.get("trials_by_id", {})
    reg_names = {r["generic"].lower() for r in state.get("regulatory", [])}
    stems = set(state.get("core_stems") or [])

    # 1. every identifier must exist in the ResearchState
    for m in re.finditer(r"\bPMID\b[:\s]*(\d[0-9A-Za-z\-/]*)", text):      # e.g. 'PMID 2025-77457-001' is not a PubMed ID at all
        if not re.fullmatch(r"\d{6,9}", m.group(1)):
            issues.append(_issue("malformed_pmid", "high", f"'{m.group(0)}' is not a valid PubMed ID", m.group(0), "remove_lines", regex=re.escape(m.group(0))))
    for pmid in sorted(set(PMID_RE.findall(text)) - K["pmids"]):
        issues.append(_issue("unretrieved_pmid", "high", f"PMID {pmid} is not in the retrieved data", f"PMID {pmid}", "remove_lines"))
    for n in sorted({"NCT" + x for x in NCT_RE.findall(text)} - K["ncts"]):
        issues.append(_issue("unretrieved_nct", "high", f"{n} is not in the retrieved registry data", n[3:], "remove_lines", regex=r"NCT\s?" + n[3:]))
    for doi in sorted({d.rstrip(".,;)]").lower() for d in DOI_RE.findall(_without_foreign_urls(text))} - K["dois"]):
        issues.append(_issue("unretrieved_doi", "high", f"DOI {doi} was not retrieved", doi, "strip_token"))
    for url in sorted({u.rstrip(".,;:)") for u in URL_RE.findall(text)}):
        if _norm_url(url) not in K["urls"]:
            issues.append(_issue("unretrieved_url", "high", f"URL {url} was not retrieved", url, "strip_token"))

    # 2. citations must match their retrieved record
    for pmid, label in citation_mismatches(text, state):
        issues.append(_issue("citation_mismatch", "high", f"citation label '{label}' does not match the retrieved record for PMID {pmid}", f"PMID {pmid}", "remove_lines_mismatch", pmid=pmid))
    _, unresolved = normalize_citations(text, state)
    for u in unresolved:
        issues.append(_issue("unresolved_citation", "high", f"'{u}' does not match any retrieved paper", u, "remove_lines"))

    # 3. unsupported numbers / doses / leftover markers
    for token in unsupported_numbers(text, state):
        issues.append(_issue("unsupported_number", "high", f"'{token}' does not appear in any retrieved source", token, "fix_number"))
    if "[value not in retrieved data]" in text:
        issues.append(_issue("marker_leftover", "high", "an unfilled value marker remains in the answer", "[value not in retrieved data]", "fix_number"))

    # 4. claim-level checks on the model-written lines (not on the Sources list, which is rebuilt from the text)
    claim_text = text.split("\n### Sources")[0]
    in_verified_tables = False
    for line in claim_text.splitlines():
        if line.startswith("## Treatment Drug Landscape"):
            in_verified_tables = True
        elif line.startswith("## ") or line.startswith("---"):
            in_verified_tables = False
        if not line.strip() or line.startswith("#") or in_verified_tables:
            continue      # code-built tables (a source sentence, a registry record or an FDA label): identifiers in them are checked separately above
        row_of_code_table = _is_row(line) and ("NCT" in line.split("|")[1] if len(line.split("|")) > 2 else False)
        ids = sorted(set(PMID_RE.findall(line)) & K["pmids"])
        if len(ids) == 1:
            p = papers[ids[0]]
            low = line.lower()
            src_text = f"{p.get('title') or ''} {p.get('abstract') or ''}".lower()
            for term in OUTCOME_TERMS:
                if term in low and term not in src_text:
                    issues.append(_issue("outcome_not_in_source", "medium", f"the claim mentions '{term}', which the cited source (PMID {ids[0]}) does not report", term, "remove_sentence", pmid=ids[0], line=line))
                    break
            if p.get("protocol") and EFFICACY_WORDS.search(re.sub(r"\([^)]*\)", "", line)) and not re.search(r"protocol|no efficacy|does not establish|not (?:yet )?(?:available|reported)|pending|planned", low):
                issues.append(_issue("protocol_as_efficacy", "high", f"PMID {ids[0]} is a protocol and cannot show efficacy", ids[0], "remove_sentence_pmid", pmid=ids[0], line=line))
            if is_indirect(p, stems) and not INDIRECT_LABEL.search(line):
                issues.append(_issue("indirect_unlabeled", "medium", f"PMID {ids[0]} studies a different or related population but is not labelled indirect", ids[0], "tag_indirect", pmid=ids[0], line=line))
        if any(p.search(line) for p, _ in DOSE_LANGUAGE):
            issues.append(_issue("dose_response_as_prescription", "medium", "a dose-response finding is worded as if it were a validated dose effect", "dose-response", "soften_dose", line=line))
        if OVERCLAIM.search(line):
            issues.append(_issue("overclaim_language", "medium", "'confirmed/proven' wording is stronger than a single cited study supports", OVERCLAIM.search(line).group(0), "soften_word", line=line))
        if not row_of_code_table:
            for nct in {"NCT" + x for x in NCT_RE.findall(line)} & K["ncts"]:
                t = trials[nct]
                for sentence in _sentences(line):
                    if nct not in sentence:
                        continue
                    if re.search(r"\bunpublished\b|\bnot published\b", sentence, re.I) or (
                            (t.get("results_posted") and re.search(r"\bno (?:posted |published |reported )?results\b|results (?:were |are |remain )?(?:not|never) (?:posted|available|reported|published)|\bwithout (?:posted )?results\b", sentence, re.I))
                            or (not t.get("results_posted") and re.search(r"\b(?:results|data) (?:were |are |have been )?(?:posted|published|reported)\b|\bposted results\b", sentence, re.I)
                                and not re.search(r"\bno\b|\bnot\b|\bpending\b|\bawait", sentence, re.I))):
                        issues.append(_issue("trial_results_contradiction", "high", f"{nct}: a statement about results/publication contradicts or goes beyond the registry record", sentence[:80], "remove_sentence", line=line, sentence=sentence))
                if EFFICACY_WORDS.search(line) and not t.get("results_posted") and not re.search(r"not (?:proof|evidence)|cannot be inferred|no results|pending|not yet|under investigation|does not|research activity", line.lower()):
                    issues.append(_issue("trial_efficacy_language", "high", f"{nct} has no posted results but is described with efficacy language", nct[3:], "remove_lines", regex=r"NCT\s?" + nct[3:], line=line))
        # approval claims need a regulatory record
        for sentence in ([line] if _is_row(line) else _sentences(line)):
            if APPROVAL_POSITIVE.search(sentence) and not NEGATION.search(sentence) and not any(n.split()[0] in sentence.lower() for n in reg_names):
                issues.append(_issue("unsupported_approval_claim", "high", "an approval claim is not backed by a retrieved regulatory source", sentence[:80], "fix_approval", line=line, sentence=sentence))
            low_s = sentence.lower()
            if "off-label" in low_s and any(_fda_relevant(r) and r["generic"].lower().split()[0] in low_s for r in state.get("regulatory", [])):
                issues.append(_issue("off_label_vs_approved", "high", "a drug with a labelled indication matching the condition is called off-label", sentence[:80], "remove_sentence", line=line, sentence=sentence))

    # 4b. 'Established' needs a guideline (or consistent high-quality evidence); one study or one review of variable quality is not enough
    def _is_guideline(pm: str) -> bool:
        p = papers.get(pm) or {}
        return any("guideline" in str(t).lower() for t in (p.get("type") or [])) or "guideline" in (p.get("title") or "").lower()

    cl = claim_text.splitlines()
    for ln in cl:      # an 'Established' evidence level in a table row needs a cited guideline (a single review is 'evidence-supported but limited')
        if _is_row(ln) and re.search(r"\|\s*Established\s*\|", ln):
            cited = set(PMID_RE.findall(ln)) & K["pmids"]
            if cited and not any(_is_guideline(x) for x in cited):
                issues.append(_issue("established_without_guideline", "high", "a table row is labelled 'Established' but no cited source is a guideline", ln.strip()[:60], "relabel_established", line=ln))
    for i, ln in enumerate(cl):
        if ESTABLISHED_HEADING.match(ln.strip()):
            block = []
            for nxt in cl[i + 1:]:
                if not nxt.strip() or nxt.startswith(("#", "---")) or re.match(r"^\W*\*{2}[^*]+\*{2}\s*:?\s*$", nxt.strip()):
                    break
                block.append(nxt)
            cited = set(PMID_RE.findall(" ".join(block))) & K["pmids"]
            if cited and not any(_is_guideline(x) for x in cited):
                issues.append(_issue("established_without_guideline", "high", "labelled 'Established' but no cited source is a guideline", ln.strip()[:60], "relabel_established", line=ln))
        elif ESTABLISHED_INLINE.search(ln):
            cited = set(PMID_RE.findall(ln)) & K["pmids"]
            if cited and not any(_is_guideline(x) for x in cited):
                issues.append(_issue("established_without_guideline", "high", "described as established but no cited source is a guideline", ln.strip()[:60], "relabel_established", line=ln))

    # 5. contradictions / retrieval-failure semantics (sentence level)
    failed = _failed_sources(state)
    drug_failed = [k for k in failed if k.startswith(("fda", "web"))]
    for line in claim_text.splitlines():
        if _is_row(line) or line.startswith("#"):
            continue
        for sentence in _sentences(line):
            if NO_DRUGS_CLAIM.search(sentence) and state.get("regulatory") and not SCOPED.search(sentence):
                issues.append(_issue("contradiction_no_drugs", "high", "says no drug treatment exists/was found while regulatory drug records were retrieved", sentence[:80], "remove_sentence", line=line, sentence=sentence))
            elif NON_EXISTENCE.search(sentence) and drug_failed and re.search(r"drug|medication|pharmacolog|treatment|approved", sentence, re.I) and not SCOPED.search(sentence):
                issues.append(_issue("failure_as_absence", "high", "states non-existence although a drug/regulatory source failed or was partial", sentence[:80], "scope_sentence", line=line, sentence=sentence))

    # 6. mandatory sections and disclosure
    if "drug landscape" not in text.lower():
        issues.append(_issue("missing_drug_landscape", "high", "the Treatment Drug Landscape section is missing", "", "none"))
    if state.get("drug_intelligence_status") in ("partial", "failed") and not re.search(r"drug landscape is incomplete|incomplete because", text, re.I):
        issues.append(_issue("drug_incomplete_not_disclosed", "high", "the drug landscape is incomplete but the answer does not say so", "", "add_drug_disclosure"))
    if "evidence searched on" not in text.lower():
        issues.append(_issue("missing_search_date", "medium", "the 'Evidence searched on' date is missing", "", "none"))

    # 7. established care: every row must cite a retrieved guideline or say that none was retrieved
    known_guideline_urls = {_norm_url(g["url"]) for g in state.get("guidelines", []) if g.get("url")}
    in_care = False
    for ln in text.splitlines():
        if ln.startswith("#"):
            in_care = bool(re.match(r"#+\s*Established care", ln, re.I))
            continue
        if in_care and _is_row(ln) and not set(ln.strip()) <= set("|-: ") and not ln.lstrip().startswith("| Topic"):
            urls = {_norm_url(u) for u in URL_RE.findall(ln)}
            if not (urls and urls <= known_guideline_urls) and "No guideline retrieved" not in ln and "search unavailable" not in ln:
                issues.append(_issue("established_care_unsourced", "high", "an Established care row cites no retrieved guideline", ln.strip()[:60], "remove_lines", regex=re.escape(ln.strip()[:60])))

    # 8. diagnosis / personal-advice wording: FLAGGED and a general-information note appended; the sentence itself is never reworded
    if PERSONAL_ADVICE.search(text) and GENERAL_NOTE not in text:
        hit = PERSONAL_ADVICE.search(text).group(0)
        issues.append(_issue("personal_advice", "low", f"wording that reads like a diagnosis or a personal treatment decision ('{hit}')", hit, "append_note"))

    # 9. formatting (a truncated or mis-numbered list must never reach the reader)
    issues.extend(lint_report(text))
    return issues


# ------------------------------------------------------------------ repair (exact spans only)
def _remove_lines(text: str, regex: str) -> tuple[str, int]:
    kept = [ln for ln in text.splitlines() if not re.search(regex, ln)]
    return "\n".join(kept), len(text.splitlines()) - len(kept)


def _clean_line_after_edit(line: str) -> str | None:
    core = re.sub(r"[\s\-*•:|]+", "", line)
    return None if len(core) < 12 else line


def _remove_sentence(text: str, needle: str, line: str | None = None) -> str:
    out = []
    for ln in text.splitlines():
        if needle in ln and (line is None or ln == line):
            if _is_row(ln) or is_list_item(ln):
                continue  # a table row or a list item is one claim: it is removed whole, never left as a stub
            kept = [s for s in _sentences(ln) if needle not in s]
            new = " ".join(kept).strip()
            if new and _clean_line_after_edit(new) and not re.fullmatch(r"[\s\-*•]*\*?\([^()]*\)\*?\s*", new):   # not a stranded "(Direct, 2026)" annotation
                out.append(new)
            continue
        out.append(ln)
    return "\n".join(out)


def _set_cell(text: str, needle: str, value: str) -> str:
    out = []
    for ln in text.splitlines():
        if _is_row(ln) and needle in ln:
            cells = ln.strip().strip("|").split("|")
            cells = [(" " + value + " ") if needle in c else c for c in cells]
            out.append("|" + "|".join(cells) + "|")
        else:
            out.append(ln)
    return "\n".join(out)


def _drop_empty_tables(text: str) -> str:
    """A table whose every row was removed by a repair (header + separator only) is replaced by 'None retrieved.'"""
    lines, out, i = text.splitlines(), [], 0
    while i < len(lines):
        is_header = lines[i].lstrip().startswith("|")
        has_sep = i + 1 < len(lines) and lines[i + 1].lstrip().startswith("|") and set(lines[i + 1].strip()) <= set("|-: ")
        no_rows = i + 2 >= len(lines) or not lines[i + 2].lstrip().startswith("|")
        if is_header and has_sep and no_rows:
            out.append("None retrieved.")
            i += 2
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


LINE_EDITS = {"soften_dose", "soften_word", "relabel_established", "tag_indirect"}


def _current_line(text: str, original: str) -> str:
    """Several issues can sit on one line and each repair edits it. Find the line as it is NOW (exact, else by its first characters)."""
    lines = text.splitlines()
    if original in lines:
        return original
    prefix = original[:40]
    return next((ln for ln in lines if prefix and ln.startswith(prefix)), original)


def _merge_duplicate_headings(text: str) -> str:
    """After a relabel two bold-only headings can be identical inside one section: keep the first, so the bullets merge under it."""
    seen, out = set(), []
    for ln in text.splitlines():
        if ln.startswith(("#", "---")):
            seen = set()
        elif re.fullmatch(r"\*\*[^*]+\*\*", ln.strip()):
            if ln.strip() in seen:
                continue
            seen.add(ln.strip())
        out.append(ln)
    return "\n".join(out)


def apply_repairs(text: str, issues: list[dict], state: dict) -> tuple[str, list[str]]:
    """ONE repair pass over the exact failed spans. Nothing is invented: a failed claim is removed, scoped or set to 'Not retrieved'."""
    actions: list[str] = []
    # phase 1: removals and replacements of whole claims / cells / identifiers
    for it in [i for i in issues if i["action"] not in LINE_EDITS]:
        a, needle = it["action"], it["needle"]
        if it.get("line"):
            it["line"] = _current_line(text, it["line"])      # an earlier repair may already have edited this line
        before = text
        if a == "remove_lines":
            text, n = _remove_lines(text, it.get("regex") or re.escape(needle))
            if n:
                actions.append(f"removed {n} line(s): {it['message']}")
        elif a == "remove_lines_mismatch":
            text, n = _remove_lines(text, r"PMID[:\s]*" + it["pmid"])
            if n:
                actions.append(f"removed {n} line(s) whose citation contradicted the retrieved record for PMID {it['pmid']}")
        elif a == "strip_token":
            text = re.sub(r"\[([^\]]*)\]\(" + re.escape(needle) + r"\)", r"\1", text)  # markdown link -> keep its text
            text = text.replace(needle, "")
            text = re.sub(r"\bDOI[:\s]*\.?(?=\s*(?:\n|$|\|))", "", text)  # no dangling 'DOI' label
            actions.append(f"removed unretrieved identifier/link: {needle}")
        elif a == "fix_number":
            if needle == "[value not in retrieved data]" or needle in text:
                if any(_is_row(ln) and needle in ln for ln in text.splitlines()):
                    text = _set_cell(text, needle, "Not retrieved")
                    actions.append(f"set table cell with unsupported value '{needle}' to 'Not retrieved'")
                else:
                    text = _remove_sentence(text, needle)
                    actions.append(f"removed sentence with unsupported value '{needle}'")
        elif a == "remove_sentence":
            text = _remove_sentence(text, it.get("sentence") or needle, it.get("line"))
            actions.append(f"removed unsupported claim: {it['message']}")
        elif a == "remove_sentence_pmid":
            text = _remove_sentence(text, f"PMID {it['pmid']}", it.get("line"))
            actions.append(f"removed protocol-as-efficacy claim (PMID {it['pmid']})")
        elif a == "fix_approval":
            if _is_row(it.get("line", "")):
                cells = it["line"].strip().strip("|").split("|")
                new_cells = [" Approval status not verified in retrieved regulatory sources " if APPROVAL_POSITIVE.search(c) else c for c in cells]
                text = text.replace(it["line"], "|" + "|".join(new_cells) + "|", 1)
            else:
                text = _remove_sentence(text, it["sentence"], it.get("line"))
            actions.append("replaced an unsourced approval claim")
        elif a == "tag_indirect":
            new = it["line"].rstrip() + " *(indirect evidence: different or related population)*" if not _is_row(it["line"]) else it["line"]
            text = text.replace(it["line"], new, 1)
            actions.append(f"labelled PMID {it['pmid']} as indirect evidence")
        elif a == "relabel_established":
            line = it["line"]
            new = re.sub(r"\*\*established\*\*(?=\s*:)", "**Evidence-supported but limited**", line, flags=re.I)        # a label followed by a colon
            new = re.sub(r"\*\*established\*\*", "supported by limited evidence", new, flags=re.I)                   # inside a sentence
            new = re.sub(r"\b(is|are) established\b", r"\1 supported by limited evidence", new, flags=re.I)
            new = re.sub(r"\bestablished (to|for|treatment)\b", r"supported by limited evidence \1", new, flags=re.I)
            if ESTABLISHED_HEADING.match(line.strip()):
                new = "**Evidence-supported but limited**"
            text = text.replace(line, new, 1)
            text = _merge_duplicate_headings(text)
            actions.append("relabelled 'Established' as 'Evidence-supported but limited' (no guideline among the cited sources)")
        elif a == "soften_dose":
            line = it["line"]
            new = line
            for pattern, replacement in DOSE_LANGUAGE:
                new = pattern.sub(replacement, new)
            text = text.replace(line, new, 1)
            actions.append("reworded dose-response language as an exploratory finding")
        elif a == "soften_word":
            line = it["line"]
            new = OVERCLAIM.sub(lambda m: SOFTER.get(m.group(0).lower(), m.group(0)), line)
            text = text.replace(line, new, 1)
            actions.append(f"softened '{needle}' (a single study cannot 'confirm' or 'prove')")
        elif a == "scope_sentence":
            names = ", ".join(_failed_sources(state)[:4])
            text = text.replace(it["sentence"], f"Retrieval was incomplete ({names}); the absence of drugs or approvals cannot be concluded.", 1)
            actions.append("replaced a non-existence claim with a retrieval-failure statement")
        elif a == "append_note":
            if GENERAL_NOTE not in text:
                text = _append_body(text, f"*{GENERAL_NOTE}*")
            actions.append("flagged diagnosis/personal-advice wording and added a general-information note (the wording was not changed)")
        elif a == "add_drug_disclosure":
            text = _append_body(text, "**Drug landscape is incomplete because a regulatory/drug source failed or was only partly retrieved; this does not show that no drug exists.**")
            actions.append("added the drug-landscape incompleteness disclosure")
        if text == before and a not in ("none",):
            pass
    # phase 2: wording edits. ALL edits for one line are applied together, so one edit cannot hide the line from another.
    by_line: dict[str, list[dict]] = {}
    for it in issues:
        if it["action"] in LINE_EDITS:
            by_line.setdefault(it["line"], []).append(it)
    for line, its in by_line.items():
        line = _current_line(text, line)          # a phase-1 removal may already have shortened this line
        kinds, new = {i["action"] for i in its}, line
        if "soften_dose" in kinds:
            for pattern, replacement in DOSE_LANGUAGE:
                new = pattern.sub(replacement, new)
            actions.append("reworded dose-response language as an exploratory finding")
        if "soften_word" in kinds:
            new = OVERCLAIM.sub(lambda m: SOFTER.get(m.group(0).lower(), m.group(0)), new)
            actions.append("softened 'confirmed/proven/validated' (a single study cannot 'confirm' or 'prove')")
        if "relabel_established" in kinds:
            new = re.sub(r"\|\s*Established\s*\|", "| Evidence-supported but limited |", new)      # a table cell
            new = re.sub(r"\*\*established\*\*(?=\s*:)", "**Evidence-supported but limited**", new, flags=re.I)        # a label followed by a colon
            new = re.sub(r"\*\*established\*\*", "supported by limited evidence", new, flags=re.I)                   # inside a sentence
            new = re.sub(r"\b(is|are) established\b", r"\1 supported by limited evidence", new, flags=re.I)
            new = re.sub(r"\bestablished (to|for|treatment)\b", r"supported by limited evidence \1", new, flags=re.I)
            if ESTABLISHED_HEADING.match(line.strip()):
                new = "**Evidence-supported but limited**"
            actions.append("relabelled 'Established' as 'Evidence-supported but limited' (no guideline among the cited sources)")
        if "tag_indirect" in kinds:
            if _is_row(new):   # a table row: set the category and population cells (the model may have misstated the population)
                cells = new.strip().strip("|").split("|")
                if len(cells) >= 3:
                    cells[1] = " Indirect evidence (different or related population) "
                    cells[2] = " Different or related population (see the cited study); not condition-specific "
                    new = "|" + "|".join(cells) + "|"
            else:
                new = new.rstrip() + " *(indirect evidence: different or related population)*"
            actions.append("labelled the cited paper as indirect evidence (different or related population)")
        if new != line:
            text = text.replace(line, new, 1)
    if any(i["action"] == "relabel_established" for i in issues):
        text = _merge_duplicate_headings(text)
    text, lint_actions = lint_fix(text)          # always: removals above can leave a gap in a numbered list
    return _drop_empty_tables(text), actions + lint_actions


# ------------------------------------------------------------------ quality control + the full loop
BLOCKING_KINDS_GROUNDING = {"unretrieved_pmid", "unretrieved_nct", "unretrieved_doi", "unretrieved_url", "citation_mismatch", "unresolved_citation",
                            "unsupported_number", "marker_leftover", "outcome_not_in_source", "indirect_unlabeled"}


def quality_control(text: str, state: dict, issues: list[dict]) -> dict:
    """PASS only if the (repaired) final text has no remaining blocking issue and the retrieval status is represented honestly."""
    kinds = {i["kind"] for i in issues}
    checks = {
        "grounding": "fail" if kinds & BLOCKING_KINDS_GROUNDING else "pass",
        "drugs": "fail" if kinds & {"missing_drug_landscape", "drug_incomplete_not_disclosed", "unsupported_approval_claim", "off_label_vs_approved", "contradiction_no_drugs"} else "pass",
        "trials": "fail" if "trial_efficacy_language" in kinds or "protocol_as_efficacy" in kinds else "pass",
        "retrieval_semantics": "fail" if "failure_as_absence" in kinds else "pass",
        "completeness": "fail" if "missing_search_date" in kinds else "pass",
        "formatting": "fail" if kinds & {"lint_numbering", "lint_stray_bold", "lint_cut_off"} else "pass",
    }
    blocking = [i for i in issues if i["severity"] in ("high", "medium")]
    return {"status": "PASS" if not blocking else "NEEDS_REPAIR", "checks": checks, "remaining": [i["message"] for i in issues]}


def verify_and_repair(answer: str, state: dict, repair: bool = True, max_repairs: int = 1) -> dict:
    """WRITE -> normalise -> rebuild Sources -> VERIFY -> (one repair) -> rebuild Sources -> VERIFY AGAIN -> QC. Returns the final text and an honest status."""
    state.setdefault("repair_count", 0)
    answer, date_actions = fix_future_dates(answer, state)
    text, _ = normalize_citations(answer, state)
    text = rebuild_sources(text, state)
    issues_before = find_issues(text, state)
    actions: list[str] = []
    if issues_before and repair and state["repair_count"] < max_repairs:
        text, actions = apply_repairs(text, issues_before, state)
        state["repair_count"] += 1
        text, _ = normalize_citations(text, state)
        text = rebuild_sources(text, state)
    actions = date_actions + actions
    issues_after = find_issues(text, state)          # the SAME final string that will be shown
    qc = quality_control(text, state, issues_after)
    state["verification_status"] = qc["status"]
    cited = {"pmids": len(set(PMID_RE.findall(text))), "dois": len({d.rstrip('.,;)]').lower() for d in DOI_RE.findall(_without_foreign_urls(text))}),
             "trials": len({"NCT" + x for x in NCT_RE.findall(text)}), "urls": len({_norm_url(u) for u in URL_RE.findall(text)})}
    return {"answer": text, "status": qc["status"], "checks": qc["checks"], "issues_before": issues_before, "issues_after": issues_after,
            "repairs": [f"{a} (x{n})" if n > 1 else a for a, n in Counter(actions).items()], "repair_count": state["repair_count"], "checked": cited}
