"""Report sections built IN CODE from retrieved records, so a model cannot headline one paper, truncate a list, or invent a source.

- key_studies_markdown : authors, year, design, a VERBATIM finding sentence from the abstract, PMID. An entry missing any of the five is dropped.
- bottom_line_markdown : counts of what was retrieved by evidence tier + FDA-labelled drugs. It states no efficacy claim of its own.
- source_banner        : names the sources that STILL failed or were partial after retry, and says absence of evidence cannot be concluded from them.
"""
from __future__ import annotations

import re

from verification import DOSE_LANGUAGE, ESTABLISHED_INLINE, OVERCLAIM, _sentences, is_indirect

DISCLAIMER = "Research support, not a substitute for clinical judgement."

# rank 0 = strongest design; the rank decides which studies are shown first.
RESULT_MARK = re.compile(r"\b(?:Results?|Findings|Conclusions?|RESULTS?|FINDINGS|CONCLUSIONS?)\b\s*(?::|\.|(?=[A-Z]))")


def design_of(p: dict) -> tuple[str, int] | None:
    """Design label and rank from the record's own publication types; None when the record does not say (then the study is not listed)."""
    kinds = " | ".join(str(t) for t in (p.get("study_type") or []) + (p.get("type") or [])).lower()
    if "retracted" in kinds or "protocol" in kinds or p.get("protocol") or p.get("retracted"):
        return None
    for needle, label, rank in (("meta-analysis", "meta-analysis", 0), ("systematic review", "systematic review", 0), ("randomized controlled trial", "randomized controlled trial", 1),
                                ("randomised", "randomized controlled trial", 1), ("clinical trial", "clinical trial", 2), ("cohort", "cohort study", 2), ("case-control", "case-control study", 2),
                                ("observational", "observational study", 2), ("cross-sectional", "cross-sectional study", 2), ("review", "narrative review", 3), ("case-study", "case report", 4),
                                ("case report", "case report", 4)):
        if needle in kinds:
            return label, rank
    return None


RESULT_CUE = re.compile(r"\b(?:improv\w*|reduc\w*|increas\w*|decreas\w*|signific\w*|differ\w*|associat\w*|effect(?:s|ive|iveness)?|efficacy|superior|inferior|"
                        r"no (?:significant|difference|benefit|evidence)|did not|failed|showed|demonstrat\w*|found|resulted|outperform\w*|benefit\w*|worse|better|"
                        r"remission|response|odds|hazard|risk|SMD|OR|RR|CI)\b|\d\s?%|p\s?[<=]\s?0?\.\d", re.I)
NOT_A_RESULT = re.compile(r"\b(?:we (?:searched|aimed|conducted|included|reviewed|examined)|was conducted|were searched|searched|aims? (?:to|of)|objective|purpose of|"
                          r"this (?:review|study|paper) (?:aims|examines|reviews|investigates)|PRISMA|databases|inclusion criteria|is a (?:significant|common|prevalent)|often|"
                          r"well documented|remains? (?:one|a)|yielded|findings reflect|prevalence studies|studies tested|characterized by|prevalen\w+ of)\b", re.I)


def finding_sentence(p: dict) -> str | None:
    """One sentence copied unchanged from the Results / Conclusions of the abstract that reports an outcome; None if there is no clean one.
    Background and method sentences are never used as a 'finding'."""
    abstract = re.sub(r"\s+", " ", (p.get("abstract") or "")).strip()
    if not abstract:
        return None
    m = RESULT_MARK.search(abstract)
    sentences = [s.strip() for s in _sentences(abstract[m.end():] if m else abstract)]
    candidates = sentences if m else list(reversed(sentences[-4:]))        # no heading: the closing sentences are the conclusion
    for s in candidates:
        if not 40 <= len(s) <= 330 or s[-1] not in ".!?" or "**" in s or "|" in s:
            continue
        if NOT_A_RESULT.search(s) or not RESULT_CUE.search(s):
            continue
        if OVERCLAIM.search(s) or ESTABLISHED_INLINE.search(s) or any(pat.search(s) for pat, _ in DOSE_LANGUAGE):
            continue          # a repair would reword it; a quoted finding must stay verbatim, so it is not used
        return s
    return None


def first_author(p: dict) -> str | None:
    authors = p.get("authors") or []
    if not authors:
        return None
    surname = re.sub(r"[^A-Za-zÀ-ſ'\- ]", "", str(authors[0]).split()[0]) if str(authors[0]).split() else ""
    if len(surname) < 2:
        return None
    return surname if len(authors) == 1 else f"{surname} et al."


def key_study_entries(papers: list[dict], stems: set[str] | None = None, limit: int = 5) -> list[dict]:
    """Complete entries only: authors, year, design, finding, PMID. Direct evidence first, strongest design first, then newest."""
    entries = []
    for p in papers:
        author, design, finding = first_author(p), design_of(p), finding_sentence(p)
        year = str(p.get("year") or "")
        if not (author and design and finding and year.isdigit() and re.fullmatch(r"\d{6,9}", str(p.get("pmid") or ""))):
            continue
        entries.append({"paper": p, "author": author, "design": design[0], "rank": design[1], "finding": finding, "year": year, "pmid": p["pmid"],
                        "indirect": is_indirect(p, stems)})
    entries.sort(key=lambda e: (e["indirect"], e["rank"], -int(e["year"])))
    return entries[:limit]


def key_studies_markdown(papers: list[dict], stems: set[str] | None = None, limit: int = 5) -> str:
    entries = key_study_entries(papers, stems, limit)
    if not entries:
        return "## Key studies\n\nNo retrieved record had all of: authors, year, study design, a stated result and a PMID, so none is listed."
    lines = []
    for i, e in enumerate(entries, 1):
        note = " *(indirect evidence: different or related population)*" if e["indirect"] else ""
        lines.append(f"{i}. **{e['author']} ({e['year']})**, {e['design']}: “{e['finding']}” (PMID {e['pmid']}){note}")
    return "## Key studies\n\n" + "\n".join(lines)


def remove_section(text: str, heading: str) -> str:
    """Delete a model-written '### <heading>' section (up to the next heading), so the code-built one is the only one."""
    return re.sub(rf"(?ims)^#{{2,4}}\s*{re.escape(heading)}\s*\n.*?(?=^#{{1,4}}\s|\Z)", "", text).strip()


# ------------------------------------------------------------------ bottom line
TIER_OF = {"meta-analysis": 0, "systematic review": 0, "randomized controlled trial": 1, "clinical trial": 2, "cohort study": 2, "case-control study": 2,
           "observational study": 2, "cross-sectional study": 2, "narrative review": 3, "case report": 4}
TIER_NAMES = [("systematic review or meta-analysis", "systematic reviews or meta-analyses"), ("randomized controlled trial", "randomized controlled trials"),
              ("other clinical study", "other clinical studies"), ("narrative review", "narrative reviews"), ("case report", "case reports"), ("record of unstated design", "records of unstated design")]


def tier_counts(papers: list[dict], stems: set[str] | None = None) -> list[int]:
    counts = [0] * len(TIER_NAMES)
    for p in papers:
        if p.get("retracted") or p.get("protocol"):
            continue
        d = design_of(p)
        counts[TIER_OF.get(d[0], 5) if d else 5] += 1
    return counts


def is_broad_treatment_question(question: str, plan: dict | None, treatment_flag: bool) -> bool:
    return bool(treatment_flag) and not ((plan or {}).get("drugs"))


def bottom_line_markdown(question: str, state: dict, papers: list[dict], stems: set[str] | None, broad: bool) -> str:
    """Built from counts only: it cannot headline one paper and cannot state an efficacy result."""
    counts = tier_counts(papers, stems)
    parts = [f"{n} {TIER_NAMES[i][0] if n == 1 else TIER_NAMES[i][1]}" for i, n in enumerate(counts) if n]
    total = sum(counts)
    labelled = [r for r in state.get("regulatory", []) if r.get("matches_condition")]
    text = "**Bottom line.** "
    if total:
        text += f"The searches returned {total} usable PubMed/Europe PMC record(s) on this question: " + ", ".join(parts) + ". "
    else:
        text += "No usable PubMed/Europe PMC record was retrieved for this question. "
    if labelled:
        names = ", ".join(sorted({r["generic"] for r in labelled})[:6])
        text += f"{len(labelled)} FDA drug label(s) name this condition in their indication wording ({names}); see the drug tables for the label wording. "
    else:
        text += "No FDA label whose indication names this condition was retrieved. "
    text += "What each study found is listed under Key studies; none of this is a treatment recommendation."
    return text


# ------------------------------------------------------------------ source-failure banner
GROUPS = [("PubMed / Europe PMC", "pubmed"), ("ClinicalTrials.gov", "trials"), ("FDA drug labels", "fda"), ("Web search", "web"), ("Guideline search", "guideline")]
SUPERSEDED_BY = {"pubmed_fast": ("pubmed_recent",), "pubmed_fast_q": ("pubmed_quality",), "trials_fast": ("trials_deep",), "fda_fast": ("fda_deep",),
                 "web_fast": ("web_approvals", "web_standard")}


def source_problems(state: dict, core_only: bool = False) -> list[tuple[str, str, list[str]]]:
    """[(source name, 'failed'|'partial', [reasons])] for sources still not fully retrieved AFTER the retry (a fast job that a deep job redid and passed is not a problem)."""
    status = state.get("source_status", {})
    ok = {k for k, v in status.items() if v.get("retrieval_status") == "success"}
    problems = []
    for label, prefix in GROUPS:
        if core_only and prefix == "guideline":      # a guideline-search failure is shown inside the Established care section, not as a top banner
            continue
        jobs = {k: v for k, v in status.items() if k.startswith(prefix)}
        if not jobs:
            continue
        bad = {k: v for k, v in jobs.items() if v.get("retrieval_status") != "success" and not any(s in ok for s in SUPERSEDED_BY.get(k, ()))}
        if not bad:
            continue
        reasons = [f"{k}: {str(v.get('error') or v.get('retrieval_status'))[:70]}" for k, v in bad.items()]
        problems.append((label, "failed" if len(bad) == len(jobs) else "partial", reasons))
    if "deep_retrieval" in status and status["deep_retrieval"].get("retrieval_status") != "success":
        problems.append(("Deep retrieval stage", "failed", [str(status["deep_retrieval"].get("error"))[:90]]))
    return problems


def source_banner(state: dict) -> str:
    problems = source_problems(state, core_only=True)
    if not problems:
        return ""
    named = "; ".join(f"**{name}** ({kind})" for name, kind, _ in problems)
    return (f"> **Source warning:** {named} did not return complete results even after a retry. "
            "The absence of evidence cannot be concluded from these sources: something may exist that was not retrieved.\n")



# ------------------------------------------------------------------ model-written Key studies: keep only complete entries
DESIGN_WORDS = re.compile(r"meta-analys|systematic review|randomi[sz]ed|\bRCT\b|clinical trial|cohort|case-control|case report|case series|cross-sectional|observational|"
                          r"narrative review|\breview\b|qualitative|pilot|open-label|double-blind|quasi-experimental", re.I)
ITEM_MARK = re.compile(r"^\s*(?:\d{1,2}[.)]|[-*•])\s+")


def clean_model_key_studies(text: str, papers: list[dict] | None = None, stems: set[str] | None = None) -> str:
    """The model writes Key studies. Every entry must show authors, a year, a study design, a finding and a PMID; any other entry is dropped,
    and the list is renumbered. If nothing complete is left, the code-built list (verbatim abstract sentences) is used instead."""
    m = re.search(r"(?ims)^#{2,4}\s*Key studies\s*\n(.*?)(?=^#{1,4}\s|\Z)", text)
    before = text[:m.start()].rstrip() if m else text.rstrip()
    after = text[m.end():].lstrip() if m else ""
    items = []
    for ln in (m.group(1).splitlines() if m else []):
        if not ITEM_MARK.match(ln):
            continue
        body = ITEM_MARK.sub("", ln).strip()
        words = re.sub(r"PMID\s*\d+|\(\s*(?:19|20)\d\d\s*\)|[*_]", "", body).split()
        complete = (re.search(r"[A-Z][A-Za-zÀ-ſ'\-]+(?: et al\.?| and [A-Z]|,)", body) and re.search(r"(?:19|20)\d\d", body) and DESIGN_WORDS.search(body)
                    and re.search(r"PMID\s*\d{6,9}", body) and body.count("**") % 2 == 0 and len(words) >= 10 and body.rstrip()[-1:] in ".)\"”")
        if complete:
            items.append(body)
    if items:
        section = "## Key studies\n\n" + "\n".join(f"{i}. {b}" for i, b in enumerate(items[:5], 1))
    else:
        section = key_studies_markdown(papers or [], stems)
    return "\n\n".join(x for x in (before, section, after) if x)
