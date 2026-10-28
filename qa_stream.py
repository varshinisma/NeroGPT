"""NeuroGPT streaming Q/A: a useful first answer in about 5 seconds, then progressive deep research.

Follows NeuroGPT_8_Skills_Specification. Every skill's SKILL.md is loaded lazily, only at the stage that needs it.

    question
      -> [parallel] FAST retrieval (PubMed + trials + web, 3 s timeout)   and   orchestrator plan (LLM)
      -> INITIAL              fast answer streamed (clinical-answer-writer): bottom line, recent finding, top drugs
      -> [background, started as soon as the plan is ready] DEEP retrieval (latest-evidence, drug web search, clinical-trials)
      -> EVIDENCE_ENRICHING   evidence-synthesis streamed
      -> DRUGS_ENRICHING      drug-intelligence streamed (Treatment Drug Landscape)
      -> TRIALS_ENRICHING     clinical-trials table from the registry records (no model, nothing invented)
      -> VERIFICATION         citation-verification + quality-control, in code, against everything retrieved
      -> FINAL

Events (dicts) are yielded as the work happens. Run from the command line:  python qa_stream.py "your question"
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
import os
import re
import sys
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from datetime import datetime
from pathlib import Path

import phi
import server
from urllib.parse import urlparse
import chunking
import fulltext
from ratelimit import limiter
from merge import find_paper, normalize_url, search_log
from clinical_tools import _article, page_date, fda_label_link, remove_age_expressions, search_pubmed_quick, search_clinical_trials, search_europepmc_quick, search_fda_labels, search_pubmed
from landscape import (patient_age_years, drug_intelligence_status, family_status, landscape_markdown, limitations_markdown, record_source_status, source_family,
                       top_drugs_markdown, trials_markdown, dose_candidates, verify_dose_picks, dose_summary_ok, verify_paper_triage, verify_trial_triage, dose_fallback, fda_relevant, numbered_sentences, verify_fda_decisions, verify_drug_decisions,
                       verify_regulator_decisions, REGULATOR_HOSTS)
from pipeline import save_pdf, strip_preliminary
from presentation import closing_note, numberize
import hybrid_search
import retrieval_routing as rr
from hybrid_search import cross_rerank, rerank as hybrid_rerank
from tracing import traced
from source_figures import apply_population_check, apply_strict_check, search_paper_figures, cut_for_model, display_records, explain_figures, select_figures
from guidelines import filter_authoritative, find_guidelines, load_config as load_care_config, topics_for
from report import DISCLAIMER, clean_model_key_studies, design_of, source_problems
from verification import date_after, is_indirect, known_sources, verify_and_repair

FAST_TIMEOUT_S = 4.5          # per retrieval skill, FAST path (the FDA table is part of the first answer, so the FDA search is waited for)
DEEP_TIMEOUT_S = 25.0         # per retrieval skill, DEEP path
CLAUDE_MODE = os.getenv("NEUROGPT_LLM") == "claude_files"   # the text-writing steps are answered by Claude through files, not by an API model
DEEP_BUDGET_S = 1e5 if CLAUDE_MODE else 100.0   # total deep-research budget (not applied while Claude writes the text)
MAX_CONTEXT_PAPERS = 16
FULL_TEXT_PAPERS = int(os.getenv("FULL_TEXT_PAPERS", "6"))      # open-access papers among the best ranked whose methods, results and conclusion are read (fulltext.py)
FULL_TEXT_WAIT_S = float(os.getenv("FULL_TEXT_WAIT_S", "6"))
FAST_TOKENS, DEEP_TOKENS = 330, 2200
PLAN_TOKENS = 1000      # the plan also states what is asked and why and the case facts; a reply cut off by the limit is not valid JSON, so the limit is generous
STAGES = ["INITIAL", "EVIDENCE_ENRICHING", "DRUGS_ENRICHING", "TRIALS_ENRICHING", "VERIFICATION", "FINAL"]

STOP = set("""a an the of in on for to and or with without about from by at as is are was were be been do does did can could should would
how what which who whom why when where tell me give show explain describe please latest recent new newest current best top any some
treat treating treated treatment treatments therapy therapies manage managing management drug drugs medication medications medicine
cure option options approach approaches evidence research study studies update updates know patient my his her their mrs mr ms""".split())
TREATMENT_WORDS = {"treat", "treating", "treated", "treatment", "treatments", "therapy", "therapies", "manage", "managing", "management",
                   "drug", "drugs", "medication", "medications", "medicine", "cure", "option", "options"}
RECENT_WORDS = {"latest", "recent", "new", "newest", "current", "update", "updates", "2025", "2026"}


# ------------------------------------------------------------------ small utilities
class Clock:
    def __init__(self):
        self.t0, self.marks = time.time(), {}

    def now(self) -> float:
        return round(time.time() - self.t0, 2)

    def mark(self, name: str) -> float:
        self.marks.setdefault(name, self.now())
        return self.marks[name]


_cache: dict[str, tuple[float, str]] = {}
_cache_lock = threading.Lock()


def cached(kind: str, key: str, ttl: float, fn):
    """TTL cache for tool calls (stable metadata is reused; time-sensitive data expires quickly). Errors are never cached."""
    full = f"{kind}:{key}"
    with _cache_lock:
        hit = _cache.get(full)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
    value = fn()
    if not str(value).startswith(("Web search failed", "No web results")) and '"error"' not in str(value)[:20]:
        with _cache_lock:
            _cache[full] = (time.time(), value)
    return value


_tavily_client = None


@traced("web_search", "tool")
def web_search(query: str, snippet_chars: int = 500) -> str:
    """Web search: Tavily first (fast, relevant, regulator pages), DDGS as the fallback. Same JSON shape as server.search_live_web."""
    global _tavily_client
    if os.getenv("TAVILY_API_KEY"):
        try:
            if _tavily_client is None:
                from tavily import TavilyClient
                _tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
            limiter.acquire("tavily")
            found = _tavily_client.search(query, max_results=5, search_depth="basic").get("results", [])
            items = [{"title": r.get("title"), "url": r.get("url"), "snippet": (r.get("content") or "")[:snippet_chars]} for r in found]
            if items:
                return json.dumps(items, ensure_ascii=False)
        except Exception as error:
            if "usage limit" in str(error) or "Forbidden" in type(error).__name__ or "Unauthorized" in type(error).__name__:
                os.environ.pop("TAVILY_API_KEY", None)      # the plan limit is reached or the key is refused: stop asking Tavily in this run of the server, use the fallback search
    return server.search_live_web(query)


def guideline_search(query: str) -> str:
    """Web search limited to the authoritative guideline organisations (config/care_topics.json); results are re-checked against that list afterwards."""
    global _tavily_client
    domains = list(load_care_config()["authoritative_domains"])
    if os.getenv("TAVILY_API_KEY"):
        try:
            if _tavily_client is None:
                from tavily import TavilyClient
                _tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
            limiter.acquire("tavily")
            found = _tavily_client.search(query, max_results=5, search_depth="basic", include_domains=domains).get("results", [])
            return json.dumps([{"title": r.get("title"), "url": r.get("url"), "snippet": (r.get("content") or "")[:1500]} for r in found], ensure_ascii=False)
        except Exception as error:
            if "usage limit" in str(error) or "Forbidden" in type(error).__name__ or "Unauthorized" in type(error).__name__:
                os.environ.pop("TAVILY_API_KEY", None)
            pass                       # Tavily unavailable or over its usage limit: fall back to the DuckDuckGo search below
    sites = " OR ".join(f"site:{d}" for d in ("aap.org", "nice.org.uk", "cdc.gov", "nih.gov", "who.int", "aacap.org", "asha.org", "aota.org"))
    return server.search_live_web(f"{query} ({sites})")      # results are re-checked against the full allow-list afterwards


REGULATOR_DOMAINS = ["fda.gov", "ema.europa.eu", "cdsco.gov.in"]


def regulator_search(query: str) -> str:
    """Web search limited to the regulators' own sites. Same JSON shape as web_search; the pages are re-checked against REGULATOR_DOMAINS afterwards."""
    global _tavily_client
    if os.getenv("TAVILY_API_KEY"):
        try:
            if _tavily_client is None:
                from tavily import TavilyClient
                _tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
            found = _tavily_client.search(query, max_results=6, search_depth="basic", include_domains=REGULATOR_DOMAINS).get("results", [])
            items = [{"title": r.get("title"), "url": r.get("url"), "snippet": (r.get("content") or "")[:1500]} for r in found]
            if items:
                return json.dumps(items, ensure_ascii=False)
        except Exception:
            pass
    return server.search_live_web(f"{query} (" + " OR ".join(f"site:{d}" for d in REGULATOR_DOMAINS) + ")")


def regulator_search_many(queries: list[str]) -> str:
    """Several regulator-site searches merged (each page once)."""
    seen, merged = set(), []
    with ThreadPoolExecutor(max_workers=len(queries)) as ex:
        results = list(ex.map(regulator_search, queries))
    for raw in results:
        data = _json(raw)
        for item in data if isinstance(data, list) else []:
            if item.get("url") and item["url"] not in seen:
                seen.add(item["url"])
                merged.append(item)
    return json.dumps(merged, ensure_ascii=False) if merged else "Web search failed: No results found."


def ingest_guidelines(state: dict, raw: dict) -> None:
    """guidelines_deep -> state['guideline_rows'] (one row per topic) and state['guidelines'] (the rows that have a retrieved guideline)."""
    text = raw.get("guidelines_deep")
    if text is None:
        return
    data = _json(text)
    if isinstance(data, dict) and data.get("results") is not None:
        routed = [g for g in state.get("guidelines", []) if str(g.get("topic", "")).startswith("web_guideline_op")]      # rows from the routed guideline searches stay
        state["guideline_rows"] = data["results"] + routed
        state["guidelines"] = [r for r in data["results"] if r.get("title")] + [g for g in routed if g.get("url") not in {r.get("url") for r in data["results"]}]
        state["guideline_search_ok"] = True
        if data.get("failed"):
            state["source_status"]["guidelines_deep"] = {"retrieval_status": "partial", "evidence_status": "unavailable",
                                                         "error": f"{data['failed']} of {data.get('topics')} topic searches failed"}
    else:
        state["guideline_rows"], state["guidelines"], state["guideline_search_ok"] = [], [], False


CITE_TAIL = re.compile(r"\s*((?:\[(?:PMID\s*)?\d+(?:[,;]\s*(?:PMID\s*)?\d+)*\]\s*[,;]?\s*)+)\.?\s*$")


def fix_reference_column(text: str) -> str:
    """A small model sometimes puts the citation at the end of the 'What the evidence shows' cell and leaves the Reference cell out: the row then has one cell too few
    and the Reference column prints empty. The trailing citation is moved into its own Reference cell."""
    out, width = [], 0
    for line in text.split(chr(10)):
        if line.lstrip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if re.search(r"reference", line, re.I) and not width and not re.fullmatch(r"[\s|:\-]+", line):
                width = len(cells)
            elif width and len(cells) == width - 1 and not re.fullmatch(r"[\s|:\-]+", line):
                m = CITE_TAIL.search(cells[-1])
                shows = cells[-1][:m.start()].rstrip() if m else cells[-1]
                cells = cells[:-1] + [shows + ("." if m and shows and not shows.endswith(".") else ""), m.group(1).strip() if m else ""]
                line = "| " + " | ".join(cells) + " |"
        else:
            width = 0
        out.append(line)
    return chr(10).join(out)


def rows_to_table(text: str, papers: dict | None = None) -> str:
    """The Current evidence rows the model wrote as 'Approach || Evidence level || What the evidence shows || Reference' lines -> the Markdown table, built here so no padding is possible. A line whose
    citation is at the end of the evidence cell (three cells) gets it moved to the Reference column. Text that is already a table, or has no such lines, is returned unchanged."""
    rows = []
    for line in text.split(chr(10)):
        cells = [c.strip().strip("|").strip() for c in line.split("||")]
        if len(cells) == 4 and cells[3] == "" and all(cells[:3]):
            cells = cells[:3]
        if len(cells) == 5 and cells[4] == "" and all(cells[:4]):
            cells = cells[:4]
        if len(cells) == 3 and all(cells):
            ids = re.findall(r"PMID\s*\[?(\d+)", cells[2])
            if ids:
                body = re.sub(r"\s*\([^()]*PMID[^()]*\)[\s.]*$", "", cells[2]).strip()
                cells = cells[:2] + [body or cells[2], f"(PMID {ids[-1]})"]
        if len(cells) == 4 and all(cells):
            rows.append(cells)
        elif len(cells) == 5 and all(cells):
            rows.append(cells)
    if not rows and papers is not None:
        for line in text.splitlines():
            if not line.lstrip().startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) == 4 and all(cells) and not all(re.fullmatch(r"[-: ]+", c) for c in cells):
                if [c.lower() for c in cells] in (["approach", "evidence level", "what the evidence shows", "reference"],
                                                   ["approach", "relevance", "evidence level", "what the evidence shows", "reference"]):
                    continue
                rows.append(cells)
    if not rows:
        return text
    heading = next((l for l in text.split(chr(10)) if l.lstrip().startswith("#")), "## Current evidence")
    if papers is None:
        return heading.strip() + chr(10) * 2 + "| Approach | Evidence level | What the evidence shows | Reference |" + chr(10) + "|---|---|---|---|" + chr(10) + chr(10).join("| " + " | ".join(row) + " |" for row in rows)
    groups = {"direct": [], "indirect": [], "background": [], "mixed": [], "unclassified": []}
    for row in rows:
        # The model supplies four reader-facing cells.  Relevance remains an
        # internal sort key and is deliberately not rendered as a column.
        if len(row) == 5:
            row = [row[0], row[2], row[3], row[4]]
        ids = re.findall(r"PMID\s*\[?(\d{6,9})", " ".join(row[1:]))
        labels = {(papers.get(i) or {}).get("triage", {}).get("relevance") for i in ids}
        labels.discard(None)
        if labels == {"direct"}:
            category = "direct"
        elif labels == {"indirect"}:
            category = "indirect"
        elif labels == {"background"}:
            category = "background"
        elif "direct" in labels and "indirect" in labels:
            category = "mixed"
        else:
            category = "unclassified"
        groups[category].append(row)
    from report import design_of
    for values in groups.values():
        def table_order(row):
            ids = re.findall(r"PMID\s*\[?(\d{6,9})", " ".join(row[1:]))
            cited = next(((papers.get(i) or {}) for i in ids if i in papers), {})
            hybrid = cited.get("hybrid") or {}
            score = hybrid.get("cross") if hybrid.get("cross") is not None else hybrid.get("fused")
            design = design_of(cited)
            return (-(float(score) if isinstance(score, (int, float)) else 0.0), design[1] if design else 99)
        values.sort(key=table_order)
    ordered = [row for category in ("direct", "indirect", "background", "mixed", "unclassified") for row in groups[category]]
    rendered = "| Approach | Evidence level | What the evidence shows | Reference |\n|---|---|---|---|\n" + "\n".join("| " + " | ".join(row) + " |" for row in ordered)
    return heading.strip() + "\n\n" + rendered


def tidy_model_sections(text: str) -> str:
    """Remove what a small model sometimes adds around its sections: letter labels like '(A)' / '**(B)**' and loose '---' separators."""
    text = re.sub(r"(?m)^[ \t]*\*{0,2}\(?[A-D]\)\*{0,2}[ \t]*$\n?", "", text)
    text = re.sub(r"(?m)^[ \t]*-{3,}[ \t]*$\n?", "", text)
    text = chr(10).join(re.sub(r" {2,}", " ", re.sub(r"-{4,}", "---", ln)) if ln.lstrip().startswith("|") else ln for ln in text.split(chr(10)))      # table padding is only whitespace: single spaces, short separator row
    text = fix_reference_column(text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def load_skill_text(name: str) -> str:
    if name not in server.SKILL_ORDER:      # a skill used only inside the program (not offered to the orchestrator), such as figure-selection
        return server.load_skills([name])
    return server.load_skill(name)  # lazy: reads only this skill, logs "[skills] loaded on demand"


def keywords(question: str) -> tuple[list[str], bool, bool]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}|20\d\d", question.lower())
    keep = [w for w in words if w not in STOP]
    return keep[:5], any(w in TREATMENT_WORDS for w in words), any(w in RECENT_WORDS for w in words)


def pubmed_query_from_question(question: str) -> tuple[str, int]:
    kw, treatment, recent = keywords(question)
    base = " AND ".join(kw) if kw else question
    if treatment:
        base += " AND (treatment OR therapy OR management OR intervention)"
    return base, 1 if recent else 3


def _json(text):
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return {"error": str(text)[:200]}


# ------------------------------------------------------------------ ResearchState
def new_state(question: str) -> dict:
    """The FULL ResearchState: the internal source of truth. Model calls never receive it whole, only the filtered build_*_context views."""
    return {
        "query": question, "intent": None, "entities": {}, "core_stems": [],
        "papers": {},          # latest-evidence records, keyed by PMID (title, date, study type, PMID, DOI, URL, abstract, retracted, protocol, retrieved_at ...)
        "trials_by_id": {},    # clinical-trials records, keyed by NCT ID (status, last_update, interventions, results_available, results_source ...)
        "web": [],             # web sources
        "regulatory": [],      # FDA drug-label records (official wording, label date, URL)
        "drug_records": [],    # structured drug entries built from regulatory records
        "fda_terms": [],
        "fast_evidence": [], "deep_evidence": [], "drugs": "", "trials": [], "synthesis": "", "citations": [],
        "source_status": {},   # per source: retrieval_status success|partial|failed, evidence_status found|not_found_after_search|unavailable, error
        "retrieval_errors": {},
        "drug_intelligence_status": "pending", "trial_status": "pending", "verification_status": "pending", "repair_count": 0,
        "verification": {}, "qc": {}, "answer_status": "INITIAL",
        "timestamps": {"request_started": datetime.now().isoformat(timespec="seconds")},
    }


def ingest(papers: dict, trials: dict, web: list, raw: dict) -> dict:
    """Merge raw tool output into the de-duplicated evidence pools. Returns {source: error} for failed sources."""
    errors = {}
    for key, text in raw.items():
        data = _json(text)
        if key.startswith("pubmed"):
            if isinstance(data, dict) and data.get("error"):
                errors[key] = data["error"]
                continue
            for p in data.get("results", []):
                same = find_paper(papers, p)      # the same paper by PMID, else DOI, else normalized title: one copy is kept
                if same is not None:
                    found_by = papers[same].setdefault("found_by", [papers[same].get("origin")])      # found again by another search: every search that returned it is recorded
                    if key not in found_by:
                        found_by.append(key)
                    continue
                surname = (p.get("authors") or ["Unknown"])[0].split(" ")[0]
                types = [t for t in p.get("study_type", []) if t != "Journal Article"][:3]
                papers[p["pmid"]] = {
                    "source_id": f"pmid:{p['pmid']}", "pmid": p["pmid"], "doi": p.get("doi"), "pmcid": p.get("pmcid"), "url": p.get("url"), "year": p.get("year"),
                    "date": p.get("publication_date"), "journal": p.get("journal"), "authors": (p.get("authors") or [])[:3], "title": p.get("title"),
                    "type": types, "study_type": types, "abstract": p.get("abstract_excerpt") or "", "population": None, "finding": None, "limitations": None,
                    "cite": f"{surname} et al., {p.get('journal')}, {p.get('year')} [PMID {p['pmid']}]", "origin": key, "found_by": [key],
                    "retracted": any("retract" in str(t).lower() for t in p.get("study_type", [])),
                    "protocol": "protocol" in (p.get("title") or "").lower() or any("protocol" in str(t).lower() for t in p.get("study_type", [])),
                    "retrieved_at": datetime.now().isoformat(timespec="seconds")}
        elif key.startswith("trials"):
            if isinstance(data, dict) and data.get("error"):
                errors[key] = data["error"]
                continue
            for t in data.get("results", []):
                known = trials.get(t["nct_id"])
                if known is not None:
                    found_by = known.setdefault("found_by", [])
                    if key not in found_by:
                        found_by.append(key)
                    continue
                trials.setdefault(t["nct_id"], {"found_by": [key], **t, "source_id": f"nct:{t['nct_id']}", "results_available": bool(t.get("results_posted")),
                                                "results_source": "ClinicalTrials.gov registry", "evidence_status": "registry_status_only"})
        elif key.startswith("web"):
            if isinstance(data, list):
                known = {normalize_url(w["url"]): w for w in web}
                for r in data:
                    again = known.get(normalize_url(r.get("url")))
                    if again is not None:      # the same page from another search: one copy, every search recorded
                        again.setdefault("found_by", [again.get("source")])
                        if key not in again["found_by"]:
                            again["found_by"].append(key)
                        continue
                    web.append({"title": r.get("title"), "url": r.get("url"), "snippet": r.get("snippet") or "", "source": key, "found_by": [key]})
                    known[normalize_url(r.get("url"))] = web[-1]
            else:
                errors[key] = data.get("error", "no results")
    return errors


def ingest_regulatory(state: dict, raw: dict) -> None:
    """FDA drug-label records -> state['regulatory'] and structured state['drug_records'] (dose and route stay None unless sourced)."""
    for key, text in raw.items():
        if not key.startswith("fda"):
            continue
        data = _json(text)
        if isinstance(data, dict) and data.get("error"):
            state["retrieval_errors"][key] = data["error"]
            continue
        state["fda_terms"] = data.get("searched_terms") or state["fda_terms"]
        seen = {r["generic"].lower() for r in state["regulatory"]}
        for r in data.get("results", []):
            if r["generic"].lower() in seen:
                continue
            seen.add(r["generic"].lower())
            state["regulatory"].append(r)
            state["drug_records"].append({
                "drug": r["generic"], "category": "FDA-approved for the labelled indication" if fda_relevant(r) else "FDA label found; indication does not mention the condition",
                "indication": r["indication_excerpt"], "population": None, "approval_status": "approved (labelled indication)" if fda_relevant(r) else "not_verified",
                "approval_jurisdiction": "FDA", "approval_date": None, "label_date": r.get("label_date"), "dose": None, "route": None, "dose_source": None,
                "evidence_source": r.get("url"), "verification_status": "verified_in_regulatory_source"})


def snippet(text: str, chars: int) -> str:
    """Abstract excerpt for the model. Prefer the RESULTS / CONCLUSION part, so the model never has to guess an outcome from the background."""
    if len(text) <= chars:
        return text
    # headings arrive either as "RESULTS: ..." or, once tags are stripped, as "Results After two months ..."
    m = re.search(r"\b(?:Results?|Findings|Conclusions?|RESULTS?|FINDINGS|CONCLUSIONS?)\b\s*(?::|\.|(?=[A-Z]))", text)
    if m and m.start() > 0:
        return text[m.start():][:chars]
    return text[:100] + " ... " + text[-(chars - 105):]


def paper_view(p: dict, chars: int, stems: set[str] | None = None) -> dict:
    triage = p.get("triage") or {}
    view = {"cite": p["cite"], "year": p["year"], "date": p["date"], "type": p["type"], "title": p["title"], "abstract": snippet(p["abstract"], chars),
            "relation": {"indirect": "indirect (different or related population)", "background": "background"}.get(triage.get("relevance"), "direct"), "protocol_no_results": bool(p.get("protocol"))}
    if p.get("passages"):      # the passages of this paper that bear on the question (chunking.py): the writer reads those, with only the start of the abstract for context
        view["abstract"] = snippet(p["abstract"], 260)
        view["relevant_passages"] = [{"from": c["section"], "text": c["text"]} for c in p["passages"]]
    elif p.get("body"):
        view["full_text_sections"] = p["body"]      # the free full text of this paper (methods, results, conclusion), cut short; the abstract above stays the paper's summary
    if triage:
        view["triage"] = {"evidence_role": triage.get("role"), "population": triage.get("population"),
                           "relevance_rationale": triage.get("rationale") or triage.get("why_relevant"),
                           "evidence_sentence": triage.get("evidence")}
    if date_after(p.get("date") or p.get("year"), datetime.now()):
        view["after_search_date"] = True      # its date is later than today: an advance publication
    return view


def trial_view(t: dict) -> dict:
    return {"id": t["nct_id"], "title": t["title"], "phase": t.get("phase"), "status": t.get("recruitment_status"),
            "last_update": t.get("last_update"), "interventions": (t.get("interventions") or [])[:4],
            "primary_endpoint": t.get("primary_endpoint"), "results_posted": t.get("results_posted"), "why_stopped": t.get("why_stopped")}


GENERIC = {"children", "child", "pediatric", "paediatric", "adolescent", "adolescents", "adults", "adult", "patients", "patient", "people", "kids", "infants"}


def core_stems(question: str, plan: dict | None = None) -> set[str]:
    """Stems of the question's own subject words (and drugs the planner found). Used to drop off-topic records, not to rank."""
    words = keywords(question)[0] + [d.lower() for d in ((plan or {}).get("drugs") or [])]
    return {w[:5] for w in words if w not in GENERIC and len(w) >= 4}


def usable(p: dict, stems: set[str] | None = None) -> bool:
    """Only papers explicitly judged direct or meaningfully indirect can support report content; failed or missing triage fails closed."""
    return not p.get("retracted") and (p.get("triage") or {}).get("relevance") in ("direct", "indirect")


RELEVANCE_RANK = {"direct": 0, "indirect": 1, "background": 2}


def ordered_papers(papers: dict, stems: set[str] | None = None, *, allow_untriaged: bool = False, include_background: bool = False) -> list[dict]:
    """Question relevance is the first tier; within a tier, preserve reranker relevance and use study design as a tie-breaker.
    Background is available only to a caller that explicitly asks for contextual sources. Irrelevant and retracted records never enter."""
    pool = [p for p in papers.values() if (usable(p, stems) or
            (include_background and not p.get("retracted") and (p.get("triage") or {}).get("relevance") == "background" and (p.get("triage") or {}).get("evidence")) or
            (allow_untriaged and not p.get("triage") and not p.get("retracted")))]
    from report import design_of
    original_order = {str(p.get("pmid")): i for i, p in enumerate(pool)}
    def rank(p):
        relevance = RELEVANCE_RANK.get((p.get("triage") or {}).get("relevance"), 3)
        score = (p.get("hybrid") or {}).get("cross")
        if score is None:
            score = (p.get("hybrid") or {}).get("fused")
        score = float(score) if isinstance(score, (int, float)) else 0.0
        design = design_of(p)
        strength = design[1] if design else 99
        return relevance, -score, strength, original_order.get(str(p.get("pmid")), 0)
    return sorted(pool, key=rank)


# ------------------------------------------------------------------ model streaming
_bridge_lock, _bridge_n = threading.Lock(), [0]


def bridge_llm(instructions: str, prompt: str):
    """Claude replaces the API model: write the exact prompt (the skill's rules + this stage's data) to a file, wait for the reply file."""
    folder = Path(os.environ["NEUROGPT_BRIDGE_DIR"])
    folder.mkdir(parents=True, exist_ok=True)
    with _bridge_lock:
        _bridge_n[0] += 1
        n = _bridge_n[0]
    body = "=== INSTRUCTIONS (the SKILL.md loaded for this stage) ===" + chr(10) + instructions + chr(10) + chr(10) + "=== PROMPT ===" + chr(10) + prompt + chr(10)
    (folder / f"stage{n}_prompt.md").write_text(body, encoding="utf-8")
    reply = folder / f"stage{n}_response.md"
    while not reply.exists() or reply.stat().st_size == 0:
        time.sleep(0.5)
    time.sleep(0.3)
    text = reply.read_text(encoding="utf-8")
    for i in range(0, len(text), 60):  # delivered in small pieces, like a stream
        yield text[i:i + 60]


FIRST_CHUNK_S, IDLE_CHUNK_S = 20.0, 25.0      # a model call that has produced nothing for this long has stalled (a normal call starts within 1-3 s)


def _with_deadline(source, first: float = FIRST_CHUNK_S, idle: float = IDLE_CHUNK_S):
    """Yield what a (possibly stalling) generator produces. It runs in its own thread; if it is silent for longer than the limit the call is abandoned with a TimeoutError,
    so one stalled model call can never hold the whole answer."""
    box: queue.Queue = queue.Queue()

    def pump():
        try:
            for item in source:
                box.put(("chunk", item))
            box.put(("done", None))
        except Exception as error:
            box.put(("error", error))
    threading.Thread(target=pump, daemon=True).start()
    waited = first
    while True:
        try:
            kind, value = box.get(timeout=waited)
        except queue.Empty:
            raise TimeoutError(f"the model gave no answer for {waited:.0f} s")
        if kind == "chunk":
            waited = idle
            yield value
        elif kind == "error":
            raise value
        else:
            return


def stream_llm(instructions: str, prompt: str, max_tokens: int, attempts: int = 2, temperature: float | None = None, purpose: str = "writer"):
    """Yield text chunks from a FRESH agent (no shared history). Bounded retries only before the first chunk; a stalled call is abandoned (see _with_deadline)."""
    if CLAUDE_MODE:
        yield from bridge_llm(instructions, prompt)
        return
    from agno.agent import Agent

    def one_attempt():
        agent = Agent(model=model_for(max_tokens, temperature, purpose), instructions=[instructions], markdown=True)
        got = False
        for event in agent.run(prompt, stream=True):
            kind, content = str(getattr(event, "event", "")), getattr(event, "content", None)
            if "Error" in kind or (not got and str(content or "").startswith(("API error", "Error"))):
                raise RuntimeError(str(content or kind)[:300])
            if kind == "RunContent" and content:
                got = True
                yield str(content)
    last, tries, limited = None, 0, 0
    while tries < attempts:
        tries += 1
        got = False
        try:
            for chunk in _with_deadline(one_attempt()):
                got = True
                yield chunk
            return
        except Exception as error:
            last = error
            if got:
                raise
            if ("429" in str(error) or "rate limit" in str(error).lower()) and limited < 2:      # the provider's per-minute limit: wait a few seconds and ask again (not counted as an attempt)
                limited += 1
                tries -= 1
                time.sleep(3.0 * limited)
    raise RuntimeError(f"The model provider returned an error: {last}")


_models: dict[tuple, object] = {}


def planner_choice() -> tuple[str | None, str | None]:
    """(provider, model) of the plan call when PLANNER_PROVIDER is set (for example groq with PLANNER_MODEL=openai/gpt-oss-120b); (None, None) keeps the provider and model of the other calls.
    The account's Mistral tier may not include the larger Mistral models, so the plan call can use a stronger model of another provider that the project already supports."""
    provider = os.getenv("PLANNER_PROVIDER", "").strip().lower()
    if provider not in {"groq", "mistral"}:
        return None, None
    return provider, os.getenv("PLANNER_MODEL", "").strip() or None


def model_id_for(purpose: str = "writer") -> str | None:
    """Return the configured model for a pipeline stage without changing providers.

    Planning and evidence decisions benefit from a stronger model.  The visible
    prose writer can use a less expensive Mistral model, while still receiving
    only the already screened, verified evidence state.
    """
    if os.getenv("MODEL_PROVIDER", "groq").lower() != "mistral":
        return None
    if purpose == "planning":      # the plan call (intent, information needed, medicines needed, skills) may have its own, stronger model (planner_choice)
        return planner_choice()[1] or os.getenv("MISTRAL_REASONING_MODEL", "").strip() or os.getenv("MISTRAL_MODEL", "").strip() or None
    if purpose == "triage":
        return os.getenv("MISTRAL_REASONING_MODEL", "").strip() or os.getenv("MISTRAL_MODEL", "").strip() or None
    return os.getenv("MISTRAL_WRITER_MODEL", "").strip() or os.getenv("MISTRAL_MODEL", "").strip() or None


def model_for(max_tokens: int, temperature: float | None = None, purpose: str = "writer"):
    """Reuse connections per model, stage, output size and temperature."""
    model_id = model_id_for(purpose)
    provider = planner_choice()[0] if purpose == "planning" else None
    key = (provider, model_id, max_tokens, temperature, purpose)
    if key not in _models:
        _models[key] = server.build_model(max_tokens=max_tokens, temperature=temperature, model_id=model_id, provider=provider)
    return _models[key]


def warm_up() -> None:
    """Open the model connections before the first question (a cold first call costs about 3 seconds)."""
    from agno.agent import Agent
    import clinical_tools
    threading.Thread(target=phi.warm_up, daemon=True).start()  # local name detector
    threading.Thread(target=hybrid_search.load_cross_encoder, daemon=True).start()      # the cross-encoder model (downloaded once, then loaded from disk)
    threading.Thread(target=clinical_tools.warm_connections, daemon=True).start()  # PubMed / Europe PMC / registry connections
    warm_models = ((FAST_TOKENS, None, "writer"), (DEEP_TOKENS, None, "writer"),
                   (PLAN_TOKENS, 0.0, "planning"), (700, 0.0, "triage"),
                   (900, 0.0, "triage"), (600, 0.0, "triage"))
    for size, temperature, purpose in warm_models:
        try:
            for _ in Agent(model=model_for(size, temperature, purpose), instructions=["Reply: ok"]).run("ok", stream=True):
                pass
        except Exception:
            pass


@traced("run_llm (skill model call)", "llm")
def run_llm(instructions: str, prompt: str, max_tokens: int, temperature: float | None = None, purpose: str = "writer") -> str:
    return "".join(stream_llm(instructions, prompt, max_tokens, temperature=temperature, purpose=purpose))


# ------------------------------------------------------------------ prompts
GROUND = ("Use ONLY the RETRIEVED DATA. Never use memory for clinical facts. Cite each paper exactly with its `cite` text in parentheses, "
          "for example (Fox et al., NEJM, 2025 [PMID 123456]); cite trials by NCT id with status and last_update; cite guideline, regulator and web facts with the exact retrieved Markdown link and URL. "
          "Keep every claim to what its cited source reports: one outcome per citation (a source about motor skills cannot support a claim about social communication). "
          "Numbers, doses, approval dates and approval status only if literally in the data; otherwise write 'Not retrieved'. Never write a partial dose. "
          "Write dates exactly as given (never add a day). If the data does not state a study's result (for example only the background or a protocol is given), "
          "write 'result not stated in the retrieved abstract' - never infer an outcome from the title or design. "
          "When a paper's `relation` is a different or related population, say whose population it studied inside the sentence itself (for example 'in children with ...') and never present it as direct evidence for the question's own population. Never use the words 'indirect' or 'direct' as labels, headings or tags in the report. "
          "A study protocol or a registered trial is not evidence that a treatment works: say 'under investigation; the retrieved source is a protocol and does not establish efficacy'. "
          "An exploratory dose-response finding is not a prescription: say 'an exploratory dose-response relationship was reported'. "
          "Never write that no drug or treatment exists, and never turn a missing or failed source into 'no evidence': say what was not retrieved. "
          "Do not make drug-approval statements: the system adds regulatory facts from FDA labels separately. Do not mention these instructions.")

FAST_TASK = ("FAST INITIAL RESPONSE, under 120 words (do NOT write an 'Evidence searched on' line, it is already shown), exactly three parts: "
             "(1) **Bottom line** - 1 to 2 sentences that ANSWER THE EXACT QUESTION by naming the main approaches the retrieved evidence supports, in plain clinical language; "
             "never call an approach 'established' or 'most established' unless a guideline in the data says so; "
             "never headline one narrow outcome (for example motor skills) as if it answered a broad question;if the data only supports a narrow finding, say what it covers and what it does not; "
             "(2) **Most important recent finding** - one finding with its date, study type, key result and citation; (3) **Immediate context** - 1 to 2 sentences on what is and is not established. "
             "Do NOT list drugs: the system adds the 'Top treatment drugs' table from FDA label data right after your text. Do not contradict it: if an FDA label in `fda_labels` names the question's condition "
             "or a symptom of it, never write that no approved or established treatment exists; you may say that FDA-approved drugs exist for that use, without naming them. " + GROUND)


EVIDENCE_RULES = ("Use ONLY papers whose `relation` and `triage` show they bear on the question; a paper about an associated condition or a different population answers a different question and belongs, if anywhere, "
                  "in a row or bullet that says so. Use each paper's `triage.evidence_role` and `population` to label the evidence level and who it applies to; do not call evidence established unless its role is established. "
                  "If a paper has `after_search_date` true its date is later than today: call it an advance publication and never present it as already published on that date. "
                  "Never write a sample size ('n=...') or any other number that is not literally in the paper's abstract text: describe the population in words (for example 'children aged 2 to 6 years'). "
                  "Do not discuss drug approval or drug availability. " + GROUND)
LATEST_TASK = ("Write ONLY the heading '## Latest findings' followed by at most 5 bullets, ordered with the closest matches to the question first and then useful related-population studies; within each group order newest first, and finish every bullet; each bullet gives the exact date as given in the data, the study type, and TWO sentences: "
               "who or what was studied (population and design) and what it found, with the main limitation when the data states one; end with (PMID <the PMID printed in that study's cite field>). "
               "Use the closest-matching papers first, then useful related-population papers that report a result (never a protocol or a study without results). State a related-population study's population in the sentence itself; write no label, heading or tag such as 'indirect evidence'. Background evidence must not support a recommendation. Every bullet starts with its own subject (what was studied), never with 'It' or 'The study'. " + EVIDENCE_RULES)
WORKUP_ADDENDUM = ("After the Latest findings bullets, add '## Clinical approach' only because this question's stated information needs call for a diagnostic or management plan. "
                   "Give a short, ordered plan that answers the user's actual question using only the supplied papers and guideline records. Include urgent assessment or targeted work-up only when a retrieved source supports it and the case facts make it relevant. "
                   "For each clinical action, cite the source inline: papers by PMID, guidelines by the exact Markdown URL link in the data. If a step is reasonable general context but no retrieved source supports it, explicitly label it 'General clinical context; not directly supported by the retrieved sources.' "
                   "Do not diagnose the patient or prescribe a patient-specific treatment. If no relevant guidance was retrieved, say that the work-up evidence is incomplete instead of filling gaps from memory.")
CURRENT_TASK = ("Write ONLY the heading '## Current evidence' followed by ONE LINE per treatment approach, each line in exactly this form: Approach || Evidence level || What the evidence shows || Reference "
                "(the code turns the lines into the table; write no table, no '|' bars, no separator row and no padding). One line per treatment approach "
                "(for example an intervention type or a symptom-directed treatment), at most 8 rows, strongest evidence first, each approach ONCE. "
                "Evidence level must be one of: Established (only a guideline or consistent high-quality evidence; a systematic review of studies of variable quality is NOT established), "
                "Evidence-supported but limited, Emerging, Experimental or preliminary, Protocol - no results, Insufficient evidence. "
                "'What the evidence shows' is ONE or TWO sentences of what the cited source reports: the population, the result and, when stated, its limit. "
                "Reference is (PMID <the PMID printed in that study's cite field>). "
                "Give a row only when a closely matching paper reports a result. A related-population study may have its own row after the closest matches, with its population stated in the 'What the evidence shows' cell (no label such as 'indirect evidence'); do not use it as the sole support for a recommendation. Background sources do not support treatment rows. Do not add rows for protocols. "
                + EVIDENCE_RULES)
TAIL_TASK = ("Write ONLY these parts, in this order, and finish every part. "
             "FIRST, '## Key studies': a numbered list of up to 5 studies, each in the form '**<first author> et al. (<year>)**, <design>: <what it found, in one or two sentences, with the population; copy any number exactly from the abstract and never write one that is not there> (PMID <the PMID printed in that study's cite field>)' "
             "- fill every <...> from a study in the data; the design is for example meta-analysis, systematic review, randomized controlled trial or cohort study. "
             "SECOND, '## Conflicting evidence' (only a real disagreement between two DIFFERENT sources, or negative results: 2 to 3 sentences naming what disagrees and on what; if there is none, one sentence saying what was compared "
             "and that no disagreement was retrieved). "
             "THIRD, '## What remains under investigation' (at most 6 bullets, no sub-bullets, each 1 to 2 sentences naming the specific open question: the population, outcome or comparison not yet answered). "
             "Distinguish efficacy, effectiveness, feasibility and association. " + EVIDENCE_RULES)
EVIDENCE_TASK = LATEST_TASK + " || " + CURRENT_TASK + " || " + TAIL_TASK      # the three tasks, in the document's order (each is sent as its own model call)

HOLD_FROM = re.compile(r"(?im)^#{2,4}\s*(?:Key studies|Conflicting|What remains)")      # the specification puts Key studies, Conflicting evidence and What remains AFTER the drugs and trials

DOSE_TASK = ("Below is the dosing text of an FDA drug label for ONE use of the drug. Write 3 to 4 short plain-English sentences for a clinician that cover: the starting dose, how it is adjusted "
             "(steps and timing), the usual or maximum dose range, and the age or weight group it applies to, but only where the text states them. Use ONLY what the text says. "
             "Copy every number and unit exactly as written; never add a number, a warning, advice or any fact that is not in the text, and never comment on what the label does NOT say. Keep every qualifier exactly as written (for example 'no less than', 'at least', 'up to', 'a minimum of') and do not add a timing the text does not give. Reply with the sentences only: plain text, no bold, no heading, no list.")

PAPER_TRIAGE_TASK = ("Decide what each retrieved paper is worth for the exact question. Each paper has numbered `sentences` (sentence 1 is the title). Reply with ONE JSON object {\"papers\": [...]} and nothing else, "
                     "with one object per paper: {\"id\": the offered id, \"relevance\": one of direct|indirect|background|not_relevant|uncertain, \"role\": one of established|emerging|experimental|insufficient_or_negative|conflicting|ongoing|background, \"need\": one exact value from clinical_profile.information_needs for direct or indirect evidence (otherwise empty), \"scope\": one of generalizable|case_specific|context_only, \"sentence\": a supporting abstract or full-text sentence number (never title sentence 1 for relevant evidence), \"rationale\": a concise explanation of which question need it informs and why}. relevance is direct (it addresses the clinical problem, population and requested intent), "
                     "indirect (the population or intervention differs, BUT the study still answers one of the question's stated information needs and its result can reasonably inform that need), background (limited context but no evidence answering a stated need), not_relevant, uncertain. "
                     "role describes certainty/status only: established, emerging, experimental, insufficient_or_negative, conflicting, ongoing, background. The design quality is separate and comes from the publication type. "
                     "For each decision assess condition, age/population applicability, question intent and stated information need, intervention (when treatment is asked), and outcomes. A different age group, pediatric population, or associated diagnosis alone does not make a paper indirect. A paper on an unrelated disease, outcome or treatment is not_relevant even if the age or a generic term overlaps. Do not invent an unmentioned differential diagnosis merely to retain a paper. Use indirect only when you can name the specific question need it informs; otherwise use not_relevant. Use background only for brief context, never as evidence for a treatment or finding. "
                     "Keep relevance separate from evidence strength: a strong study design does not make an indirect study direct. A case report about a different, named disorder is case_specific when it merely co-occurs with broad symptoms; it does not establish a general work-up or management step. Mark it generalizable only when the source actually evaluates an approach applicable to the question's population, or when the question itself is about that specific disorder. The rationale must name the specific question need answered, use no unsupported facts, and fit in one sentence. Judge only from the supplied text and the question; never invent.")
TRIAL_TRIAGE_TASK = ("Decide what each registry record is worth for the exact question and its information needs. Each record has numbered `sentences`. Reply with ONE JSON object {\"trials\": [...]} and nothing else, "
                     "with ONE STRING per record in exactly this form: id|relevance|kind|sentence_number. relevance is one of: direct, indirect, background, not_relevant, uncertain. "
                     "kind is one of: treatment_trial (tests an intervention as treatment for the question), diagnostic_or_assessment, other. sentence_number is the number of the sentence that shows it. Use clinical_profile to assess the requested condition, patient facts, and information needs. A record about a different condition, population, intervention, or outcome is not relevant merely because it has a generic overlapping term. "
                     "Example: \"NCT01234567|direct|treatment_trial|3\". "
                     "Age overlap or generic words alone do not make a registry record relevant; an unrelated condition is not relevant. Judge only from the supplied text and the question; never invent.")
FDA_TASK = ("Using ONLY the RETRIEVED DATA, decide for each FDA drug label whether its indication is relevant to the question. Each label is given as numbered `sentences` (the first is number 1). "
            "Reply with ONE JSON object {\"question_asks\": ..., \"fda_labels\": [...]} and nothing else. "
            "question_asks: what the QUESTION asks for, one of \"treat_or_manage\" (how to treat or manage a condition) | \"prevent\" | \"diagnose\" | \"other\". "
            "fda_labels: one object per label: {\"id\": its id, \"purpose\": what the LABEL's indication is for, one of \"treat_or_manage\" (treating the condition or managing its symptoms, complications or "
            "the side effects of its treatment) | \"prevent\" | \"diagnose\" | \"other\", \"relevant\": true if that indication is about the question's condition (the condition itself, or a symptom, complication or "
            "co-occurring condition of it), otherwise false; a label that only mentions the condition in passing (a risk it lowers, a population, a setting) is false, "
            "\"relation\": \"treats_condition\" | \"treats_associated_symptom_or_comorbidity\", \"sentence\": the number of the sentence that shows it}. "
            "The sentence you point at must itself state what the drug is indicated for. Never invent a sentence number.")
DRUG_INTERPRET_TASK = ("Using ONLY the RETRIEVED DATA, interpret the drug evidence for the question. Every source is given as a numbered list of `sentences` (the first sentence is number 1). "
                       "Reply with ONE JSON object {\"drugs\": [...]} and nothing else. "
                       "drugs: the drugs the `sources` describe as used in clinical practice for the question's condition, its symptoms or co-occurring conditions, other than the drugs already in `fda_labelled`. "
                       "One object per drug, at most 14: {\"drug\": the name of ONE specific drug exactly as the sentence writes it, \"drug_kind\": \"generic\" | \"brand\" | \"class\" (a group of drugs such as a family or a type), \"source_id\": the exact id of the source, \"sentence\": the number of the sentence in that source that names the drug and what it is used for, "
                       "\"relevance\": \"treats_condition\" | \"treats_associated_symptom_or_comorbidity\" | \"not_relevant\" | \"undetermined\", "
                       "\"statement_type\": \"used_in_practice\" (the sentence says the drug is used, prescribed, given, recommended or licensed for it in practice, guidance or a label) | "
                       "\"study_result\" (it reports what happened in a study or analysis: improved, better than placebo, an effect size) | \"other\", "
                       "\"category\": \"Guideline-recommended\" (only if that source is a guideline or an organisation's recommendation) | \"Off-label\", "
                       "\"indication\": what the drug is used for, a short phrase (at most 8 words) copied from that sentence, \"population\": the patients, in the words of that sentence, or \"\", \"dose_sentence\": the number of a sentence in the SAME source that states this drug's dose, route or frequency for this use, or 0 if no sentence does}. "
                       "A drug counts if the sentence says it is USED, PRESCRIBED, GIVEN or RECOMMENDED in practice (statement_type used_in_practice), or reports a clinical study of it in patients with the condition (statement_type study_result); never call a study result used_in_practice. "
                       "Leave out drugs that are only being tested in a trial or protocol, animal or laboratory work, drugs named as a risk or an exposure, side effects, supplements, therapies and devices. "
                       "A drug counts only if the sentence says what it is used for in specific terms (a named symptom, behaviour or condition): a general phrase such as \"symptoms\" or \"co-occurring symptoms\" is not enough, leave that drug out. "
                       "Use \"undetermined\" when the text does not make the use clear. Never invent a drug, an id or a sentence number; if there are none, \"drugs\" is [].")
DOSE_PICK_TASK = ("For each drug, choose the ONE candidate sentence that states the drug's dose, route of administration or frequency for the question's condition (and the patients the question names, "
                  "if any). Reply with ONE JSON object {\"doses\": [...]} and nothing else, one object per drug: {\"drug\": the drug as given, \"candidate\": the id of the chosen candidate}. "
                  "Leave a drug out when no candidate states a dose, route or frequency for this use (a candidate about a different condition or patient group does not count). Never invent an id.")
TRIAGE_PAPERS = 36      # how many retrieved papers the relevance reading (triage) reads
@lru_cache(maxsize=None)
def skill_step(name: str, number: int) -> str:
    """The text of `## Step <number>:` of a skill's SKILL.md (the rules live in that file, not in the code)."""
    text = (Path(__file__).parent / "skills" / name / "SKILL.md").read_text(encoding="utf-8")
    found = re.search(rf"^## Step {number}:.*?(?=^## )", text, re.S | re.M)
    return found.group(0).strip() if found else ""


def with_rules(task: str, step: int) -> str:
    """A figure task followed by the rules of its step in skills/figure-selection/SKILL.md. The rules go in the SAME message as the data (the text model followed them there and ignored them when they
    were only the system instructions: it answered with no figure at all)."""
    return task + "\n\nRULES (the figure-selection skill):\n" + skill_step("figure-selection", step)


FIGURE_JOBS: dict = {}
FIGURE_FOLDER = Path(__file__).parent / "demo_output" / "figures"
FIGURE_TASK = ("Follow the figure-selection skill, Step 1: original figures from published open-access papers are offered for a doctor's question; choose only those that pass all five tests (at most 5, most useful first). "
               "Reply with ONE JSON object {\"figures\": [{\"id\": the exact id, \"helps_answer\": true/false, \"clinically_meaningful\": true/false, \"adds_beyond_report\": true/false, "
               "\"not_decorative\": true/false, \"answers_what_is_asked\": true/false, \"reason\": 4 to 15 consecutive words copied exactly from that figure's caption}]} and nothing else. If none qualifies, \"figures\" is [].")
POPULATION_TASK = ("Follow the figure-selection skill, Step 2: decide whether each paper studied the patient population the question names. Reply with ONE JSON object and nothing else: "
                   "{\"question_population\": ..., \"papers\": [{\"pmid\": the paper's pmid, \"study_population\": ..., \"evidence_quote\": ..., \"verdict\": \"match\" | \"broader\" | \"mismatch\" | \"unclear\"}]}.")
EXPLAIN_TASK = ("Follow the figure-selection skill, Step 5: each of `figures` is an ORIGINAL figure that will be shown to the doctor; you are given its caption, its paper's title and abstract, and `image_description`, "
                "what a vision model saw in the picture. Write the three lines for each. Reply with ONE JSON object and nothing else: {\"figures\": [{\"id\": the exact id, \"how_to_read\": ..., \"how_to_read_quote\": ..., "
                "\"shows_for_question\": ..., \"shows_quote\": ..., \"helps_treatment\": ..., \"helps_quote\": ...}]}.")
STRICT_TASK = ("Follow the figure-selection skill, Step 3: decide for each figure whether the figure ITSELF presents the kind of information the question asks for. Reply with ONE JSON object and nothing else: "
               "{\"asked_kind\": ..., \"figures\": [{\"id\": the exact id, \"presents_it\": true/false, \"what_it_presents\": up to 12 words}]}.")
VISION_PROMPT = ("This image is a figure from a published paper. The question asked: {question} The paper's caption: {caption} "
                 "Look at the image itself. Reply with JSON only: {{\"shows\": what the image actually shows, in one sentence, \"matches_caption\": true if it is what the caption describes, "
                 "\"relevant_to_question\": true if it shows results, data or a diagram that bears on the question, \"legible\": true if its text, labels and axes are readable}}. "
                 "Do not say whether a treatment works; only describe what the image shows.")


GROQ_VISION_MODEL = "qwen/qwen3.8-27b"
MISTRAL_VISION_MODEL = "mistral-medium-latest"


def vision_settings() -> tuple[str, str]:
    """(provider, model) of the vision step. VISION_PROVIDER (groq or mistral) wins; without it Groq is used when a GROQ_API_KEY is set, otherwise Mistral. VISION_MODEL overrides the model."""
    provider = os.getenv("VISION_PROVIDER", "").strip().lower() or ("groq" if os.getenv("GROQ_API_KEY") else "mistral")
    default = {"groq": GROQ_VISION_MODEL, "mistral": MISTRAL_VISION_MODEL}.get(provider, "")
    return provider, os.getenv("VISION_MODEL", "").strip() or default


def vision_image(figure: dict) -> tuple[str, str]:
    """(mime type, base64) of the image to show the vision model: the saved original (or its lossless PNG copy); a very large one is shrunk first because providers limit the request size."""
    import base64
    import io
    from PIL import Image
    path = Path(figure.get("pdf_path") or figure["path"])
    raw = path.read_bytes()
    mime = "image/png" if path.suffix == ".png" else "image/jpeg"
    if len(raw) > 2_500_000:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        image.thumbnail((1800, 1800))
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=88)
        raw, mime = buffer.getvalue(), "image/jpeg"
    return mime, base64.b64encode(raw).decode()


def vision_ask(text: str, figures: list[dict], max_tokens: int = 400):
    """One call to the vision model with one or more of the saved pictures (the originals, or their lossless PNG copies). Returns the reply text, or None when it could not be reached or refused (rate
    limit, error, no key). A provider's token limit is waited out: when Groq says 'try again in N seconds' the call is repeated."""
    try:
        provider, model = vision_settings()
        images = [vision_image(f) for f in figures]
        if provider == "groq":
            from groq import Groq
            client = Groq(api_key=os.environ["GROQ_API_KEY"], timeout=40.0, max_retries=0)
            extra = {"reasoning_effort": "none"} if model.startswith("qwen") else {}      # the answer is a short JSON: no thinking phase
            content = [{"type": "text", "text": text}] + [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}} for mime, data in images]
            waited = 0.0
            for attempt in range(4):
                try:
                    reply = client.chat.completions.create(model=model, temperature=0, max_tokens=max_tokens, messages=[{"role": "user", "content": content}], **extra)
                    return reply.choices[0].message.content
                except Exception as error:
                    pause = re.search(r"try again in ([\d.]+)s", str(error))
                    if "rate" not in str(error).lower() or not pause or attempt == 3 or waited > 24:
                        raise
                    time.sleep(min(float(pause.group(1)) + 0.5, 12.0))      # a free-tier token limit: wait as long as Groq says, then ask again
                    waited += float(pause.group(1)) + 0.5
        try:
            from mistralai import Mistral
        except ImportError:      # mistralai 2.x keeps the client in a sub-package
            from mistralai.client import Mistral
        client = Mistral(api_key=os.environ["MISTRAL_API_KEY"], timeout_ms=40000)
        content = [{"type": "text", "text": text}] + [{"type": "image_url", "image_url": f"data:{mime};base64,{data}"} for mime, data in images]
        reply = client.chat.complete(model=model, temperature=0, max_tokens=max_tokens, messages=[{"role": "user", "content": content}])
        return reply.choices[0].message.content
    except Exception:
        return None


def vision_inspect(figure: dict, question: str):
    """A vision model looks at the downloaded ORIGINAL image. Returns its reply as text, or None when it could not be reached or refused: the figure is then skipped. Used only to verify the image,
    never as clinical evidence."""
    return vision_ask(VISION_PROMPT.format(question=question, caption=cut_for_model(figure["caption"], 600)), [figure])


def written_evidence(markdown: str, limit: int = 3500) -> str:
    """What the report already says in words about the evidence: its summary, current evidence and key studies sections."""
    parts = [m.group(0) for m in re.finditer(r"(?ms)^## (?:Clinical summary|Current evidence|Key studies)\s*\n.*?(?=^## |\Z)", markdown or "")]
    return cut_for_model(" ".join("\n".join(parts).split()), limit)


def find_figures(question: str, papers: dict, report: str, found: list, trace: dict, condition: str = "") -> list:
    """The whole selection of figures, run BESIDE the writing of the report (not after it): the gates, the population check, the relevance test, the vision check and the two plain-language lines.
    `report` is the evidence the report has written so far (used by the relevance test to see what a figure adds). Any failure, refusal or timeout only means fewer figures or none."""
    direct = {pmid for pmid, p in papers.items() if (p.get("triage") or {}).get("relevance") == "direct"}
    indirect = {pmid for pmid, p in papers.items() if (p.get("triage") or {}).get("relevance") == "indirect"}      # a related condition or population: allowed, and labelled as indirect evidence

    def rubric(available: list) -> list:
        ordered = sorted(available, key=lambda f: (f.get("relation") != "direct", available.index(f)))      # direct papers first
        chunks = [ordered[k:k + 10] for k in range(0, min(len(ordered), 40), 10)]      # a reply about many more figures than this does not fit the model's output limit: several calls, side by side

        def judge(chunk: list) -> list:
            payload = {"question": question, "written_report": report, "figures": [{"id": f["id"], "paper": (f.get("title") or "")[:140], "design": f.get("design") or "not stated",
                                                                                      "relation": f.get("relation") or "direct", "caption": cut_for_model(f["caption"], 600)} for f in chunk]}
            try:
                return ask_json(question, with_rules(FIGURE_TASK, 1), payload, 900, skill="evidence-synthesis").get("figures") or []
            except Exception:
                return []
        with ThreadPoolExecutor(max_workers=max(len(chunks), 1)) as judge_pool:
            return [item for part in judge_pool.map(judge, chunks) for item in part]

    def population(available: list) -> list:
        related = [{**f, "population_note": "related condition or population: " + (((papers.get(str(f["pmid"])) or {}).get("triage") or {}).get("population") or "see the paper")}
                   for f in available if f.get("relation") == "indirect"]      # indirect evidence is labelled, not matched to the question's population
        available = [f for f in available if f.get("relation") != "indirect"]
        if not available:
            return related
        texts, views = {}, []
        for pmid in sorted({str(f["pmid"]) for f in available}):
            p = papers.get(pmid) or {}
            note = (p.get("triage") or {}).get("population") or ""
            texts[pmid] = " ".join(x for x in (p.get("title"), p.get("abstract"), note) if x)
            views.append({"pmid": pmid, "title": p.get("title"), "abstract": cut_for_model(p.get("abstract") or "", 1500), "population_noted_by_relevance_step": note or None})
        return apply_population_check(available, ask_json(question, with_rules(POPULATION_TASK, 2), {"question": question, "papers": views}, 700, skill="evidence-synthesis"), texts, question, trace, condition=condition) + related
    figures: list = []
    try:
        check = (lambda fig: vision_inspect(fig, question)) if os.getenv("FIGURE_IMAGE_CHECK", "off").strip().lower() == "on" else None
        figures = select_figures(found, set(), direct, rubric, check, FIGURE_FOLDER, trace=trace, population=population, indirect_pmids=indirect)
    except Exception as error:
        trace["error"] = f"{type(error).__name__}: {str(error)[:80]}"
    if figures:
        try:      # a second, focused judgement: does each figure itself present the kind of information the question asks for? (a figure that only describes the problem is dropped)
            payload = {"question": question, "figures": [{"id": f["id"], "caption": cut_for_model(f["caption"], 500), "paper_title": f.get("title"), "image_description": f.get("image_check") or ""} for f in figures]}
            answer = {}
            for attempt in range(2):
                try:
                    answer = ask_json(question, with_rules(STRICT_TASK, 3), payload, 700, skill="evidence-synthesis")
                except Exception:
                    answer = {}
                if answer.get("figures"):
                    break
            before = list(figures)
            figures = apply_strict_check(figures, answer, trace)
            for f in before:
                if not any(k["id"] == f["id"] for k in figures):
                    trace.setdefault("figure_fates", []).append({"id": f["id"], "label": f.get("label"), "fate": "dropped by the strict check: it does not itself present what the question asks"})
        except Exception as error:
            trace["strict_check_error"] = f"{type(error).__name__}: {str(error)[:80]}"      # the extra check could not run: the figures stay
    if figures:
        try:      # two plain-language lines per figure, each kept only if it can be verified against the figure's caption and the paper's own text (source_figures.explain_figures)
            texts = {str(f["pmid"]): " ".join(x for x in ((papers.get(str(f["pmid"])) or {}).get("title"), (papers.get(str(f["pmid"])) or {}).get("abstract")) if x) for f in figures}

            def one(f: dict):      # one figure per call, side by side: a single reply covering several figures can be cut off at the model's output limit
                payload = {"question": question, "figures": [{"id": f["id"], "caption": cut_for_model(f["caption"], 900), "paper_title": f.get("title"),
                                                             "paper_abstract": cut_for_model(texts[str(f["pmid"])], 1800), "image_description": f.get("image_check") or ""}]}
                for attempt in range(2):      # a reply that is missing or unusable is asked for once more
                    try:
                        got = ask_json(question, with_rules(EXPLAIN_TASK, 5), payload, 1300, skill="evidence-synthesis").get("figures") or []
                    except Exception:
                        got = []
                    if got:
                        return got
                return []
            with ThreadPoolExecutor(max_workers=len(figures)) as explain_pool:
                answers = [item for part in explain_pool.map(one, figures) for item in part]
            figures = explain_figures(figures, {"figures": answers}, texts, trace)
        except Exception as error:
            trace["explanations_error"] = f"{type(error).__name__}: {str(error)[:80]}"
    return figures


def collect_figures(state: dict, markdown: str = "", timeout: float = 15.0) -> list:
    """The figures found beside the writing: normally ready by the time the report is final, so this is a short wait (at most `timeout` seconds). If they are not ready by then the report goes out
    without them: the report is never held up for figures. What happened is kept in state['figure_trace']."""
    job = FIGURE_JOBS.get(state.get("figure_job"))
    if job is None:
        return state.get("source_figures") or []
    thread, box = job
    thread.join(timeout=timeout)
    ready = not thread.is_alive()
    state["figure_trace"] = {**box.get("trace", {}), "ready_when_the_report_was_final": ready}
    if not ready:      # not ready yet: the report goes out now, the job is kept and its figures are delivered later (late_figures)
        state["source_figures"] = []
        return []
    FIGURE_JOBS.pop(state.get("figure_job"), None)
    state["source_figures"] = list(box.get("figures") or [])
    return state["source_figures"]


def figures_pending(final: dict) -> bool:
    """True when the figure job of this answer has not finished yet (its figures will be delivered later)."""
    return final["state"].get("figure_job") in FIGURE_JOBS


def deliver_outputs(question: str, final: dict):
    """After the final event, the events that follow it, shared by every front end (the web page and the Chainlit app): `figures` (the papers' original figures, unless still being prepared),
    `pdf` (the saved report), and, when the figures were not ready, a late `figures` event (late=True) followed by a `pdf` event for the PDF rebuilt with them."""
    def items(records: list) -> list:
        return [{k: v for k, v in f.items() if k != "pdf_path"} for f in records]
    figures = prepare_figures(final)      # found beside the writing: normally ready already, otherwise a short wait
    pending = figures_pending(final)
    if not pending:
        yield {"type": "figures", "items": items(figures)}
    paths = save_outputs(question, final, figures=figures)
    yield {"type": "pdf", "name": paths["pdf"].name if paths["pdf"] else None}
    if pending:
        late = late_figures(final)
        yield {"type": "figures", "items": items(late), "late": True}
        if late:
            paths = save_outputs(question, final, figures=late)
            yield {"type": "pdf", "name": paths["pdf"].name if paths["pdf"] else None}


def late_figures(final: dict, timeout: float = 150.0) -> list:
    """The records of the figures that were not ready when the report was final, waited for at most `timeout` more seconds (the report itself was shown long before). [] when they never come."""
    job = FIGURE_JOBS.get(final["state"].get("figure_job"))
    if job is None:
        return []
    thread, box = job
    thread.join(timeout=timeout)
    if thread.is_alive():
        return []
    FIGURE_JOBS.pop(final["state"].get("figure_job"), None)
    final["state"]["figure_trace"] = {**box.get("trace", {}), "ready_when_the_report_was_final": False, "delivered_late": True}
    final["state"]["source_figures"] = list(box.get("figures") or [])
    return display_records(final["state"]["source_figures"], final["state"].get("reference_numbers"), final["markdown"])


def prepare_figures(final: dict) -> list:
    """The records of the figures for the page and the PDF (one list for both): what is ready when the report is final, with where each belongs in the report."""
    return display_records(collect_figures(final["state"], final["markdown"]), final["state"].get("reference_numbers"), final["markdown"])


REGULATOR_TASK = ("Using ONLY the RETRIEVED DATA, read each regulator page in `items` for the question. Each page is given as numbered `sentences` (the first is number 1). "
                  "Reply with ONE JSON object {\"items\": [...]} and nothing else. One object per page: {\"id\": its id, "
                  "\"classification\": \"approval\" | \"label_update\" | \"safety_communication\" | \"other_regulatory_action\" | \"not_relevant\" | \"uncertain\", "
                  "\"relation\": how the action relates to the question: \"treats_condition\" (the action is about treating the question's condition) | \"treats_associated_symptom_or_comorbidity\" | "
                  "\"risk_or_safety_related\" (a risk or safety matter about the question's condition or its treatment) | \"other_indication\" (the action is about a different indication) | \"unclear\", "
                  "\"concerns\": what the action concerns (drug and indication), in the words of the page, \"drug\": the drug's name as written in the page, or \"\", "
                  "\"sentences\": [the numbers of the one or two sentences that show it], \"date\": the announcement date exactly as written in the page, or \"\"}. "
                  "A page that merely mentions the question's condition is not an approval for it: read what the action actually concerns. "
                  "Medical devices, diagnostic tests, meetings, transcripts, testimony, written requests, review documents, guidance, calendars and general information pages are not_relevant: an action is a decision "
                  "by the regulator about a drug (approval, new indication, label change, safety communication, restriction or withdrawal). Never invent a date, a drug or a sentence number.")
def lookup_doses(rows: list[dict], condition: str, question: str, cap: float = 9.0, stems: list[str] | None = None) -> None:
    """Dose and route for every listed drug that has none yet: one extra web search per drug (in parallel, at most `cap` seconds), the sentences that name the drug and state an amount, a route
    or a frequency are offered to the model, which picks one per drug (landscape.verify_dose_picks). A drug without a verified sentence keeps 'not stated'."""
    targets = [r for r in rows if not r.get("dose")][:14]
    if not targets:
        return
    pool = ThreadPoolExecutor(max_workers=len(targets))
    futures = {pool.submit(web_search, f"{r['drug']} dose and route of administration for {condition}", 1500): r for r in targets}
    candidates: dict[str, list] = {}
    try:
        done, _ = wait(list(futures), timeout=cap)
        for future in done:
            try:
                results = _json(future.result())
            except Exception:
                continue
            found = dose_candidates(futures[future]["drug"], results if isinstance(results, list) else [], stems=stems)
            if found:
                candidates[futures[future]["drug"]] = found
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    if not candidates:
        return
    payload = {"condition": condition, "drugs": [{"drug": d, "candidates": [{"id": c["id"], "text": c["text"]} for c in cs]} for d, cs in candidates.items()]}
    chosen = verify_dose_picks(ask_json(question, DOSE_PICK_TASK, payload, 600).get("doses"), candidates)
    for row in targets:
        pick = chosen.get(row["drug"])
        if pick:
            row["dose"], row["dose_url"] = pick["text"], pick["url"]


@traced("ask_json (skill model call)", "llm")
def ask_json(question: str, task: str, payload: dict, max_tokens: int, skill: str = "drug-intelligence", purpose: str = "triage") -> dict:
    """One drug-intelligence call that must answer with a JSON object; the object is returned as parsed (nothing in it is trusted until the caller has verified it against the source)."""
    prompt = f"TASK: {task} Write the JSON compactly on a single line, with no indentation or line breaks.\n\nQUESTION: {question}\nRETRIEVED DATA (JSON):\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    raw = "".join(stream_llm(load_skill_text(skill), prompt, max_tokens, temperature=0.0, purpose=purpose))      # reading evidence into a structured decision: the same input gives the same decision
    found = re.search(r"\{.*\}", raw, re.S)
    text = found.group(0) if found else ""
    try:
        data = json.loads(text) if text else {}
    except json.JSONDecodeError:      # the answer ran into the token limit: keep the complete objects before the cut, never a half-written one
        data = {}
        for key in ("fda_labels", "drugs", "items", "verdicts", "papers", "trials", "queries"):
            part = re.search(rf'"{key}"\s*:\s*\[(.*)', raw, re.S)
            if part:
                objects, depth, start = [], 0, None
                for i, ch in enumerate(part.group(1)):
                    if ch == "{":
                        start, depth = (i if depth == 0 else start), depth + 1
                    elif ch == "}" and depth:
                        depth -= 1
                        if depth == 0 and start is not None:
                            try:
                                objects.append(json.loads(part.group(1)[start:i + 1]))
                            except json.JSONDecodeError:
                                pass
                data[key] = objects
    data = data if isinstance(data, dict) else {}
    data.setdefault("_raw", raw[:2500])      # kept for the audit trail of what the model actually returned
    return data


DRUG_TASK = ("Write ONLY this block. Heading '#### Other agents named in the retrieved evidence (not in the FDA label records)' followed by a Markdown table with columns: "
             "Drug (generic/brand) | Category | Population studied | Dose / route | Approval status | Key evidence. "
             "Category must be one of: Investigational, Indirect evidence (related population), Guideline-recommended (only if a retrieved guideline source says so), "
             "Off-label (only if a retrieved source states off-label use for this condition). Do NOT list any drug in `regulatory_records_already_listed`. "
             "List only SUBSTANCES named in the data (drugs, supplements, probiotics, biologics). Do NOT list exercise, dance, music, diet, therapy, behavioural or device interventions: they belong to the non-drug section. Do NOT list a drug that appears only as a risk or exposure (for example prenatal exposure). Write exactly six cells per row. If there are none, write 'None retrieved.' Then, only if the data has any, '**Safety alerts:**' with a source. "
             "Under 220 words. " + GROUND)

INTENTS = ("disease_research", "latest_findings", "drug_treatment", "clinical_trial", "comparison", "mechanism", "diagnosis", "safety", "case_evaluation", "comprehensive_research")
PLAN_TASK = ("Follow the skill: classify the question and extract entities. Reply with ONE JSON object only. Keys: \"is_clinical\" (true if a medical question), "
             "\"intent\" (disease_research|latest_findings|drug_treatment|clinical_trial|comparison|mechanism|diagnosis|safety|case_evaluation|comprehensive_research), \"secondary_intents\" (a list of additional applicable intents from the same list, excluding the primary intent), \"condition\", "
             "\"drugs\" (drug names in the question), \"pubmed_query\" (PubMed terms with AND/OR), \"trial_condition\", \"trial_intervention\" (drug or empty), "
             "\"asked\" (what the doctor is asking, in one short sentence), \"purpose\" (why they are asking: the clinical decision or need behind the question, in one short sentence), "
             "\"info_needed\" (the kinds of information needed, from: treatment_options|efficacy|safety|dosing|diagnosis|prognosis|mechanism|guideline_recommendation|trial_status|regulatory_status|drug_interaction), "
             "\"medicines_needed\" (true only when the question names a medicine, asks which medicines or doses to use, asks about drug safety or approvals, compares drugs, or is a plain question of how to treat a condition for which medicines are a main treatment; false when it asks for a work-up, a diagnosis, how something works, or the management of a case without asking about medicines), "
             "\"answer_format\" (how the answer is best organised: options_comparison|stepwise_plan|summary|table|yes_no_with_evidence), "
             "\"case_facts\" (when the question describes a patient or case: the facts that matter, each copied WORD FOR WORD from the question as a short phrase; [] when there is no case), "
             "\"semantic_variants\" (3 to 5 short phrases, 2 to 8 words each, that published papers might use for the SAME clinical problem in place of the question's own words: synonyms, older or newer terms, closely related terminology; never a different condition and never a drug the question did not name), "
             "\"skills\" (which of the skills in the SKILL CATALOG must write expanded searches for this question: a list of their exact names; a case study may need several, a question about one drug's safety may need only one).")
INFO_NEEDED = ("treatment_options", "efficacy", "safety", "dosing", "diagnosis", "prognosis", "mechanism", "guideline_recommendation", "trial_status", "regulatory_status", "drug_interaction")
ANSWER_FORMATS = ("options_comparison", "stepwise_plan", "summary", "table", "yes_no_with_evidence")
def _squash(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def verified_understanding(plan: dict, question: str) -> dict:
    """What the model understood about the question, kept only as far as the question itself supports it: `case_facts` must be word for word in the question; the kinds of information, the answer
    format and the sources must come from the fixed lists; the drug names of a sub-question must be written in the question. `asked` and `purpose` are the model's own reading (short sentences) and are
    labelled as interpretation, never as fact about the patient."""
    own = _squash(question)

    def sentence(value) -> str:
        return " ".join(str(value or "").split())[:220]
    facts = [" ".join(str(f).split())[:140] for f in (plan.get("case_facts") if isinstance(plan.get("case_facts"), list) else []) if str(f).strip()]
    facts = [f for f in dict.fromkeys(facts) if _squash(f) and _squash(f) in own][:12]
    info = [i for i in (plan.get("info_needed") if isinstance(plan.get("info_needed"), list) else []) if i in INFO_NEEDED]
    # A malformed planner reply must not silently erase the user's broad
    # request.  These are general request-intent cues, never condition or age
    # rules; they only preserve the types of evidence the question explicitly
    # asks the retrievers to cover.
    lower_question = question.lower()
    if not info:
        intent_cues = (
            (("diagnos", "work-up", "workup", "evaluat", "assess"), "diagnosis"),
            (("treat", "manage", "interven", "rehabilit", "follow-up"), "treatment_options"),
            (("guideline", "recommend"), "guideline_recommendation"),
            (("safety", "adverse", "harm", "risk"), "safety"),
            (("trial", "study recruiting", "clinicaltrials"), "trial_status"),
            (("fda", "approval", "label", "approved"), "regulatory_status"),
            (("dose", "dosing"), "dosing"),
        )
        info = [need for cues, need in intent_cues if any(cue in lower_question for cue in cues)]
    secondary = [i for i in (plan.get("secondary_intents") if isinstance(plan.get("secondary_intents"), list) else []) if i in INTENTS and i != plan.get("intent")]
    fmt = plan.get("answer_format") if plan.get("answer_format") in ANSWER_FORMATS else "summary"
    if fmt == "summary" and "diagnosis" in info:
        fmt = "stepwise_plan"
    medicines = plan.get("medicines_needed") if isinstance(plan.get("medicines_needed"), bool) else None      # unknown stays unknown: the earlier behaviour
    return {"medicines_needed": medicines, "asked": sentence(plan.get("asked")), "purpose": sentence(plan.get("purpose")), "info_needed": list(dict.fromkeys(info)), "secondary_intents": list(dict.fromkeys(secondary)), "answer_format": fmt, "case_facts": facts}


EXPANDER_SKILLS = ("latest-evidence", "clinical-trials", "drug-intelligence")      # the skills that write expanded searches (each has a "Query transformation by intent" section)
EXPANSION_TOTAL_S = float(os.getenv("ROUTING_DEADLINE_S", "30"))      # HARD DEADLINE: from the start of the deep search, the evidence group waits at most this long for the routed searches; unfinished work is recorded, never waited for
EXPAND_TASK = ("Follow the section below to write the expanded searches of the {skill} skill for this question. `intent` and `understanding` say what the question needs; keep the patient and the condition of the "
               "question in every search. Reply with ONE JSON object and nothing else, in the format the section gives.")


OFF_WORDS = {"off", "0", "false", "no"}


def routing_enabled() -> bool:
    """ENABLE_DEEP_ROUTING=false turns the model-written, source-aware deep searches off (the older DEEP_ROUTING / QUERY_EXPANSION=off still work when it is not set)."""
    if os.getenv("ENABLE_DEEP_ROUTING", "").strip():
        return os.getenv("ENABLE_DEEP_ROUTING", "").strip().lower() not in OFF_WORDS
    return all(os.getenv(name, "on").strip().lower() not in OFF_WORDS for name in ("DEEP_ROUTING", "QUERY_EXPANSION"))


def legacy_deep_enabled() -> bool:
    """ENABLE_LEGACY_SEARCH=false skips the legacy deep search jobs, so the deep search is the routed one only (the older LEGACY_DEEP=off still works when it is not set; the fast search is never affected)."""
    return os.getenv("ENABLE_LEGACY_SEARCH", os.getenv("LEGACY_DEEP", "on")).strip().lower() not in OFF_WORDS


def routing_outcome(stage: dict | None) -> dict | None:
    """The source status of the deep routing when it did not do its job, else None. `failed`: it errored, was skipped (plan not ready) or wrote no valid search;
    `partial`: it ran out of time and only the searches already started were used. The legacy results still make the report either way."""
    stage = stage or {}
    trace = stage.get("trace") or {}
    if "error" in trace:
        return {"retrieval_status": "failed", "evidence_status": "unavailable", "error": str(trace["error"])[:120]}
    if "skipped" in trace and trace["skipped"] != "deep routing is off":
        return {"retrieval_status": "failed", "evidence_status": "unavailable", "error": str(trace["skipped"])[:120]}
    if not stage:
        return {"retrieval_status": "failed", "evidence_status": "unavailable", "error": "the routing stage did not report"}
    outcomes = [v.get("outcome") for v in (trace.get("skills") or {}).values()]
    if outcomes and all(o in ("malformed", "error", "all_rejected") for o in outcomes) and not [o for o in (stage.get("ops") or []) if getattr(o, "origin", "") not in ("baseline",)]:
        return {"retrieval_status": "failed", "evidence_status": "unavailable", "error": "no valid search was written by the skills"}
    if stage.get("unfinished") or trace.get("partial"):
        return {"retrieval_status": "partial", "evidence_status": "unavailable", "error": "timed out: the routed searches did not all finish within the deadline"}
    return None


ROUTING_BUDGET_S = float(os.getenv("ROUTING_BUDGET_S", "18"))      # the time the routed searches get after their queries are written


@lru_cache(maxsize=None)
def skill_description(name: str) -> str:
    text = (Path(__file__).parent / "skills" / name / "SKILL.md").read_text(encoding="utf-8")
    found = re.search(r"(?ms)^description:\s*(.+?)\n---", text)
    return " ".join(found.group(1).split()) if found else ""


def retrieval_catalog() -> str:
    """The catalog the quick plan call reads: only each retrieval skill's name and one-line description (progressive loading: the full files are loaded afterwards, only for the skills chosen)."""
    return "\n".join(f"- {name}: {skill_description(name)[:260]}" for name in EXPANDER_SKILLS)


@lru_cache(maxsize=None)
def skill_section(name: str, heading: str) -> str:
    """The text of the `## <heading>` section of a skill's SKILL.md (the rules live in the file, not in the code)."""
    text = (Path(__file__).parent / "skills" / name / "SKILL.md").read_text(encoding="utf-8")
    found = re.search(rf"^## {re.escape(heading)}.*?(?=^## |\Z)", text, re.S | re.M)
    return found.group(0).strip() if found else ""


@lru_cache(maxsize=None)
def skill_catalog() -> dict:
    return rr.load_skill_catalog(Path(__file__).parent / "skills")


def ingest_guideline_ops(state: dict, raw: dict) -> None:
    """The routed `guideline_sites` searches (keys web_guideline_op_*) -> state['guidelines'] rows (authoritative organisations only, the condition named in the page), merged with any rows already there."""
    from guidelines import rows_from_results
    stems = [x for x in (state.get("core_stems") or []) if len(x) >= 4]
    known = {g.get("url") for g in state.get("guidelines", [])}
    for key, text in raw.items():
        if not key.startswith("web_guideline_op"):
            continue
        data = _json(text)
        if isinstance(data, dict) and data.get("error"):
            continue      # recorded as a failed source by record_source_status; never an empty result
        rows = [r for r in rows_from_results(data if isinstance(data, list) else [], load_care_config(), stems, topic=key, label="Guideline (searched for this question)") if r["url"] not in known]
        for r in rows:
            known.add(r["url"])
            state.setdefault("guidelines", []).append(r)
            state.setdefault("guideline_rows", []).append(r)
        if isinstance(data, list):
            state["guideline_search_ok"] = True


def real_adapters(anchors: set[str]) -> dict:
    """The seven targets, wired to the existing search functions and the pipeline's result cache."""
    return rr.make_adapters(search_pubmed=search_pubmed, search_europepmc_quick=search_europepmc_quick, search_clinical_trials=search_clinical_trials, search_fda_labels=search_fda_labels,
                            guideline_search=guideline_search, regulator_search=regulator_search, web_search=web_search, anchors=anchors, cache=cached,
                            guideline_filter=filter_authoritative)


def _write_queries(question: str, plan: dict, skill: str, anchors: set[str], ask, attempts: int = 2) -> dict:
    """One retrieval skill, loaded from its SKILL.md, writes its searches. Returns {'ops', 'rejected', 'outcome', 'seconds'}. outcome: wrote | declined (the source is not needed: not an error) |
    all_rejected | malformed | error. A reply that could not be used is retried once; a decline is never retried."""
    began = time.time()
    rules = skill_section(skill, "Query transformation by intent")
    if not rules:
        return {"ops": [], "rejected": [], "outcome": "error", "seconds": 0.0, "error": "the skill has no query transformation section"}
    payload = {"question": question, "intent": plan.get("intent"), "understanding": {k: v for k, v in (plan.get("understanding") or {}).items() if v}}
    task = EXPAND_TASK.format(skill=skill) + "\n\nSECTION (the " + skill + " skill):\n" + rules
    outcome, rejected, error = "error", [], ""
    for _ in range(attempts):
        try:
            reply = ask(question, task, payload, 1500, skill=skill)
        except Exception as exc:
            outcome, error = "error", f"{type(exc).__name__}: {str(exc)[:80]}"
            continue
        ops, rejected = rr.parse_operations(skill, reply, question, anchors)
        if ops:
            return {"ops": ops, "rejected": rejected, "outcome": "wrote", "seconds": round(time.time() - began, 1)}
        if rr.declined(reply) or (isinstance(reply, dict) and reply.get("queries") == []):
            return {"ops": [], "rejected": [], "outcome": "declined", "seconds": round(time.time() - began, 1)}
        outcome = "all_rejected" if isinstance(reply, dict) and isinstance(reply.get("queries"), list) else "malformed"
    return {"ops": [], "rejected": rejected, "outcome": outcome, "seconds": round(time.time() - began, 1), "error": error}


COVERAGE_CALL_S = float(os.getenv("COVERAGE_CALL_S", "8"))      # the time the one bounded call that fills coverage gaps may take
MAX_COVERAGE_ADDITIONS = int(os.getenv("ROUTING_MAX_COVERAGE_ADDITIONS", "4"))      # the most searches that call may add
COVERAGE_TASK = ("Some information needs of this question have too few searches before retrieval starts. Write the missing searches for the gaps below, at most {n} in all. Each must differ in purpose, source or substance "
                 "from the searches already written (`already_written`), keep the condition and the patient details of the question, and use ONLY a target from the list in the section. For a `discovery_shortfall` write `missing` "
                 "`web` searches, one for each of the listed `purposes` that the `have` searches do not already cover. For a `need_without_suitable_source` write one search for that need on one of the `suitable` sources "
                 "and set `serves` to that need. For an `authority_source_missing` write one search for each target in `missing_targets` (the guideline organisations' pages, or the regulators' pages), about what the question asks (its evaluation, management or medicines) for the "
                 "case's findings and patient group, using the findings in the case's own words. For a `single_source_need` decide which of its `other_sources` (see `offers`) could really hold evidence for that need, and write one search for each of those, "
                 "using the same `need_id`, `need` and `serves`; skip a source that cannot hold it. A gap that cannot be filled gets nothing. Reply with ONE JSON object and nothing else: "
                 "{{\"queries\": [{{\"need_id\": ..., \"need\": ..., \"serves\": ..., \"target\": ..., \"query\": ..., \"args\": {{}}, \"priority\": \"essential\" or \"supporting\"}}]}}.")


@lru_cache(maxsize=None)
def source_capabilities() -> dict:
    return rr.load_capabilities()


def coverage_check_enabled() -> bool:
    return os.getenv("COVERAGE_CHECK", "on").strip().lower() not in {"off", "0", "false", "no"}


def fill_coverage(question: str, plan: dict, gaps: list[dict], ops: list, anchors: set[str], ask=ask_json, call_s: float | None = None) -> dict:
    """ONE bounded model call that writes the missing searches for the gaps found before retrieval. Everything it writes is validated like any other search (allowlisted target, the question's terms kept) and
    near-duplicates of the searches already written are dropped. Returns {'ops', 'outcome', 'seconds', 'rejected'}; a slow or failed call adds nothing and says so."""
    began = time.time()
    limit = COVERAGE_CALL_S if call_s is None else call_s
    section = skill_section("latest-evidence", "Query transformation by intent")
    sources = section[section.find("### Sources you may target"):section.find("### Output")] if "### Sources you may target" in section else section
    discovery = any(g["kind"] == "discovery_shortfall" for g in gaps)
    payload = {"question": question, "understanding": {k: v for k, v in (plan.get("understanding") or {}).items() if v}, "gaps": gaps,
               "already_written": [{"need": o.need, "serves": o.serves, "target": o.target, "query": o.query[:110]} for o in ops]}
    box: dict = {}

    def call():
        try:
            box["reply"] = ask(question, COVERAGE_TASK.format(n=MAX_COVERAGE_ADDITIONS) + "\n\nSECTION:\n" + sources, payload, 900, skill="drug-intelligence" if discovery else "latest-evidence")
        except Exception as error:
            box["error"] = f"{type(error).__name__}: {str(error)[:80]}"
    thread = threading.Thread(target=call, daemon=True)
    thread.start()
    thread.join(timeout=limit)
    seconds = round(time.time() - began, 1)
    if thread.is_alive():
        return {"ops": [], "outcome": "timed_out", "seconds": seconds, "rejected": 0}
    if "reply" not in box:
        return {"ops": [], "outcome": "error", "seconds": seconds, "rejected": 0, "error": box.get("error", "")}
    new, rejected = rr.parse_operations("coverage", box["reply"], question, anchors, max_ops=MAX_COVERAGE_ADDITIONS)
    kept = []
    for op in new:
        if any(rr.near_duplicate(op, existing) or op.key == existing.key for existing in list(ops) + kept):
            rejected.append({"item": None, "reason": "duplicate of a search already written"})
            continue
        op.origin = "coverage"
        if discovery and op.target == "web":
            op.skill = "drug-intelligence"      # a discovery search of the drug skill: it counts toward the minimum and belongs to the drug section
        kept.append(op)
    return {"ops": kept, "outcome": "added" if kept else "nothing_added", "seconds": seconds, "rejected": len(rejected)}


@traced("route deep search (skills write source-aware queries)", "chain")
def routing_anchors(question: str, plan: dict) -> set[str]:
    return rr.anchor_words(question, plan.get("condition", ""), (plan.get("understanding") or {}).get("case_facts"))


def route_deep(question: str, plan: dict, ask=ask_json, submit=None) -> dict:
    """The model-driven part of the deep search: validate the plan's skills against what the information needs require, let each chosen skill write its searches IN PARALLEL (it may write none),
    validate every search and its target, merge duplicates, and add the minimal baselines (the original question as a literature search; a minimal search for a skill whose queries could not be written).
    `submit(ops)`, when given, starts searches the moment they are known: the baselines at once, each skill's searches when that skill has answered, the coverage additions when they are written. The
    searches therefore run while the other skills are still writing, and the returned list is exactly what was submitted."""
    understanding = plan.get("understanding") or {}
    skills, select_trace = rr.select_skills(plan.get("skills"), understanding.get("info_needed"), skill_catalog())
    anchors = routing_anchors(question, plan)
    literature = rr.baseline_operations(question, skills, fallback=bool(select_trace.get("fallback")))
    ops, duplicates = rr.merge_operations(literature)
    if submit:
        submit(ops)

    def add(new: list, cap: int = rr.MAX_OPERATIONS) -> None:
        nonlocal ops, duplicates
        merged, more = rr.merge_operations(ops, new, cap=cap)
        fresh = merged[len(ops):]
        ops, duplicates = merged, duplicates + more
        if submit and fresh:
            submit(fresh)
    written: dict = {}
    with ThreadPoolExecutor(max_workers=max(1, len(skills))) as pool:
        futures = {pool.submit(_write_queries, question, plan, s, anchors, ask): s for s in skills}
        for future in as_completed(futures):
            written[futures[future]] = future.result()
            add(written[futures[future]]["ops"])      # this skill's searches start now, while the others are still being written
    written = {s: written[s] for s in skills}
    baselines = []
    for skill, result in written.items():
        if result["outcome"] in ("malformed", "error", "all_rejected"):      # a query-generation failure, NOT a decline and NOT an empty result
            baselines += rr.baseline_for(skill, question, anchors, plan.get("condition", ""), plan.get("drugs"))
    add(baselines)
    coverage: dict = {"gaps": [], "outcome": "not_needed"}
    if coverage_check_enabled():
        gaps = rr.coverage_gaps(ops, understanding.get("info_needed"), skills, source_capabilities())      # before anything is retrieved
        coverage = {"gaps": gaps, "outcome": "no_gaps" if not gaps else "pending"}
        if gaps:
            filled = fill_coverage(question, plan, gaps, ops, anchors, ask=ask)
            before = len(ops)
            add(filled["ops"], cap=rr.MAX_OPERATIONS + MAX_COVERAGE_ADDITIONS)
            coverage.update({"outcome": filled["outcome"], "added": len(ops) - before, "seconds": filled["seconds"], "rejected": filled["rejected"]})
    return {"ops": ops, "anchors": anchors, "skills": skills, "rejected": [x for r in written.values() for x in r["rejected"]],
            "trace": {"select_skills": select_trace, "coverage": coverage, "skills": {s: {k: v for k, v in r.items() if k in ("outcome", "seconds", "error")} | {"operations": len(r["ops"])} for s, r in written.items()},
                      "duplicates_removed": len(duplicates), "baselines": [o.id for o in literature + baselines]}}


ROUTING_WORKERS = int(os.getenv("ROUTING_WORKERS", "10"))      # routed searches that run at the same time


def run_routing_stage(question: str, plan: dict, make_adapters=real_adapters, budget_s: float = ROUTING_BUDGET_S, ask=ask_json, box: dict | None = None) -> dict:
    """Write the routed searches and run them in parallel under the budget (each starts as soon as it is written), and return everything the pipeline and the saved state need."""
    anchors = routing_anchors(question, plan)
    execution = rr.Execution([], make_adapters(anchors), budget_s, workers=ROUTING_WORKERS, hold=True)
    if box is not None:      # what is already running can be used if the deadline ends before the stage is complete
        box.update(execution=execution, anchors=anchors)
    routed = route_deep(question, plan, ask=ask, submit=execution.submit)
    execution.finish()
    return {**routed, "execution": execution, "raw": rr.raw_for_ingest(routed["ops"], execution.results)}


@traced("plan: understanding", "chain")
def plan_with_skill(question: str) -> dict:
    """clinical-research-orchestrator: classify and extract entities (skill loaded on demand)."""
    plan, status = {}, "ok"
    for attempt in (1, 2):      # the plan is the base of the query decomposition: a reply that is missing, cut off or not JSON is asked for once more before the question's own words are used
        try:
            text = run_llm(load_skill_text("clinical-research-orchestrator"), f"TASK: {PLAN_TASK}\nSKILL CATALOG (name: what it is for):\n{retrieval_catalog()}\nQuestion: {question}", PLAN_TOKENS, temperature=0.0, purpose="planning" if attempt == 1 or not planner_choice()[0] else "planning_backup")      # the second try goes to the main provider when the planner's own provider failed or stalled
            plan = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
            status = "ok" if attempt == 1 else "retried"
            break
        except Exception:
            plan, status = {}, "fallback"
    plan["plan_status"] = status      # ok / retried / fallback (the question could not be analysed: searches use its own words)
    plan.setdefault("is_clinical", True)
    plan.setdefault("intent", "disease_research")
    kw, _, _ = keywords(question)
    plan.setdefault("condition", " ".join(kw) or question)
    plan.setdefault("drugs", [])
    plan.setdefault("pubmed_query", pubmed_query_from_question(question)[0])
    plan["semantic_variants"] = clean_variants(plan.get("semantic_variants"))      # wording alternatives for a search by meaning: plain phrases only
    plan.setdefault("trial_condition", plan["condition"])
    plan.setdefault("trial_intervention", ", ".join(map(str, plan["drugs"][:1])) if isinstance(plan["drugs"], list) else str(plan["drugs"]))
    plan = normalize_plan(plan)
    if plan["intent"] not in INTENTS:
        plan["intent"] = "disease_research"
    plan["understanding"] = verified_understanding(plan, question)
    proposed = plan.get("skills")
    plan["skills"] = [s for s in proposed if isinstance(s, str)] if isinstance(proposed, list) else []      # what the model proposed; rr.select_skills validates it against the catalog and the information needs
    return plan


def trim_part(text: str) -> tuple[str, bool]:
    """A writing part that stopped mid-sentence (output limit) loses its last, incomplete ENTRY (the text after the last blank line), so no heading or author line is left hanging."""
    t = text.rstrip()
    if not t or t.endswith((".", ")", "*", "|", ":", "]", "!", "?", "\"")):
        return text, False
    cut = t.rfind("\n\n")
    return (t[:cut].rstrip() if cut > 0 else trim_incomplete(t)[0]), True


def trim_incomplete(text: str) -> tuple[str, bool]:
    """If the model stopped mid-sentence (output limit), drop the last incomplete line instead of showing a half-word. Returns (text, was_cut)."""
    t = text.rstrip()
    if not t or t.endswith((".", ")", "*", "|", ":", "]", "!", "?")):
        return text, False
    cut = t.rfind("\n")
    return (t[:cut].rstrip() if cut > 0 else t), True


def normalize_plan(plan: dict) -> dict:
    """A model may return a list where text is expected (or the reverse). The rest of the pipeline relies on these types."""
    def text(v) -> str:
        if isinstance(v, dict):  # e.g. {"primary": "...", "synonyms": [...], "normalized": "..."}: take the normalised name, never the keys
            for key in ("normalized", "primary", "name", "term", "label"):
                if isinstance(v.get(key), str) and v[key].strip():
                    return v[key].strip()
            return " ".join(text(x) for x in v.values() if isinstance(x, str))
        if isinstance(v, (list, tuple)):
            return " ".join(dict.fromkeys(text(x) for x in v if text(x)))
        return "" if v is None else str(v)
    for key in ("intent", "condition", "pubmed_query", "trial_condition", "trial_intervention"):
        plan[key] = text(plan.get(key))
    secondary = plan.get("secondary_intents")
    plan["secondary_intents"] = [text(v) for v in secondary if text(v)] if isinstance(secondary, (list, tuple)) else []
    drugs = plan.get("drugs")
    plan["drugs"] = [str(d) for d in drugs] if isinstance(drugs, (list, tuple)) else ([str(drugs)] if drugs else [])
    plan["is_clinical"] = bool(plan.get("is_clinical", True))
    return plan


# ------------------------------------------------------------------ retrieval
def run_jobs(jobs: dict, timeout: float, essential: list[str] | None = None, extra_until: float = 0.6) -> tuple[dict, dict]:
    """Run retrieval skills in parallel. A skill that exceeds its timeout is marked partial; the others continue."""
    pool = ThreadPoolExecutor(max_workers=len(jobs))
    started = time.time()

    def timed(fn):
        t0 = time.time()
        try:
            return fn(), time.time() - t0
        except Exception as error:
            return json.dumps({"error": str(error)}), time.time() - t0

    futures = {name: pool.submit(timed, fn) for name, fn in jobs.items()}
    if essential:  # proceed as soon as the essential skills finish; slower ones get only a short grace period
        wait([futures[n] for n in essential if n in futures], timeout=timeout)
        wait(list(futures.values()), timeout=max(0.0, min(extra_until, timeout) - (time.time() - started)))
    else:
        wait(list(futures.values()), timeout=timeout)
    raw, secs = {}, {}
    for name, future in futures.items():
        if future.done():
            raw[name], secs[name] = future.result()[0], round(future.result()[1], 1)
        else:
            raw[name], secs[name] = json.dumps({"error": f"timed out after {timeout:.0f}s (partial)"}), timeout
    pool.shutdown(wait=False, cancel_futures=True)
    return raw, secs


def with_retry(fn, retries: int = 2, pause: float = 0.4):
    """Bounded retries (max 2) for the drug, regulatory and trial sources: a result that is an error is retried, then returned as it is."""
    def run():
        last = None
        for _ in range(retries + 1):
            last = fn()
            if '"error"' not in str(last)[:40] and not str(last).startswith("Web search failed"):
                return last
            time.sleep(pause)
        return last
    return run


def primary_with_fallback(primary, fallback, grace: float = 3.0):
    """Two searches of the same question side by side. The primary's answer is used when it is back within `grace` seconds (the same papers every time it answers in time); the fallback's papers
    are added behind it when they are ready. If the primary is slow, the fallback answers alone, so the first answer is never left without papers."""
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=2)
    f_primary, f_fallback = pool.submit(primary), pool.submit(fallback)

    def papers(future, wait):
        try:
            data = json.loads(future.result(timeout=wait))
            return data if isinstance(data, dict) and data.get("results") else None
        except Exception:
            return None
    try:
        first = papers(f_primary, grace)
        second = papers(f_fallback, 0.5 if first else max(grace, 6.0))
        if first is None and second is None:
            return f_primary.result(timeout=max(1.0, grace))      # neither had results: the primary's own answer (or its error) is what the caller sees
        merged = list((first or second)["results"])
        have = {r.get("pmid") for r in merged}
        if first and second:
            merged += [r for r in second["results"] if r.get("pmid") not in have]
        return json.dumps({**(first or second), "results": merged}, ensure_ascii=False)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def fast_jobs(question: str) -> dict:
    return {name: with_retry(fn, retries=1, pause=0.4) for name, fn in _fast_jobs(question).items()}     # one retry with backoff for EVERY source in the fast pass


def _fast_jobs(question: str) -> dict:
    query, years = pubmed_query_from_question(question)
    kw, _, _ = keywords(question)
    cond = " ".join(kw) or question
    return {
        "pubmed_fast": lambda: cached("pm", f"{query}|{years}", 900, lambda: primary_with_fallback(
            lambda: search_europepmc_quick(query, max_results=6, years_back=years, sort="relevance"), lambda: search_pubmed_quick(query, 4, years, "relevance"))),
        "pubmed_fast_q": lambda: cached("pm", f"{query}|{years}|q", 900, lambda: search_europepmc_quick(
            f"({query}) AND (PUB_TYPE:\"Systematic Review\" OR PUB_TYPE:\"Meta-Analysis\" OR PUB_TYPE:\"Randomized Controlled Trial\" OR PUB_TYPE:\"Practice Guideline\")",
            max_results=4, years_back=years + 1, sort="relevance")),
        "trials_fast": lambda: cached("ct", cond, 300, lambda: search_clinical_trials(cond, "", "", 4)),
        "trials_drugs_fast": lambda: cached("ct", f"{cond}|drug-fast", 300, lambda: search_clinical_trials(cond, "", "", 4, drug_only=True)),
        "fda_fast": lambda: cached("fda", cond, 3600, lambda: search_fda_labels(kw, None, 4)),
        "web_fast": lambda: cached("web", cond, 300, lambda: web_search(f"{cond} FDA approved treatment drugs")),
    }


def retry_jobs(question: str, plan: dict) -> dict:
    """Retries of the fast sources that FAILED, built from the planner's condition. The fast pass takes the first words of the question, which for a question that opens with the patient's age or
    sex are not a condition (a registry search then has no usable condition and fails); the planner has already named the condition. The fast functions themselves are not changed."""
    kw, _, _ = keywords(remove_age_expressions(str(plan.get("condition") or "")))
    if not kw:
        return {}
    cond = " ".join(kw)
    return {
        "trials_fast": lambda: cached("ct", cond, 300, lambda: search_clinical_trials(cond, "", "", 4)),
        "trials_drugs_fast": lambda: cached("ct", f"{cond}|drug-fast", 300, lambda: search_clinical_trials(cond, "", "", 4, drug_only=True)),
        "fda_fast": lambda: cached("fda", cond, 3600, lambda: search_fda_labels(kw, None, 4)),
        "web_fast": lambda: cached("web", cond, 300, lambda: web_search(f"{cond} FDA approved treatment drugs")),
    }


def deep_jobs(plan: dict, question: str) -> dict:
    year, cond = datetime.now().year, plan["condition"]
    built = planned_pubmed_query(plan) if plan.get("plan_status") in ("ok", "retried") else None
    pq = built[1] if built else pubmed_query_from_question(question)[0]      # the planner's subject (the condition's concepts) when the plan is usable; else the question's own words + treatment clause
    quality = "(randomized controlled trial[pt] OR meta-analysis[pt] OR systematic review[pt] OR guideline[pt])"
    negative = '("no significant"[tiab] OR failed[tiab] OR negative[tiab] OR hepatotoxicity[tiab] OR "liver injury"[tiab] OR futility[tiab] OR "safety signal"[tiab])'
    nondrug = '(behavioral[tiab] OR psychosocial[tiab] OR "early intervention"[tiab] OR "parent-mediated"[tiab] OR rehabilitation[tiab] OR exercise[tiab] OR surgery[tiab] OR diet[tiab])'
    planned_terms = keywords(remove_age_expressions(cond))[0] if plan.get("plan_status") in ("ok", "retried") else []      # a usable plan: the words of the planner's condition (the question's age and filler words are not the condition)
    fda_terms = planned_terms or keywords(remove_age_expressions(question))[0] or re.findall(r"[A-Za-z]{4,}", cond)   # else the question's own words first: never free-text planner prose
    jobs = {
        "pubmed_recent": lambda: cached("pm", f"{pq}|r", 900, lambda: search_pubmed(pq, max_results=8, years_back=2, sort="date")),
        "pubmed_quality": lambda: cached("pm", f"{pq}|q", 900, lambda: search_pubmed(f"({pq}) AND {quality}", max_results=5, years_back=5, sort="relevance")),
        "pubmed_bestmatch": lambda: cached("pm", f"{question}|bm", 900, lambda: search_pubmed(re.sub(r"[?\"]", " ", question).strip(), max_results=6, years_back=5, sort="relevance")),      # the plain question, ranked by PubMed's own Best Match: a second, non-keyword-AND retriever for the hybrid pool
        "pubmed_freefull": lambda: cached("pm", f"{pq}|ff", 900, lambda: search_pubmed(f"({pq}) AND pubmed pmc[sb] AND {quality}", max_results=5, years_back=5, sort="relevance")),      # papers with a free PubMed Central copy: they can show their figures
        "pubmed_negative": lambda: cached("pm", f"{pq}|n", 900, lambda: search_pubmed(f"({pq}) AND {negative}", max_results=4, years_back=5, sort="relevance")),
        "pubmed_drugs": lambda: cached("pm", f"{pq}|dr", 900, lambda: search_pubmed(
            f"({pq}) AND (pharmacological[tiab] OR medication[tiab] OR \"drug therapy\"[tiab] OR drug[tiab]) AND {quality}", max_results=5, years_back=5, sort="relevance")),
        "pubmed_drugs_review": lambda: cached("pm", f"{pq}|drr", 900, lambda: search_pubmed(
            f"({pq}) AND (treatment[ti] OR therapy[ti] OR management[ti]) AND (review[pt] OR guideline[pt] OR \"clinical trial\"[pt])", max_results=5, years_back=5, sort="relevance")),
        "trials_drugs": with_retry(lambda: cached("ct", f"{plan['trial_condition']}|drug-only", 300, lambda: search_clinical_trials(plan["trial_condition"], "", "", 12, drug_only=True))),
        "trials_deep": with_retry(lambda: cached("ct", f"{plan['trial_condition']}|{plan['trial_intervention']}", 300,
                                                 lambda: search_clinical_trials(plan["trial_condition"], plan["trial_intervention"], "", 10))),
        "fda_deep": with_retry(lambda: cached("fda", f"{' '.join(fda_terms)}|{','.join(plan.get('drugs') or [])}", 3600,
                                              lambda: search_fda_labels(fda_terms, plan.get("drugs") or []))),
        "web_approvals": with_retry(lambda: cached("web", f"{cond}|a", 300, lambda: web_search(f"{cond} newly approved drug FDA EMA approval {year - 1} {year}"))),
        "web_medications": with_retry(lambda: cached("web", f"{cond}|m", 300, lambda: web_search(f"medications commonly prescribed for {cond}: which drugs are used for its symptoms and related conditions", snippet_chars=1500))),
        "web_named_drugs": with_retry(lambda: cached("web", f"{cond}|nd", 300, lambda: web_search(f"{cond} treatment: the specific drugs by generic name, including anti-infective, immune-directed, anti-inflammatory and symptom-directed medications", snippet_chars=1500))),
        "web_symptom_drugs": with_retry(lambda: cached("web", f"{cond}|sd", 300, lambda: web_search(f"{cond} co-occurring conditions and associated symptoms: which medications are prescribed for each and in which patients", snippet_chars=1500))),
        "pubmed_drugs_symptoms": lambda: cached("pm", f"{pq}|dsy", 900, lambda: search_pubmed(
            f"({pq}) AND (comorbid*[tiab] OR symptom*[tiab] OR \"co-occurring\"[tiab]) AND (pharmacotherapy[tiab] OR medication*[tiab] OR drug*[tiab])", max_results=5, years_back=5, sort="relevance")),
        "web_regulatory": with_retry(lambda: cached("web", f"{cond}|reg", 300, lambda: regulator_search_many([f"FDA press announcement action treatment {cond}", f"{cond} drug approval label expansion safety communication FDA"]))),
        "web_standard": with_retry(lambda: cached("web", f"{cond}|s", 300, lambda: web_search(f"{cond} treatment guideline first-line drugs dosage"))),
    }
    gl_words = [w for w in keywords(question)[0] if w not in GENERIC] or keywords(question)[0]
    jobs["guidelines_deep"] = with_retry(lambda: cached("gl", f"{' '.join(gl_words)}|{[t['id'] for t in topics_for(question)]}", 3600,
                                                        lambda: json.dumps(find_guidelines(question, gl_words, guideline_search), ensure_ascii=False)))
    if not plan.get("drugs"):  # non-drug treatments matter when the question is not about one specific drug
        jobs["pubmed_nondrug"] = lambda: cached("pm", f"{pq}|d", 900, lambda: search_pubmed(f"({pq}) AND {nondrug} AND {quality}", max_results=4, years_back=5, sort="relevance"))
    jobs.update(plan_pubmed_jobs(plan))
    return jobs


NEED_TERMS = {"diagnosis": "diagnosis OR diagnostic OR evaluation OR assessment OR screening OR \"work-up\"", "treatment_options": "treatment OR therapy OR management OR intervention",
              "efficacy": "efficacy OR effectiveness OR outcome*", "safety": "safety OR \"adverse events\" OR \"side effects\"", "dosing": "dose OR dosage OR dosing",
              "prognosis": "prognosis OR outcome* OR \"follow-up\"", "mechanism": "mechanism OR pathophysiology OR etiology OR cause*",
              "guideline_recommendation": "guideline* OR recommendation* OR consensus OR \"practice parameter\"", "drug_interaction": "\"drug interaction*\" OR interaction*"}      # plain wording of each information need (no conditions or drugs)
INTENT_WORDS = re.compile(r"diagnos|work-?up|manage|treat|therap|intervention|guideline|evaluat|assess|screen|prognos|outcome|recommend", re.I)


def clean_variants(value, limit: int = 5) -> list[str]:
    """The planner's alternative wordings, as plain phrases: 3 to 80 characters, no search operators, no repeats."""
    out = []
    for v in value if isinstance(value, list) else []:
        text = " ".join(re.sub(r"[()\[\]\"]|\b(?:AND|OR|NOT)\b", " ", str(v)).split())
        text = " ".join(re.sub(r"[()\[\]\"]|\b(?:AND|OR|NOT)\b", " ", str(v)).split())
        if 3 <= len(text) <= 80 and text.lower() not in {x.lower() for x in out}:
            out.append(text)
    return out[:limit]


def intent_queries(state: dict, plan: dict) -> list[str]:
    """The intent-based queries of this question as plain phrases, for the ranking: one per information need (the condition and the need), the natural-language searches the skills wrote
    (routing), and the planner's alternative wordings. Each retrieved paper is compared with every one of them, not only with the question."""
    condition = " ".join(str(plan.get("condition") or "").split())
    out = [f"{condition} {need.replace('_', ' ')}".strip() for need in ((plan.get("understanding") or {}).get("info_needed") or []) if condition][:4]
    out += [str(row.get("query") or "") for row in (state.get("routing") or {}).get("operations", []) if row.get("target") in ("pubmed", "europepmc", "guideline_sites", "web") and row.get("status") != "not_run"]
    out += [f"{condition} {v}".strip() if len(v.split()) < 4 else v for v in (plan.get("semantic_variants") or [])]
    return [q for q in dict.fromkeys(" ".join(q.split()) for q in out) if len(q) > 8][:9]


def planned_pubmed_query(plan: dict) -> tuple[str, str] | None:
    """(the planner's PubMed query, its subject part) when the plan gave a well-formed query, else None. The subject part is the groups that name the condition (the groups made of intent words,
    such as work-up or management, are left out) so that one search can be written per information need."""
    query = str(plan.get("pubmed_query") or "").strip()
    if not (12 <= len(query) <= 600) or query.count("(") != query.count(")") or query.count('"') % 2:
        return None
    groups = [g.strip().strip("()") for g in re.split(r"\)\s+AND\s+\(", query)]
    subject = [g for g in groups if g and not INTENT_WORDS.search(g)]
    return (query, " AND ".join(f"({g})" for g in subject)) if subject else None


def plan_pubmed_jobs(plan: dict) -> dict:
    """Legacy-style PubMed searches built from the plan instead of the question's raw words: the planner's own query, its subject newest first, and one search per information need (the subject AND that need's wording)."""
    if plan.get("plan_status") not in ("ok", "retried"):
        return {}
    built = planned_pubmed_query(plan)
    if not built:
        return {}
    query, subject = built
    jobs = {"pubmed_plan_main": lambda: cached("pm", f"{query}|pm", 900, lambda: search_pubmed(query, max_results=12, years_back=5, sort="relevance")),
            "pubmed_plan_recent": lambda: cached("pm", f"{subject}|pr", 900, lambda: search_pubmed(subject, max_results=6, years_back=2, sort="date"))}
    if os.getenv("ENABLE_FULLTEXT_SEARCH", "true").strip().lower() not in OFF_WORDS:      # the words are also searched in the FULL TEXT of open-access articles (Europe PMC), which PubMed cannot do
        jobs["pubmed_plan_epmc_fulltext"] = lambda: cached("pm", f"{subject}|eft", 900, lambda: search_europepmc_quick(subject, max_results=10, years_back=5, sort="relevance", fulltext=True))
    variants = plan.get("semantic_variants") or []
    if variants:      # the same problem in other words (the planner's alternatives): papers that do not use the question's own words are found as well
        spoken = " OR ".join(f'"{v}"' for v in variants)
        jobs["pubmed_plan_variants"] = lambda: cached("pm", f"{spoken}|v", 900, lambda: search_pubmed(spoken, max_results=8, years_back=5, sort="relevance"))
        if "pubmed_plan_epmc_fulltext" in jobs:
            jobs["pubmed_plan_epmc_variants"] = lambda: cached("pm", f"{spoken}|evf", 900, lambda: search_europepmc_quick(spoken, max_results=8, years_back=5, sort="relevance", fulltext=True))
    needs = [n for n in ((plan.get("understanding") or {}).get("info_needed") or []) if n in NEED_TERMS][:3]
    for need in needs:
        text = f"{subject} AND ({NEED_TERMS[need]})"
        jobs[f"pubmed_plan_{need}"] = (lambda t=text: cached("pm", f"{t}|pn", 900, lambda: search_pubmed(t, max_results=8, years_back=5, sort="relevance")))
    return jobs


LEGACY_BY_NEED ={"drugs": {"pubmed_drugs", "pubmed_drugs_review", "pubmed_drugs_symptoms", "fda_deep", "web_approvals", "web_medications", "web_named_drugs", "web_symptom_drugs", "web_regulatory"},
                  "trials": {"trials_drugs", "trials_deep"}}      # the legacy jobs that belong to one kind of need; the literature, web standard-care and guideline jobs always run


def legacy_jobs_for_intent(plan: dict, jobs: dict) -> tuple[dict, dict]:
    """Keep the legacy deep jobs that the question's intent needs: the drug jobs only when the plan selects the drug skill (or says medicines are needed), the trial jobs only when it selects the clinical-trials skill.
    A plan that could not be made (fallback) or that selects nothing usable keeps every job, as before. Returns (jobs, trace)."""
    understanding = plan.get("understanding") or {}
    skills, select_trace = rr.select_skills(plan.get("skills"), understanding.get("info_needed"), skill_catalog())
    if plan.get("plan_status") == "fallback" or select_trace.get("fallback"):
        return jobs, {"skills": skills, "dropped": [], "kept_all": True}
    wanted = {"drugs": "drug-intelligence" in skills or understanding.get("medicines_needed") is True, "trials": "clinical-trials" in skills}
    dropped = sorted(name for need, names in LEGACY_BY_NEED.items() if not wanted[need] for name in names if name in jobs)
    return {n: fn for n, fn in jobs.items() if n not in dropped}, {"skills": skills, "dropped": dropped, "kept_all": False}


# ------------------------------------------------------------------ skill-specific context views (read-only: they never modify the ResearchState)
def compact_status(state: dict) -> dict:
    return {k: f"{v['retrieval_status']}/{v['evidence_status']}" for k, v in state["source_status"].items()}


def build_fast_context(state: dict, question: str) -> dict:
    """clinical-answer-writer, Phase 1: a few papers, trials and web items. No drug data (the system adds the FDA table itself)."""
    stems = core_stems(question)
    return {"question": question, "papers": [paper_view(p, 330, stems) for p in ordered_papers(state["papers"], stems, allow_untriaged=True)[:6]],
            "trials": [trial_view(t) for t in list(state["trials_by_id"].values())[:3]],
            "web": [{"title": w["title"], "url": w["url"], "snippet": w["snippet"][:220]} for w in state["web"][:3]],
            "fda_labels": [{"drug": r["generic"], "indication": (r.get("indication_excerpt") or "")[:240]} for r in state["regulatory"][:6]],      # what the FDA table that follows is built from: read it, judge it, do not contradict it
            "source_status": compact_status(state)}


def build_evidence_context(state: dict, question: str) -> dict:
    """evidence-synthesis: usable papers (not retracted, on topic) with their relation to the question, plus a few trials."""
    stems = set(state["core_stems"])
    understood = {k: v for k, v in (state.get("understanding") or {}).items() if v}
    return {"question": question, **({"how_the_question_was_understood": understood} if understood else {}),
            "papers": [paper_view(p, 800, stems) for p in ordered_papers(state["papers"], stems)[:MAX_CONTEXT_PAPERS]],
            "guidelines": [{"organisation": g.get("organisation"), "title": g.get("title"), "year": g.get("year"), "url": g.get("url"),
                           "supporting_text": g.get("text") or g.get("excerpt") or ""} for g in state.get("guidelines", []) if g.get("url")],
            "trials": [trial_view(t) for t in [t for t in state["trials_by_id"].values() if (t.get("triage") or {}).get("relevance") in ("direct", "indirect")][:4]],
            "source_status": compact_status(state)}


def build_drug_context(state: dict, question: str) -> dict:
    """drug-intelligence: web drug sources, trial interventions, short evidence snippets and the names the FDA records already cover.
    NOT the writer output, NOT the full papers."""
    stems = set(state["core_stems"])
    return {"question": question, "regulatory_records_already_listed": [r["generic"] for r in state["regulatory"]] + list(state.get("used_drugs", [])),
            "web": [{"title": w["title"], "url": w["url"], "snippet": w["snippet"][:300]} for w in state["web"][:8]],
            "trial_interventions": sorted({i for t in state["trials_by_id"].values() for i in (t.get("interventions") or []) if i})[:12],
            "papers": [{"cite": p["cite"], "title": p["title"], "text": snippet(p["abstract"], 220),
                        "relation": "indirect (different or related population)" if is_indirect(p, stems) else "direct"}
                       for p in ordered_papers(state["papers"], stems)[:MAX_CONTEXT_PAPERS]],
            "source_status": compact_status(state)}


def build_trials_context(state: dict) -> dict:
    """clinical-trials: registry records only (the trials table is built by code from these)."""
    return {"trials": [trial_view(t) for t in state["trials_by_id"].values()], "source_status": compact_status(state).get("trials_deep")}


def build_verification_context(state: dict) -> dict:
    """citation-verification runs in code against the FULL state; this is the set of identifiers it accepts."""
    return known_sources(state)


def build_qc_context(state: dict) -> dict:
    return {"source_status": state["source_status"], "drug_intelligence_status": state["drug_intelligence_status"], "repair_count": state["repair_count"]}


def evidence_lifecycle(state: dict, report_text: str, verification_status: str) -> dict:
    """Audit the paper lifecycle by stable PMID, preserving every retrieval route and separating screening from citation checks."""
    papers = state.get("papers", {})
    records = list(papers.values())
    screened = [p for p in records if p.get("triage")]
    included = [p for p in records if usable(p)]
    cited = sorted(set(re.findall(r"\bPMID\s*[:\[]?\s*(\d{6,9})", report_text)) & set(papers))
    retrieved_occurrences = sum(max(1, len(p.get("found_by") or [p.get("origin")])) for p in records)
    verified_decisions = [p for p in included if (p.get("triage") or {}).get("evidence")]
    return {
        "papers": {
            "retrieved_occurrences": retrieved_occurrences,
            "deduplicated": len(records),
            "screened": len(screened),
            "included": len(included),
            "excluded_after_screen": sum(1 for p in screened if p not in included),
            "unreviewed": len(records) - len(screened),
            "citation_checked": len(cited),
            "citation_check_status": verification_status,
            "ids": {"retrieved": sorted(p.get("source_id") or f"pmid:{p.get('pmid')}" for p in records),
                    "included": sorted(p.get("source_id") or f"pmid:{p.get('pmid')}" for p in included),
                    "cited": [f"pmid:{pmid}" for pmid in cited],
                    "triage_evidence_verified": sorted(p.get("source_id") or f"pmid:{p.get('pmid')}" for p in verified_decisions)},
            "provenance": {str(p.get("pmid")): list(p.get("found_by") or [p.get("origin")]) for p in records}
        }
    }


def interpret_fda_labels(question: str, records: list[dict]) -> dict:
    """drug-intelligence reads every retrieved FDA label that has no decision yet and decides whether its indication is relevant to the question (and whether it is the condition itself or an
    associated symptom or comorbidity). Code checks the sentence number a 'relevant' decision points at. A label with no verified decision is left undecided (and is not shown)."""
    out = {"confirmed": [], "excluded": [], "unreviewed": [], "reviewer_rejected": []}
    offered = {}
    for r in records:
        text = (r.get("indication_statement") or r.get("indication_excerpt") or "")[:900]
        if r.get("model_relevant") is None and text:
            offered[f"fda:{r['generic']}:{str(r.get('set_id') or len(offered))[:8]}"] = {"indication_text": text, "record": r, "sentences": numbered_sentences(text, 900, 12)}
    offered = dict(list(offered.items())[:12])
    for attempt in (1, 2):      # the answer varies between runs: ask once more for the labels that got no verified decision
        todo = {k: v for k, v in offered.items() if v["record"].get("model_relevant") is None}
        if not todo:
            break
        answer = ask_json(question, FDA_TASK, {"fda_labels": [{"id": k, "sentences": [f"{i}. {t}" for i, t in enumerate(v["sentences"], 1)]} for k, v in todo.items()]}, 600)
        part = verify_fda_decisions(answer.get("fda_labels") or [], todo, answer.get("question_asks"))
        out["confirmed"] += part["confirmed"]
        out["excluded"] += part["excluded"]
    out["unreviewed"] = [v["record"]["generic"] for v in offered.values() if v["record"].get("model_relevant") is None]
    return out


def triage_trials(state: dict, question: str, trials: list[dict]) -> None:
    """evidence-synthesis reads registry records (title, conditions, interventions, primary outcome) and decides relevance and kind; code checks the sentence number and stores the decision."""
    offered_t = {t["nct_id"]: {"sentences": [t["title"] or "", "Conditions: " + "; ".join(t.get("conditions") or []), "Interventions: " + "; ".join(t.get("interventions") or []),
                                             "Primary outcome: " + str(t.get("primary_endpoint") or "not stated")]} for t in trials if not t.get("triage")}
    if not offered_t:
        return
    answer = ask_json(question, TRIAL_TRIAGE_TASK, {"trials": [{"id": k, "sentences": [f"{n}. {x}" for n, x in enumerate(v["sentences"], 1)]} for k, v in offered_t.items()]}, 700, skill="evidence-synthesis")
    decisions = verify_trial_triage(answer.get("trials") or [], offered_t)
    for k in offered_t:
        state["trials_by_id"][k]["triage"] = decisions.get(k) or {"relevance": "uncertain", "kind": "other", "evidence": ""}


def fast_drug_decisions(state: dict, question: str) -> None:
    """Runs next to the fast answer (it does not delay it): the model decides which retrieved FDA labels and which late-phase registry drug trials bear on the question, so the first answer's
    top-drugs table shows only decided rows. Failures leave records undecided (and therefore not listed); they are kept in state['fast_decision_trace'] and the table says so."""
    from landscape import is_late_phase
    state["fast_decision_trace"] = []
    candidates = sorted((t for t in state["trials_by_id"].values() if t.get("drug_interventions") and is_late_phase(t)), key=lambda t: t.get("last_update") or "", reverse=True)[:6]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(interpret_fda_labels, question, state["regulatory"]), pool.submit(triage_trials, state, question, candidates)]
        for f in futures:
            try:
                state["fast_decision_trace"].append(f.result())
            except Exception as error:
                state["fast_decision_trace"].append({"error": f"{type(error).__name__}: {str(error)[:100]}"})


def triage_evidence(state: dict, question: str, skipped: list[str], only: set[str] | None = None) -> None:
    """evidence-synthesis reads ALL retrieved papers and trial records (not a keyword-filtered subset) and decides, for the question: relevance, evidence role, population.
    Code checks every decision against the record's own text (landscape.verify_*_triage) and stores the verified decision on the record (p['triage'] / t['triage']).
    A part whose call fails is left without a decision (its records are then offered to the model as they are) and is reported."""
    from landscape import numbered_sentences
    today = f"{datetime.now():%Y-%m-%d}"
    if only is None:
        papers = [p for p in state["papers"].values() if not p.get("retracted")][:TRIAGE_PAPERS]
    else:      # only the NEW records (found after the first reading): the others keep their decisions, and nothing outside this set is marked uncertain
        papers = [state["papers"][i] for i in only if i in state["papers"] and not state["papers"][i].get("retracted") and not state["papers"][i].get("triage")][:TRIAGE_PAPERS]      # retrieval order (the search groups interleave recent, high-quality, negative-result and drug papers), not date or design
    offered_p = {p["pmid"]: {"sentences": [p["title"]] + numbered_sentences(p["abstract"], 700),
                              "study_type": p.get("study_type") or p.get("type") or []} for p in papers}
    information_needs = list((state.get("understanding") or {}).get("info_needed") or [])
    clinical_profile = {"condition": str((state.get("entities") or {}).get("condition") or ""),
                        "case_facts": list((state.get("understanding") or {}).get("case_facts") or []),
                        "information_needs": information_needs}
    if only is None:
        trials = [t for t in list(state["trials_by_id"].values())[:12] if not t.get("triage")]      # the fast pass has already read some of them
    else:
        trials = [state["trials_by_id"][i] for i in only if i in state["trials_by_id"] and not state["trials_by_id"][i].get("triage")][:12]
    offered_t = {t["nct_id"]: {"sentences": [t["title"] or "", "Conditions: " + "; ".join(t.get("conditions") or []), "Interventions: " + "; ".join(t.get("interventions") or []),
                                             "Primary outcome: " + str(t.get("primary_endpoint") or "not stated")]} for t in trials}
    jobs = []
    keys = list(offered_p)
    for i in range(0, len(keys), 6):      # small parts run side by side: each call has little to read and little to write
        part = keys[i:i + 6]
        jobs.append(("papers", part, {"today": today, "clinical_profile": clinical_profile, "papers": [{"id": k, "year": state["papers"][k]["year"], "type": state["papers"][k]["type"],
                                                                   "sentences": [f"{n}. {t}" for n, t in enumerate(offered_p[k]["sentences"], 1)]} for k in part]}, PAPER_TRIAGE_TASK))
    if offered_t:
        jobs.append(("trials", list(offered_t), {"clinical_profile": clinical_profile,
                                                   "trials": [{"id": k, "sentences": [f"{n}. {t}" for n, t in enumerate(v["sentences"], 1)]} for k, v in offered_t.items()]}, TRIAL_TRIAGE_TASK))

    trace: list = []

    def run(job):
        kind, part, payload, task = job
        began = time.time()
        try:
            answer = ask_json(question, task, payload, 900, skill="evidence-synthesis")
        except Exception as error:
            trace.append({"kind": kind, "error": str(error)[:100], "seconds": round(time.time() - began, 1)})
            return kind, part, None
        trace.append({"kind": kind, "seconds": round(time.time() - began, 1), "sample": (answer.get(kind) or [])[:2], "count": len(answer.get(kind) or []), "raw_tail": answer.get("_raw", "")[-600:], "sent": len(payload.get("papers") or payload.get("trials") or [])})
        items = answer.get(kind) or []
        apply_decisions(kind, part, items)      # applied as soon as this part is back: the writer is not held up by the slowest part
        return kind, part, items

    def apply_decisions(kind, part, items):
        decisions = verify_paper_triage(items, offered_p, information_needs=information_needs,
                                        broad_question=len(information_needs) > 1) if kind == "papers" else verify_trial_triage(items, offered_t)
        for k in part:
            record = state["papers"][k] if kind == "papers" else state["trials_by_id"][k]
            record["triage"] = decisions.get(k) or ({"relevance": "uncertain", "role": None, "population": "", "evidence": ""} if kind == "papers" else {"relevance": "uncertain", "kind": "other", "evidence": ""})
    with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as pool:
        results = list(pool.map(run, jobs))
    state["triage_trace"] = trace if only is None else list(state.get("triage_trace") or []) + trace      # kept in the saved state for audit
    failed = 0
    for kind, part, items in results:
        if items is None:
            failed += 1
    if not failed and only is None:
        for p in state["papers"].values():      # records beyond the triaged set were not read: they are not offered to the writer
            if not p.get("triage") and not p.get("retracted"):
                p["triage"] = {"relevance": "uncertain", "role": None, "population": "", "evidence": ""}
    if only is not None:
        for p in papers:      # a record of this set that got no verified decision is not shown as relevant
            p.setdefault("triage", {"relevance": "uncertain", "role": None, "population": "", "evidence": ""})
        for t in trials:
            t.setdefault("triage", {"relevance": "uncertain", "kind": "other", "evidence": ""})
    if failed:
        skipped.append(f"the relevance reading of {failed} part(s) of the retrieved evidence did not complete; those records were offered to the writer as retrieved")


# ------------------------------------------------------------------ the streaming pipeline
ESCALATION_DEADLINE_S = float(os.getenv("ESCALATION_DEADLINE_S", "25"))      # HARD DEADLINE for the whole phase before the evidence writer: the relevance reading, the escalation searches and the reading of their results
ESCALATION_MIN_S = 2.0      # an escalation step is not started with less than this left of the deadline
SCREEN_THRESHOLD = float(os.getenv("SCREEN_THRESHOLD", "-8.0"))      # CALIBRATED on 285 saved web snippets: scores run from -11.5 to +8.4 (median -4.7, 25th percentile -9.3). Pages that were later USED as sources had a median of -4.3, so the
SCREEN_MARGIN = float(os.getenv("SCREEN_MARGIN", "2.0"))      # screen only rejects the clearly off-topic bottom (below -10); results from -10 to -6 are borderline and count as relevant for coverage, left to the model-based assessment
ALTERNATES_TASK = ("Each need below was searched in one source and no relevant evidence came back (the search found nothing, found nothing relevant, or failed). For each need write ONE search in a DIFFERENT, suitable "
                   "source from the list in the section, keeping the condition and the patient details of the question. A regulatory approval or drug label need must use a regulatory or label source; a literature need may "
                   "use another literature index or a targeted web search. Reply with ONE JSON object and nothing else, in the same format as the section gives, with the same `need_id` for each; write no search for a need that has no suitable second source.")


def propose_alternates(question: str, plan: dict, wanted: list, anchors: set[str], all_ops: list, ask=ask_json) -> dict:
    """ONE batched model call for the essential needs that have no second source yet. Every search is validated like any other (allowlisted target, anchored query) and must use a different source than the one tried."""
    if not wanted:
        return {}
    section = skill_section("latest-evidence", "Query transformation by intent")
    sources = section[section.find("### Sources you may target"):section.find("### Output")] if "### Sources you may target" in section else section
    payload = {"question": question, "understanding": {k: v for k, v in (plan.get("understanding") or {}).items() if v},
               "needs": [{"need_id": n.need_id, "need": n.need, "tried": [{"target": o.target, "query": o.query[:120]} for o in all_ops if o.id in n.op_ids], "outcome": n.state} for n in wanted]}
    try:
        reply = ask(question, ALTERNATES_TASK + "\n\nSECTION:\n" + sources, payload, 700, skill="latest-evidence")
    except Exception:
        return {}
    names = {n.need_id: n.need for n in wanted}
    items = reply.get("queries") if isinstance(reply, dict) and isinstance(reply.get("queries"), list) else (reply.get("alternates") if isinstance(reply, dict) and isinstance(reply.get("alternates"), list) else [])
    items = [{**i, "need": i.get("need") or names.get(i.get("need_id"), "")} for i in items if isinstance(i, dict)]
    ops, _ = rr.parse_operations("alternates", {"queries": items}, question, anchors)
    tried = {n.need_id: {o.target for o in all_ops if o.id in n.op_ids} for n in wanted}
    out = {}
    for op in ops:
        if op.need_id in tried and op.target not in tried[op.need_id] and op.need_id not in out:
            out[op.need_id] = {"target": op.target, "query": op.query, "args": op.args}
    return out


def relevance_counter(state: dict, question: str, scorer=None, *, optimistic: bool = False):
    """How many records of a routed search were judged relevant. Before the relevance reading (`optimistic`) any record counts, so only empty and failed searches escalate. Afterwards: papers and trials by their
    verified reading; web, guideline and regulator results by the local cross-encoder screen (borderline results count: they are left to the model-based assessment); FDA labels count as found (their
    decision comes with the drug stage)."""
    def count(op, result) -> int:
        if result.status != "completed_found":
            return 0
        if optimistic or op.target == "fda_label":
            return result.n
        try:
            data = json.loads(result.raw)
        except (TypeError, ValueError):
            return 0
        rows = data if isinstance(data, list) else (data.get("results") or [])
        if op.target in ("pubmed", "europepmc"):
            return sum(1 for r in rows if (state["papers"].get(r.get("pmid"), {}).get("triage") or {}).get("relevance") in ("direct", "indirect"))
        if op.target == "trials":
            return sum(1 for r in rows if (state["trials_by_id"].get(r.get("nct_id"), {}).get("triage") or {}).get("relevance") in ("direct", "indirect"))
        if op.target == "guideline_sites":      # already restricted to the authoritative organisations: a page from one is what the need asks for (a topical score would reject official pages)
            return len(rows)
        if op.target == "regulator":      # regulators' own pages (checked against their hosts): relevance to the drug is the model's decision in the drug stage, not a web-search score
            return sum(1 for r in rows if any((urlparse(r.get("url") or "").hostname or "").lower().endswith(h) for h in REGULATOR_HOSTS))
        screened = rr.screen_results(op.query or question, rows, scorer, SCREEN_THRESHOLD, SCREEN_MARGIN) if scorer else {"relevant": [], "borderline": rows}
        return len(screened["relevant"]) + len(screened["borderline"])
    return count


def ingest_routed(state: dict, raw: dict) -> set[str]:
    """Merge routed results into the state exactly as the first round was merged; returns the ids (PMIDs and NCT ids) of the records that are NEW."""
    before = set(state["papers"]) | set(state["trials_by_id"])
    state.setdefault("_search_raw", {}).update(raw)
    state["retrieval_errors"].update(ingest(state["papers"], state["trials_by_id"], state["web"], raw))
    ingest_regulatory(state, raw)
    ingest_guideline_ops(state, raw)
    record_source_status(state, raw)
    return (set(state["papers"]) | set(state["trials_by_id"])) - before


SECTION_SOURCES = {"drugs": {"targets": {"fda_label", "regulator"}, "skills": {"drug-intelligence"}}, "trials": {"targets": {"trials"}, "skills": {"clinical-trials"}},
                   "literature": {"targets": {"pubmed", "europepmc"}, "skills": {"latest-evidence"}}}


def compute_section_states(record: dict) -> dict:
    """Per report section: covered | not_applicable | not_retrieved | no_relevant_evidence | failed | timed_out | not_generated, from the needs the routed searches served. A section no search was written for
    (its skill was not chosen, or declined) is not applicable; a section whose searches failed, timed out or never ran is never reported as 'no evidence'."""
    ops, needs = record.get("all_ops") or [], record.get("needs") or {}
    out = {}
    for section, spec in SECTION_SOURCES.items():
        ids = {nid for o in ops if o.target in spec["targets"] or o.skill in spec["skills"] for nid in o.needs()}
        states = [needs[n]["state"] for n in ids if n in needs]
        out[section] = {"state": rr.need_section_state(states, applicable=bool(states)), "needs": sorted(ids)}
    return out


@traced("escalation phase (relevance assessment + one alternate source per essential need)", "chain")
def run_escalation_phase(state: dict, question: str, plan: dict, stage: dict, triage_thread, skipped: list[str], *, make_adapters=real_adapters, scorer=None, ask=ask_json,
                         deadline_s: float = ESCALATION_DEADLINE_S, clock=time.monotonic) -> dict:
    """Before the evidence is written: (1) searches of essential needs that found nothing or failed are tried once in an alternate source WHILE the relevance reading runs; (2) when the reading is done, needs
    whose records were all judged irrelevant get one alternate-source search too; (3) the new records are read (never marking the old ones uncertain). Each need escalates at most once, supporting needs
    never, and everything is bounded by one hard deadline: unfinished work is recorded and reported in the limitations, never waited for."""
    record = {"waves": [], "skipped": [], "unfinished": [], "needs": {}}
    began = clock()
    deadline = began + deadline_s

    def remaining() -> float:
        return max(0.0, deadline - clock())
    if not stage or not stage.get("execution"):
        triage_thread.join(timeout=deadline_s)      # no routed stage (it is off, or it did not finish): only papers the relevance reading has judged reach the writer, so it is waited for (bounded by the same hard deadline)
        if triage_thread.is_alive():
            skipped.append("the relevance reading did not finish before the deadline; papers it had not judged were not used as evidence")
        return record
    ops, first_results = stage["ops"], dict(stage["execution"].results)
    all_ops = list(ops)
    results = dict(first_results)
    adapters = make_adapters(stage["anchors"])
    executed = {o.key for o in ops}
    escalated: set[str] = set()
    jobs: list[dict] = []

    def launch(wave: int, new_ops: list) -> dict:
        """Run one wave's searches in a thread so that the relevance reading can go on beside it."""
        job = {"wave": wave, "ops": new_ops, "execution": None}

        def work():
            job["execution"] = rr.Execution(new_ops, adapters, max(0.05, remaining()))      # never longer than what is left of the hard deadline
        thread = threading.Thread(target=work, daemon=True)
        job["thread"] = thread
        thread.start()
        return job

    def plan_wave(needs: dict, wave: int) -> list:
        room = rr.MAX_ESCALATIONS - len(escalated)      # the cap is for the whole phase, not for each wave
        new_ops, missing, skips = rr.plan_escalations(needs, all_ops, executed, already=escalated, max_escalations=max(0, room))
        record["skipped"] += [{**s, "wave": wave} for s in skips]
        if missing and remaining() > ESCALATION_MIN_S:
            wanted = [needs[m] for m in missing]
            alternates = propose_alternates(question, plan, wanted, stage["anchors"], all_ops, ask=ask)
            for need_id, alt in alternates.items():
                st = needs[need_id]
                source = next((o for o in all_ops if o.id in st.op_ids), None)
                op = rr.Operation(id=f"escalation:{need_id}", need_id=need_id, need=st.need, target=alt["target"], query=alt["query"], args=alt["args"], priority="essential", origin="escalation",
                                  skill=source.skill if source else "", escalation={"from": source.id if source else "", "from_target": source.target if source else "", "reason": "failed" if st.state == "failed" else st.reason or "no_relevant", "round": 1})
                if op.key not in executed and len(new_ops) < room:
                    new_ops.append(op)
        new_ops = [o for o in new_ops if o.key not in executed][:max(0, room)]
        for op in new_ops:
            executed.add(op.key)
            escalated.add(op.need_id)
            all_ops.append(op)
        return new_ops

    def finish(job: dict, wait_s: float) -> set[str]:
        job["thread"].join(timeout=wait_s)
        execution = job["execution"]
        if execution is None:
            record["unfinished"].append(f"wave {job['wave']}: the alternate-source searches did not finish before the deadline and were not used")
            return set()
        results.update(execution.results)
        unfinished = [o for o in job["ops"] if execution.results[o.id].status in ("timed_out", "not_run")]
        if unfinished:
            record["unfinished"].append(f"wave {job['wave']}: {len(unfinished)} alternate-source search(es) did not finish within the deadline and were not used")
        raw = rr.raw_for_ingest(job["ops"], execution.results, index_in=all_ops)
        new_ids = ingest_routed(state, raw)
        record["waves"].append({"wave": job["wave"], "searches": [{"need_id": o.need_id, "need": o.need, "from": o.escalation, "target": o.target, "query": o.query[:160], "status": execution.results[o.id].status,
                                                                    "n": execution.results[o.id].n, "error": execution.results[o.id].error[:100]} for o in job["ops"]], "new_records": len(new_ids)})
        return new_ids

    def read_new(new_ids: set[str]) -> None:
        if not new_ids or remaining() <= 0.5:
            for i in new_ids:      # no time to read them: they are not offered as relevant
                for table in ("papers", "trials_by_id"):
                    if i in state[table]:
                        state[table][i].setdefault("triage", {"relevance": "uncertain", "role": None, "population": "", "evidence": "", "kind": "other"})
            if new_ids:
                record["unfinished"].append(f"the relevance reading of {len(new_ids)} record(s) found by the alternate-source searches did not finish before the deadline; they are not shown as relevant")
            return
        thread = threading.Thread(target=lambda: triage_evidence(state, question, skipped, only=new_ids), daemon=True)
        thread.start()
        thread.join(timeout=remaining())
        if thread.is_alive():
            record["unfinished"].append("the relevance reading of the records found by the alternate-source searches did not finish before the deadline")

    # ---- wave 1: needs whose searches found nothing or failed (no relevance reading is needed to know that); runs beside the relevance reading
    needs = rr.evaluate_needs(all_ops, results, relevance_counter(state, question, optimistic=True))
    wave1 = plan_wave(needs, 1)
    job1 = launch(1, wave1) if wave1 else None
    triage_thread.join(timeout=remaining())
    triage_done = not triage_thread.is_alive()
    if not triage_done:
        record["unfinished"].append("the relevance reading of the retrieved records did not finish before the deadline; records without a verified decision were offered as retrieved")
    new_ids = finish(job1, remaining()) if job1 else set()
    read_new(new_ids)
    # ---- wave 2: needs whose records were all judged irrelevant (needs the finished reading)
    if triage_done and remaining() > ESCALATION_MIN_S:
        scorer = scorer if scorer is not None else hybrid_search.cross_scorer()
        needs = rr.evaluate_needs(all_ops, results, relevance_counter(state, question, scorer))
        wave2 = plan_wave(needs, 2)
        if wave2:
            job2 = launch(2, wave2)
            read_new(finish(job2, remaining()))
    final = rr.evaluate_needs(all_ops, results, relevance_counter(state, question, scorer if scorer is not None else None, optimistic=not triage_done))
    record["needs"] = {k: {"need": v.need, "priority": v.priority, "state": v.state, "reason": v.reason, "relevant": v.relevant} for k, v in final.items()}
    record["seconds"] = round(clock() - began, 1)
    record["all_ops"] = all_ops
    record["results"] = results
    for note in record["unfinished"]:
        skipped.append(note)
    return record


@traced("stream_answer", "chain")
def stream_answer(question: str):
    scrubbed = phi.scrub(question)          # identifiers never leave this machine: everything below sees only the scrubbed question
    question = scrubbed.text or question
    clock, state = Clock(), new_state(question)
    state["privacy"] = {"removed": scrubbed.removed, "detector": scrubbed.detector}
    papers, trials, web = state["papers"], state["trials_by_id"], state["web"]   # the containers live INSIDE the ResearchState
    skipped: list[str] = []
    pool = ThreadPoolExecutor(max_workers=3)

    def event(kind: str, **data) -> dict:
        return {"type": kind, "t": clock.now(), **data}

    yield event("stage", stage="INITIAL")
    if scrubbed.changed:
        yield event("warning", category="privacy", message=phi.notice(scrubbed))

    # --- FAST retrieval now. The orchestrator plan (a second model call) starts only AFTER the first token of the fast answer,
    #     so the two model calls do not compete; the plan then triggers DEEP retrieval in the background.
    plan_box: dict = {}
    plan_ready, deep_done = threading.Event(), threading.Event()
    evidence_ready = threading.Event()      # set when the escalation phase is over: the drug landscape reads its evidence only then

    def plan_worker():
        try:
            plan_box["plan"] = plan_with_skill(question)
        except Exception as error:
            plan_box["error"] = str(error)[:200]
        finally:
            plan_ready.set()

    # Deep retrieval runs as FOUR independent groups, each with its own 'done' signal, so a section can be shown the moment ITS research is ready
    # (evidence -> drugs -> trials -> guidelines) instead of waiting for the slowest source of all.
    GROUPS_ORDER = ("evidence", "drugs", "trials", "guidelines")
    group_done = {g: threading.Event() for g in GROUPS_ORDER}

    def group_of(name: str) -> str:
        return "evidence" if name.startswith("pubmed") else "trials" if name.startswith("trials") else "guidelines" if name.startswith("guideline") else "drugs"

    def deep_worker():
        try:
            plan_ready.wait(timeout=3600 if CLAUDE_MODE else 60)
            if "plan" not in plan_box:
                plan_box["error"] = plan_box.get("error", "planning did not finish")
                return
            jobs = deep_jobs(plan_box["plan"], question) if legacy_deep_enabled() else {}
            if jobs:      # only the legacy jobs the question's intent needs
                jobs, legacy_trace = legacy_jobs_for_intent(plan_box["plan"], jobs)
                state["legacy_selection"] = legacy_trace
            # A retry happens only after planning finished, so use the planner's
            # normalized clinical concepts instead of repeating the provisional
            # FAST transformation.
            retry_pool = {**fast_jobs(question), **retry_jobs(question, plan_box["plan"])}      # the planner's condition replaces the question's first words for the sources that failed
            for name, st in list(state["source_status"].items()):      # a fast-pass source that still failed gets one more try here, with the longer timeout
                if st.get("retrieval_status") != "success" and name in retry_pool and name not in jobs:
                    jobs[name] = retry_pool[name]
                    plan_box.setdefault("retried", []).append(name)
            by_group: dict[str, dict] = {g: {} for g in GROUPS_ORDER}
            for name, fn in jobs.items():
                if name in EARLY_EVIDENCE_JOBS or name.startswith("pubmed_plan_"):
                    continue          # already running since the fast pass finished (they do not need the plan), or run with the evidence group once the plan is in
                by_group["drugs" if name.startswith("pubmed") else group_of(name)][name] = fn      # a retried fast PubMed job is merged with the drug group

            def run_group(g: str):
                try:
                    if not by_group[g]:
                        plan_box["raw_" + g] = {}
                    elif g == "evidence":      # show evidence as soon as the two main PubMed searches are in; the others get a short grace period
                        plan_box["raw_" + g], _ = run_jobs(by_group[g], DEEP_TIMEOUT_S, essential=[n for n in ("pubmed_recent", "pubmed_quality") if n in by_group[g]], extra_until=8.0)
                    elif g == "drugs" and "fda_deep" in by_group[g]:      # the FDA labels are essential; the web searches get a grace period (their results are used if they arrive)
                        plan_box["raw_" + g], _ = run_jobs(by_group[g], DEEP_TIMEOUT_S, essential=["fda_deep"], extra_until=12.0)
                    else:
                        plan_box["raw_" + g], _ = run_jobs(by_group[g], DEEP_TIMEOUT_S)
                except Exception as error:
                    plan_box["error_" + g] = f"{g} retrieval crashed: {type(error).__name__}: {str(error)[:150]}"
                finally:
                    group_done[g].set()
            for g in GROUPS_ORDER:
                if g != "evidence":      # the evidence group was started earlier by start_early_evidence()
                    threading.Thread(target=run_group, args=(g,), daemon=True).start()
        except Exception as error:  # a crash here must be RECORDED as a failed deep retrieval, never swallowed
            plan_box["error"] = f"deep retrieval crashed: {type(error).__name__}: {str(error)[:150]}"
        finally:
            if "plan" not in plan_box or plan_box.get("error"):
                for ev_ in group_done.values():
                    ev_.set()
            deep_done.set()

    EARLY_EVIDENCE_JOBS = {"pubmed_recent", "pubmed_quality", "pubmed_negative", "pubmed_drugs", "pubmed_nondrug", "pubmed_bestmatch"}

    def start_early_evidence() -> None:
        """The deep PubMed searches are built from the question's own words, not from the planner, so they start as soon as the fast search is
        done and run while the first answer is being written; the evidence section can then start right after the first answer."""
        provisional = {"drugs": [], "condition": " ".join(keywords(question)[0]) or question, "trial_condition": "", "trial_intervention": "", "intent": "comprehensive_research"}
        early = {n: fn for n, fn in deep_jobs(provisional, question).items() if n in EARLY_EVIDENCE_JOBS} if legacy_deep_enabled() else {}

        routing_box: dict = {}
        started = time.time()

        def run_routing():
            """Once the quick plan call has chosen the skills, each loads its SKILL.md and writes its searches (in parallel, possibly none); they are routed to their sources and run beside the main searches."""
            try:
                if not routing_enabled():
                    routing_box["trace"] = {"skipped": "deep routing is off"}
                elif not (plan_ready.wait(timeout=EXPANSION_TOTAL_S) and plan_box.get("plan")):
                    routing_box["trace"] = {"skipped": "the plan was not ready in time"}
                else:
                    routing_box["t_plan_ready"] = round(time.time() - started, 1)
                    routing_box.update(run_routing_stage(question, plan_box["plan"], box=routing_box))
                    routing_box["t_executed"] = round(time.time() - started, 1)
            except Exception as error:
                routing_box["trace"] = {"error": f"{type(error).__name__}: {str(error)[:80]}"}
        routing_thread = threading.Thread(target=run_routing, daemon=True)
        routing_thread.start()

        def run():
            try:
                jobs_now = early
                if legacy_deep_enabled() and plan_ready.wait(timeout=max(0.0, EXPANSION_TOTAL_S - (time.time() - started))) and plan_box.get("plan", {}).get("plan_status") in ("ok", "retried"):
                    kept, _ = legacy_jobs_for_intent(plan_box["plan"], deep_jobs(plan_box["plan"], question))      # a usable plan: the evidence searches are built from the plan (its condition and one search per information need), not from the question's raw words
                    jobs_now = {n: fn for n, fn in kept.items() if n in EARLY_EVIDENCE_JOBS or n.startswith("pubmed_plan_")}
                plan_box["raw_evidence"] = run_jobs(jobs_now, DEEP_TIMEOUT_S, essential=[n for n in ("pubmed_recent", "pubmed_quality") if n in jobs_now], extra_until=8.0)[0] if jobs_now else {}
                routing_thread.join(timeout=max(0.0, EXPANSION_TOTAL_S - (time.time() - started)))      # the routed searches are waited for a bounded time
                if routing_thread.is_alive() and routing_box.get("execution") is not None:      # the deadline ended while the stage was still going: the searches already started are used (their finished results), the rest is recorded as unfinished
                    execution = routing_box["execution"]
                    execution.budget_s = 0.0
                    execution.finish()
                    submitted = list(execution.ops)
                    routing_box.update(ops=submitted, skills=[], rejected=[], raw=rr.raw_for_ingest(submitted, execution.results),
                                       trace={"partial": "the deadline ended while the searches were still being written; the searches already started were used"})
                plan_box["raw_evidence"].update(routing_box.get("raw") or {})
                routing_box["unfinished"] = routing_thread.is_alive()      # the deadline passed before the routed searches were in: recorded, and reported in the limitations
                routing_box["waited_s"] = round(time.time() - started, 1)
                plan_box["routing"] = dict(routing_box)      # a snapshot: whatever finishes later is not silently merged into a finished state
            except Exception as error:
                plan_box["error_evidence"] = f"evidence retrieval crashed: {type(error).__name__}: {str(error)[:150]}"
            finally:
                group_done["evidence"].set()
        threading.Thread(target=run, daemon=True).start()

    threading.Thread(target=deep_worker, daemon=True).start()
    raw_fast, _ = run_jobs(fast_jobs(question), FAST_TIMEOUT_S, essential=["pubmed_fast", "trials_fast", "fda_fast"], extra_until=2.5)
    state.setdefault("_search_raw", {}).update(raw_fast)
    state["retrieval_errors"].update(ingest(papers, trials, web, raw_fast))
    ingest_regulatory(state, raw_fast)
    record_source_status(state, raw_fast)
    state["core_stems"] = sorted(core_stems(question))
    state["patient_age_years"] = patient_age_years(question)      # a label's age range is compared with it (landscape.age_note)
    start_early_evidence()
    state["fast_evidence"] = list(papers.values())
    clock.mark("retrieval_fast_s")
    if not papers and not trials and not state["regulatory"] and not state["retrieval_errors"]:  # nothing medical was found: do not invent an answer
        note = "I answer clinical research questions using PubMed, ClinicalTrials.gov and the web. I found no medical literature for this question - please rephrase it as a clinical question."
        yield event("token", stage="INITIAL", text=note)
        yield event("final", markdown=note, timings=dict(clock.marks), state=state, skipped=[])
        pool.shutdown(wait=False, cancel_futures=True)
        return
    state["searched_on"] = f"{datetime.now():%Y-%m-%d}"
    header = f"**Evidence searched on {state['searched_on']}** (PubMed, ClinicalTrials.gov, FDA drug labels, web)\n\n"
    summary_head = "## Clinical summary\n\n"
    yield event("token", stage="INITIAL", text=header + summary_head)

    # --- INITIAL: the fast answer, streamed (clinical-answer-writer sees only its own view)
    fast_prompt = (f"TASK: {FAST_TASK}\n\nQUESTION: {question}\nRETRIEVED DATA (JSON):\n"
                   f"{json.dumps(build_fast_context(state, question), ensure_ascii=False, separators=(',', ':'))}")
    decisions = threading.Thread(target=fast_drug_decisions, args=(state, question), daemon=True)      # runs while the fast answer is written
    decisions.start()
    fast_text = header  # the date line is part of the saved answer, not only of the stream
    try:
        for chunk in stream_llm(load_skill_text("clinical-answer-writer"), fast_prompt, FAST_TOKENS, purpose="writer"):
            if "first_token_s" not in clock.marks:
                clock.mark("first_token_s")
                threading.Thread(target=plan_worker, daemon=True).start()
            fast_text += chunk
            yield event("token", stage="INITIAL", text=chunk)
    except Exception as error:
        yield event("error", message=f"The fast answer could not be written: {error}")
        pool.shutdown(wait=False, cancel_futures=True)
        return
    model_text = fast_text[len(header):]                       # the model-written Phase-1 text (Bottom line / Most important recent finding / Immediate context)
    # the FDA table (built by code from the FDA label records) follows the three model-written parts, as in the original layout
    clock.mark("first_useful_s")
    state["timestamps"]["first_response"] = clock.now()
    yield event("milestone", name="first_useful_answer", seconds=clock.marks["first_useful_s"], first_token=clock.marks.get("first_token_s"))
    decisions.join(timeout=12.0)      # the table needs the model's decisions on the labels and trials (the deep search is not done before about 10 s, so waiting costs the report nothing)
    plan_ready.wait(timeout=max(0.0, 8.0 - clock.now()))      # the plan says whether the question asks for medicines at all; a plan that is not ready keeps the earlier behaviour
    if plan_box.get("plan"):
        state["medicines_needed"] = ((plan_box["plan"].get("understanding") or {}).get("medicines_needed"))
    yield event("token", stage="INITIAL", text="\n\n" + top_drugs_markdown(state))
    clock.mark("top_drugs_s")

    # --- DEEP retrieval: each group is absorbed into the ResearchState the moment IT is done (the report grows section by section)
    plan_box_view = plan_box

    def absorb(group: str) -> None:
        """Wait (within the 100 s budget) for one retrieval group, then merge its results into the ResearchState."""
        finished = group_done[group].wait(timeout=max(3.0, DEEP_BUDGET_S - clock.now()))
        if group != "evidence" and not deep_done.is_set():      # the evidence group does not depend on the plan
            deep_done.wait(timeout=max(1.0, DEEP_BUDGET_S - clock.now()))
        plan = plan_box_view.get("plan")
        if group != "evidence" and plan_box_view.get("error") and not state["source_status"].get("deep_retrieval"):
            state["retrieval_errors"]["deep_retrieval"] = plan_box_view["error"]
            state["source_status"]["deep_retrieval"] = {"retrieval_status": "failed", "evidence_status": "unavailable", "error": plan_box_view["error"]}
        if not finished:
            state["source_status"][f"deep_{group}"] = {"retrieval_status": "failed", "evidence_status": "unavailable",
                                                       "error": f"{group} retrieval did not finish within the {DEEP_BUDGET_S:.0f} s budget"}
            return
        if plan_box_view.get("error_" + group):
            state["source_status"][f"deep_{group}"] = {"retrieval_status": "failed", "evidence_status": "unavailable", "error": plan_box_view["error_" + group]}
        raw = plan_box_view.get("raw_" + group, {})
        if not plan and group != "evidence":
            return
        if plan:
            state.update(intent=plan["intent"], entities=plan)
            state["understanding"] = plan.get("understanding") or {}
            state["medicines_needed"] = (plan.get("understanding") or {}).get("medicines_needed")
            if plan.get("plan_status") == "fallback":
                skipped.append("the question could not be analysed (the planning call failed twice); the searches use the question's own words, so the intent-based search planning is reduced")
            stage = plan_box.get("routing") or {}
            state["routing"] = {"question": question, "intent": plan.get("intent"), "understanding": plan.get("understanding") or {},
                                "planner": {"status": plan.get("plan_status"), "condition": plan.get("condition"), "secondary_intents": plan.get("secondary_intents") or [], "skills": plan.get("skills") or []},
                                "timing": {k: stage.get(k) for k in ("t_plan_ready", "t_executed", "waited_s") if stage.get(k) is not None}, "unfinished": bool(stage.get("unfinished")),
                                **(rr.routing_record(stage["ops"], stage["execution"].results, rejected=stage.get("rejected"), trace=stage.get("trace"),
                                                     late=stage["execution"].late_results()) if stage.get("execution") else {"trace": stage.get("trace") or {}})}
        if group == "evidence" and routing_enabled():
            failure = routing_outcome(plan_box_view.get("routing"))
            if failure:      # the legacy results still make the whole report; the failure is a source status and a line in the limitations
                state["source_status"]["deep_routing"] = failure
                state["retrieval_errors"]["deep_routing"] = failure["error"]
        state.setdefault("_search_raw", {}).update(raw)
        state["retrieval_errors"].update(ingest(papers, trials, web, raw))
        ingest_regulatory(state, raw)      # FDA label records from any group (the routed fda_label searches run in the evidence group)
        ingest_guideline_ops(state, raw)      # guideline pages from the routed guideline_sites searches
        record_source_status(state, raw)
        if group == "guidelines":
            ingest_guidelines(state, raw)          # after the status: a guideline search that failed for SOME topics is recorded as partial
        state["core_stems"] = sorted(core_stems(question, plan))

    # the saved answer re-builds the Phase-1 drug block from the FINAL regulatory records, so a fast-pass timeout that the deep pass resolved is not left in the text
    # no banner at the top of the report: failed or partial sources are named in the 'Evidence limitations' section at the end
    opening = model_text
    sections: list[str] = []          # filled in the order the sections are STREAMED, so the final report is the same document that was shown growing


    guidelines_ready = threading.Event()

    def prepare_landscape() -> str:
        """Merge the drug / trial / guideline retrieval, name the used-in-practice drugs, look up labels, condense doses and build the landscape. No streaming here:
        it runs in a background thread while the model finishes the held-back sections."""
        marks = {"thread_started": round(clock.now(), 1)}      # where the landscape time goes (kept in the saved state)
        if state.get("medicines_needed") is False:      # the question does not ask about medicines: no label reading, no dose lookups
            absorb("guidelines")
            guidelines_ready.set()
            absorb("drugs")
            absorb("trials")
            state["trial_status"] = "complete" if family_status(state, "trials").get("retrieval_status") == "success" else "partial"
            state["trials"] = list(trials.values())
            state["landscape_marks"] = {**marks, "skipped": "medicines not needed"}
            sections[0] = header + summary_head + opening + "\n\n" + top_drugs_markdown(state)      # the opening is kept; its drug table says that medicines were not asked about
            return landscape_markdown(state, [], [])
        absorb("guidelines")
        guidelines_ready.set()
        marks["guidelines_in"] = round(clock.now(), 1)
        absorb("drugs")
        marks["drugs_in"] = round(clock.now(), 1)
        absorb("trials")
        marks["data_in"] = round(clock.now(), 1)
        state["trial_status"] = "complete" if family_status(state, "trials").get("retrieval_status") == "success" else "partial"
        state["trials"] = list(trials.values())
        label_pool = ThreadPoolExecutor(max_workers=14)      # label lookups must not queue behind the slower dose-summary model calls
        fda_recs: list = []
        fda_label_futures: list = []
        dose_futures: list = []

        def condense(rec: dict):      # the label's dosing section in 3-4 sentences; kept only if every number is in the label text
            prompt = (f"TASK: {DOSE_TASK}\n\nDRUG: {rec['generic']}\nUSE: {question}\nLABEL DOSING TEXT:\n{rec['dose_statement']}")
            try:
                return rec, "".join(stream_llm(load_skill_text("drug-intelligence"), prompt, 300)).replace("**", "").strip()      # no markdown emphasis in a table cell
            except Exception:
                return rec, ""

        def start_fda_work():
            """The label link (with its boxed warning) and the dose summary for every label now judged relevant, each started once."""
            for rec in state["regulatory"]:
                if fda_relevant(rec) and not any(rec is x for x in fda_recs):
                    fda_recs.append(rec)
                    fda_label_futures.append(label_pool.submit(fda_label_link, rec["generic"]))
                    if len(rec.get("dose_statement") or "") > 350:
                        dose_futures.append(label_pool.submit(condense, rec))
        start_fda_work()      # the labels the fast pass already judged relevant

        # drug-intelligence INTERPRETS the retrieved evidence (which drugs the sources describe as used, how relevant, which category, which regulator pages are approvals or safety items);
        # code then VERIFIES every answer against the source text (landscape.verify_*). Two interpretation calls run in parallel, in this background thread: the first answer is never waiting for them.
        offered: dict = {}      # the sources the model may cite: id -> text, label, address, kind
        for p in sorted((p for p in list(papers.values()) if not p.get("retracted")), key=lambda p: not str(p.get("origin", "")).startswith("pubmed_drugs"))[:24]:      # every retrieved paper, drug-treatment searches first; each source is judged by the model
            if not p.get("retracted") and not p.get("protocol") and p.get("abstract"):
                offered[p["pmid"]] = {"title": p["title"][:160], "text": snippet(p["abstract"], 900), "label": f"PMID {p['pmid']} ({p['cite'].split(' [')[0]})", "url": "", "kind": "paper"}
        for w in state["web"][:22]:
            if w.get("url") and source_family(w.get("source") or "") != "regulator" and w.get("snippet"):      # regulator pages are classified by their own call below
                offered[w["url"]] = {"title": (w.get("title") or "web page")[:160], "text": w["snippet"], "label": f"[{(w.get('title') or 'web page')[:70]}]({w['url']})", "url": w["url"], "kind": "web"}
        for g in state.get("guidelines", []):
            offered[g["url"]] = {"title": g["title"][:160], "text": g.get("text") or g.get("excerpt") or "", "label": f"[{g['organisation']}: {g['title'][:60]}]({g['url']})", "url": g["url"], "kind": "guideline"}
        offered_reg: dict = {}
        for w in state["web"]:
            page = (w.get("url") or "").split("#", 1)[0].split("?", 1)[0].rstrip("/").lower()
            host = (urlparse(w.get("url") or "").hostname or "").lower()
            if source_family(w.get("source") or "") == "regulator" and w.get("snippet") and any(host == h or host.endswith("." + h) for h in REGULATOR_HOSTS) \
                    and not any(v["url"].split("#", 1)[0].split("?", 1)[0].rstrip("/").lower() == page for v in offered_reg.values()) and len(offered_reg) < 8:
                offered_reg[f"reg:{len(offered_reg) + 1}"] = {"title": re.sub(r"\s*/\s*FDA\s*$", "", re.sub(r"\s+", " ", w.get("title") or w["url"]).strip()), "url": w["url"], "text": w["snippet"][:1200]}
        for v in offered.values():      # the model is shown each source as numbered sentences and answers with a number: the quote shown is always the source's own sentence
            v["sentences"] = numbered_sentences(v["text"], complete_only=v["kind"] != "paper")      # a snippet's cut-off end is not a sentence
        for v in offered_reg.values():
            v["text"] = v["text"][:1200]
            head = v["title"][:40].lower()
            v["sentences"] = [x for x in numbered_sentences(v["text"], 1200, complete_only=True) if not x.lower().startswith(head)]      # the page title repeated at the top of the text is not a quotable sentence
        condition = (state.get("entities") or {}).get("condition") or question

        def interpret_part(keys: list) -> dict:
            """One interpretation call over some of the sources. Raises on a model failure; the caller decides what a failure means."""
            payload = {"condition": condition, "fda_labelled": [r["generic"] for r in state["regulatory"] if fda_relevant(r)],
                       "sources": [{"id": k, "kind": offered[k]["kind"], "title": offered[k]["title"], "sentences": [f"{i}. {t}" for i, t in enumerate(offered[k]["sentences"], 1)]}
                                   for k in keys if offered[k]["sentences"]]}
            return ask_json(question, DRUG_INTERPRET_TASK, payload, 700)

        def interpret_drugs() -> dict:
            result = {"rows": [], "rejected": [], "items": [], "error": None}
            keys = list(offered)
            parts = [keys[i:i + 4] for i in range(0, len(keys), 4)]      # a few sources per call: each source is judged on its own, so one slip cannot hide the rest

            def safe(part: list) -> dict:
                try:
                    return interpret_part(part)
                except Exception as error:
                    return {"_error": f"{type(error).__name__}: {str(error)[:80]}"}
            for attempt in (1, 2):      # the second ask (only when fewer than 3 rows passed the source check) adds what the first one missed; the check itself never changes
                with ThreadPoolExecutor(max_workers=8) as part_pool:
                    answers = list(part_pool.map(safe, parts))
                if answers and all("_error" in a for a in answers):
                    result["error"] = answers[0]["_error"]
                    break
                result["items"] += [d for a in answers for d in (a.get("drugs") or [])]
                result["rows"], result["rejected"] = verify_drug_decisions(result["items"], offered, state, limit=14)
                if result["rows"] or not offered or clock.now() > 24.0:      # a second ask only when nothing passed the source check, and only while it still costs little
                    break
            return result

        def interpret_regulator() -> dict:
            result = {"items": [], "rejected": [], "error": None}
            if not offered_reg:
                return result
            try:
                answer = ask_json(question, REGULATOR_TASK, {"condition": condition, "items": [{"id": k, "title": v["title"], "sentences": [f"{i}. {t}" for i, t in enumerate(v["sentences"], 1)]} for k, v in offered_reg.items()]}, 700)
                result["raw"] = answer.get("items") or []
                result["items"], result["rejected"] = verify_regulator_decisions(result["raw"], offered_reg)
                undated = [x for x in result["items"] if x["date"] == "date not stated"]
                if undated:      # the date the page itself states (read from the page, never guessed)
                    with ThreadPoolExecutor(max_workers=6) as date_pool:
                        for x, found in zip(undated, date_pool.map(lambda x: page_date(x["url"]), undated)):
                            if found:
                                x["date"] = found
            except Exception as error:
                result["error"] = f"{type(error).__name__}: {str(error)[:80]}"
            return result

        def interpret_fda() -> dict:
            try:
                return interpret_fda_labels(question, state["regulatory"])
            except Exception as error:
                return {"confirmed": [], "excluded": [], "unreviewed": [r["generic"] for r in state["regulatory"] if r.get("model_relevant") is None], "reviewer_rejected": [],
                        "error": f"{type(error).__name__}: {str(error)[:80]}"}

        with ThreadPoolExecutor(max_workers=3) as interpret_pool:      # the drug call (itself two calls), the FDA-label call and the regulator call run together
            f_drugs, f_reg, f_fda = interpret_pool.submit(interpret_drugs), interpret_pool.submit(interpret_regulator), interpret_pool.submit(interpret_fda)
            drug_result, reg_result, fda_result = f_drugs.result(), f_reg.result(), f_fda.result()
        marks["interpreted"] = round(clock.now(), 1)
        start_fda_work()      # labels newly judged relevant by the deep pass
        sections[0] = header + summary_head + opening + "\n\n" + top_drugs_markdown(state)      # same order as streamed; the FDA table now reflects every decided label
        fda_names = [r["generic"] for r in fda_recs]
        if drug_result["error"]:
            skipped.append(f"drugs used in practice: the drug interpretation did not complete ({drug_result['error']}); none are listed")
        if reg_result["error"]:
            skipped.append(f"regulator announcements: the interpretation did not complete ({reg_result['error']}); none are listed")
        if fda_result.get("error"):
            skipped.append(f"FDA labels: the interpretation did not complete ({fda_result['error']}); labels without a decision are not listed")
        if fda_result["unreviewed"]:
            skipped.append(f"{len(fda_result['unreviewed'])} retrieved FDA label(s) got no verified relevance decision from the interpretation step and are not listed")
        used_rows, regulator_items = drug_result["rows"], reg_result["items"]
        try:      # one more search per listed drug: its dose and route for this condition; the model picks a sentence, the code keeps it only if it names the drug and states an amount, a route or a frequency
            lookup_doses(used_rows, condition, question, stems=[x for x in (state.get("core_stems") or []) if len(x) >= 4])
        except Exception as error:
            skipped.append(f"dose lookup for the listed drugs did not complete ({type(error).__name__}); doses are shown only where a source gave one")
        known_urls = {w.get("url") for w in state["web"]}
        for row in used_rows:      # a page that gave a dose is a retrieved source from now on: its numbers and its address pass the verification
            if row.get("dose_url") and row["dose_url"] not in known_urls:
                state["web"].append({"title": "dose page for " + row["drug"], "url": row["dose_url"], "snippet": row["dose"], "source": "web_dose"})
                known_urls.add(row["dose_url"])
        marks["doses"] = round(clock.now(), 1)

        trial_rows: list = []      # trials are shown in the Clinical trials section (registry facts); they are not repeated as drug rows
        state['trial_drugs'] = []
        label_jobs = {r['drug']: label_pool.submit(fda_label_link, r['drug']) for r in used_rows}      # each listed drug's OWN DailyMed page and boxed warning: looked up while the reviewer works
        marks["reviewed"] = round(clock.now(), 1)
        state["regulator_items"] = regulator_items
        state["used_drugs_trace"] = {"model_items": drug_result["items"], "sources_offered": len(offered), "accepted": len(used_rows), "rejected": drug_result["rejected"],
                                     "fda_decisions": fda_result, "model_regulator_items": reg_result.get("raw"), "regulator_rejected": reg_result["rejected"]}      # kept in the saved state for audit
        state["used_drugs"] = [r["drug"] for r in used_rows]
        listed = used_rows + trial_rows
        from concurrent.futures import wait
        wait(list(label_jobs.values()) + fda_label_futures + dose_futures, timeout=4.0)      # the label links and dose summaries get a few seconds at most: the answer is never held up by a slow lookup

        def finished(future):
            try:
                return future.result(timeout=0) if future.done() else None
            except Exception:
                return None
        listed_labels = [finished(label_jobs[r['drug']]) for r in listed]      # a drug whose label lookup did not finish is listed as plain text
        fda_labels = [finished(f) for f in fda_label_futures]
        for row, found in zip(listed, listed_labels):
            row['fda_label'] = (found or {}).get('url')
            generic = ((found or {}).get('generic_name') or "").lower()
            if generic and generic != row['drug'].lower():
                row['drug'] = f"{row['drug']} ({generic})"      # a brand name, with the generic name the FDA label gives for it
        shown_fda = [(n, f) for n, f, rec in zip(fda_names, fda_labels, fda_recs) if fda_relevant(rec)]      # a label the interpretation excluded is not listed
        safety_labels = [{'drug': n, 'url': f['url'], 'boxed_warning': f.get('boxed_warning')} for n, f in list(zip([r['drug'] for r in listed], listed_labels)) + shown_fda if f]
        state['drug_labels'] = [{'drug': x['drug'], 'url': x['url'], 'boxed_warning': x.get('boxed_warning')} for x in safety_labels]      # retrieved addresses: the verifier must not strip them
        state["listed_drugs"] = [r['drug'] for r in listed] + [rec['generic'] for rec in fda_recs if fda_relevant(rec)]      # every drug of the landscape: the safety section accounts for each one
        marks["labels"] = round(clock.now(), 1)
        for future in dose_futures:
            rec, text = finished(future) or (None, "")
            if rec is None:      # the summary did not finish in time: the label's own first sentences are shown instead
                continue
            if dose_summary_ok(text, rec["dose_statement"]):
                rec["dose_summary"] = text
            else:
                rec["dose_summary"] = dose_fallback(rec["dose_statement"])      # never an unchecked number: complete sentences copied from the label
                skipped.append(f"dose summary for {rec['generic']} was not faithful to the label text; the label's own first sentences are shown")
        for rec in fda_recs:
            if rec.get("dose_statement") and not rec.get("dose_summary"):
                rec["dose_summary"] = dose_fallback(rec["dose_statement"])
        marks["dose"] = round(clock.now(), 1)
        state["landscape_marks"] = marks
        landscape_md = landscape_markdown(state, used_rows, trial_rows)      # ONE categorized table + regulator announcements + boxed warnings
        return landscape_md

    landscape_box: dict = {}

    def run_prepare():
        try:
            evidence_ready.wait(timeout=ESCALATION_DEADLINE_S + 10)      # the alternate-source results (labels, regulator pages) are part of its evidence
            landscape_box["md"] = prepare_landscape()
        except Exception as error:
            landscape_box["error"] = f"{type(error).__name__}: {str(error)[:120]}"
        finally:
            guidelines_ready.set()
    landscape_thread = threading.Thread(target=run_prepare, daemon=True)
    landscape_started: list = []

    def start_landscape():
        if not landscape_started:
            landscape_started.append(True)
            landscape_thread.start()

    # 2. EVIDENCE_ENRICHING: starts the moment the PubMed deep search is in (it does NOT wait for web, FDA, trials or guidelines)
    state["answer_status"] = "ENRICHING"
    yield event("stage", stage="EVIDENCE_ENRICHING")
    absorb("evidence")
    try:
        hybrid_rerank(question, papers, state.setdefault("hybrid_trace", {}), queries=intent_queries(state, plan_box.get("plan") or {}))      # hybrid search on EVERY retrieved paper: semantic + lexical + the sources' own order, fused (hybrid_search.py); the triage reads the best TRIAGE_PAPERS first
        cross_rerank(question, papers, state["hybrid_trace"], top=TRIAGE_PAPERS)      # second stage: a cross-encoder reads the question and each of the first papers together
    except Exception as error:
        state["hybrid_trace"] = {"error": f"{type(error).__name__}: {str(error)[:80]}"}
    state["deep_evidence"] = list(papers.values())
    fulltext_box: dict = {}

    def run_fulltext():
        try:      # beside the triage and the escalation: the free full text of the best-ranked open-access papers (methods, results, conclusion) for the writer; others stay abstract-only
            best = [p for p in list(papers.values())[:TRIAGE_PAPERS] if p.get("pmcid") and not p.get("retracted")][:FULL_TEXT_PAPERS]
            if fulltext.enabled():
                fulltext_box["record"] = fulltext.attach(best, _article, cap_s=FULL_TEXT_WAIT_S)
                state["full_text"] = fulltext_box["record"]
            if chunking.enabled():      # then the passages of the best-ranked papers are scored against the question (local cross-encoder), beside the relevance reading and the escalation
                scorer = hybrid_search.cross_scorer(wait=2.0)
                state["passage_ranking"] = chunking.select(question, list(papers.values())[:MAX_CONTEXT_PAPERS], scorer) if scorer else {"skipped": "the cross-encoder was not ready"}
        except Exception as error:
            fulltext_box["error"] = f"{type(error).__name__}: {str(error)[:80]}"
    fulltext_thread = threading.Thread(target=run_fulltext, daemon=True)
    if fulltext.enabled() or chunking.enabled():
        fulltext_thread.start()
    clock.mark("retrieval_deep_s")
    yield event("milestone", name="retrieval_complete", seconds=clock.marks["retrieval_deep_s"])
    def run_triage():
        triage_evidence(state, question, skipped)      # evidence-synthesis decides which retrieved papers and trials bear on the question; the prose waits for it, the drug landscape does not
        clock.mark("triage_s")
    triage_thread = threading.Thread(target=run_triage, daemon=True)
    triage_thread.start()
    stems = set(state["core_stems"])
    sections.append("")      # slot 0: the opening + top drugs, filled in below once the FDA results are in
    parts: dict = {}      # head, landscape, trials, tail: put into the report in the document's order at the end
    landscape_emitted: list = []

    def emit_landscape_and_trials():
        """The drug landscape and the trials are shown as soon as they are ready, even while the model is still writing the held-back sections."""
        if landscape_emitted or not landscape_started or landscape_thread.is_alive():
            return
        landscape_emitted.append(True)
        yield event("stage", stage="DRUGS_ENRICHING")
        if "md" not in landscape_box:      # the preparation failed or timed out: show what is known, never nothing
            skipped.append("drug landscape: " + str(landscape_box.get("error", "preparation did not finish in time")))
            landscape_box["md"] = landscape_markdown(state, [], [])
        landscape_md = landscape_box["md"]
        yield event("token", stage="DRUGS_ENRICHING", text="\n\n" + landscape_md + "\n\n")
        state["drug_intelligence_status"] = drug_intelligence_status(state, not any(x.startswith("drugs used in practice") for x in skipped))
        state["drugs"] = landscape_md
        parts["landscape"] = landscape_md
        clock.mark("drugs_done_s")
        yield event("stage", stage="TRIALS_ENRICHING")      # clinical-trials: registry records, no model
        parts["trials"] = trials_markdown(state)
        yield event("token", stage="TRIALS_ENRICHING", text="\n\n" + parts["trials"])
        clock.mark("trials_done_s")

    start_landscape()      # the landscape reads its own evidence (every retrieved source, each judged by the model): it does not wait for the triage
    yield event("token", stage="EVIDENCE_ENRICHING", text="\n\n")
    figure_box: dict = {"figures": []}

    def run_figures() -> None:
        """BESIDE the writing: reads the figures of the retrieved open-access papers that were judged relevant, waits for the evidence the report has written, and runs the whole selection
        (find_figures), so that the figures are ready about when the report is. A failure means no figures."""
        trace = figure_box.setdefault("trace", {})
        try:
            triage_thread.join(timeout=25)      # the relevance judgement of every paper
            order = {"direct": 0, "indirect": 1}
            ranked = sorted((p for p in papers.values() if p.get("pmcid") and not p.get("retracted") and (p.get("triage") or {}).get("relevance") in order),
                            key=lambda p: order.get((p.get("triage") or {}).get("relevance"), 2))      # every retrieved paper the relevance reading judged direct or indirect, direct first
            views = [{"pmid": p["pmid"], "pmcid": p["pmcid"], "doi": p.get("doi"), "title": p.get("title"), "cite": p.get("cite"),
                      "design": ", ".join(p.get("type") or p.get("study_type") or []) or None} for p in ranked[:24]]
            fetched = json.loads(search_paper_figures(json.dumps(views), want=16, batches=3))
            trace.setdefault('reading', {})['page_status'] = fetched.get('page_status', [])
            found = fetched.get('figures') or []
            writing["current"]["done"].wait(timeout=30)      # the relevance test compares a figure with what the report already says
            report = cut_for_model(" ".join("".join(writing[n]["produced"]) for n in ("latest", "current")), 3500)
            figure_box["figures"] = find_figures(question, papers, report, found, trace, condition=(state.get("entities") or {}).get("condition") or "")
        except Exception as error:
            trace["error"] = f"{type(error).__name__}: {str(error)[:80]}"
    figures_thread = threading.Thread(target=run_figures, daemon=True)
    try:      # option (a): the relevance reading and the escalation finish BEFORE the evidence is written, under one hard deadline (ESCALATION_DEADLINE_S); unfinished work is reported in the limitations
        escalation = run_escalation_phase(state, question, plan_box.get("plan") or {}, plan_box.get("routing") or {}, triage_thread, skipped)
        state["escalations"] = {k: v for k, v in escalation.items() if k not in ("all_ops", "results")}
        if escalation.get("all_ops"):
            state["section_states"] = compute_section_states(escalation)      # the report shows WHY a section has no content (not applicable, not retrieved, no evidence found, failed)
        if escalation.get("all_ops"):
            stage = plan_box.get("routing") or {}
            late = stage["execution"].late_results() if stage.get("execution") else {}
            state["routing"] = {**(state.get("routing") or {}), **rr.routing_record(escalation["all_ops"], escalation["results"], rejected=stage.get("rejected"), trace=stage.get("trace"), late=late)}
    except Exception as error:
        skipped.append(f"the escalation phase did not complete ({type(error).__name__}); the evidence is written from what was retrieved")
        triage_thread.join(timeout=3.5)
    if (plan_box.get("routing") or {}).get("unfinished"):
        skipped.append("not all source-aware searches were written and finished before the deadline; those that had finished were used and the rest were not")
    evidence_ready.set()
    if fulltext_thread.is_alive():
        fulltext_thread.join(timeout=4.0)      # normally finished long ago (it ran beside the triage); a late one is simply not used
    understanding = state.get("understanding") or {}
    needs_workup = understanding.get("answer_format") == "stepwise_plan" or "diagnosis" in (understanding.get("info_needed") or [])
    if needs_workup and not guidelines_ready.wait(timeout=max(0.0, DEEP_BUDGET_S - clock.now())):
        skipped.append("guideline retrieval did not finish before the report deadline; the clinical approach may be incomplete")
    context = build_evidence_context(state, question)
    writing = {name: {"queue": queue.Queue(), "text": "", "produced": [], "done": threading.Event()} for name in ("latest", "current", "tail")}
    figures_thread.start()      # the figures are found while the three parts are written

    def write_part(name: str, task: str) -> None:
        """One writing task in its own thread: its text goes into a queue as the model produces it. The three run side by side, so the report takes as long as the slowest of them, not the sum."""
        prompt = f"TASK: {task}\n\nQUESTION: {question}\nRETRIEVED DATA (JSON):\n{json.dumps(context, ensure_ascii=False, separators=(',', ':'))}"
        trace = writing[name].setdefault("trace", {"started": round(clock.now(), 1)})
        compact = "\n\nWrite the table compactly: ONE space on each side of every '|' and a plain '|---|---|' separator row. Never pad cells with spaces to align columns."
        held: list[str] = []      # the Current evidence part is released only when it is complete: a retry must not leave a half table on the screen

        def emit(text: str) -> None:
            if name == "current":
                held.append(text)
            else:
                writing[name]["produced"].append(text)
                writing[name]["queue"].put(text)

        def run(task_prompt: str, temperature: float | None) -> bool:
            """Stream one attempt, squeezing runs of spaces (table padding is only whitespace). Returns False when the model fell into endless padding and the attempt was abandoned."""
            blanks, last = 0, "\n"
            for chunk in stream_llm(load_skill_text("evidence-synthesis"), task_prompt, DEEP_TOKENS, temperature=temperature, purpose="writer"):
                trace.setdefault("first_chunk", round(clock.now(), 1))
                squeezed = re.sub(r"[ \t]{2,}", " ", chunk)
                if last in " \t" and squeezed.startswith(" "):
                    squeezed = squeezed[1:]
                blanks = blanks + len(chunk) if not chunk.strip(" \t") else 0
                if blanks > 400:
                    return False
                if squeezed:
                    last = squeezed[-1]
                    emit(squeezed)
            return True
        try:
            finished = run(prompt, None)
            if name == "current":
                held[:] = [rows_to_table("".join(held), state["papers"])]      # final table order and relevance labels come from the same screened records as all other evidence sections
                if not finished or not re.search(r"(?m)^\|\s*-{3}", held[0]):
                    trace["retried"] = True      # endless padding, or no rows at all: the part is written again once, deterministically
                    held.clear()
                    run(prompt + compact, 0.0)
                    held[:] = [rows_to_table("".join(held), state["papers"])]
                writing[name]["produced"][:] = ["".join(held)]
                writing[name]["queue"].put("".join(held))
            elif not finished:
                trace["retried"] = True
                run(prompt + compact, 0.0)
            trace["ended"] = round(clock.now(), 1)
        except Exception as error:
            skipped.append(f"evidence writing, part {name} ({str(error)[:100]})")
        finally:
            writing[name]["done"].set()
            writing[name]["queue"].put(None)
    understanding = state.get("understanding") or {}
    needs_workup = understanding.get("answer_format") == "stepwise_plan" or "diagnosis" in (understanding.get("info_needed") or [])
    latest_task = LATEST_TASK + (" " + WORKUP_ADDENDUM if needs_workup else "")
    for name, task in (("latest", latest_task), ("current", CURRENT_TASK), ("tail", TAIL_TASK)):
        threading.Thread(target=write_part, args=(name, task), daemon=True).start()

    def drain(name: str, live: bool = True):
        """Everything one writing task produces, in order: shown as it arrives (live) or collected silently."""
        while True:
            chunk = writing[name]["queue"].get()
            if chunk is None:
                return
            writing[name]["text"] += chunk
            if live:
                yield event("token", stage="EVIDENCE_ENRICHING", text=chunk)

    yield from drain("latest")      # Latest findings first, live; the other two parts are already being written
    yield event("token", stage="EVIDENCE_ENRICHING", text="\n\n")
    yield from drain("current")
    clock.mark("evidence_done_s")
    # the drug landscape and the trials follow Current evidence as soon as they are ready; Key studies ... What remains come after them
    landscape_thread.join(timeout=max(10.0, DEEP_BUDGET_S - clock.now()))      # nothing else may read or change the papers while the thread works
    yield from emit_landscape_and_trials()
    latest_text, latest_cut = trim_part(writing["latest"]["text"])
    if latest_cut:
        skipped.append("the Latest findings reached the model's output limit; its last, incomplete entry was removed")
    gen_text = "\n\n".join(x for x in (latest_text.strip(), writing["current"]["text"].strip()) if x)
    yield from drain("tail", live=False)      # normally finished long ago; its cleaned text is shown below
    gen_text = gen_text + ("\n\n" + writing["tail"]["text"].strip() if writing["tail"]["text"].strip() else "")
    state["writing_raw"] = {k: v["text"] for k, v in writing.items()}      # exactly what each writing task returned, kept in the saved state for audit
    state["writing_trace"] = {k: v.get("trace") for k, v in writing.items()}
    state["synthesis"], cut_off = trim_incomplete(gen_text or "")
    state["synthesis"] = tidy_model_sections(state["synthesis"])      # no stray labels such as "(A)" and no loose separators
    if cut_off:
        skipped.append("evidence section reached the model's output limit; its last, incomplete line was removed")
    synthesis = clean_model_key_studies(state["synthesis"], ordered_papers(papers, stems, include_background=True), stems)   # model-written Key studies; deterministically grouped, ordered and renumbered from screened records
    cut = HOLD_FROM.search(synthesis)
    evidence_head, evidence_tail = (synthesis[:cut.start()].rstrip(), synthesis[cut.start():]) if cut else (synthesis, "")
    parts["head"] = evidence_head
    clock.mark("evidence_done_s")

    # 5. Key studies, Conflicting evidence, What remains under investigation (model-written, held back until the drugs and trials were shown)
    if evidence_tail.strip():
        parts["tail"] = evidence_tail.strip()
        yield event("token", stage="TRIALS_ENRICHING", text="\n\n" + parts["tail"])
    sections.extend(parts[k] for k in ("head", "landscape", "trials", "tail") if parts.get(k))      # the document's order

    # limitations (deterministic) - keeps 'source failed' apart from 'searched, nothing found'
    retracted = [p for p in papers.values() if p.get("retracted")]
    excluded_after_screen = [p for p in papers.values() if not p.get("retracted") and p.get("triage") and not usable(p, stems)]
    unreviewed_papers = [p for p in papers.values() if not p.get("retracted") and not p.get("triage")]
    notes = [f"Excluded retracted publication: {p['title'][:90]} (PMID {p['pmid']}, {p['year']}). Retracted work is not used as evidence." for p in retracted]
    future = [p for p in papers.values() if str(p.get("year") or "").isdigit() and int(p["year"]) > datetime.now().year]
    notes += [f"PMID {p['pmid']} carries a journal issue date of {p['year']} (later than today); treat its date as an advance-publication date." for p in future]
    if excluded_after_screen:
        notes.append(f"{len(excluded_after_screen)} screened record(s) were excluded because the relevance reading judged them not to answer the question.")
    if unreviewed_papers:
        notes.append(f"{len(unreviewed_papers)} retrieved record(s) were not screened before the deadline and were not used as evidence.")
    lifecycle = evidence_lifecycle(state, "", "pending")
    paper_counts = lifecycle["papers"]
    duplicate_n = max(0, paper_counts["retrieved_occurrences"] - paper_counts["deduplicated"])
    notes.append(f"Evidence lifecycle: {paper_counts['retrieved_occurrences']} retrieved record occurrence(s), {paper_counts['deduplicated']} unique records after merging provenance ({duplicate_n} duplicate occurrence(s)), "
                 f"{paper_counts['screened']} screened, {paper_counts['included']} included for synthesis, {paper_counts['unreviewed']} unreviewed.")
    limits_md = limitations_markdown(state, skipped, notes)
    yield event("token", stage="TRIALS_ENRICHING", text="\n\n" + limits_md)

    # VERIFICATION + QC on the FINAL answer string: verify -> repair the exact failed spans -> verify again -> QC
    yield event("stage", stage="VERIFICATION")
    full = strip_preliminary("\n\n".join(s for s in sections + [limits_md] if s))
    result = verify_and_repair(full, state)
    full = result["answer"]
    state["verification"] = {"status": result["status"], "checked": result["checked"], "repairs": result["repairs"], "repair_count": result["repair_count"],
                             "issues_before_repair": [i["message"] for i in result["issues_before"]]}
    state["evidence_lifecycle"] = evidence_lifecycle(state, full, result["status"])
    cited_n = state["evidence_lifecycle"]["papers"]["citation_checked"]
    lifecycle_line = f"Evidence lifecycle: {paper_counts['retrieved_occurrences']} retrieved occurrence(s), {paper_counts['deduplicated']} unique, {paper_counts['screened']} screened, {paper_counts['included']} included, {paper_counts['unreviewed']} unreviewed; {cited_n} source identifier(s) cited after verification ({result['status']})."
    full = re.sub(r"\n\n### Sources", "\n\n" + lifecycle_line + "\n\n### Sources", full, count=1)
    final_check = verify_and_repair(full, state, repair=False)
    full = final_check["answer"]
    result = final_check
    state["verification"].update({"status": result["status"], "checked": result["checked"], "final_check": True})
    state["qc"] = {"status": result["status"], "checks": result["checks"], "remaining": [i["message"] for i in result["issues_after"]], "view": build_qc_context(state)}
    clock.mark("verification_done_s")
    routing_keys = {row["key"] for row in (state.get("routing") or {}).get("operations", [])}
    state["search_log"] = search_log(state.pop("_search_raw", {}), routing_keys)      # per search: the mode that wrote it, its source, how many records came back, how many the other mode did not find
    logging.getLogger("neurogpt.search").info("search summary: %s", state["search_log"]["summary_line"])
    yield event("verification", status=result["status"], issues=[i["message"] for i in result["issues_after"]], repaired=result["repairs"],
                checked=result["checked"], checks=result["checks"])

    # FINAL
    state["answer_status"] = "FINAL"
    clock.mark("final_s")
    state["timestamps"].update(clock.marks)
    state["citations"] = [papers[p]["cite"] for p in set(re.findall(r"PMID[:\s]*([0-9]{6,9})", full)) if p in papers]
    full, cited_order = numberize(full, papers)      # [1], [2] in the text and one reference list; the mapping is kept in the state
    state["reference_numbers"] = {str(i): pmid for i, pmid in enumerate(cited_order, 1)}
    state["figure_job"] = str(id(figure_box))      # the figures are still being found and read; the answer is not held for them. They are collected when the PDF is made (collect_figures)
    FIGURE_JOBS[state["figure_job"]] = (figures_thread, figure_box)
    footer = closing_note(result["status"])
    pool.shutdown(wait=False, cancel_futures=True)
    yield event("final", markdown=full + footer, timings=dict(clock.marks), state=state, skipped=skipped)


# ------------------------------------------------------------------ save + command line
def save_outputs(question: str, final: dict, subfolder: str = "", figures: list | None = None) -> dict:
    """Browser runs (the server) save to demo_output/. Command-line test runs pass subfolder="cli_test_runs", so a test can never overwrite a browser answer or its PDF."""
    out = Path(__file__).parent / "demo_output" / subfolder
    out.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", question.lower())[:50].strip("-") or "answer"
    (out / f"{slug}.md").write_text(f"# {question}\n\n{final['markdown']}\n", encoding="utf-8")
    (out / f"{slug}.state.json").write_text(json.dumps(final["state"], indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    if figures is None:
        figures = prepare_figures(final)      # the page and the PDF get the SAME records
    (out / f"{slug}.state.json").write_text(json.dumps(final["state"], indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    pdf = save_pdf(question, final["markdown"], out, slug, figures=figures)
    return {"md": out / f"{slug}.md", "pdf": pdf, "figures": figures}


def main() -> None:
    # Windows terminals often default to cp1252; source text may contain Unicode names and punctuation.
    # Keep the CLI report stream usable instead of aborting before the final markdown/PDF is saved.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    question = " ".join(sys.argv[1:]) or "What is the latest evidence on tolebrutinib in multiple sclerosis?"
    warm = threading.Thread(target=warm_up, daemon=True)  # the server does this once at start-up; the command line does it here so timings match
    warm.start()
    warm.join(timeout=8)
    final = None
    for ev in stream_answer(question):
        if ev["type"] == "stage":
            print(f"\n[{ev['t']:6.2f}s] --- {ev['stage']} ---", flush=True)
        elif ev["type"] == "token":
            print(ev["text"], end="", flush=True)
        elif ev["type"] in ("milestone", "verification", "error", "warning"):
            print(f"\n[{ev['t']:6.2f}s] {ev['type'].upper()}: { {k: v for k, v in ev.items() if k not in ('type', 't')} }", flush=True)
        elif ev["type"] == "final":
            final = ev
    if final:
        print("\n\n=== SEARCH SUMMARY ===")
        print(" ", ((final.get("state") or {}).get("search_log") or {}).get("summary_line", "no search log"))
        print("\n\n=== TIMINGS (seconds from request start) ===")
        for k, v in final["timings"].items():
            print(f"  {k:22} {v}")
        paths = save_outputs(question, final, subfolder="cli_test_runs")      # NOT demo_output/: the browser run owns the report and PDF of a question
        print(f"\nSaved: {paths['md']}")


if __name__ == "__main__":
    main()
