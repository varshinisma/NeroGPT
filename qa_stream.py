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
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path

import phi
import server
from clinical_tools import fda_label_link, search_clinical_trials, search_europepmc_quick, search_fda_labels, search_pubmed
from landscape import (drug_intelligence_status, landscape_markdown, limitations_markdown, record_source_status,
                       top_drugs_markdown, trials_markdown, used_in_practice_rows, candidate_sentences, trial_drug_rows, dose_summary_ok, dose_fallback)
from pipeline import save_pdf, strip_preliminary
from presentation import closing_note, numberize
from guidelines import find_guidelines, load_config as load_care_config, topics_for
from report import DISCLAIMER, clean_model_key_studies, design_of, source_problems
from verification import is_indirect, known_sources, verify_and_repair

FAST_TIMEOUT_S = 4.5          # per retrieval skill, FAST path (the FDA table is part of the first answer, so the FDA search is waited for)
DEEP_TIMEOUT_S = 25.0         # per retrieval skill, DEEP path
CLAUDE_MODE = os.getenv("NEUROGPT_LLM") == "claude_files"   # the text-writing steps are answered by Claude through files, not by an API model
DEEP_BUDGET_S = 1e5 if CLAUDE_MODE else 100.0   # total deep-research budget (not applied while Claude writes the text)
MAX_CONTEXT_PAPERS = 16
FAST_TOKENS, DEEP_TOKENS = 330, 1700   # two output sizes only, so two warm connections
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


def web_search(query: str, snippet_chars: int = 500) -> str:
    """Web search: Tavily first (fast, relevant, regulator pages), DDGS as the fallback. Same JSON shape as server.search_live_web."""
    global _tavily_client
    if os.getenv("TAVILY_API_KEY"):
        try:
            if _tavily_client is None:
                from tavily import TavilyClient
                _tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
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
        state["guideline_rows"] = data["results"]
        state["guidelines"] = [r for r in data["results"] if r.get("title")]
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


def tidy_model_sections(text: str) -> str:
    """Remove what a small model sometimes adds around its sections: letter labels like '(A)' / '**(B)**' and loose '---' separators."""
    text = re.sub(r"(?m)^[ \t]*\*{0,2}\(?[A-D]\)\*{0,2}[ \t]*$\n?", "", text)
    text = re.sub(r"(?m)^[ \t]*-{3,}[ \t]*$\n?", "", text)
    text = fix_reference_column(text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def load_skill_text(name: str) -> str:
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
                if p["pmid"] in papers:
                    continue
                surname = (p.get("authors") or ["Unknown"])[0].split(" ")[0]
                types = [t for t in p.get("study_type", []) if t != "Journal Article"][:3]
                papers[p["pmid"]] = {
                    "source_id": f"pmid:{p['pmid']}", "pmid": p["pmid"], "doi": p.get("doi"), "url": p.get("url"), "year": p.get("year"),
                    "date": p.get("publication_date"), "journal": p.get("journal"), "authors": (p.get("authors") or [])[:3], "title": p.get("title"),
                    "type": types, "study_type": types, "abstract": p.get("abstract_excerpt") or "", "population": None, "finding": None, "limitations": None,
                    "cite": f"{surname} et al., {p.get('journal')}, {p.get('year')} [PMID {p['pmid']}]", "origin": key,
                    "retracted": any("retract" in str(t).lower() for t in p.get("study_type", [])),
                    "protocol": "protocol" in (p.get("title") or "").lower() or any("protocol" in str(t).lower() for t in p.get("study_type", [])),
                    "retrieved_at": datetime.now().isoformat(timespec="seconds")}
        elif key.startswith("trials"):
            if isinstance(data, dict) and data.get("error"):
                errors[key] = data["error"]
                continue
            for t in data.get("results", []):
                trials.setdefault(t["nct_id"], {**t, "source_id": f"nct:{t['nct_id']}", "results_available": bool(t.get("results_posted")),
                                                "results_source": "ClinicalTrials.gov registry", "evidence_status": "registry_status_only"})
        elif key.startswith("web"):
            if isinstance(data, list):
                seen = {w["url"] for w in web}
                web += [{"title": r.get("title"), "url": r.get("url"), "snippet": r.get("snippet") or "", "source": key}
                        for r in data if r.get("url") not in seen]
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
                "drug": r["generic"], "category": "FDA-approved for the labelled indication" if r["matches_condition"] else "FDA label found; indication does not mention the condition",
                "indication": r["indication_excerpt"], "population": None, "approval_status": "approved (labelled indication)" if r["matches_condition"] else "not_verified",
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
    return {"cite": p["cite"], "year": p["year"], "date": p["date"], "type": p["type"], "title": p["title"], "abstract": snippet(p["abstract"], chars),
            "relation": "indirect (different or related population)" if is_indirect(p, stems) else "direct", "protocol_no_results": bool(p.get("protocol"))}


def trial_view(t: dict) -> dict:
    return {"id": t["nct_id"], "title": t["title"], "phase": t.get("phase"), "status": t.get("recruitment_status"),
            "last_update": t.get("last_update"), "interventions": (t.get("interventions") or [])[:4],
            "primary_endpoint": t.get("primary_endpoint"), "results_posted": t.get("results_posted"), "why_stopped": t.get("why_stopped")}


GENERIC = {"children", "child", "pediatric", "paediatric", "adolescent", "adolescents", "adults", "adult", "patients", "patient", "people", "kids", "infants"}


def core_stems(question: str, plan: dict | None = None) -> set[str]:
    """Stems of the question's own subject words (and drugs the planner found). Used to drop off-topic records, not to rank."""
    words = keywords(question)[0] + [d.lower() for d in ((plan or {}).get("drugs") or [])]
    return {w[:5] for w in words if w not in GENERIC and len(w) >= 4}


def usable(p: dict, stems: set[str]) -> bool:
    """Not retracted, and about the question's subject (a stem appears in the title or abstract)."""
    return not p.get("retracted") and (not stems or any(st in f"{p['title']} {p['abstract']}".lower() for st in stems))


def evidence_tier(p: dict) -> int:
    """0 guideline, 1 systematic review / meta-analysis, 2 randomized trial, 3 other clinical study, 4 narrative review, 5 case report, 6 design not stated."""
    kinds = " ".join(str(t) for t in (p.get("type") or [])) + " " + (p.get("title") or "")
    if "guideline" in kinds.lower():
        return 0
    design = design_of(p)
    return design[1] + 1 if design else 6


def ordered_papers(papers: dict, stems: set[str] | None = None) -> list[dict]:
    """What the model sees first: the strongest study design first (guideline, systematic review / meta-analysis, randomized trial ...), the newest first within a design,
    keeping retrieval order within a year. Retracted and off-topic records are left out of what the model sees (they stay in the ResearchState). One sort, no reranking model."""
    pool = [p for p in papers.values() if stems is None or usable(p, stems)]
    return sorted(pool, key=lambda p: (evidence_tier(p), -(int(p["year"]) if str(p.get("year") or "").isdigit() else 0)))


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


def stream_llm(instructions: str, prompt: str, max_tokens: int, attempts: int = 2):
    """Yield text chunks from a FRESH agent (no shared history). Bounded retries only before the first chunk."""
    if CLAUDE_MODE:
        yield from bridge_llm(instructions, prompt)
        return
    from agno.agent import Agent
    last = None
    for _ in range(attempts):
        got = False
        try:
            agent = Agent(model=model_for(max_tokens), instructions=[instructions], markdown=True)
            for event in agent.run(prompt, stream=True):
                kind, content = str(getattr(event, "event", "")), getattr(event, "content", None)
                if "Error" in kind or (not got and str(content or "").startswith(("API error", "Error"))):
                    raise RuntimeError(str(content or kind)[:300])
                if kind == "RunContent" and content:
                    got = True
                    yield str(content)
            return
        except Exception as error:
            last = error
            if got:
                raise
    raise RuntimeError(f"The model provider returned an error: {last}")


_models: dict[int, object] = {}


def model_for(max_tokens: int):
    """One model object per output size, reused across requests, so the HTTP connection stays open (no fresh handshake per call)."""
    if max_tokens not in _models:
        _models[max_tokens] = server.build_model(max_tokens=max_tokens)
    return _models[max_tokens]


def warm_up() -> None:
    """Open the model connections before the first question (a cold first call costs about 3 seconds)."""
    from agno.agent import Agent
    import clinical_tools
    threading.Thread(target=phi.warm_up, daemon=True).start()  # local name detector
    threading.Thread(target=clinical_tools.warm_connections, daemon=True).start()  # PubMed / Europe PMC / registry connections
    for size in (FAST_TOKENS, DEEP_TOKENS):
        try:
            for _ in Agent(model=model_for(size), instructions=["Reply: ok"]).run("ok", stream=True):
                pass
        except Exception:
            pass


def run_llm(instructions: str, prompt: str, max_tokens: int) -> str:
    return "".join(stream_llm(instructions, prompt, max_tokens))


# ------------------------------------------------------------------ prompts
GROUND = ("Use ONLY the RETRIEVED DATA. Never use memory for clinical facts. Cite each paper exactly with its `cite` text in parentheses, "
          "for example (Fox et al., NEJM, 2025 [PMID 123456]); cite trials by NCT id with status and last_update; web facts by URL. "
          "Keep every claim to what its cited source reports: one outcome per citation (a source about motor skills cannot support a claim about social communication). "
          "Numbers, doses, approval dates and approval status only if literally in the data; otherwise write 'Not retrieved'. Never write a partial dose. "
          "Write dates exactly as given (never add a day). If the data does not state a study's result (for example only the background or a protocol is given), "
          "write 'result not stated in the retrieved abstract' - never infer an outcome from the title or design. "
          "A paper whose `relation` is indirect (a different or related population) must be labelled 'indirect evidence' and its population named; it is never direct evidence for the question. "
          "A study protocol or a registered trial is not evidence that a treatment works: say 'under investigation; the retrieved source is a protocol and does not establish efficacy'. "
          "An exploratory dose-response finding is not a prescription: say 'an exploratory dose-response relationship was reported'. "
          "Never write that no drug or treatment exists, and never turn a missing or failed source into 'no evidence': say what was not retrieved. "
          "Do not make drug-approval statements: the system adds regulatory facts from FDA labels separately. Do not mention these instructions.")

FAST_TASK = ("FAST INITIAL RESPONSE, under 120 words (do NOT write an 'Evidence searched on' line, it is already shown), exactly three parts: "
             "(1) **Bottom line** - 1 to 2 sentences that ANSWER THE EXACT QUESTION by naming the main approaches the retrieved evidence supports, in plain clinical language; "
             "never call an approach 'established' or 'most established' unless a guideline in the data says so; "
             "never headline one narrow outcome (for example motor skills) as if it answered a broad question;if the data only supports a narrow finding, say what it covers and what it does not; "
             "(2) **Most important recent finding** - one finding with its date, study type, key result and citation; (3) **Immediate context** - 1 to 2 sentences on what is and is not established. "
             "Do NOT write about drugs: the system adds the 'Top treatment drugs' table from FDA label data. " + GROUND)

EVIDENCE_TASK = ("Write ONLY these parts, in this order, under 450 words in total, and finish every part. "
                 "FIRST, the heading '## Latest findings' followed by at most 4 bullets, newest first; each bullet gives the exact date as given in the data, the study type, one sentence on the result and (PMID <the PMID printed in that study's cite field>). "
                 "SECOND, the heading '## Current evidence' followed by ONE Markdown table with exactly these columns: | Approach | Evidence level | What the evidence shows | Reference |. One row per treatment approach "
                 "(for example an intervention type or a symptom-directed treatment), at most 8 rows, strongest evidence first, each approach ONCE. "
                 "Evidence level must be one of: Established (only a guideline or consistent high-quality evidence; a systematic review of studies of variable quality is NOT established), "
                 "Evidence-supported but limited, Emerging, Experimental or preliminary, Indirect evidence (different or related population), Protocol - no results, Insufficient evidence. "
                 "'What the evidence shows' is ONE sentence of what the cited source reports. Reference is (PMID <the PMID printed in that study's cite field>). "
                 "THIRD, '## Key studies': a numbered list of up to 5 studies, each in the form '**<first author> et al. (<year>)**, <design>: <what it found, one sentence> (PMID <the PMID printed in that study's cite field>)' "
                 "- fill every <...> from a study in the data; the design is for example meta-analysis, systematic review, randomized controlled trial or cohort study. "
                 "FOURTH, '## Conflicting evidence' (only a real disagreement between two DIFFERENT sources, or negative results; otherwise say none was retrieved). "
                 "FIFTH, '## What remains under investigation' (at most 5 short bullets, no sub-bullets). Distinguish efficacy, effectiveness, feasibility and association. "
                 "Do not discuss drug approval or drug availability. " + GROUND)

HOLD_FROM = re.compile(r"(?im)^#{2,4}\s*(?:Key studies|Conflicting|What remains)")      # the specification puts Key studies, Conflicting evidence and What remains AFTER the drugs and trials

DOSE_TASK = ("Below is the dosing text of an FDA drug label for ONE use of the drug. Write 3 to 4 short plain-English sentences for a clinician that cover: the starting dose, how it is adjusted "
             "(steps and timing), the usual or maximum dose range, and the age or weight group it applies to, but only where the text states them. Use ONLY what the text says. "
             "Copy every number and unit exactly as written; never add a number, a warning, advice or any fact that is not in the text, and never comment on what the label does NOT say. Keep every qualifier exactly as written (for example 'no less than', 'at least', 'up to', 'a minimum of') and do not add a timing the text does not give. Reply with the sentences only: plain text, no bold, no heading, no list.")

USED_TASK = ("List the drugs that the SOURCES below describe as USED IN CLINICAL PRACTICE to treat or manage this condition or its symptoms (for example in a review, guideline or "
             "practice article), beyond any drug already in `fda_labelled_for_condition`. Reply with ONE JSON array only, at most 6 objects, each {\"drug\": generic name, "
             "\"source\": the exact `id` of the source that says so}. Do NOT include drugs that are only being tested in a trial or protocol, animal or laboratory studies, "
             "drugs named as a risk or exposure, supplements, therapies or devices. Never invent a drug or an id; if there are none, reply [].")

DRUG_TASK = ("Write ONLY this block. Heading '#### Other agents named in the retrieved evidence (not in the FDA label records)' followed by a Markdown table with columns: "
             "Drug (generic/brand) | Category | Population studied | Dose / route | Approval status | Key evidence. "
             "Category must be one of: Investigational, Indirect evidence (related population), Guideline-recommended (only if a retrieved guideline source says so), "
             "Off-label (only if a retrieved source states off-label use for this condition). Do NOT list any drug in `regulatory_records_already_listed`. "
             "List only SUBSTANCES named in the data (drugs, supplements, probiotics, biologics). Do NOT list exercise, dance, music, diet, therapy, behavioural or device interventions: they belong to the non-drug section. Do NOT list a drug that appears only as a risk or exposure (for example prenatal exposure). Write exactly six cells per row. If there are none, write 'None retrieved.' Then, only if the data has any, '**Safety alerts:**' with a source. "
             "Under 220 words. " + GROUND)

PLAN_TASK = ("Follow the skill: classify the question and extract entities. Reply with ONE JSON object only. Keys: \"is_clinical\" (true if a medical question), "
             "\"intent\" (disease_research|latest_findings|drug_treatment|clinical_trial|comparison|mechanism|comprehensive_research), \"condition\", "
             "\"drugs\" (drug names in the question), \"pubmed_query\" (PubMed terms with AND/OR), \"trial_condition\", \"trial_intervention\" (drug or empty).")


def plan_with_skill(question: str) -> dict:
    """clinical-research-orchestrator: classify and extract entities (skill loaded on demand)."""
    try:
        text = run_llm(load_skill_text("clinical-research-orchestrator"), f"TASK: {PLAN_TASK}\nQuestion: {question}", FAST_TOKENS)
        plan = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    except Exception:
        plan = {}
    kw, _, _ = keywords(question)
    plan.setdefault("is_clinical", True)
    plan.setdefault("intent", "disease_research")
    plan.setdefault("condition", " ".join(kw) or question)
    plan.setdefault("drugs", [])
    plan.setdefault("pubmed_query", pubmed_query_from_question(question)[0])
    plan.setdefault("trial_condition", plan["condition"])
    plan.setdefault("trial_intervention", ", ".join(map(str, plan["drugs"][:1])) if isinstance(plan["drugs"], list) else str(plan["drugs"]))
    return normalize_plan(plan)


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


def fast_jobs(question: str) -> dict:
    return {name: with_retry(fn, retries=1, pause=0.4) for name, fn in _fast_jobs(question).items()}     # one retry with backoff for EVERY source in the fast pass


def _fast_jobs(question: str) -> dict:
    query, years = pubmed_query_from_question(question)
    kw, _, _ = keywords(question)
    cond = " ".join(kw) or question
    return {
        "pubmed_fast": lambda: cached("pm", f"{query}|{years}", 900, lambda: search_europepmc_quick(query, max_results=6, years_back=years, sort="relevance")),
        "pubmed_fast_q": lambda: cached("pm", f"{query}|{years}|q", 900, lambda: search_europepmc_quick(
            f"({query}) AND (PUB_TYPE:\"Systematic Review\" OR PUB_TYPE:\"Meta-Analysis\" OR PUB_TYPE:\"Randomized Controlled Trial\" OR PUB_TYPE:\"Practice Guideline\")",
            max_results=4, years_back=years + 1, sort="relevance")),
        "trials_fast": lambda: cached("ct", cond, 300, lambda: search_clinical_trials(cond, "", "", 4)),
        "trials_drugs_fast": lambda: cached("ct", f"{cond}|drug-fast", 300, lambda: search_clinical_trials(cond, "", "", 4, drug_only=True)),
        "fda_fast": lambda: cached("fda", cond, 3600, lambda: search_fda_labels(kw, None, 4)),
        "web_fast": lambda: cached("web", cond, 300, lambda: web_search(f"{cond} FDA approved treatment drugs")),
    }


def deep_jobs(plan: dict, question: str) -> dict:
    year, cond = datetime.now().year, plan["condition"]
    pq = pubmed_query_from_question(question)[0]  # the question's own words + treatment clause: avoids the noise of broad planner synonyms
    quality = "(randomized controlled trial[pt] OR meta-analysis[pt] OR systematic review[pt] OR guideline[pt])"
    negative = '("no significant"[tiab] OR failed[tiab] OR negative[tiab] OR hepatotoxicity[tiab] OR "liver injury"[tiab] OR futility[tiab] OR "safety signal"[tiab])'
    nondrug = '(behavioral[tiab] OR psychosocial[tiab] OR "early intervention"[tiab] OR "parent-mediated"[tiab] OR rehabilitation[tiab] OR exercise[tiab] OR surgery[tiab] OR diet[tiab])'
    fda_terms = keywords(question)[0] or re.findall(r"[A-Za-z]{4,}", cond)   # the question's own words first: never free-text planner prose
    jobs = {
        "pubmed_recent": lambda: cached("pm", f"{pq}|r", 900, lambda: search_pubmed(pq, max_results=8, years_back=2, sort="date")),
        "pubmed_quality": lambda: cached("pm", f"{pq}|q", 900, lambda: search_pubmed(f"({pq}) AND {quality}", max_results=5, years_back=5, sort="relevance")),
        "pubmed_negative": lambda: cached("pm", f"{pq}|n", 900, lambda: search_pubmed(f"({pq}) AND {negative}", max_results=4, years_back=5, sort="relevance")),
        "pubmed_drugs": lambda: cached("pm", f"{pq}|dr", 900, lambda: search_pubmed(
            f"({pq}) AND (pharmacological[tiab] OR medication[tiab] OR \"drug therapy\"[tiab] OR drug[tiab]) AND {quality}", max_results=5, years_back=5, sort="relevance")),
        "trials_drugs": with_retry(lambda: cached("ct", f"{plan['trial_condition']}|drug-only", 300, lambda: search_clinical_trials(plan["trial_condition"], "", "", 12, drug_only=True))),
        "trials_deep": with_retry(lambda: cached("ct", f"{plan['trial_condition']}|{plan['trial_intervention']}", 300,
                                                 lambda: search_clinical_trials(plan["trial_condition"], plan["trial_intervention"], "", 10))),
        "fda_deep": with_retry(lambda: cached("fda", f"{' '.join(fda_terms)}|{','.join(plan.get('drugs') or [])}", 3600,
                                              lambda: search_fda_labels(fda_terms, plan.get("drugs") or []))),
        "web_approvals": with_retry(lambda: cached("web", f"{cond}|a", 300, lambda: web_search(f"{cond} newly approved drug FDA EMA approval {year - 1} {year}"))),
        "web_medications": with_retry(lambda: cached("web", f"{cond}|m", 300, lambda: web_search(f"medications commonly prescribed to children with {cond}: which drugs are used for irritability, sleep, anxiety and ADHD", snippet_chars=1500))),
        "web_regulatory": with_retry(lambda: cached("web", f"{cond}|reg", 300, lambda: regulator_search_many([f"FDA press announcement action treatment {cond}", f"{cond} drug approval label expansion safety communication FDA"]))),
        "web_standard": with_retry(lambda: cached("web", f"{cond}|s", 300, lambda: web_search(f"{cond} treatment guideline first-line drugs dosage"))),
    }
    gl_words = [w for w in keywords(question)[0] if w not in GENERIC] or keywords(question)[0]
    jobs["guidelines_deep"] = with_retry(lambda: cached("gl", f"{' '.join(gl_words)}|{[t['id'] for t in topics_for(question)]}", 3600,
                                                        lambda: json.dumps(find_guidelines(question, gl_words, guideline_search), ensure_ascii=False)))
    if not plan.get("drugs"):  # non-drug treatments matter when the question is not about one specific drug
        jobs["pubmed_nondrug"] = lambda: cached("pm", f"{pq}|d", 900, lambda: search_pubmed(f"({pq}) AND {nondrug} AND {quality}", max_results=4, years_back=5, sort="relevance"))
    return jobs


# ------------------------------------------------------------------ skill-specific context views (read-only: they never modify the ResearchState)
def compact_status(state: dict) -> dict:
    return {k: f"{v['retrieval_status']}/{v['evidence_status']}" for k, v in state["source_status"].items()}


def build_fast_context(state: dict, question: str) -> dict:
    """clinical-answer-writer, Phase 1: a few papers, trials and web items. No drug data (the system adds the FDA table itself)."""
    stems = core_stems(question)
    return {"question": question, "papers": [paper_view(p, 330, stems) for p in ordered_papers(state["papers"], stems)[:6]],
            "trials": [trial_view(t) for t in list(state["trials_by_id"].values())[:3]],
            "web": [{"title": w["title"], "url": w["url"], "snippet": w["snippet"][:220]} for w in state["web"][:3]],
            "source_status": compact_status(state)}


def build_evidence_context(state: dict, question: str) -> dict:
    """evidence-synthesis: usable papers (not retracted, on topic) with their relation to the question, plus a few trials."""
    stems = set(state["core_stems"])
    return {"question": question, "papers": [paper_view(p, 450, stems) for p in ordered_papers(state["papers"], stems)[:MAX_CONTEXT_PAPERS]],
            "trials": [trial_view(t) for t in list(state["trials_by_id"].values())[:4]], "source_status": compact_status(state)}


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


# ------------------------------------------------------------------ the streaming pipeline
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
            jobs = deep_jobs(plan_box["plan"], question)
            retry_pool = fast_jobs(question)
            for name, st in list(state["source_status"].items()):      # a fast-pass source that still failed gets one more try here, with the longer timeout
                if st.get("retrieval_status") != "success" and name in retry_pool and name not in jobs:
                    jobs[name] = retry_pool[name]
                    plan_box.setdefault("retried", []).append(name)
            by_group: dict[str, dict] = {g: {} for g in GROUPS_ORDER}
            for name, fn in jobs.items():
                if name in EARLY_EVIDENCE_JOBS:
                    continue          # already running since the fast pass finished (they do not need the plan)
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

    EARLY_EVIDENCE_JOBS = {"pubmed_recent", "pubmed_quality", "pubmed_negative", "pubmed_drugs", "pubmed_nondrug"}

    def start_early_evidence() -> None:
        """The deep PubMed searches are built from the question's own words, not from the planner, so they start as soon as the fast search is
        done and run while the first answer is being written; the evidence section can then start right after the first answer."""
        provisional = {"drugs": [], "condition": " ".join(keywords(question)[0]) or question, "trial_condition": "", "trial_intervention": "", "intent": "comprehensive_research"}
        early = {n: fn for n, fn in deep_jobs(provisional, question).items() if n in EARLY_EVIDENCE_JOBS}

        def run():
            try:
                plan_box["raw_evidence"], _ = run_jobs(early, DEEP_TIMEOUT_S, essential=[n for n in ("pubmed_recent", "pubmed_quality") if n in early], extra_until=8.0)
            except Exception as error:
                plan_box["error_evidence"] = f"evidence retrieval crashed: {type(error).__name__}: {str(error)[:150]}"
            finally:
                group_done["evidence"].set()
        threading.Thread(target=run, daemon=True).start()

    threading.Thread(target=deep_worker, daemon=True).start()
    raw_fast, _ = run_jobs(fast_jobs(question), FAST_TIMEOUT_S, essential=["pubmed_fast", "trials_fast", "fda_fast"], extra_until=2.5)
    state["retrieval_errors"].update(ingest(papers, trials, web, raw_fast))
    ingest_regulatory(state, raw_fast)
    record_source_status(state, raw_fast)
    state["core_stems"] = sorted(core_stems(question))
    start_early_evidence()
    state["fast_evidence"] = list(papers.values())
    clock.mark("retrieval_fast_s")
    if not papers and not trials and not state["regulatory"] and not state["retrieval_errors"]:  # nothing medical was found: do not invent an answer
        note = "I answer clinical research questions using PubMed, ClinicalTrials.gov and the web. I found no medical literature for this question - please rephrase it as a clinical question."
        yield event("token", stage="INITIAL", text=note)
        yield event("final", markdown=note, timings=dict(clock.marks), state=state, skipped=[])
        pool.shutdown(wait=False, cancel_futures=True)
        return
    header = f"**Evidence searched on {datetime.now():%Y-%m-%d}** (PubMed, ClinicalTrials.gov, FDA drug labels, web)\n\n"
    summary_head = "## Clinical summary\n\n"
    yield event("token", stage="INITIAL", text=header + summary_head)

    # --- INITIAL: the fast answer, streamed (clinical-answer-writer sees only its own view)
    fast_prompt = (f"TASK: {FAST_TASK}\n\nQUESTION: {question}\nRETRIEVED DATA (JSON):\n"
                   f"{json.dumps(build_fast_context(state, question), ensure_ascii=False, separators=(',', ':'))}")
    fast_text = header  # the date line is part of the saved answer, not only of the stream
    try:
        for chunk in stream_llm(load_skill_text("clinical-answer-writer"), fast_prompt, FAST_TOKENS):
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
    yield event("token", stage="INITIAL", text="\n\n" + top_drugs_markdown(state))
    clock.mark("first_useful_s")
    state["timestamps"]["first_response"] = clock.now()
    yield event("milestone", name="first_useful_answer", seconds=clock.marks["first_useful_s"], first_token=clock.marks.get("first_token_s"))

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
        state["retrieval_errors"].update(ingest(papers, trials, web, raw))
        if group == "drugs":
            ingest_regulatory(state, raw)
        record_source_status(state, raw)
        if group == "guidelines":
            ingest_guidelines(state, raw)          # after the status: a guideline search that failed for SOME topics is recorded as partial
        state["core_stems"] = sorted(core_stems(question, plan))

    # the saved answer re-builds the Phase-1 drug block from the FINAL regulatory records, so a fast-pass timeout that the deep pass resolved is not left in the text
    # no banner at the top of the report: failed or partial sources are named in the 'Evidence limitations' section at the end
    opening = model_text
    sections: list[str] = []          # filled in the order the sections are STREAMED, so the final report is the same document that was shown growing

    streamed_upto: dict[str, int] = {}

    def stage_stream(stage: str, skill: str, task: str, data: dict, max_tokens: int, hold: re.Pattern | None = None, on_hold=None, between=None, on_start=None):
        """One enrichment stage (a generator): ONE skill loaded lazily + that skill's own view, fresh model context, tokens yielded live.
        With `hold`, tokens are shown only up to the first match of the pattern; the rest is kept back and shown later, in its place in the document.
        Returns the text, or None if skipped/failed."""
        if clock.now() > DEEP_BUDGET_S:
            skipped.append(f"{stage} (100 s deep-research budget used)")
            return None
        prompt = f"TASK: {task}\n\nQUESTION: {question}\nRETRIEVED DATA (JSON):\n{json.dumps(data, ensure_ascii=False, separators=(',', ':'))}"
        text, emitted, held = "", 0, False
        if on_start:
            on_start()      # independent work that only needs the retrieved data starts now, in the background
        try:
            for chunk in stream_llm(load_skill_text(skill), prompt, max_tokens):
                text += chunk
                if hold is None:
                    yield event("token", stage=stage, text=chunk)
                    emitted = len(text)
                elif not held:
                    m = hold.search(text)
                    end = m.start() if m else max(emitted, len(text) - 40)        # keep a short tail back: a heading may be half-typed
                    if m and on_hold:
                        on_hold()      # the visible part is done: start the work that does not need the rest
                    held = bool(m)
                    if end > emitted:
                        yield event("token", stage=stage, text=text[emitted:end])
                        emitted = end
                if between is not None and held:
                    yield from between()      # other finished sections are shown while the model is still writing the held-back part
            if hold is not None and not held and emitted < len(text):
                yield event("token", stage=stage, text=text[emitted:])
                emitted = len(text)
        except Exception as error:
            skipped.append(f"{stage} ({str(error)[:120]})")
            streamed_upto[stage] = emitted
            return None
        streamed_upto[stage] = emitted
        return text

    def prepare_landscape() -> str:
        """Merge the drug / trial / guideline retrieval, name the used-in-practice drugs, look up labels, condense doses and build the landscape. No streaming here:
        it runs in a background thread while the model finishes the held-back sections."""
        absorb("guidelines")
        absorb("drugs")
        absorb("trials")
        sections[0] = header + summary_head + opening + "\n\n" + top_drugs_markdown(state)      # same order as streamed: Bottom line, finding, context, then the FDA table (with the FINAL FDA records)
        state["trial_status"] = "complete" if (state["source_status"].get("trials_deep") or state["source_status"].get("trials_fast") or {}).get("retrieval_status") == "success" else "partial"
        state["trials"] = list(trials.values())
        label_pool = ThreadPoolExecutor(max_workers=6)
        fda_recs = [r for r in state["regulatory"] if r.get("matches_condition")]
        fda_names = [r["generic"] for r in fda_recs]
        fda_label_futures = [label_pool.submit(fda_label_link, n) for n in fda_names]      # each FDA drug's own label (boxed warning): runs while the model works below

        def condense(rec: dict):      # the label's dosing section in 3-4 sentences; kept only if every number is in the label text
            prompt = (f"TASK: {DOSE_TASK}\n\nDRUG: {rec['generic']}\nUSE: {question}\nLABEL DOSING TEXT:\n{rec['dose_statement']}")
            try:
                return rec, "".join(stream_llm(load_skill_text("drug-intelligence"), prompt, 300)).replace("**", "").strip()      # no markdown emphasis in a table cell
            except Exception:
                return rec, ""
        dose_futures = [label_pool.submit(condense, rec) for rec in fda_recs if len(rec.get("dose_statement") or "") > 350]

        # drugs described as USED IN PRACTICE: the model only NAMES a drug and a source; code verifies the name and the 'used / prescribed / recommended' wording in that source.
        # Sources: the review articles, the web pages (standard-of-care pages included) and the retrieved guideline excerpts.
        used_sources: dict = {}
        for p in sorted(ordered_papers(papers, stems), key=lambda p: p.get("origin") != "pubmed_drugs")[:14]:      # reviews found by the drug-treatment search first
            if not p.get("retracted") and not p.get("protocol"):
                used_sources[p["pmid"]] = (snippet(p["abstract"], 700), f"PMID {p['pmid']} ({p['cite'].split(' [')[0]})")
        for w in state["web"][:12]:
            if w.get("url"):
                used_sources[w["url"]] = (w.get("snippet") or "", f"[{(w.get('title') or 'web page')[:160]}]({w['url']})")
        for g in state.get("guidelines", []):
            used_sources[g["url"]] = (g.get("text") or g.get("excerpt") or "", f"[{g['organisation']}: {g['title'][:160]}]({g['url']})")
        topic_stems = [s for s in state.get('core_stems', []) if len(s) >= 4]
        full_text = {k: (v[0] + ' ' + v[1] + ' ' + k).lower() for k, v in used_sources.items()}      # text + title + address
        used_sources = {k: (candidate_sentences(v[0]), v[1]) for k, v in used_sources.items()
                        if candidate_sentences(v[0]) and (not topic_stems or any(s in full_text[k] for s in topic_stems))}      # only sources that are about the condition      # only sources with a usable sentence are shown to the model
        used_items: list = []
        used_rows: list = []
        for attempt in ((1, 2) if used_sources else ()):      # the model's choice varies between runs: if no drug passed the source check, ask once more (the check itself never changes)
            try:
                used_prompt = (f"TASK: {USED_TASK}\n\nQUESTION: {question}\nRETRIEVED DATA (JSON):\n" + json.dumps(
                    {"fda_labelled_for_condition": [r["generic"] for r in state["regulatory"] if r.get("matches_condition")],
                     "sources": [{"id": k, "text": v[0]} for k, v in used_sources.items()]}, ensure_ascii=False, separators=(",", ":")))
                raw_used = "".join(stream_llm(load_skill_text("drug-intelligence"), used_prompt, 400))
                m_used = re.search(r"\[.*\]", raw_used, re.S)
                used_items = json.loads(m_used.group(0)) if m_used else []
            except Exception as error:
                skipped.append(f"drugs used in practice ({str(error)[:80]})")
                break
            used_rows = used_in_practice_rows(used_items, used_sources, state, limit=5)
            if used_rows or clock.now() > DEEP_BUDGET_S * 0.6:
                break
        used_rows = used_in_practice_rows(used_items, used_sources, state, limit=5)
        state["used_drugs_trace"] = {"model_items": used_items, "sources_offered": len(used_sources), "accepted": len(used_rows)}      # kept in the saved state for audit
        state["used_drugs"] = [r["drug"] for r in used_rows]
        trial_rows = trial_drug_rows(state, {r['generic'] for r in state['regulatory']} | {r['drug'] for r in used_rows})      # drugs being TESTED in registered trials, from the registry only
        state['trial_drugs'] = [r['drug'] for r in trial_rows]
        listed = used_rows + trial_rows
        listed_labels = list(label_pool.map(fda_label_link, [r['drug'] for r in listed]))      # each listed drug's OWN DailyMed page, with its boxed warning
        fda_labels = [f.result() for f in fda_label_futures]      # already done while the model was working
        for row, found in zip(listed, listed_labels):
            row['fda_label'] = (found or {}).get('url')
        safety_labels = [{'drug': n, 'url': f['url'], 'boxed_warning': f.get('boxed_warning')} for n, f in zip([r['drug'] for r in listed] + fda_names, listed_labels + fda_labels) if f]
        state['drug_labels'] = [{'drug': x['drug'], 'url': x['url'], 'boxed_warning': x.get('boxed_warning')} for x in safety_labels]      # retrieved addresses: the verifier must not strip them
        for future in dose_futures:
            rec, text = future.result()
            if dose_summary_ok(text, rec["dose_statement"]):
                rec["dose_summary"] = text
            else:
                rec["dose_summary"] = dose_fallback(rec["dose_statement"])      # never an unchecked number: complete sentences copied from the label
                skipped.append(f"dose summary for {rec['generic']} was not faithful to the label text; the label's own first sentences are shown")
        landscape_md = landscape_markdown(state, used_rows, trial_rows)      # ONE categorized table + regulator announcements + boxed warnings
        return landscape_md

    landscape_box: dict = {}

    def run_prepare():
        try:
            landscape_box["md"] = prepare_landscape()
        except Exception as error:
            landscape_box["error"] = f"{type(error).__name__}: {str(error)[:120]}"
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
    state["deep_evidence"] = list(papers.values())
    clock.mark("retrieval_deep_s")
    yield event("milestone", name="retrieval_complete", seconds=clock.marks["retrieval_deep_s"])
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

    yield event("token", stage="EVIDENCE_ENRICHING", text="\n\n")
    gen_text = yield from stage_stream("EVIDENCE_ENRICHING", "evidence-synthesis", EVIDENCE_TASK, build_evidence_context(state, question), DEEP_TOKENS,
                                       hold=HOLD_FROM, on_hold=start_landscape, between=emit_landscape_and_trials, on_start=start_landscape)
    start_landscape()      # no held-back part (the model stopped early): start it now
    landscape_thread.join(timeout=max(10.0, DEEP_BUDGET_S - clock.now()))      # nothing else may read or change the papers while the thread works
    yield from emit_landscape_and_trials()      # if it was not shown during the model's writing
    state["synthesis"], cut_off = trim_incomplete(gen_text or "")
    state["synthesis"] = tidy_model_sections(state["synthesis"])      # no stray labels such as "(A)" and no loose separators
    if cut_off:
        skipped.append("evidence section reached the model's output limit; its last, incomplete line was removed")
    synthesis = clean_model_key_studies(state["synthesis"], ordered_papers(papers, stems), stems)   # model-written Key studies; incomplete entries dropped, list renumbered
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
    off_topic = [p for p in papers.values() if not p.get("retracted") and not usable(p, stems)]
    notes = [f"Excluded retracted publication: {p['title'][:90]} (PMID {p['pmid']}, {p['year']}). Retracted work is not used as evidence." for p in retracted]
    future = [p for p in papers.values() if str(p.get("year") or "").isdigit() and int(p["year"]) > datetime.now().year]
    notes += [f"PMID {p['pmid']} carries a journal issue date of {p['year']} (later than today); treat its date as an advance-publication date." for p in future]
    if off_topic:
        notes.append(f"{len(off_topic)} retrieved record(s) were excluded because they were not about the question's subject.")
    limits_md = limitations_markdown(state, skipped, notes)
    yield event("token", stage="TRIALS_ENRICHING", text="\n\n" + limits_md)

    # VERIFICATION + QC on the FINAL answer string: verify -> repair the exact failed spans -> verify again -> QC
    yield event("stage", stage="VERIFICATION")
    full = strip_preliminary("\n\n".join(s for s in sections + [limits_md] if s))
    result = verify_and_repair(full, state)
    full = result["answer"]
    state["verification"] = {"status": result["status"], "checked": result["checked"], "repairs": result["repairs"], "repair_count": result["repair_count"],
                             "issues_before_repair": [i["message"] for i in result["issues_before"]]}
    state["qc"] = {"status": result["status"], "checks": result["checks"], "remaining": [i["message"] for i in result["issues_after"]], "view": build_qc_context(state)}
    clock.mark("verification_done_s")
    yield event("verification", status=result["status"], issues=[i["message"] for i in result["issues_after"]], repaired=result["repairs"],
                checked=result["checked"], checks=result["checks"])

    # FINAL
    state["answer_status"] = "FINAL"
    clock.mark("final_s")
    state["timestamps"].update(clock.marks)
    state["citations"] = [papers[p]["cite"] for p in set(re.findall(r"PMID[:\s]*([0-9]{6,9})", full)) if p in papers]
    full, cited_order = numberize(full, papers)      # [1], [2] in the text and one reference list; the mapping is kept in the state
    state["reference_numbers"] = {str(i): pmid for i, pmid in enumerate(cited_order, 1)}
    footer = closing_note(result["status"])
    pool.shutdown(wait=False, cancel_futures=True)
    yield event("final", markdown=full + footer, timings=dict(clock.marks), state=state, skipped=skipped)


# ------------------------------------------------------------------ save + command line
def save_outputs(question: str, final: dict, subfolder: str = "") -> dict:
    """Browser runs (the server) save to demo_output/. Command-line test runs pass subfolder="cli_test_runs", so a test can never overwrite a browser answer or its PDF."""
    out = Path(__file__).parent / "demo_output" / subfolder
    out.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", question.lower())[:50].strip("-") or "answer"
    (out / f"{slug}.md").write_text(f"# {question}\n\n{final['markdown']}\n", encoding="utf-8")
    (out / f"{slug}.state.json").write_text(json.dumps(final["state"], indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    pdf = save_pdf(question, final["markdown"], out, slug)
    return {"md": out / f"{slug}.md", "pdf": pdf}


def main() -> None:
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
        print("\n\n=== TIMINGS (seconds from request start) ===")
        for k, v in final["timings"].items():
            print(f"  {k:22} {v}")
        paths = save_outputs(question, final, subfolder="cli_test_runs")      # NOT demo_output/: the browser run owns the report and PDF of a question
        print(f"\nSaved: {paths['md']}")


if __name__ == "__main__":
    main()
