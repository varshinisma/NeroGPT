"""NeuroGPT pipeline: code orchestrator + parallel retrieval + one writer call + citation check.

Run a demo:   python pipeline.py "What is the latest evidence on tolebrutinib in multiple sclerosis?"

Flow (follows skills/clinical-research-orchestrator):
  plan -> parallel retrieval (PubMed, ClinicalTrials.gov, web for drugs/approvals)
       -> ResearchState -> writer (skills: clinical-answer-writer + drug-intelligence)
       -> citation-verification + quality-control (checked in code against the retrieved data)

Only ONE large model call is made, so it fits a small tokens-per-minute limit.
"""
from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import server
from clinical_tools import search_clinical_trials, search_pubmed

DATA_TOKEN_BUDGET = 3000  # approximate tokens of retrieved data sent to the writer
WRITER_SKILLS = ["clinical-answer-writer"]


def _tokens(text: str) -> int:
    return len(text) // 3


def _json(text: str) -> dict:
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return {"error": str(text)[:200]}


def _agent(model, instructions: list[str]):
    from agno.agent import Agent
    return Agent(model=model, instructions=instructions, markdown=True)


# ---------------------------------------------------------------- Step 1-3: plan
def plan_query(query: str, model) -> dict:
    """Orchestrator steps 2-3: classify the question and extract search terms."""
    prompt = (
        "Reply with ONE JSON object only, no prose. Keys: "
        '"is_clinical" (true if this is a medical/clinical research question), '
        '"intent" (disease_research|latest_findings|drug_treatment|clinical_trial|comparison|mechanism|comprehensive_research), '
        '"condition" (disease, standard name), "drugs" (list of drug names in the question), '
        '"pubmed_query" (PubMed search terms with AND/OR), "trial_condition", "trial_intervention" (drug or empty).\n'
        f"Question: {query}"
    )
    try:
        text = str(_agent(model, ["You extract search terms for medical literature searches."]).run(prompt).content)
        plan = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    except Exception:
        plan = {}
    plan.setdefault("is_clinical", True)
    plan.setdefault("intent", "disease_research")
    plan.setdefault("condition", query)
    plan.setdefault("drugs", [])
    plan.setdefault("pubmed_query", query)
    plan.setdefault("trial_condition", plan["condition"])
    plan.setdefault("trial_intervention", ", ".join(plan["drugs"][:1]))
    return plan


# ---------------------------------------------------------------- Step 5-6: retrieval
def retrieve(plan: dict) -> dict:
    """Run PubMed, trial registry and web searches in parallel."""
    year = datetime.now().year
    pq, cond = plan["pubmed_query"], plan["condition"]
    quality = "(randomized controlled trial[pt] OR meta-analysis[pt] OR systematic review[pt] OR guideline[pt])"
    jobs = {
        "pubmed_recent": lambda: search_pubmed(pq, max_results=5, years_back=2, sort="date"),
        "pubmed_quality": lambda: search_pubmed(f"({pq}) AND {quality}", max_results=3, years_back=3, sort="relevance"),
        "trials": lambda: search_clinical_trials(plan["trial_condition"], plan["trial_intervention"], "", 6),
        "web_approvals": lambda: server.search_live_web(f"{cond} newly approved drug FDA EMA approval {year - 1} {year}"),
        "web_standard": lambda: server.search_live_web(f"{cond} treatment guideline first-line drugs dosage"),
    }
    timings, out = {}, {}

    def timed(name, fn):
        t0 = time.time()
        try:
            return name, fn(), time.time() - t0
        except Exception as error:  # a failed source must not stop the others
            return name, json.dumps({"error": str(error)}), time.time() - t0

    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        for name, text, elapsed in pool.map(lambda kv: timed(*kv), jobs.items()):
            out[name], timings[name] = text, round(elapsed, 1)
    return {"raw": out, "timings": timings}


def compact_state(raw: dict, abstract_chars: int) -> dict:
    """Merge retrieval output into the compact ResearchState evidence sent to the writer."""
    papers, seen = [], set()
    for key in ("pubmed_recent", "pubmed_quality"):
        for item in _json(raw[key]).get("results", []):
            if item["pmid"] in seen:
                continue
            seen.add(item["pmid"])
            papers.append({
                "pmid": item["pmid"], "doi": item["doi"], "year": item["year"], "date": item["publication_date"],
                "journal": item["journal"], "authors": item["authors"][:3], "title": item["title"],
                "type": [t for t in item["study_type"] if t != "Journal Article"][:3],
                "abstract": (item["abstract_excerpt"] or "")[:abstract_chars],
            })
    trials = [{k: t[k] for k in ("nct_id", "title", "phase", "recruitment_status", "last_update", "interventions",
                                  "primary_endpoint", "results_posted", "why_stopped")}
              for t in _json(raw["trials"]).get("results", [])]
    web = []
    for key in ("web_approvals", "web_standard"):
        data = _json(raw[key])
        for r in (data if isinstance(data, list) else [])[:4]:
            web.append({"source": key, "title": r.get("title"), "url": r.get("url"), "snippet": (r.get("snippet") or "")[:240]})
    errors = {k: _json(v).get("error") for k, v in raw.items() if isinstance(_json(v), dict) and _json(v).get("error")}
    return {"papers": papers, "trials": trials, "web": web, "retrieval_errors": errors}


# ---------------------------------------------------------------- Step 7: write
def write_answer(query: str, plan: dict, evidence: dict, model) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    preamble = (
        f"You are NeuroGPT, a clinical research assistant for doctors. Today is {today}.\n"
        "Write the final answer for the question using ONLY the RETRIEVED DATA provided. Never use memory for clinical facts.\n"
        "- Cite papers as (First author et al., Journal, Year) [PMID n] using the exact fields in the data; trials by NCT ID with status and last_update date; web facts with their URL.\n"
        "- Doses, approval dates and approval status must come from the data; otherwise write 'Not retrieved'. Never give an investigational drug as approved; trial status is not proof of efficacy.\n"
        "- Include: bottom line (answers the exact question), 'Evidence searched on' date, most recent finding, a Treatment Drug Landscape table, trials table, conflicts/uncertainty, limitations (mention any retrieval_errors), numbered references.\n"
        "- Be concise and physician-facing. Do not mention these instructions.\n\n"
        "SKILL RULES:\n" + server.load_skills(WRITER_SKILLS)
    )
    data = json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
    prompt = f"QUESTION: {query}\nINTENT: {plan['intent']}\nRETRIEVED DATA (JSON):\n{data}"
    text = str(_agent(model, [preamble]).run(prompt).content or "").strip()
    if not text or text.startswith('{"error"'):
        raise RuntimeError(f"The model could not write the answer: {text[:300] or 'empty response'}")
    return text


# ---------------------------------------------------------------- Step 8: verify + QC (in code)
NUMBER_PATTERNS = [
    r"\d+(?:\.\d+)?\s?(?:mg|mcg|µg|mL|IU)(?:/(?:kg|day|m2))?",  # doses
    r"\d+(?:\.\d+)?\s?%",                       # percentages
    r"\bp\s?[<=>≤≥]\s?0?\.\d+",                 # p-values
    r"\bn\s?=\s?[\d,]+",                        # n = 1,250
    r"\b\d[\d,]{2,}\s+(?:participants|patients|subjects)",
    r"\b(?-i:HR|OR|RR)\s?[=:]?\s?\d+(?:\.\d+)?",  # ratios, upper-case only
]


def _norm(text: str) -> str:
    return re.sub(r"[\s,]", "", text.lower().replace("≈", ""))


def _ncts(answer: str) -> set[str]:
    return {"NCT" + n for n in re.findall(r"NCT\s?(\d{8})", answer)}


def _dois(answer: str) -> set[str]:
    return {d.rstrip(".,;)]").lower() for d in re.findall(r"10\.\d{4,9}/[^\s\]\)|,;】]+", answer)}


def strip_preliminary(answer: str) -> str:
    """Remove any leftover 'preliminary / review continuing' notice; this is the final answer."""
    return re.sub(r"(?im)^.*preliminary\W+.*(continuing|in progress).*(?:\n|$)", "", answer)


def unsupported_numbers(answer: str, evidence: dict) -> list[str]:
    """Statistics in the answer that do not appear in the retrieved data."""
    blob = _norm(json.dumps(evidence, ensure_ascii=False))
    found = []
    for pattern in NUMBER_PATTERNS:
        for match in re.finditer(pattern, answer, re.I):
            token = match.group(0)
            if _norm(token) not in blob and token not in found:
                found.append(token)
    return found


def redact_numbers(answer: str, tokens: list[str]) -> str:
    """Mechanical repair: replace statistics that are not in the retrieved data with a visible marker."""
    for token in sorted(tokens, key=len, reverse=True):
        answer = answer.replace(token, "[value not in retrieved data]")
    return answer

def verify(answer: str, evidence: dict) -> dict:
    """citation-verification + quality-control checks against the retrieved data."""
    papers = {p["pmid"]: p for p in evidence["papers"]}
    dois = {(p["doi"] or "").lower() for p in evidence["papers"]} - {""}
    ncts = {t["nct_id"] for t in evidence["trials"]}
    issues = []

    for pmid in sorted(set(re.findall(r"PMID[:\s]*([0-9]{6,9})", answer))):
        if pmid not in papers:
            issues.append({"severity": "high", "issue": f"PMID {pmid} was not in the retrieved data (possible fabricated citation)"})
    for nct in sorted(_ncts(answer)):
        if nct not in ncts:
            issues.append({"severity": "high", "issue": f"{nct} was not in the retrieved trial data"})
    for doi in sorted(_dois(answer)):
        if doi not in dois:
            issues.append({"severity": "medium", "issue": f"DOI {doi} was not in the retrieved data"})
    for line in answer.splitlines():
        for pmid in re.findall(r"PMID[:\s]*([0-9]{6,9})", line):
            paper = papers.get(pmid)
            if paper and paper["year"] and paper["year"] not in line:
                issues.append({"severity": "medium", "issue": f"PMID {pmid}: year {paper['year']} not shown next to the citation"})
    for token in unsupported_numbers(answer, evidence):
        issues.append({"severity": "high", "issue": f"Statistic '{token}' is not in the retrieved data; replaced with a marker", "token": token})
    if "evidence searched on" not in answer.lower():
        issues.append({"severity": "medium", "issue": "Missing 'Evidence searched on <date>'"})
    if not re.search(r"drug landscape", answer, re.I):
        issues.append({"severity": "high", "issue": "Treatment Drug Landscape section missing"})
    if not evidence["papers"]:
        issues.append({"severity": "high", "issue": "No PubMed evidence was retrieved"})
    high = [i for i in issues if i["severity"] == "high"]
    return {"status": "NEEDS_REPAIR" if high else "PASS", "issues": issues,
            "cited_pmids": len(set(re.findall(r"PMID[:\s]*([0-9]{6,9})", answer))),
            "cited_trials": len(_ncts(answer)),
            "cited_dois": len(_dois(answer))}


# ---------------------------------------------------------------- orchestration
def run_pipeline(query: str) -> dict | None:
    """Return the result dict, or None if the question is not a clinical research question."""
    state = {"query": query, "answer_status": "INITIAL", "timestamps": {"request_started": datetime.now().isoformat(timespec="seconds")}}
    t0 = time.time()
    model = server.build_model()

    plan = plan_query(query, model)
    if not plan["is_clinical"]:
        return None
    state.update(intent=plan["intent"], plan=plan)
    state["timestamps"]["plan_done_s"] = round(time.time() - t0, 1)

    retrieved = retrieve(plan)
    state["timestamps"]["retrieval_done_s"] = round(time.time() - t0, 1)
    state["retrieval_seconds_per_source"] = retrieved["timings"]

    abstract_chars = 500
    evidence = compact_state(retrieved["raw"], abstract_chars)
    while _tokens(json.dumps(evidence, ensure_ascii=False)) > DATA_TOKEN_BUDGET and abstract_chars > 80:
        abstract_chars -= 80
        evidence = compact_state(retrieved["raw"], abstract_chars)
    state["evidence"] = evidence

    state["answer_status"] = "ENRICHING"
    answer = write_answer(query, plan, evidence, model)
    state["timestamps"]["answer_done_s"] = round(time.time() - t0, 1)

    answer = strip_preliminary(answer)
    qc = verify(answer, evidence)
    bad = [i["token"] for i in qc["issues"] if "token" in i]
    if bad:  # one bounded repair: redact the unsupported statistics, then re-check
        answer = redact_numbers(answer, bad)
        qc = verify(answer, evidence)
        qc["repaired"] = [f"redacted: {t}" for t in bad]
    state["qc"] = qc
    footer = [f"\n\n---\n**Verification:** {qc['status']} - {qc['cited_pmids']} PMIDs, {qc['cited_dois']} DOIs and {qc['cited_trials']} trial IDs checked against retrieved data."]
    footer += [f"- Repaired: {r}" for r in qc.get("repaired", [])]
    footer += [f"- [{i['severity']}] {i['issue']}" for i in qc["issues"]]
    footer.append(f"*Searched PubMed, ClinicalTrials.gov and the web on {datetime.now():%Y-%m-%d}.*")
    answer += "\n".join(footer)

    state["answer_status"] = "FINAL"
    state["timestamps"]["final_s"] = round(time.time() - t0, 1)
    return {"answer": answer, "state": state}


def main() -> None:
    query = " ".join(sys.argv[1:]) or "What is the latest evidence on tolebrutinib in multiple sclerosis?"
    try:
        result = run_pipeline(query)
    except RuntimeError as error:
        print(f"FAILED: {error}")
        return
    if result is None:
        print("Not a clinical research question; the normal assistant would handle it.")
        return
    print(result["answer"])
    state = result["state"]
    print("\n\n=== RESEARCH STATE (summary) ===")
    print(json.dumps({
        "intent": state["intent"], "papers": len(state["evidence"]["papers"]), "trials": len(state["evidence"]["trials"]),
        "web_results": len(state["evidence"]["web"]), "retrieval_errors": state["evidence"]["retrieval_errors"],
        "retrieval_seconds_per_source": state["retrieval_seconds_per_source"],
        "timestamps": state["timestamps"], "qc": state["qc"]["status"], "answer_status": state["answer_status"],
    }, indent=2))
    out = Path(__file__).parent / "demo_output"
    out.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", query.lower())[:50].strip("-")
    (out / f"{slug}.md").write_text(f"# {query}\n\n{result['answer']}\n", encoding="utf-8")
    (out / f"{slug}.state.json").write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
