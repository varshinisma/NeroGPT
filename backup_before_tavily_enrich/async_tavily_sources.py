"""PubMed, Europe PMC, ClinicalTrials.gov and the DailyMed labels reached THROUGH Tavily (switch SOURCES_VIA_TAVILY=on in .env).

Each function has the same name, arguments and JSON text back as its twin in async_tools, so the rest of the pipeline is unchanged. Tavily is asked to search only the source's own web domain and returns page text
(title, link, an excerpt); the identifiers (PMID, NCT number, DailyMed set id) are read from the page link. What a web search cannot give stays empty: authors, journal, study type, trial phase and status,
label dosing sections. The searches are limited by what Tavily's index holds, not by PubMed's own query language, so the Boolean field tags are dropped from the query first."""
from __future__ import annotations

import asyncio
import json
import re

import async_core
import async_web
import clinical_tools as ct

PUBMED = ["pubmed.ncbi.nlm.nih.gov"]
EUROPEPMC = ["europepmc.org", "pubmed.ncbi.nlm.nih.gov"]
TRIALS = ["clinicaltrials.gov"]
DAILYMED = ["dailymed.nlm.nih.gov"]


def plain_query(query: str) -> str:
    """A PubMed/Boolean query as plain words: field tags, operators, quotes and brackets removed."""
    text = re.sub(r"\[[^\]]*\]", " ", str(query or ""))
    text = re.sub(r"\b(AND|OR|NOT)\b|[()\"*]", " ", text)
    text = " ".join(text.split())
    if len(text.split()) < 3:
        question = " ".join(async_web.QUESTION.get().split())
        text = f"{question} {text}".strip()
    return text[:300]


async def _search(query: str, domains: list, count: int) -> list[dict]:
    """Tavily restricted to the domains; [] when there is no key, no quota or an error (the first quota refusal is printed once in the terminal)."""
    import os
    if not os.getenv("TAVILY_API_KEY"):
        return []
    try:
        await async_core.limiter.acquire("tavily")
        depth = os.getenv("TAVILY_DEPTH", "advanced").strip() or "advanced"
        found = (await async_web._tavily().search(query, max_results=max(1, min(int(count), 20)), search_depth=depth, include_domains=domains)).get("results", [])
        return [r for r in found if r.get("url")]
    except Exception as error:
        if async_web._refused(error):
            async_web._mark_exhausted(error)
        else:
            print(f"[tavily-sources] search failed: {type(error).__name__}: {str(error)[:100]}", flush=True)
        return []


def _year(*texts) -> str | None:
    for text in texts:
        found = re.search(r"\b(20[0-3]\d|19[7-9]\d)\b", str(text or ""))
        if found:
            return found.group(1)
    return None


def _paper(item: dict) -> dict | None:
    url = item.get("url") or ""
    found = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d{5,9})", url) or re.search(r"europepmc\.org/(?:abstract|article)/MED/(\d{5,9})", url)
    if not found:
        return None
    pmid = found.group(1)
    title = re.sub(r"\s*[-|]\s*(PubMed|Europe PMC|PMC)\s*$", "", item.get("title") or "").strip()
    text = re.sub(r"\s+", " ", item.get("content") or "").strip()
    return {"title": title, "authors": [], "journal": None, "publication_date": None, "year": _year(text, title), "study_type": [], "pmid": pmid, "doi": None, "pmcid": None,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", "abstract_excerpt": text[:4000] or None}


async def _papers(query: str, domains: list, max_results: int) -> str:
    items = await _search(plain_query(query), domains, max_results)
    seen, results = set(), []
    for item in items:
        paper = _paper(item)
        if paper and paper["pmid"] not in seen:
            seen.add(paper["pmid"])
            results.append(paper)
    if not results:
        return json.dumps({"results": [], "note": "No results through Tavily.", "searched_at": ct._now()})
    return json.dumps({"results": results[:max_results], "searched_at": ct._now()}, ensure_ascii=False)


async def asearch_pubmed(query: str, max_results: int = 8, years_back: int = 2, sort: str = "date") -> str:
    return await _papers(query, PUBMED, max_results)


async def asearch_pubmed_quick(query: str, max_results: int = 6, years_back: int = 3, sort: str = "relevance") -> str:
    return await _papers(query, PUBMED, max_results)


async def asearch_europepmc_quick(query: str, max_results: int = 6, years_back: int = 3, sort: str = "relevance", fulltext: bool = False) -> str:
    return await _papers(query, EUROPEPMC, max_results)


async def asearch_clinical_trials(condition, intervention: str = "", status: str = "", max_results: int = 8, drug_only: bool = False) -> str:
    base = ct._ctgov_condition_text(condition)
    if not base:
        return json.dumps({"error": "ClinicalTrials.gov search not run: no usable condition terms"})
    query = f"clinical trial {base} {intervention or ''} {status.replace('_', ' ').lower() if status else ''}".strip()
    items = await _search(query, TRIALS, max_results)
    results, seen = [], set()
    for item in items:
        found = re.search(r"clinicaltrials\.gov/(?:study|ct2/show)/(NCT\d{8})", item.get("url") or "")
        if not found or found.group(1) in seen:
            continue
        nct = found.group(1)
        seen.add(nct)
        text = re.sub(r"\s+", " ", item.get("content") or "").strip()
        results.append({"nct_id": nct, "title": re.sub(r"\s*[|-]\s*ClinicalTrials\.gov\s*$", "", item.get("title") or "").strip(), "phase": None, "ages": None, "conditions": [base][:1],
                        "design": None, "interventions": [], "drug_interventions": [], "recruitment_status": None, "why_stopped": None, "start_date": None, "estimated_completion": None,
                        "last_update": None, "primary_endpoint": None, "sponsor": None, "results_posted": False, "brief_summary": text[:1500], "url": f"https://clinicaltrials.gov/study/{nct}"})
    if not results:
        return json.dumps({"results": [], "note": "No studies found through Tavily.", "searched_at": ct._now()})
    return json.dumps({"results": results[:max_results], "searched_at": ct._now(), "note": "Found through Tavily: status, phase and interventions are not available; read the summary."}, ensure_ascii=False)


def _set_id(url: str) -> str | None:
    found = re.search(r"setid=([0-9a-fA-F-]{36})", url or "") or re.search(r"/dailymed/(?:lookup\.cfm\?setid=|drugInfo\.cfm\?setid=)([0-9a-fA-F-]{36})", url or "")
    return found.group(1) if found else None


async def afda_label_link(generic: str) -> dict | None:
    name = re.sub(r"[^A-Za-z0-9 \-]", "", str(generic or "")).strip()
    if len(name) < 4:
        return None
    for item in await _search(f"{name} prescribing information label", DAILYMED, 5):
        set_id = _set_id(item.get("url"))
        if set_id and name.split()[0].lower() in f"{item.get('title') or ''} {item.get('content') or ''}".lower():
            return {"generic": name, "generic_name": None, "set_id": set_id, "url": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}", "brand": None, "boxed_warning": None, "label_date": None}
    return None


async def asearch_fda_labels(condition_terms: list[str], drug_names: list[str] | None = None, max_drugs: int = 8) -> str:
    condition = " ".join(str(t) for t in condition_terms or [])
    names = [str(d) for d in drug_names or []]
    queries = [f"{n} prescribing information indications dosage" for n in names] or [f"drugs indicated for {condition} prescribing information"]
    groups = await asyncio.gather(*[_search(q, DAILYMED, 3) for q in queries])
    records, seen = [], set()
    for items in groups:
        for item in items:
            set_id = _set_id(item.get("url"))
            if not set_id or set_id in seen:
                continue
            seen.add(set_id)
            title = re.sub(r"\s*[|-]\s*(DailyMed|NIH).*$", "", item.get("title") or "").strip()
            text = re.sub(r"\s+", " ", item.get("content") or "").strip()
            records.append({"generic": title.title(), "brands": [], "jurisdiction": "FDA", "indication_excerpt": text[:300], "matches_condition": bool(condition) and condition.lower().split()[0] in text.lower(),
                            "indication_statement": None, "dose_statement": None, "dose_general": None, "route": None, "label_extras": {}, "safety_text": text[:600] or None, "label_date": None,
                            "set_id": set_id, "url": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}", "source_id": f"fda:{set_id}"})
    return json.dumps({"results": records[:max_drugs + 4 * len(names)], "searched_terms": condition_terms, "searched_drugs": names, "searched_at": ct._now(),
                       "note": "Found through Tavily (DailyMed pages): dosing sections are not parsed."}, ensure_ascii=False)
