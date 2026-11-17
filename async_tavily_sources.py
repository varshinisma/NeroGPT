"""PubMed, Europe PMC, ClinicalTrials.gov and the DailyMed labels reached THROUGH Tavily (switch SOURCES_VIA_TAVILY=on in .env).

Each function has the same name, arguments and JSON text back as its twin in async_tools, so the rest of the pipeline is unchanged. Tavily is asked to search only the source's own web domain and returns page text
(title, link, an excerpt); the identifiers (PMID, NCT number, DailyMed set id) are read from the page link.

Tavily does the SEARCHING (a meaning-based search over that site's pages); the source's own API is then used only to READ the records Tavily found, by identifier. That second step is a lookup, not a search:
  * TAVILY_SOURCES_SEMANTIC (default on)  the query is sent as the person's question plus the focus of the search, Tavily is asked for more pages than needed and the pages whose text is closest in
                                          meaning to the question are kept (the same selection the web searches use, async_web._tavily_semantic);
  * TAVILY_SOURCES_ENRICH   (default on)  every PMID is read from PubMed (title, authors, journal, study type, DOI, PMC id, the real abstract), every NCT number from ClinicalTrials.gov (phase, status,
                                          design, interventions) and every DailyMed set id from openFDA (indications, dosing, warnings). Without it the records hold only what a web page excerpt shows, so there are
                                          no figures (they need the PMC id), no full text and no author or study design for the Key studies;
  * TAVILY_SOURCES_FALLBACK (default on)  when Tavily gives nothing usable (no key, quota finished, an error, no page of that source), the source's own API answers instead;
  * TAVILY_ENRICH_TIMEOUT_S (default 6)   the time the identifier lookups get; a lookup that is late leaves the Tavily-only record (never a missing record).
Set any of them to off to go back to the plainer behaviour."""
from __future__ import annotations

import asyncio
import json
import os
import re
import drug_reconcile as dr
import xml.etree.ElementTree as ET
from datetime import date

import async_core
import async_web
import clinical_tools as ct

PUBMED = ["pubmed.ncbi.nlm.nih.gov"]
EUROPEPMC = ["europepmc.org", "pubmed.ncbi.nlm.nih.gov"]
TRIALS = ["clinicaltrials.gov"]
DAILYMED = ["dailymed.nlm.nih.gov"]

DIRECT: dict = {}      # name -> the source's own API twin (async_tools fills it before it puts the Tavily versions in their place); used by the fallback


def _on(name: str, default: str = "on") -> bool:
    return os.getenv(name, default).strip().lower() not in {"off", "0", "false", "no"}


def _wait_s() -> float:
    try:
        return max(1.0, float(os.getenv("TAVILY_ENRICH_TIMEOUT_S", "6")))
    except ValueError:
        return 6.0


def plain_query(query: str) -> str:
    """A PubMed/Boolean query as plain words: field tags, operators, quotes and brackets removed."""
    text = re.sub(r"\[[^\]]*\]", " ", str(query or ""))
    text = re.sub(r"\b(AND|OR|NOT)\b|[()\"*]", " ", text)
    text = " ".join(text.split())
    if len(text.split()) < 3:
        question = " ".join(async_web.QUESTION.get().split())
        text = f"{question} {text}".strip()
    return text[:300]


def _focus(query: str, limit: int = 110) -> str:
    """The query as plain words, short enough for async_web.natural_query to put it after the question ('<question> Focus: <words>')."""
    text = plain_query(query)
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] or text[:limit]


async def _direct(name: str, *args, **kwargs):
    """The source's own API answer for the same call, or None when the fallback is off or the direct twin is not registered."""
    function = DIRECT.get(name)
    if function is None or not _on("TAVILY_SOURCES_FALLBACK"):
        return None
    print(f"[tavily-sources] {name}: nothing usable from Tavily, the source's own API answers", flush=True)
    return await function(*args, **kwargs)


async def _search(query: str, domains: list, count: int, semantic: bool | None = None) -> list[dict]:
    """Pages of the domains for the query: [{url, title, content}]; [] when there is no key, no quota or an error (the first quota refusal is printed once in the terminal).
    Semantic (default: TAVILY_SOURCES_SEMANTIC): the question as a sentence, more pages than needed, the pages closest in meaning to the question kept (async_web._tavily_semantic).
    Not semantic: the plain words as they are (used where the NAME of a drug must match, which a sentence would blur)."""
    if not os.getenv("TAVILY_API_KEY"):
        return []
    count = max(1, min(int(count), 20))
    if semantic is None:
        semantic = _on("TAVILY_SOURCES_SEMANTIC")
    if semantic:
        items = await async_web._tavily_semantic(_focus(query), count, 1500, domains)
        return [{"url": x["url"], "title": x.get("title"), "content": x.get("snippet")} for x in items if x.get("url")]
    try:
        await async_core.limiter.acquire("tavily")
        depth = os.getenv("TAVILY_DEPTH", "advanced").strip() or "advanced"
        found = (await async_web._tavily().search(query, max_results=count, search_depth=depth, include_domains=domains)).get("results", [])
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


_BOILERPLATE = re.compile(r"(?i)\b(?:these highlights|initial u\.?s\.? approval|package label|principal display panel|revised:)\b")
_HEADING = re.compile(r"(?i)^(?:highlights?(?: of prescribing information)?|(?:full |complete )?prescribing information|(?:complete |full )?boxed warning|boxed warning|indications?(?: and usage)?|dosage(?: and administration)?|description|warnings?(?: and precautions)?|contraindications|adverse reactions|how supplied|clinical pharmacology|drug label|label|dailymed label|spl|dailymed|package insert|medication guide|patient information)$")
_FORM = r"(?:tablets?|capsules?|injection|solution|suspension|cream|ointment|gel|patch|spray|inhaler|powder|lozenges?|film|syrup|elixir)"


def _clean_name(title) -> str:
    """A drug's name from a DailyMed page title, on ONE line and short; '' when the title is label boilerplate or a block of page text. Some pages are titled with the label's own opening text
    ('these highlights do not include all the information needed to use X tablets safely ... see full prescribing information for X tablets ...'): the product named in that standard sentence is used."""
    text = " ".join(str(title or "").split())
    named = re.search(r"(?i)full prescribing information for ([^.;|]{3,80}?)(?:\.|;|\||$)", text)
    if named:
        text = named.group(1)
    elif _BOILERPLATE.search(text) or len(text) > 80:
        return ""
    text = re.sub(r"\s*[®™]+\s*", " ", text)
    text = re.sub(r"(?i)\s*[-|]\s*(?:DailyMed|NIH).*$", "", text)
    text = " ".join(text.replace("|", "/").split())
    if _HEADING.match(text.strip(" .:-")) or _BOILERPLATE.search(text):      # a section heading of the page, not a drug
        return ""
    return text[:80]


def _name_from_indication(label: dict, stems=()) -> str:
    """(stems: the condition's stems.) The product named in the label's own indication sentence ('1.3 Migraine Divalproex sodium delayed-release tablets, USP are indicated for ...'): from the drug's name to the dosage form."""
    text = " ".join(" ".join((label or {}).get("indications_and_usage") or []).split())
    m = re.search(rf"((?:[\w\-]+\s+){{0,6}}?[\w\-]*(?:{_FORM}))(?:,?\s*USP)?,?\s+(?:are|is)\s+indicated", text, re.I)
    if not m:
        return ""
    words = re.sub(r"^\s*\d+(?:\.\d+)*\s*", "", m.group(1)).split()
    while len(words) > 2 and any(st and st.lower() in words[0].lower() for st in stems or ()):      # a heading word before the name: the condition ("Migraine")
        words = words[1:]
    return _clean_name(" ".join(words[-5:]))


def _label_generic(label: dict, fallback: str = "", stems=()) -> str:
    """The generic name for a label read from openFDA: the label's own generic name (a combination product's ingredients joined by commas, as openFDA writes them), else the page's cleaned name,
    else the brand name; never text of the label itself."""
    openfda = (label or {}).get("openfda") or {}
    names = [" ".join(str(n).split()) for n in (openfda.get("generic_name") or []) if str(n).strip()]
    if names:
        return ", ".join(names)[:120]
    return _clean_name(fallback) or _name_from_indication(label, stems) or _clean_name((openfda.get("brand_name") or [""])[0]) or "DailyMed label"


# ------------------------------------------------------------------ reading the records Tavily found, by identifier (a lookup, not a search)
async def _fetch_pubmed(pmids: list[str]) -> dict:
    """{pmid: full record} from PubMed (one efetch request for all of them; the same parser as the direct search)."""
    root = ET.fromstring(await async_core.aget(f"{ct.EUTILS}/efetch.fcgi", {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}))
    return ct._parse_pubmed_xml(root)


def _merge(paper: dict, full: dict | None) -> dict:
    """The PubMed record over the Tavily one (a field PubMed leaves empty keeps what the page showed)."""
    if not full:
        return paper
    merged = {**paper, **{k: v for k, v in full.items() if v not in (None, "", [])}}
    merged["abstract_excerpt"] = full.get("abstract_excerpt") or paper.get("abstract_excerpt")
    return merged


async def _enrich_papers(papers: list[dict]) -> tuple[list[dict], set]:
    """(papers with their PubMed records, the PMIDs that were read). A failed or late lookup leaves the Tavily-only papers."""
    if not papers or not _on("TAVILY_SOURCES_ENRICH"):
        return papers, set()
    try:
        by_id = await asyncio.wait_for(_fetch_pubmed([p["pmid"] for p in papers]), timeout=_wait_s())
    except Exception as error:
        print(f"[tavily-sources] PubMed record lookup skipped: {type(error).__name__}: {str(error)[:100]}", flush=True)
        return papers, set()
    return [_merge(p, by_id.get(p["pmid"])) for p in papers], set(by_id)


def _year_of(paper: dict) -> int:
    return int(paper["year"]) if str(paper.get("year") or "").isdigit() else 0


async def _papers(query: str, domains: list, max_results: int, direct_name: str, direct_args: tuple, direct_kwargs: dict | None = None, years_back: int = 0, sort: str = "relevance") -> str:
    items = await _search(plain_query(query), domains, max_results)
    seen, papers = set(), []
    for item in items:
        paper = _paper(item)
        if paper and paper["pmid"] not in seen:
            seen.add(paper["pmid"])
            papers.append(paper)
    if papers:
        papers, read = await _enrich_papers(papers)
        if years_back and read:      # the direct search limits the years; Tavily cannot, so the limit is applied to the papers whose PubMed record gave a year
            cutoff = date.today().year - int(years_back)
            papers = [p for p in papers if p["pmid"] not in read or not _year_of(p) or _year_of(p) >= cutoff]
        if sort == "date" and read:
            papers.sort(key=_year_of, reverse=True)
    if not papers:
        fallback = await _direct(direct_name, *direct_args, **(direct_kwargs or {}))
        if fallback is not None:
            return fallback
        return json.dumps({"results": [], "note": "No results through Tavily.", "searched_at": ct._now()})
    return json.dumps({"results": papers[:max_results], "searched_at": ct._now()}, ensure_ascii=False)


async def asearch_pubmed(query: str, max_results: int = 8, years_back: int = 2, sort: str = "date") -> str:
    return await _papers(query, PUBMED, max_results, "asearch_pubmed", (query, max_results, years_back, sort), years_back=years_back, sort=sort)


async def asearch_pubmed_quick(query: str, max_results: int = 6, years_back: int = 3, sort: str = "relevance") -> str:
    return await _papers(query, PUBMED, max_results, "asearch_pubmed_quick", (query, max_results, years_back, sort), years_back=years_back, sort=sort)


async def asearch_europepmc_quick(query: str, max_results: int = 6, years_back: int = 3, sort: str = "relevance", fulltext: bool = False) -> str:
    if fulltext and "asearch_europepmc_quick" in DIRECT:      # a search in the BODY of open-access papers is Europe PMC's own feature: a web search of titles and excerpts cannot do it
        return await DIRECT["asearch_europepmc_quick"](query, max_results, years_back, sort, fulltext=True)
    return await _papers(query, EUROPEPMC, max_results, "asearch_europepmc_quick", (query, max_results, years_back, sort), {"fulltext": fulltext}, years_back=years_back, sort=sort)


# ------------------------------------------------------------------ ClinicalTrials.gov
async def _fetch_trials(ncts: list[str]) -> dict:
    """{nct_id: full record} from ClinicalTrials.gov (one request for all of them: filter.ids takes a comma-separated list of NCT numbers)."""
    data = await async_core.aget_json(ct.CTGOV, {"filter.ids": ",".join(ncts), "pageSize": max(1, min(len(ncts), 100)), "format": "json"})
    return {r["nct_id"]: r for r in ct._parse_trials(data.get("studies", [])) if r.get("nct_id")}


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
    read = set()
    if results and _on("TAVILY_SOURCES_ENRICH"):
        try:
            full = await asyncio.wait_for(_fetch_trials([r["nct_id"] for r in results]), timeout=_wait_s())
        except Exception as error:
            print(f"[tavily-sources] trial record lookup skipped: {type(error).__name__}: {str(error)[:100]}", flush=True)
            full = {}
        read = set(full)
        results = [({**full[r["nct_id"]], "brief_summary": r.get("brief_summary")} if r["nct_id"] in full else r) for r in results]      # the registry record, plus the excerpt the page showed
        wanted = [x.strip().upper() for x in str(status or "").split(",") if x.strip()]
        if wanted:      # the direct search filters by status; here the trials whose real status is known are filtered the same way
            results = [r for r in results if r["nct_id"] not in read or str(r.get("recruitment_status") or "").upper() in wanted]
        if drug_only:
            results = [r for r in results if r["nct_id"] not in read or r.get("drug_interventions")]
    if not results:
        fallback = await _direct("asearch_clinical_trials", condition, intervention, status, max_results, drug_only)
        if fallback is not None:
            return fallback
        return json.dumps({"results": [], "note": "No studies found through Tavily.", "searched_at": ct._now()})
    note = ("Found by a meaning-based web search (Tavily); the records were read from ClinicalTrials.gov. Status is as of last_update. Ongoing does not mean effective." if read
            else "Found through Tavily: status, phase and interventions are not available; read the summary.")
    return json.dumps({"results": results[:max_results], "searched_at": ct._now(), "note": note}, ensure_ascii=False)


# ------------------------------------------------------------------ DailyMed labels
def _set_id(url: str) -> str | None:
    found = re.search(r"setid=([0-9a-fA-F-]{36})", url or "") or re.search(r"/dailymed/(?:lookup\.cfm\?setid=|drugInfo\.cfm\?setid=)([0-9a-fA-F-]{36})", url or "")
    return found.group(1) if found else None


async def _label_for_set_id(set_id: str) -> dict | None:
    """The openFDA label with this DailyMed set id (a lookup by identifier); None when openFDA has none."""
    import async_tools as at      # imported when called: async_tools imports this module
    data = await at._fda_get({"search": f'set_id:"{set_id}"', "limit": 1})
    return ((data or {}).get("results") or [None])[0]


async def afda_label_link(generic: str) -> dict | None:
    name = re.sub(r"[^A-Za-z0-9 \-]", "", str(generic or "")).strip()
    if len(name) < 4:
        return None
    for item in await _search(f"{name} prescribing information label", DAILYMED, 5, semantic=False):      # a drug's NAME must match: no sentence around it
        set_id = _set_id(item.get("url"))
        if set_id and name.split()[0].lower() in f"{item.get('title') or ''} {item.get('content') or ''}".lower():
            link = {"generic": name, "generic_name": None, "set_id": set_id, "url": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}", "brand": None, "boxed_warning": None, "label_date": None}
            if _on("TAVILY_SOURCES_ENRICH"):
                try:
                    label = await asyncio.wait_for(_label_for_set_id(set_id), timeout=_wait_s())
                except Exception as error:
                    label = None
                    print(f"[tavily-sources] label lookup skipped: {type(error).__name__}: {str(error)[:100]}", flush=True)
                if label:
                    when = str(label.get("effective_time") or "")
                    boxed = re.sub(r"\s+", " ", " ".join(label.get("boxed_warning") or [])).strip()
                    link.update({"generic_name": ((label.get("openfda") or {}).get("generic_name") or [None])[0], "brand": ((label.get("openfda") or {}).get("brand_name") or [None])[0],
                                 "boxed_warning": boxed or None, "label_date": f"{when[:4]}-{when[4:6]}-{when[6:8]}" if len(when) == 8 else None, **dr.label_facts(label)})
            return link
    return await _direct("afda_label_link", generic)


async def _enrich_labels(records: list[dict], condition_terms: list[str]) -> tuple[list[dict], int]:
    """(the records with their openFDA labels read, how many were read): indications, dosing, warnings. A label that is late or missing leaves the Tavily-only record."""
    stems = ct.condition_stems(condition_terms)

    async def one(record: dict) -> dict:
        label = await _label_for_set_id(record["set_id"])
        if not label:
            return record
        return ct._record_from_label(label, _label_generic(label, record.get("generic") or "", stems), stems)

    tasks = [asyncio.ensure_future(one(r)) for r in records]
    done, pending = await asyncio.wait(tasks, timeout=_wait_s()) if tasks else (set(), set())
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    out, read = [], 0
    for record, task in zip(records, tasks):
        if task in done and not task.cancelled() and task.exception() is None:
            out.append(task.result())
            read += 1
        else:
            out.append(record)
    return out, read


async def asearch_fda_labels(condition_terms: list[str], drug_names: list[str] | None = None, max_drugs: int = 8) -> str:
    condition = " ".join(str(t) for t in condition_terms or [])
    names = [str(d) for d in drug_names or []]
    queries = [f"{n} prescribing information indications dosage" for n in names] or [f"drugs indicated for {condition} prescribing information"]
    groups = await asyncio.gather(*[_search(q, DAILYMED, 3, semantic=bool(not names)) for q in queries])      # a named drug: its name as written; a condition: a meaning-based search
    records, seen = [], set()
    for position, items in enumerate(groups):
        asked = names[position] if names and position < len(names) else ""      # the drug this search was for
        for item in items:
            set_id = _set_id(item.get("url"))
            if not set_id or set_id in seen:
                continue
            seen.add(set_id)
            title = _clean_name(item.get("title")) or _clean_name(asked) or "DailyMed label"      # one short line: the drug name goes into a table cell
            text = re.sub(r"\s+", " ", item.get("content") or "").strip()
            records.append({"generic": title.title(), "brands": [], "jurisdiction": "FDA", "indication_excerpt": text[:300], "matches_condition": bool(condition) and condition.lower().split()[0] in text.lower(),
                            "indication_statement": None, "dose_statement": None, "dose_general": None, "route": None, "label_extras": {}, "safety_text": text[:600] or None, "label_date": None,
                            "set_id": set_id, "url": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}", "source_id": f"fda:{set_id}"})
    records = records[:max_drugs + 4 * len(names)]
    read = 0
    if records and _on("TAVILY_SOURCES_ENRICH"):
        try:
            records, read = await _enrich_labels(records, [str(t) for t in condition_terms or []])
        except Exception as error:
            print(f"[tavily-sources] label lookups skipped: {type(error).__name__}: {str(error)[:100]}", flush=True)
    if not records:
        fallback = await _direct("asearch_fda_labels", condition_terms, drug_names, max_drugs)
        if fallback is not None:
            return fallback
    note = ("Found by a web search (Tavily) of DailyMed pages; the labels were read from openFDA." if read else "Found through Tavily (DailyMed pages): dosing sections are not parsed.")
    return json.dumps({"results": records, "searched_terms": condition_terms, "searched_drugs": names, "searched_at": ct._now(), "note": note}, ensure_ascii=False)
