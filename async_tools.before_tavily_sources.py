"""Async data-source searches for the NeuroGPT pipeline: PubMed, Europe PMC, ClinicalTrials.gov, similar articles and the FDA labels. Each function is the async twin of the function with the same name in
clinical_tools.py (without the leading 'a'): same arguments, same JSON text back. The parsing is shared (clinical_tools._parse_pubmed_xml, _parse_trials, _parse_europepmc, _record_from_label), so
the two can not drift apart; only the waiting is different: every request is an `await` on async_core.aget, and requests that do not depend on each other run together with asyncio.gather."""
from __future__ import annotations

import asyncio
import json
import re
import xml.etree.ElementTree as ET
from datetime import date

import httpx

import async_core
import clinical_tools as ct

FETCH_ERRORS = (httpx.HTTPError, TimeoutError, ValueError, KeyError, TypeError, ET.ParseError)


async def _abstracts(pmids: list[str]) -> dict[str, str]:
    if not pmids:
        return {}
    root = ET.fromstring(await async_core.aget(f"{ct.EUTILS}/efetch.fcgi", {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}))
    found: dict[str, str] = {}
    for article in root.findall(".//PubmedArticle"):
        pmid = ct._text(article.find(".//MedlineCitation/PMID"))
        parts = []
        for node in article.findall(".//Abstract/AbstractText"):
            label = node.get("Label")
            body = ct._text(node)
            parts.append(f"{label}: {body}" if label else body)
        found[pmid] = " ".join(parts)[:4000]
    return found


async def asearch_pubmed(query: str, max_results: int = 8, years_back: int = 2, sort: str = "date") -> str:
    """Async twin of clinical_tools.search_pubmed. The summaries and the abstracts do not depend on each other, so they are asked for at the same time."""
    max_results = max(1, min(int(max_results), 15))
    params = {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json", "sort": "pub_date" if sort == "date" else "relevance"}
    if years_back and int(years_back) > 0:
        params.update({"datetype": "pdat", "mindate": str(date.today().year - int(years_back)), "maxdate": str(date.today().year)})
    try:
        ids = (await async_core.aget_json(f"{ct.EUTILS}/esearch.fcgi", params))["esearchresult"].get("idlist", [])
        if not ids:
            return json.dumps({"results": [], "note": "No PubMed results. Try broader terms or a larger years_back.", "searched_at": ct._now()})
        summary_json, abstracts = await asyncio.gather(
            async_core.aget_json(f"{ct.EUTILS}/esummary.fcgi", {"db": "pubmed", "id": ",".join(ids), "retmode": "json"}), _abstracts(ids))
        summaries = summary_json["result"]
    except FETCH_ERRORS as error:
        return json.dumps({"error": f"PubMed search failed: {error}"})
    results = []
    for pmid in ids:
        item = summaries.get(pmid, {})
        doi = next((a.get("value") for a in item.get("articleids", []) if a.get("idtype") == "doi"), None)
        pmcid = next((a.get("value") for a in item.get("articleids", []) if a.get("idtype") == "pmc"), None)
        pubdate = item.get("pubdate", "")
        year = re.search(r"(19|20)\d{2}", pubdate)
        results.append({
            "title": item.get("title"), "authors": [a.get("name") for a in item.get("authors", [])[:6]],
            "journal": item.get("fulljournalname") or item.get("source"), "publication_date": pubdate, "year": year.group(0) if year else None,
            "study_type": item.get("pubtype", []), "pmid": pmid, "doi": doi, "pmcid": pmcid, "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "abstract_excerpt": abstracts.get(pmid) or None})
    return json.dumps({"results": results, "searched_at": ct._now()}, ensure_ascii=False)


async def asearch_pubmed_quick(query: str, max_results: int = 6, years_back: int = 3, sort: str = "relevance") -> str:
    """Async twin of clinical_tools.search_pubmed_quick (esearch, then efetch)."""
    params = {"db": "pubmed", "term": query, "retmax": max(1, min(int(max_results), 15)), "retmode": "json", "sort": "pub_date" if sort == "date" else "relevance"}
    if years_back and int(years_back) > 0:
        params.update({"datetype": "pdat", "mindate": str(date.today().year - int(years_back)), "maxdate": str(date.today().year)})
    try:
        ids = (await async_core.aget_json(f"{ct.EUTILS}/esearch.fcgi", params))["esearchresult"].get("idlist", [])
        if not ids:
            return json.dumps({"results": [], "note": "No PubMed results.", "searched_at": ct._now()})
        root = ET.fromstring(await async_core.aget(f"{ct.EUTILS}/efetch.fcgi", {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}))
    except FETCH_ERRORS as error:
        return json.dumps({"error": f"PubMed search failed: {error}"})
    by_id = ct._parse_pubmed_xml(root)
    return json.dumps({"results": [by_id[i] for i in ids if i in by_id], "searched_at": ct._now()}, ensure_ascii=False)


async def asearch_europepmc_quick(query: str, max_results: int = 6, years_back: int = 3, sort: str = "relevance", fulltext: bool = False) -> str:
    """Async twin of clinical_tools.search_europepmc_quick: one call; PubMed is the fallback when Europe PMC fails (not for a full-text search)."""
    first_year = date.today().year - int(years_back) if years_back else 1900
    q = f"({query}) AND SRC:MED AND FIRST_PDATE:[{first_year}-01-01 TO {date.today().year}-12-31]"
    if fulltext:
        q = f"BODY:({query}) AND SRC:MED AND OPEN_ACCESS:y AND FIRST_PDATE:[{first_year}-01-01 TO {date.today().year}-12-31]"
    if sort == "date":
        q += " sort_date:y"
    params = {"query": q, "resultType": "core", "format": "json", "pageSize": max(1, min(int(max_results), 15))}
    try:
        data = (await async_core.aget_json("https://www.ebi.ac.uk/europepmc/webservices/rest/search", params))["resultList"]["result"]
    except FETCH_ERRORS:
        if fulltext:
            return json.dumps({"error": "Europe PMC full-text search failed"})
        return await asearch_pubmed_quick(query, max_results, years_back, sort)
    return json.dumps({"results": ct._parse_europepmc(data), "searched_at": ct._now()}, ensure_ascii=False)


async def asearch_clinical_trials(condition, intervention: str = "", status: str = "", max_results: int = 8, drug_only: bool = False) -> str:
    """Async twin of clinical_tools.search_clinical_trials."""
    max_results = max(1, min(int(max_results), 20))
    condition_query = ct._ctgov_condition_text(condition)
    if not condition_query:
        return json.dumps({"error": "ClinicalTrials.gov search not run: no usable condition terms", "query": {"source": "ClinicalTrials.gov", "endpoint": ct.CTGOV}})
    allowed_status = {"RECRUITING", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING", "COMPLETED", "TERMINATED", "WITHDRAWN", "SUSPENDED", "ENROLLING_BY_INVITATION", "UNKNOWN"}
    status_values = [x.strip().upper() for x in str(status or "").split(",") if x.strip()]
    if status_values and any(x not in allowed_status for x in status_values):
        return json.dumps({"error": "ClinicalTrials.gov search not run: invalid overall-status filter", "query": {"source": "ClinicalTrials.gov", "condition": condition_query, "status": status}})
    params = {"query.cond": condition_query, "pageSize": max_results, "sort": "LastUpdatePostDate:desc", "format": "json"}
    if intervention:
        params["query.intr"] = intervention
    if drug_only:
        params["query.term"] = "AREA[InterventionType](DRUG OR BIOLOGICAL)"
    if status:
        params["filter.overallStatus"] = ",".join(status_values)
    query_trace = {"source": "ClinicalTrials.gov", "endpoint": ct.CTGOV, "params": params}
    try:
        studies = (await async_core.aget_json(ct.CTGOV, params)).get("studies", [])
    except FETCH_ERRORS as error:
        return json.dumps({"error": f"ClinicalTrials.gov search failed: {error}", "query_trace": [{**query_trace, "status": "failed", "reason": f"{type(error).__name__}: {str(error)[:160]}"}]})
    if not studies:
        return json.dumps({"results": [], "note": "No studies matched this completed query.", "query_trace": [{**query_trace, "status": "no_match"}], "searched_at": ct._now()})
    return json.dumps({"results": ct._parse_trials(studies), "query_trace": [{**query_trace, "status": "found"}], "searched_at": ct._now(),
                       "note": "Status is as of last_update. Ongoing does not mean effective."}, ensure_ascii=False)


async def asimilar_articles(pmids: list, per_seed: int = 4, max_results: int = 10, years_back: int = 5) -> str:
    """Async twin of clinical_tools.similar_articles."""
    seeds = [str(p) for p in dict.fromkeys(pmids) if p]
    if not seeds:
        return json.dumps({"results": [], "note": "no seed papers", "searched_at": ct._now()})
    try:
        linked = await async_core.aget_json(f"{ct.EUTILS}/elink.fcgi", {"dbfrom": "pubmed", "db": "pubmed", "cmd": "neighbor_score", "retmode": "json", "id": seeds})
        best: dict[str, float] = {}
        for linkset in linked.get("linksets", []):
            links = next((db.get("links", []) for db in linkset.get("linksetdbs", []) if db.get("linkname") == "pubmed_pubmed"), [])
            taken = 0
            for link in links:
                pmid = str(link.get("id"))
                if pmid in seeds:
                    continue
                best[pmid] = max(best.get(pmid, 0), float(link.get("score", 0)))
                taken += 1
                if taken >= per_seed:
                    break
    except FETCH_ERRORS as error:
        return json.dumps({"error": f"PubMed similar-articles search failed: {error}"})
    ids = sorted(best, key=best.get, reverse=True)[:max(1, min(int(max_results), 15))]
    if not ids:
        return json.dumps({"results": [], "note": "no similar articles", "searched_at": ct._now()})
    return await asearch_pubmed_quick(" OR ".join(f"{i}[uid]" for i in ids), max_results=len(ids), years_back=years_back, sort="relevance")


# ------------------------------------------------------------------ FDA drug labels
async def _fda_get(params: dict) -> dict | None:
    """One openFDA call. 404 means 'nothing matched' (a normal answer), not an error."""
    try:
        return await async_core.aget_json(ct.FDA_LABEL_API, params)
    except httpx.HTTPStatusError as error:
        if error.response.status_code == 404:
            return None
        raise


async def _fda_records(generic: str, stems: list[str], limit: int = 1) -> list[dict]:
    data = await _fda_get({"search": f'openfda.generic_name.exact:"{generic}"', "limit": limit})
    out, seen = [], set()
    for label in (data or {}).get("results", []):
        record = ct._record_from_label(label, generic, stems)
        key = (record["indication_excerpt"][:120], tuple(record["brands"]))
        if key not in seen:
            seen.add(key)
            out.append(record)
    return out


async def afda_label_link(generic: str) -> dict | None:
    """Async twin of clinical_tools.fda_label_link: the DailyMed label of ONE named drug."""
    name = re.sub(r"[^A-Za-z0-9 \-]", "", str(generic or "")).strip()
    if len(name) < 4:
        return None
    data = await _fda_get({"search": f'openfda.generic_name:"{name}"', "limit": 1})
    label = ((data or {}).get("results") or [None])[0]
    generic_names = " ".join(((label or {}).get("openfda") or {}).get("generic_name") or []).lower()
    via_brand = False
    if not label or not label.get("set_id") or name.split()[0].lower() not in generic_names:
        data = await _fda_get({"search": f'openfda.brand_name:"{name}"', "limit": 1})
        label = ((data or {}).get("results") or [None])[0]
        brands = [b.lower() for b in ((label or {}).get("openfda") or {}).get("brand_name") or []]
        if not label or not label.get("set_id") or name.lower() not in brands:
            return None
        via_brand = True
        generic_names = " ".join(label.get("openfda", {}).get("generic_name") or []).lower()
    set_id = label.get("set_id")
    boxed = re.sub(r"\s+", " ", " ".join(label.get("boxed_warning") or [])).strip()
    when = str(label.get("effective_time") or "")
    return {"generic": name, "generic_name": (label.get("openfda", {}).get("generic_name") or [None])[0] if via_brand else None, "set_id": set_id,
            "url": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}",
            "brand": (label.get("openfda", {}).get("brand_name") or [None])[0], "boxed_warning": boxed or None,
            "label_date": f"{when[:4]}-{when[4:6]}-{when[6:8]}" if len(when) == 8 else None}


async def asearch_fda_labels(condition_terms: list[str], drug_names: list[str] | None = None, max_drugs: int = 8) -> str:
    """Async twin of clinical_tools.search_fda_labels: the generic names found for the condition, then every name's label records at the same time."""
    stems = ct.condition_stems(condition_terms)
    if not stems and not drug_names:
        return json.dumps({"error": f"FDA label search not run: no usable condition terms in {condition_terms!r}"})
    try:
        names: list[str] = []
        used = stems
        if stems:
            subsets = [stems] + ([stems[:i] + stems[i + 1:] for i in range(len(stems))] if len(stems) > 1 else [])
            if len(stems) > 2:
                subsets += [[s] for s in stems]
            for subset in subsets:
                if not subset:
                    continue
                query = " AND ".join(f"indications_and_usage:{s}*" for s in subset)
                data = await _fda_get({"search": query, "count": "openfda.generic_name.exact", "limit": 25})
                found = [r["term"] for r in (data or {}).get("results", [])]
                if found:
                    used = subset
                    for term in found:
                        base = " ".join(w for w in term.split() if w not in ct._LABEL_NOISE) or term
                        if base not in names:
                            names.append(base)
                    break
        asked = [d.upper() for d in drug_names or []]
        names = asked + [n for n in names if n.upper() not in asked][:max_drugs]
        asked_set = set(asked)
        groups = await asyncio.gather(*[_fda_records(n, used, 4 if n in asked_set else 1) for n in names])
        records = [r for group in groups for r in group]
    except FETCH_ERRORS as error:
        return json.dumps({"error": f"FDA label search failed: {error}"})
    return json.dumps({"results": records[:max_drugs + 4 * len(drug_names or [])], "searched_terms": used, "searched_drugs": drug_names or [],
                       "searched_at": ct._now(), "note": "empty results = searched, none found"}, ensure_ascii=False)
