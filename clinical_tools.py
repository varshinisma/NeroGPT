"""Clinical research tools for the NeuroGPT agent: PubMed and ClinicalTrials.gov.

Both use free public APIs (no key needed) and the standard library only.
Every result carries a publication year / last-update date and an identifier
(PMID, DOI, NCT ID) so answers can be cited and verified.
"""
from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
CTGOV = "https://clinicaltrials.gov/api/v2/studies"
TIMEOUT = 15
USER_AGENT = "NeuroGPT-agent/1.0"


_ncbi_lock = threading.Lock()
_ncbi_last_call = 0.0
NCBI_MIN_GAP = 0.4  # NCBI allows ~3 requests/second without an API key


def _get(url: str, params: dict) -> bytes:
    """GET with spacing for NCBI and retries on rate-limit / temporary server errors."""
    global _ncbi_last_call
    query = urllib.parse.urlencode(params, doseq=True)
    request = urllib.request.Request(f"{url}?{query}", headers={"User-Agent": USER_AGENT})
    for attempt in range(4):
        if url.startswith(EUTILS):
            with _ncbi_lock:  # serialise NCBI calls across threads
                wait = NCBI_MIN_GAP - (time.time() - _ncbi_last_call)
                if wait > 0:
                    time.sleep(wait)
                _ncbi_last_call = time.time()
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503) or attempt == 3:
                raise
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError("unreachable")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _text(element: ET.Element | None) -> str:
    return "".join(element.itertext()).strip() if element is not None else ""


def _abstracts(pmids: list[str]) -> dict[str, str]:
    """Fetch abstracts for the given PMIDs (efetch XML)."""
    if not pmids:
        return {}
    root = ET.fromstring(_get(f"{EUTILS}/efetch.fcgi", {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}))
    found: dict[str, str] = {}
    for article in root.findall(".//PubmedArticle"):
        pmid = _text(article.find(".//MedlineCitation/PMID"))
        parts = []
        for node in article.findall(".//Abstract/AbstractText"):
            label = node.get("Label")
            body = _text(node)
            parts.append(f"{label}: {body}" if label else body)
        found[pmid] = " ".join(parts)[:1500]
    return found


def search_pubmed(query: str, max_results: int = 8, years_back: int = 2, sort: str = "date") -> str:
    """Search PubMed for peer-reviewed medical literature. Returns JSON with title, authors, journal,
    publication date and year, study type, DOI, PMID and abstract excerpt for each paper.

    Use this first for any clinical or medical research question. Cite every finding with its year and PMID/DOI.

    Args:
        query: PubMed search terms, e.g. "multiple sclerosis AND ocrelizumab". Add words like
            "randomized controlled trial", "meta-analysis", "systematic review" or "guideline" to target study types.
        max_results: Number of papers to return (1-15).
        years_back: Only papers published in the last N years (use 1 for "latest", 2 by default, 0 for no limit).
        sort: "date" for newest first, "relevance" for best match.
    """
    max_results = max(1, min(int(max_results), 15))
    params = {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json",
              "sort": "pub_date" if sort == "date" else "relevance"}
    if years_back and int(years_back) > 0:
        params.update({"datetype": "pdat", "mindate": str(date.today().year - int(years_back)),
                       "maxdate": str(date.today().year)})
    try:
        ids = json.loads(_get(f"{EUTILS}/esearch.fcgi", params))["esearchresult"].get("idlist", [])
        if not ids:
            return json.dumps({"results": [], "note": "No PubMed results. Try broader terms or a larger years_back.",
                               "searched_at": _now()})
        summaries = json.loads(_get(f"{EUTILS}/esummary.fcgi", {"db": "pubmed", "id": ",".join(ids), "retmode": "json"}))["result"]
        abstracts = _abstracts(ids)
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, ET.ParseError) as error:
        return json.dumps({"error": f"PubMed search failed: {error}"})

    results = []
    for pmid in ids:
        item = summaries.get(pmid, {})
        doi = next((a.get("value") for a in item.get("articleids", []) if a.get("idtype") == "doi"), None)
        pubdate = item.get("pubdate", "")
        year = re.search(r"(19|20)\d{2}", pubdate)
        results.append({
            "title": item.get("title"),
            "authors": [a.get("name") for a in item.get("authors", [])[:6]],
            "journal": item.get("fulljournalname") or item.get("source"),
            "publication_date": pubdate,
            "year": year.group(0) if year else None,
            "study_type": item.get("pubtype", []),
            "pmid": pmid,
            "doi": doi,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "abstract_excerpt": abstracts.get(pmid) or None,
        })
    return json.dumps({"results": results, "searched_at": _now()}, ensure_ascii=False)


def search_clinical_trials(condition: str, intervention: str = "", status: str = "", max_results: int = 8) -> str:
    """Search ClinicalTrials.gov for current and recent clinical trials. Returns JSON with NCT ID, phase,
    CURRENT recruitment status, primary endpoint, sponsor, dates, last-update date and whether results are posted.

    Use this for every clinical question to find ongoing and completed trials. A recruiting or active trial
    is research in progress, NOT evidence that the drug works.

    Args:
        condition: Disease or condition, e.g. "relapsing multiple sclerosis".
        intervention: Optional drug or intervention name, e.g. "tolebrutinib".
        status: Optional comma-separated filter from RECRUITING, NOT_YET_RECRUITING, ACTIVE_NOT_RECRUITING,
            COMPLETED, TERMINATED, WITHDRAWN, SUSPENDED. Empty means all.
        max_results: Number of trials to return (1-20).
    """
    max_results = max(1, min(int(max_results), 20))
    params = {"query.cond": condition, "pageSize": max_results, "sort": "LastUpdatePostDate:desc", "format": "json"}
    if intervention:
        params["query.intr"] = intervention
    if status:
        params["filter.overallStatus"] = status.replace(" ", "")
    try:
        studies = json.loads(_get(CTGOV, params)).get("studies", [])
    except (urllib.error.URLError, TimeoutError, ValueError) as error:
        return json.dumps({"error": f"ClinicalTrials.gov search failed: {error}"})
    if not studies:
        return json.dumps({"results": [], "note": "No trials found. Try broader terms.", "searched_at": _now()})

    results = []
    for study in studies:
        proto = study.get("protocolSection", {})
        ident, status_mod = proto.get("identificationModule", {}), proto.get("statusModule", {})
        design, arms = proto.get("designModule", {}), proto.get("armsInterventionsModule", {})
        nct = ident.get("nctId")
        primary = proto.get("outcomesModule", {}).get("primaryOutcomes", [])
        results.append({
            "nct_id": nct,
            "title": ident.get("briefTitle"),
            "phase": design.get("phases"),
            "design": design.get("studyType"),
            "interventions": [i.get("name") for i in arms.get("interventions", [])][:6],
            "recruitment_status": status_mod.get("overallStatus"),
            "why_stopped": status_mod.get("whyStopped"),
            "start_date": status_mod.get("startDateStruct", {}).get("date"),
            "estimated_completion": status_mod.get("primaryCompletionDateStruct", {}).get("date"),
            "last_update": status_mod.get("lastUpdatePostDateStruct", {}).get("date"),
            "primary_endpoint": primary[0].get("measure") if primary else None,
            "sponsor": proto.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name"),
            "results_posted": study.get("hasResults", False),
            "url": f"https://clinicaltrials.gov/study/{nct}",
        })
    return json.dumps({"results": results, "searched_at": _now(),
                       "note": "Status is as of last_update. Ongoing does not mean effective."}, ensure_ascii=False)
