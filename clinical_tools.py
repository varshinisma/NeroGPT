"""Clinical research tools for the NeuroGPT agent: PubMed and ClinicalTrials.gov.

Both use free public APIs (no key needed) and the standard library only.
Every result carries a publication year / last-update date and an identifier
(PMID, DOI, NCT ID) so answers can be cited and verified.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, wait
import os
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


from ratelimit import limiter, source_of_url


import httpx

from tracing import traced
_http = httpx.Client(timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}, follow_redirects=True)  # keep-alive: connections are reused


def _get(url: str, params: dict) -> bytes:
    """GET over a pooled keep-alive connection, at most the allowed requests per second to each source (ratelimit.py), and one retry on a rate-limit / temporary server error."""
    if url.startswith(EUTILS) and os.getenv("NCBI_API_KEY"):
        params = {**params, "api_key": os.environ["NCBI_API_KEY"]}
    source = source_of_url(url)
    for attempt in range(2):      # one retry: a request that still fails is reported as a failed source
        if source != "other":
            limiter.acquire(source)      # one shared limit per source for the legacy and the routed searches together
        try:
            response = _http.get(url, params=params)
        except httpx.HTTPError as error:  # callers catch urllib.error.URLError
            raise urllib.error.URLError(str(error)) from error
        if response.status_code in (429, 500, 502, 503) and attempt < 1:
            time.sleep(1.0)
            continue
        if response.status_code >= 400:
            raise urllib.error.HTTPError(url, response.status_code, response.reason_phrase, None, None)
        return response.content
    raise RuntimeError("unreachable")


def warm_connections() -> None:
    """Open (and keep open) the connections to the three data sources, so the first question does not pay the handshake."""
    for url, params in ((f"{EUTILS}/einfo.fcgi", {"db": "pubmed", "retmode": "json"}),
                        ("https://www.ebi.ac.uk/europepmc/webservices/rest/search", {"query": "treatment", "format": "json", "pageSize": 1}),
                        (CTGOV, {"query.cond": "disease", "pageSize": 1}),
                        ("https://api.fda.gov/drug/label.json", {"search": "openfda.product_type:\"HUMAN PRESCRIPTION DRUG\"", "limit": 1})):
        try:
            _get(url, params)
        except Exception:
            pass


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
        found[pmid] = " ".join(parts)[:4000]
    return found


@traced("search_pubmed", "tool")
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
        pmcid = next((a.get("value") for a in item.get("articleids", []) if a.get("idtype") == "pmc"), None)
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
            "pmcid": pmcid,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "abstract_excerpt": abstracts.get(pmid) or None,
        })
    return json.dumps({"results": results, "searched_at": _now()}, ensure_ascii=False)


@traced("search_clinical_trials", "tool")
def _ctgov_condition_text(condition) -> str:
    """Serialize model-produced condition concepts into the string required by API v2 `query.cond`.

    The routed planner can return several concepts as a list. Passing that Python list directly
    to the HTTP client produced a value like "['condition A', 'condition B']", which the API
    rejects. Flatten structured values and join normalized phrases as ordinary condition text.
    """
    def flatten(value):
        if isinstance(value, dict):
            for key in ("normalized", "primary", "condition", "name", "term", "label"):
                if value.get(key):
                    yield from flatten(value[key])
                    return
            for child in value.values():
                yield from flatten(child)
        elif isinstance(value, (list, tuple, set)):
            for child in value:
                yield from flatten(child)
        elif value is not None:
            yield str(value)

    generic = {"what", "which", "could", "would", "should", "please", "patient", "patients", "child", "children", "adult", "adults",
               "infant", "infants", "workup", "work-up", "management", "diagnosis", "diagnostic", "treatment", "therapy", "trial", "trials",
               "year", "years", "month", "months", "week", "weeks", "day", "days", "old", "has", "have", "with", "the", "and", "for", "this"}
    if isinstance(condition, str) and " OR " in condition:      # an explicit OR between concepts: each concept is cleaned on its own and the operator is kept (the registry reads OR in capitals)
        parts = [_ctgov_condition_text(part) for part in condition.split(" OR ")]
        return " OR ".join(p for p in parts if p)
    phrases = []
    for raw in flatten(condition):
        cleaned = remove_age_expressions(raw)
        for clause in re.split(r"[,;.!?\n]+", cleaned):
            words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9-]*", clause) if w.lower() not in generic]
            phrase = " ".join(words).strip()
            if phrase and phrase not in phrases:
                phrases.append(phrase)
    return " ".join(phrases)


def search_clinical_trials(condition: str | list | dict, intervention: str = "", status: str = "", max_results: int = 8, drug_only: bool = False) -> str:
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
    condition_query = _ctgov_condition_text(condition)
    if not condition_query:
        return json.dumps({"error": "ClinicalTrials.gov search not run: no usable condition terms", "query": {"source": "ClinicalTrials.gov", "endpoint": CTGOV}})
    allowed_status = {"RECRUITING", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING", "COMPLETED", "TERMINATED", "WITHDRAWN", "SUSPENDED", "ENROLLING_BY_INVITATION", "UNKNOWN"}
    status_values = [x.strip().upper() for x in str(status or "").split(",") if x.strip()]
    if status_values and any(x not in allowed_status for x in status_values):
        return json.dumps({"error": "ClinicalTrials.gov search not run: invalid overall-status filter", "query": {"source": "ClinicalTrials.gov", "condition": condition_query, "status": status}})
    params = {"query.cond": condition_query, "pageSize": max_results, "sort": "LastUpdatePostDate:desc", "format": "json"}
    if intervention:
        params["query.intr"] = intervention
    if drug_only:      # only trials whose registered intervention type is a drug or a biological
        params["query.term"] = "AREA[InterventionType](DRUG OR BIOLOGICAL)"
    if status:
        params["filter.overallStatus"] = ",".join(status_values)
    query_trace = {"source": "ClinicalTrials.gov", "endpoint": CTGOV, "params": params}
    try:
        studies = json.loads(_get(CTGOV, params)).get("studies", [])
    except (urllib.error.URLError, TimeoutError, ValueError, TypeError, KeyError) as error:
        return json.dumps({"error": f"ClinicalTrials.gov search failed: {error}", "query_trace": [{**query_trace, "status": "failed", "reason": f"{type(error).__name__}: {str(error)[:160]}"}]})
    if not studies:
        return json.dumps({"results": [], "note": "No studies matched this completed query.", "query_trace": [{**query_trace, "status": "no_match"}], "searched_at": _now()})

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
            "ages": ", ".join(x for x in (
                (proto.get("eligibilityModule", {}).get("minimumAge") and "from " + proto["eligibilityModule"]["minimumAge"]),
                (proto.get("eligibilityModule", {}).get("maximumAge") and "up to " + proto["eligibilityModule"]["maximumAge"]),
                ("/".join(proto.get("eligibilityModule", {}).get("stdAges") or []).lower() or None)) if x) or None,      # the ages the trial enrols, as registered
            "conditions": (proto.get("conditionsModule", {}).get("conditions") or [])[:6],      # what the trial registered as its condition
            "design": design.get("studyType"),
            "interventions": [i.get("name") for i in arms.get("interventions", [])][:6],
            "drug_interventions": [i.get("name") for i in arms.get("interventions", []) if i.get("type") in ("DRUG", "BIOLOGICAL")][:6],      # registered type, not guessed
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
    return json.dumps({"results": results, "query_trace": [{**query_trace, "status": "found"}], "searched_at": _now(),
                       "note": "Status is as of last_update. Ongoing does not mean effective."}, ensure_ascii=False)


@traced("search_pubmed_quick", "tool")
def search_pubmed_quick(query: str, max_results: int = 6, years_back: int = 3, sort: str = "relevance") -> str:
    """FAST PubMed search: 2 calls (esearch + efetch) instead of 3. Same result fields as search_pubmed."""
    params = {"db": "pubmed", "term": query, "retmax": max(1, min(int(max_results), 15)), "retmode": "json",
              "sort": "pub_date" if sort == "date" else "relevance"}
    if years_back and int(years_back) > 0:
        params.update({"datetype": "pdat", "mindate": str(date.today().year - int(years_back)), "maxdate": str(date.today().year)})
    try:
        ids = json.loads(_get(f"{EUTILS}/esearch.fcgi", params))["esearchresult"].get("idlist", [])
        if not ids:
            return json.dumps({"results": [], "note": "No PubMed results.", "searched_at": _now()})
        root = ET.fromstring(_get(f"{EUTILS}/efetch.fcgi", {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}))
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, ET.ParseError) as error:
        return json.dumps({"error": f"PubMed search failed: {error}"})
    by_id = {}
    for art in root.findall(".//PubmedArticle"):
        pmid = _text(art.find(".//MedlineCitation/PMID"))
        pub = art.find(".//Journal/JournalIssue/PubDate")
        date_text = " ".join(x for x in (_text(pub.find("Year")) if pub is not None else "", _text(pub.find("Month")) if pub is not None else "",
                                         _text(pub.find("Day")) if pub is not None else "") if x) or (_text(pub.find("MedlineDate")) if pub is not None else "")
        year = re.search(r"(19|20)\d{2}", date_text)
        parts = []
        for node in art.findall(".//Abstract/AbstractText"):
            label = node.get("Label")
            parts.append(f"{label}: {_text(node)}" if label else _text(node))
        authors = [f"{_text(a.find('LastName'))} {_text(a.find('Initials'))}".strip() for a in art.findall(".//AuthorList/Author") if a.find("LastName") is not None][:6]
        doi = next((_text(a) for a in art.findall(".//PubmedData/ArticleIdList/ArticleId") if a.get("IdType") == "doi"), None)
        pmcid = next((_text(a) for a in art.findall(".//PubmedData/ArticleIdList/ArticleId") if a.get("IdType") == "pmc"), None)
        by_id[pmid] = {"pmcid": pmcid,
            "title": _text(art.find(".//ArticleTitle")), "authors": authors, "journal": _text(art.find(".//Journal/Title")),
            "publication_date": date_text, "year": year.group(0) if year else None,
            "study_type": [_text(t) for t in art.findall(".//PublicationTypeList/PublicationType")], "pmid": pmid, "doi": doi,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", "abstract_excerpt": (" ".join(parts)[:4000]) or None}
    return json.dumps({"results": [by_id[i] for i in ids if i in by_id], "searched_at": _now()}, ensure_ascii=False)


@traced("search_europepmc_quick", "tool")
def search_europepmc_quick(query: str, max_results: int = 6, years_back: int = 3, sort: str = "relevance", fulltext: bool = False) -> str:
    """FAST literature search in ONE call via Europe PMC (restricted to PubMed/MEDLINE records, so PMIDs are PubMed PMIDs).
    Same result fields as search_pubmed. Falls back to search_pubmed_quick if Europe PMC fails."""
    first_year = date.today().year - int(years_back) if years_back else 1900
    q = f"({query}) AND SRC:MED AND FIRST_PDATE:[{first_year}-01-01 TO {date.today().year}-12-31]"
    if fulltext:      # the words are searched in the FULL TEXT of the open-access articles (methods, results, discussion), not only in the title and abstract
        q = f"BODY:({query}) AND SRC:MED AND OPEN_ACCESS:y AND FIRST_PDATE:[{first_year}-01-01 TO {date.today().year}-12-31]"
    if sort == "date":
        q += " sort_date:y"
    params = {"query": q, "resultType": "core", "format": "json", "pageSize": max(1, min(int(max_results), 15))}
    try:
        data = json.loads(_get("https://www.ebi.ac.uk/europepmc/webservices/rest/search", params))["resultList"]["result"]
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError):
        if fulltext:      # PubMed cannot search full texts: the failure is reported, not answered with a different kind of search
            return json.dumps({"error": "Europe PMC full-text search failed"})
        return search_pubmed_quick(query, max_results, years_back, sort)
    results = []
    for r in data:
        if not r.get("pmid"):
            continue
        abstract = re.sub(r"<[^>]+>", " ", r.get("abstractText") or "")
        results.append({
            "title": re.sub(r"<[^>]+>", "", r.get("title") or "").strip(), "authors": [a.strip() for a in (r.get("authorString") or "").rstrip(".").split(",")][:6],
            "journal": ((r.get("journalInfo") or {}).get("journal") or {}).get("title") or r.get("journalTitle"),
            "publication_date": r.get("firstPublicationDate") or r.get("pubYear"), "year": r.get("pubYear"),
            "study_type": ((r.get("pubTypeList") or {}).get("pubType")) or [], "pmid": r["pmid"], "doi": r.get("doi"), "pmcid": r.get("pmcid"),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{r['pmid']}/", "abstract_excerpt": re.sub(r"\s+", " ", abstract).strip()[:4000] or None})
    return json.dumps({"results": results, "searched_at": _now()}, ensure_ascii=False)


# ------------------------------------------------------------------ regulatory source: FDA drug labels (openFDA)
FDA_LABEL_API = "https://api.fda.gov/drug/label.json"
_CONDITION_FILLER = {"disorder", "disease", "syndrome", "spectrum", "condition", "chronic", "acute", "symptom", "symptoms", "children", "child",
                     "pediatric", "paediatric", "adolescent", "adults", "treatment", "therapy", "treat", "management", "drug", "drugs"}


_LABEL_NOISE = {"ORAL", "ORALLY", "DISINTEGRATING", "TABLETS", "TABLET", "INJECTION", "SOLUTION", "EXTENDED", "RELEASE", "SODIUM", "HYDROCHLORIDE", "ER"}


def remove_age_expressions(text: str) -> str:
    """Remove demographic age spans (digits or written numbers) from a search query without a number-word denylist."""
    value = re.sub(r"\b(?:\d+|[a-z]+(?:[- ][a-z]+)?)\s*[- ]?(?:years?|months?|weeks?|days?)\s*[- ]old\b", " ", str(text or ""), flags=re.I)
    return re.sub(r"\b(?:aged|age)\s+\d+\b|\b\d+\s*(?:y/o|yo)\b", " ", value, flags=re.I)


def condition_stems(terms: list[str], limit: int = 3) -> list[str]:
    """Wildcard stems of the condition words (a condition word is cut to its stem), so other wordings of the condition in a label still match."""
    words = [w for w in re.findall(r"[a-z]{4,}", " ".join(terms).lower()) if w not in _CONDITION_FILLER]
    return [w[:5] for w in dict.fromkeys(words)][:limit]


def _fda_get(params: dict) -> dict | None:
    """One openFDA call. 404 means 'nothing matched' (a normal answer), not an error."""
    try:
        return json.loads(_get(FDA_LABEL_API, params))
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def indication_statement(text: str, stems: list[str], generic: str, limit: int = 520) -> str | None:
    """The label's own COMPLETE indication sentence that names the condition (for example '<Drug> tablets are indicated for the treatment of <symptom>
    associated with <condition>, including ...'), copied unchanged from 'Indications and Usage'. It starts at the beginning of the sentence and
    ends at its end (or at a list-item boundary if the sentence is very long), never in the middle of a phrase."""
    body = re.sub(r"^\s*\d+(?:\.\d+)*\s*INDICATIONS?\s*(?:AND|&)\s*USAGE\s*", "", text or "", flags=re.I)
    cut = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])|\s+(?=" + re.escape(generic) + r"\s+(?:is|are|tablets?|oral|injection|extended)\b)", re.I)
    units = [u.strip() for u in cut.split(body) if u and u.strip()]
    named = [i for i, u in enumerate(units) if any(s in u.lower() for s in stems)]
    if not named:
        return None
    formal = [i for i in named if re.search(r"\bindicated\b", units[i], re.I) and not re.search(r"\(\s*\d+\.\d+\s*\)", units[i])]      # not the 'highlights' list with section numbers
    pick = (formal or [i for i in named if re.search(r"\bindicated\b", units[i], re.I)] or named)[0]
    best = re.sub(r"^\d+(?:\.\d+)+\s+.*?\s+(?=" + re.escape(generic) + r"\b)", "", units[pick], count=1, flags=re.I)      # drop a leading section heading such as '1.3 <indication title>'
    if pick + 1 < len(units) and units[pick + 1].lower().startswith("efficacy was established") and len(best) + len(units[pick + 1]) + 1 <= limit:
        best += " " + units[pick + 1]      # the next sentence gives the population studied (for example the ages)
    best = re.sub(r"\s*\[see [^\]]*\]\s*", " ", best)      # label cross-references such as "[see Clinical Studies (14.4) ]" are not part of the statement
    best = re.sub(r"\s+([.,;])", r"\1", best).strip()
    if len(best) <= limit:
        return best
    window = best[:limit]
    boundary = max(window.rfind(" • "), window.rfind("; "), window.rfind(", "))
    return (window[:boundary] if boundary > 80 else window.rsplit(" ", 1)[0]).rstrip(" ,;:•") + " (the label continues)"


def dosage_statement(text: str, stems: list[str], limit: int = 1300) -> str | None:
    """The label's own dosing text FOR THIS CONDITION: the numbered 'Dosage and Administration' section whose title names the condition (for example
    '2.4 <indication title>'), copied unchanged and cut only where a sentence ends. None when the label has no such section
    (the dose is then not stated for this condition in the retrieved label text)."""
    t = re.sub(r"\s+", " ", text or "").strip()
    heads = list(re.finditer(r"(?<![\d.(])2\.\d+\s+(?=[A-Z])", t))
    for i, h in enumerate(heads):
        title = re.split(r"\.\s", t[h.end():h.end() + 80])[0]      # the heading itself (never the next section's heading)
        if not any(s in title.lower() for s in stems):
            continue
        end = heads[i + 1].start() if i + 1 < len(heads) else len(t)
        stem_pattern = "|".join(re.escape(x.rstrip("*")) for x in stems if x)
        block = re.sub(rf"^[A-Z][^.]{{0,100}}?\b(?:{stem_pattern})\w*(?:\s+\w+)?\s+", "", t[h.end():end], count=1, flags=re.I).strip() if stem_pattern else t[h.end():end].strip()      # drop the section title itself
        block = re.sub(r"\s+([.,;])", r"\1", re.sub(r"\s*\[see [^\]]*\]\s*", " ", block.lstrip("-\u2013 "))).strip()      # no '[see Clinical Studies (14.4)]' cross-references
        if len(block) < 40:
            continue
        if len(block) <= limit:
            return block
        cut = block.rfind(". ", 0, limit)
        return (block[:cut + 1] if cut > 150 else block[:limit].rsplit(" ", 1)[0]) + " (the label continues)"
    return None


def general_dosing(text: str, limit: int = 700) -> str | None:
    """The opening of the label's own Dosage and Administration section, copied unchanged and cut at a sentence end. Used only when no subsection is titled with the condition: it is the
    label's general dosing, not a dose for this use, and is shown as such."""
    t = re.sub(r"\s+", " ", text or "").strip()
    t = re.sub(r"\s*\[see [^\]]*\]\s*", " ", t).strip()
    if len(t) < 60:
        return None
    if len(t) <= limit:
        return t
    cut = t.rfind(". ", 0, limit)
    return (t[:cut + 1] if cut > 150 else t[:limit].rsplit(" ", 1)[0]) + " (the label continues)"


def _fda_record(generic: str, stems: list[str]) -> dict | None:
    records = _fda_records(generic, stems, 1)
    return records[0] if records else None


def _fda_records(generic: str, stems: list[str], limit: int = 1) -> list[dict]:
    """The FDA label records of one generic name. A drug the question names can have several labels for different indications (for example one for each of two uses): up to `limit` distinct labels are returned and the model decides which of them bear on the question."""
    data = _fda_get({"search": f'openfda.generic_name.exact:"{generic}"', "limit": limit})
    out, seen = [], set()
    for label in (data or {}).get("results", []):
        record = _record_from_label(label, generic, stems)
        key = (record["indication_excerpt"][:120], tuple(record["brands"]))
        if key not in seen:
            seen.add(key)
            out.append(record)
    return out


def _record_from_label(label: dict, generic: str, stems: list[str]) -> dict:
    text = re.sub(r"\s+", " ", " ".join(label.get("indications_and_usage") or [])).strip()
    low, hit = text.lower(), -1
    for stem in stems:
        hit = low.find(stem)
        if hit >= 0:
            break
    if hit >= 0:
        start = max(0, hit - 170)
        start = text.find(" ", start) + 1 if start > 0 and text.find(" ", start) >= 0 else start  # begin at a word boundary
        excerpt = text[start:hit + 230]
    else:
        excerpt = text[:300]
    set_id, when = label.get("set_id"), str(label.get("effective_time") or "")
    return {"generic": generic.title(), "brands": (label.get("openfda", {}).get("brand_name") or [])[:3], "jurisdiction": "FDA",
            "indication_excerpt": excerpt.strip(), "matches_condition": hit >= 0,
            "indication_statement": indication_statement(text, stems, generic) if hit >= 0 else None,
            "dose_statement": dosage_statement(" ".join(label.get("dosage_and_administration") or []), stems) if hit >= 0 else None,
            "dose_general": general_dosing(" ".join(label.get("dosage_and_administration") or [])) if hit >= 0 else None,
            "route": ", ".join(r.title() for r in (label.get("openfda", {}).get("route") or []))[:80] or None,
            "label_date": f"{when[:4]}-{when[4:6]}-{when[6:8]}" if len(when) == 8 else None, "set_id": set_id,
            "url": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}" if set_id else None, "source_id": f"fda:{set_id}"}


_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_FULL_DATE = rf"(?:{_MONTHS}) \d{{1,2}}, (?:19|20)\d\d|\b\d{{1,2}}/\d{{1,2}}/(?:19|20)\d\d\b"
_LABELLED_DATE = re.compile(rf"(?:Content current as of|Posted|Published|Updated|Last updated|Release|Release date|Date)[^A-Za-z0-9]{{0,20}}({_FULL_DATE})")


def page_date(url: str) -> str | None:
    """The publication date a web page states about itself ('Release: September 22, 2025'), read from the page's own visible text; None when the page is not HTML or states none.
    Parsing a date is mechanical: nothing here judges the page."""
    try:
        reply = _http.get(url, timeout=6)
        if reply.status_code != 200 or "html" not in (reply.headers.get("content-type") or ""):
            return None
        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", re.sub(r"(?is)<(?:script|style).*?</(?:script|style)>", " ", reply.content[:400000].decode("utf-8", "ignore"))))
    except Exception:
        return None
    found = _LABELLED_DATE.search(text) or re.search(f"({_FULL_DATE})", text[:6000])
    return (found.group(1) if found else None)


def fda_label_link(generic: str) -> dict | None:
    """The DailyMed label of ONE named drug (any indication), so a table row can link to that drug's own page. None when openFDA has no label for the name."""
    name = re.sub(r"[^A-Za-z0-9 \-]", "", str(generic or "")).strip()
    if len(name) < 4:
        return None
    data = _fda_get({"search": f'openfda.generic_name:"{name}"', "limit": 1})
    label = ((data or {}).get("results") or [None])[0]
    generic_names = " ".join(((label or {}).get("openfda") or {}).get("generic_name") or []).lower()
    via_brand = False
    if not label or not label.get("set_id") or name.split()[0].lower() not in generic_names:
        data = _fda_get({"search": f'openfda.brand_name:"{name}"', "limit": 1})      # the name may be a brand ('Strattera'): the label states its generic name
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


@traced("search_fda_labels", "tool")
def search_fda_labels(condition_terms: list[str], drug_names: list[str] | None = None, max_drugs: int = 8) -> str:
    """OFFICIAL regulatory facts from FDA drug labels: which drugs have a labelled indication that mentions the condition, with the label's own wording.
    Searches by condition (any disease) and/or by named drugs. 'results' empty with no 'error' means searched and none found."""
    stems = condition_stems(condition_terms)
    if not stems and not drug_names:  # nothing valid to search: that is a failure to search, NOT "searched and none found"
        return json.dumps({"error": f"FDA label search not run: no usable condition terms in {condition_terms!r}"})
    try:
        names: list[str] = []
        used = stems
        if stems:
            # all stems, then each pair, and finally each single stem, before concluding that nothing matches
            subsets = [stems] + ([stems[:i] + stems[i + 1:] for i in range(len(stems))] if len(stems) > 1 else [])
            if len(stems) > 2:
                subsets += [[s] for s in stems]
            for subset in subsets:
                if not subset:
                    continue
                query = " AND ".join(f"indications_and_usage:{s}*" for s in subset)
                data = _fda_get({"search": query, "count": "openfda.generic_name.exact", "limit": 25})
                found = [r["term"] for r in (data or {}).get("results", [])]
                if found:
                    used = subset
                    for term in found:
                        base = " ".join(w for w in term.split() if w not in _LABEL_NOISE) or term
                        if base not in names:
                            names.append(base)
                    break
        asked = [d.upper() for d in drug_names or []]
        names = asked + [n for n in names if n.upper() not in asked][:max_drugs]      # the drugs the question names come first and are never cut off
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=6) as pool:
            asked_set = set(asked)
            records = [r for group in pool.map(lambda n: _fda_records(n, used, 4 if n in asked_set else 1), names) for r in group]
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as error:
        return json.dumps({"error": f"FDA label search failed: {error}"})
    return json.dumps({"results": records[:max_drugs + 4 * len(drug_names or [])], "searched_terms": used, "searched_drugs": drug_names or [],
                       "searched_at": _now(), "note": "empty results = searched, none found"}, ensure_ascii=False)


# ---- original figures of open-access papers (the PubMed Central Open Access data service: article XML with the captions and licence, plus the original image files)
PMC_PAGE = "https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"      # only a link for the reader; it is never fetched
OA_BUCKET = "https://pmc-oa-opendata.s3.amazonaws.com/"      # the PubMed Central Open Access Subset
OPEN_LICENCE = re.compile(r"https?://creativecommons\.org/(?:licenses/(by(?:-nc)?(?:-sa)?(?:-nd)?)/(\d\.\d)|publicdomain/zero/(1\.0))", re.I)
_oa_http = httpx.Client(timeout=12, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36 NeuroGPT-research"})


def _fig_text(fragment: str) -> str:
    import html
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", fragment)).split())


def open_licence(page_html: str) -> dict | None:
    """The explicit open licence the article page states (a Creative Commons licence or a CC0 dedication): {'url','label'}. None when the page states none: reuse rights are then unclear."""
    found = OPEN_LICENCE.search(page_html or "")
    if not found:
        return None
    if found.group(3):
        return {"url": found.group(0), "label": f"CC0 {found.group(3)}"}
    return {"url": found.group(0), "label": f"CC {found.group(1).upper()} {found.group(2)}"}


def parse_figures(xml_text: str, pmcid: str, keys: list[str], limit: int = 6) -> list[dict]:
    """The figures of one open-access article: [{'id','label','caption','image','page_url'}], from the article XML and the bucket's file list (`keys`, the file names of that article). A figure whose
    image file is not in the bucket is not a candidate. The caption is the original, complete text."""
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return []
    href = "{http://www.w3.org/1999/xlink}href"
    folder = keys[0].split("/")[0] if keys else ""
    by_stem = {}
    for key in keys:
        stem, dot, ext = key.split("/", 1)[-1].rpartition(".")
        if dot and "." + ext.lower() in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
            by_stem.setdefault(stem, key)
    figures = []
    for fig in root.iter("fig"):
        graphic = next((g for g in fig.iter("graphic") if g.get(href)), None)
        name = graphic.get(href) if graphic is not None else None      # the XML names the file with or without its extension
        key = (by_stem.get(name) or by_stem.get(name.rpartition(".")[0]) or next((k for k in keys if k.endswith("/" + name)), None)) if name else None
        if not key:
            continue
        label = _fig_text(" ".join(fig.find("label").itertext())) if fig.find("label") is not None else ""
        caption_node = fig.find("caption")
        caption = _fig_text(" ".join(caption_node.itertext())) if caption_node is not None else ""
        text = caption if re.match(r"(Figure|Fig\.?)\s*\d", caption) or not label else f"{label.rstrip('.')}. {caption}".strip()
        number = re.search(r"(\d+)", label or fig.get("id", ""))
        figures.append({"id": f"{pmcid}:{fig.get('id') or key}", "label": label.rstrip(".:") or (f"Figure {int(number.group(1))}" if number else "Figure"), "caption": text,
                        "image": OA_BUCKET + key, "page_url": PMC_PAGE.format(pmcid=pmcid), "folder": folder})
        if len(figures) >= limit:
            break
    return figures


def _article(pmcid: str) -> tuple[str, list[str]] | None:
    """(article XML text, the file names of that article in the open-access bucket) for the newest version of the article; None when the article is not in the bucket."""
    listing = _oa_http.get(OA_BUCKET, params={"list-type": "2", "prefix": pmcid + ".", "max-keys": "200"})
    if listing.status_code != 200:
        time.sleep(1.0)
        listing = _oa_http.get(OA_BUCKET, params={"list-type": "2", "prefix": pmcid + ".", "max-keys": "200"})
    if listing.status_code != 200:
        return None
    keys = re.findall(r"<Key>(.*?)</Key>", listing.text)
    versions = sorted({k.split("/")[0] for k in keys}, key=lambda v: int(v.rsplit(".", 1)[-1]) if v.rsplit(".", 1)[-1].isdigit() else 0)
    if not versions:
        return None
    newest = versions[-1]
    mine = [k for k in keys if k.startswith(newest + "/")]
    xml_key = f"{newest}/{newest}.xml"
    if xml_key not in mine:
        return None
    response = _oa_http.get(OA_BUCKET + xml_key)
    return (response.text, mine) if response.status_code == 200 else None


def _batch(batch: list[dict], cap: float, statuses: list) -> list[dict]:
    """The figures of one batch of open-access papers, their articles fetched side by side for at most `cap` seconds. Each figure carries its paper's own details."""
    pool = ThreadPoolExecutor(max_workers=min(len(batch), 4))
    futures = {pool.submit(_article, p["pmcid"]): p for p in batch}
    out = []
    try:
        done, _ = wait(list(futures), timeout=cap)
        for future in done:
            paper = futures[future]
            try:
                article = future.result()
                statuses.append(200 if article else 404)
                if not article:
                    continue
                xml_text, keys = article
                licence = open_licence(xml_text)
                for fig in parse_figures(xml_text, paper["pmcid"], keys):
                    out.append({**fig, "pmid": paper.get("pmid"), "pmcid": paper["pmcid"], "doi": paper.get("doi"), "title": paper.get("title") or None, "cite": paper.get("cite") or None,
                                "design": paper.get("design") or None, "licence": licence})
            except Exception:
                continue
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return out


def candidates(papers: list[dict], want: int = 12, batch_size: int = 8, batches: int = 2, cap: float = 7.0, trace: dict | None = None) -> list[dict]:
    """Figures of the open-access papers, in the order given, a batch of papers at a time until there are `want` figures (many papers have none) or the papers or `batches` run out."""
    chosen, seen = [], set()
    for p in papers:
        if p.get("pmcid") and not p.get("retracted") and p["pmcid"] not in seen:
            seen.add(p["pmcid"])
            chosen.append(p)
    statuses = trace.setdefault("page_status", []) if trace is not None else []
    out = []
    for k in range(batches):
        batch = chosen[k * batch_size:(k + 1) * batch_size]
        if not batch or len(out) >= want:
            break
        out += _batch(batch, cap, statuses)
    return out


@traced("search_paper_figures", "tool")
def search_paper_figures(papers: str, want: int = 12, batches: int = 2) -> str:
    """Retrieve the ORIGINAL figures (caption, licence, image address) of open-access papers. `papers` is a JSON list of {"pmid","pmcid","title",...}. Returns JSON {"figures": [...]};
    the figures are only retrieved here, whether any of them is shown is decided later by the figure-selection skill."""
    try:
        listed = json.loads(papers) if isinstance(papers, str) else list(papers)
    except Exception:
        listed = []
    trace: dict = {}
    figures = candidates(listed, want=want, batches=batches, trace=trace)
    return json.dumps({"figures": figures, "page_status": trace.get("page_status", [])}, ensure_ascii=False)
