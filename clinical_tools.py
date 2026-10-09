"""Clinical research tools for the NeuroGPT agent: PubMed and ClinicalTrials.gov.

Both use free public APIs (no key needed) and the standard library only.
Every result carries a publication year / last-update date and an identifier
(PMID, DOI, NCT ID) so answers can be cited and verified.
"""
from __future__ import annotations

import json
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


_ncbi_lock = threading.Lock()
_ncbi_last_call = 0.0
NCBI_MIN_GAP = 0.11 if os.getenv("NCBI_API_KEY") else 0.35  # NCBI allows ~3 requests/second without an API key, 10 with one


import httpx

_http = httpx.Client(timeout=TIMEOUT, headers={"User-Agent": USER_AGENT}, follow_redirects=True)  # keep-alive: connections are reused


def _get(url: str, params: dict) -> bytes:
    """GET over a pooled keep-alive connection, with spacing for NCBI and retries on rate-limit / temporary server errors."""
    global _ncbi_last_call
    if url.startswith(EUTILS) and os.getenv("NCBI_API_KEY"):
        params = {**params, "api_key": os.environ["NCBI_API_KEY"]}
    for attempt in range(4):
        if url.startswith(EUTILS):
            with _ncbi_lock:  # serialise NCBI calls across threads
                wait = (0.11 if os.getenv("NCBI_API_KEY") else 0.35) - (time.time() - _ncbi_last_call)      # about 10 requests a second with a key, 3 without
                if wait > 0:
                    time.sleep(wait)
                _ncbi_last_call = time.time()
        try:
            response = _http.get(url, params=params)
        except httpx.HTTPError as error:  # callers catch urllib.error.URLError
            raise urllib.error.URLError(str(error)) from error
        if response.status_code in (429, 500, 502, 503) and attempt < 3:
            time.sleep(1.0 * (attempt + 1))
            continue
        if response.status_code >= 400:
            raise urllib.error.HTTPError(url, response.status_code, response.reason_phrase, None, None)
        return response.content
    raise RuntimeError("unreachable")


def warm_connections() -> None:
    """Open (and keep open) the connections to the three data sources, so the first question does not pay the handshake."""
    for url, params in ((f"{EUTILS}/einfo.fcgi", {"db": "pubmed", "retmode": "json"}),
                        ("https://www.ebi.ac.uk/europepmc/webservices/rest/search", {"query": "autism", "format": "json", "pageSize": 1}),
                        (CTGOV, {"query.cond": "autism", "pageSize": 1}),
                        ("https://api.fda.gov/drug/label.json", {"search": "openfda.generic_name.exact:\"RISPERIDONE\"", "limit": 1})):
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


def search_clinical_trials(condition: str, intervention: str = "", status: str = "", max_results: int = 8, drug_only: bool = False) -> str:
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
    if drug_only:      # only trials whose registered intervention type is a drug or a biological
        params["query.term"] = "AREA[InterventionType](DRUG OR BIOLOGICAL)"
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
    return json.dumps({"results": results, "searched_at": _now(),
                       "note": "Status is as of last_update. Ongoing does not mean effective."}, ensure_ascii=False)


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
        by_id[pmid] = {
            "title": _text(art.find(".//ArticleTitle")), "authors": authors, "journal": _text(art.find(".//Journal/Title")),
            "publication_date": date_text, "year": year.group(0) if year else None,
            "study_type": [_text(t) for t in art.findall(".//PublicationTypeList/PublicationType")], "pmid": pmid, "doi": doi,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", "abstract_excerpt": (" ".join(parts)[:4000]) or None}
    return json.dumps({"results": [by_id[i] for i in ids if i in by_id], "searched_at": _now()}, ensure_ascii=False)


def search_europepmc_quick(query: str, max_results: int = 6, years_back: int = 3, sort: str = "relevance") -> str:
    """FAST literature search in ONE call via Europe PMC (restricted to PubMed/MEDLINE records, so PMIDs are PubMed PMIDs).
    Same result fields as search_pubmed. Falls back to search_pubmed_quick if Europe PMC fails."""
    first_year = date.today().year - int(years_back) if years_back else 1900
    q = f"({query}) AND SRC:MED AND FIRST_PDATE:[{first_year}-01-01 TO {date.today().year}-12-31]"
    if sort == "date":
        q += " sort_date:y"
    params = {"query": q, "resultType": "core", "format": "json", "pageSize": max(1, min(int(max_results), 15))}
    try:
        data = json.loads(_get("https://www.ebi.ac.uk/europepmc/webservices/rest/search", params))["resultList"]["result"]
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError):
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
            "study_type": ((r.get("pubTypeList") or {}).get("pubType")) or [], "pmid": r["pmid"], "doi": r.get("doi"),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{r['pmid']}/", "abstract_excerpt": re.sub(r"\s+", " ", abstract).strip()[:4000] or None})
    return json.dumps({"results": results, "searched_at": _now()}, ensure_ascii=False)


# ------------------------------------------------------------------ regulatory source: FDA drug labels (openFDA)
FDA_LABEL_API = "https://api.fda.gov/drug/label.json"
_CONDITION_FILLER = {"disorder", "disease", "syndrome", "spectrum", "condition", "chronic", "acute", "symptom", "symptoms", "children", "child",
                     "pediatric", "paediatric", "adolescent", "adults", "treatment", "therapy", "treat", "management", "drug", "drugs"}
_LABEL_NOISE = {"ORAL", "ORALLY", "DISINTEGRATING", "TABLETS", "TABLET", "INJECTION", "SOLUTION", "EXTENDED", "RELEASE", "SODIUM", "HYDROCHLORIDE", "ER"}


def condition_stems(terms: list[str], limit: int = 3) -> list[str]:
    """Wildcard stems of the condition words (autism -> 'autis'), so label wording such as 'autistic disorder' still matches."""
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
    """The label's own COMPLETE indication sentence that names the condition (for example 'Risperidone tablets are indicated for the treatment of irritability
    associated with autistic disorder, including symptoms of ...'), copied unchanged from 'Indications and Usage'. It starts at the beginning of the sentence and
    ends at its end (or at a list-item boundary if the sentence is very long), never in the middle of a phrase."""
    body = re.sub(r"^\s*\d+(?:\.\d+)*\s*INDICATIONS?\s*(?:AND|&)\s*USAGE\s*", "", text or "", flags=re.I)
    cut = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])|\s+(?=" + re.escape(generic) + r"\s+(?:is|are|tablets?|oral|injection|extended)\b)", re.I)
    units = [u.strip() for u in cut.split(body) if u and u.strip()]
    named = [i for i, u in enumerate(units) if any(s in u.lower() for s in stems)]
    if not named:
        return None
    formal = [i for i in named if re.search(r"\bindicated\b", units[i], re.I) and not re.search(r"\(\s*\d+\.\d+\s*\)", units[i])]      # not the 'highlights' list with section numbers
    pick = (formal or [i for i in named if re.search(r"\bindicated\b", units[i], re.I)] or named)[0]
    best = re.sub(r"^\d+(?:\.\d+)+\s+.*?\s+(?=" + re.escape(generic) + r"\b)", "", units[pick], count=1, flags=re.I)      # drop a leading section heading such as '1.3 Irritability ...'
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
    '2.4 Irritability Associated with Autistic Disorder'), copied unchanged and cut only where a sentence ends. None when the label has no such section
    (the dose is then not stated for this condition in the retrieved label text)."""
    t = re.sub(r"\s+", " ", text or "").strip()
    heads = list(re.finditer(r"(?<![\d.(])2\.\d+\s+(?=[A-Z])", t))
    for i, h in enumerate(heads):
        title = re.split(r"\.\s", t[h.end():h.end() + 80])[0]      # the heading itself (never the next section's heading)
        if not any(s in title.lower() for s in stems):
            continue
        end = heads[i + 1].start() if i + 1 < len(heads) else len(t)
        block = re.sub(r"^[A-Z][^.]{0,100}?\bautis\w*(?:\s+\w+)?\s+", "", t[h.end():end], count=1, flags=re.I).strip()      # drop the section title itself
        block = re.sub(r"\s+([.,;])", r"\1", re.sub(r"\s*\[see [^\]]*\]\s*", " ", block.lstrip("-\u2013 "))).strip()      # no '[see Clinical Studies (14.4)]' cross-references
        if len(block) < 40:
            continue
        if len(block) <= limit:
            return block
        cut = block.rfind(". ", 0, limit)
        return (block[:cut + 1] if cut > 150 else block[:limit].rsplit(" ", 1)[0]) + " (the label continues)"
    return None


def _fda_record(generic: str, stems: list[str]) -> dict | None:
    records = _fda_records(generic, stems, 1)
    return records[0] if records else None


def _fda_records(generic: str, stems: list[str], limit: int = 1) -> list[dict]:
    """The FDA label records of one generic name. A drug the question names can have several labels for different indications (for example one for diabetes and one for weight
    management): up to `limit` distinct labels are returned and the model decides which of them bear on the question."""
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
