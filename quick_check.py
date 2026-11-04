"""A quick, grounded first answer to 'are those drugs also used for <another condition>?', shown before the full research starts.

For each drug carried over from the earlier answers it looks at two things only: the drug's FDA label (does the indication name the condition?) and PubMed titles and abstracts (does a paper
mention both the drug and the condition?). The wording is built by code from what was found; no model writes it, so nothing is invented, and a failed lookup says so. Nothing here names a
disease or a drug."""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, wait

import conditions as cond_groups

MAX_DRUGS = 10
DEADLINE_S = 8.0


def _records(raw) -> list[dict]:
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return []
    return [r for r in (data or {}).get("results", []) if isinstance(r, dict)] if isinstance(data, dict) else []


def _text(record: dict) -> str:
    return " ".join(str(v) for v in record.values() if isinstance(v, (str, int)))


def check_drug(drug: str, condition: str, fda_lookup, pubmed_lookup) -> dict:
    """{'drug', 'label': the FDA indication sentence naming the condition or '', 'papers': how many abstracts mention both, 'error': what could not be looked up}."""
    group = [{"name": condition, "stems": cond_groups.stems_of(condition)}]
    out = {"drug": drug, "label": "", "papers": 0, "pmids": [], "error": []}
    try:
        for r in _records(fda_lookup(drug)):
            text = str(r.get("indication_statement") or r.get("indication_excerpt") or "")
            if cond_groups.tag_indexes(text, group):
                sentence = next((s for s in re.split(r"(?<=[.!?])\s+", text) if cond_groups.tag_indexes(s, group)), text)
                out["label"] = " ".join(sentence.split())[:260]
                break
    except Exception:
        out["error"].append("FDA label")
    try:
        name_stem = drug.lower().split()[0][:6]
        for p in _records(pubmed_lookup(drug, condition)):
            text = _text(p)
            if name_stem in text.lower() and cond_groups.tag_indexes(text, group):
                out["papers"] += 1
                if p.get("pmid"):
                    out["pmids"].append(str(p["pmid"]))
    except Exception:
        out["error"].append("PubMed")
    return out


def markdown(results: list[dict], condition: str) -> str:
    """The quick answer: one sentence that says what was found overall, then one line per drug."""
    if not results:
        return ""
    approved = [r for r in results if r["label"]]
    studied = [r for r in results if not r["label"] and r["papers"]]
    unknown = [r for r in results if not r["label"] and not r["papers"] and r["error"]]
    none = [r for r in results if not r["label"] and not r["papers"] and not r["error"]]
    if approved:
        head = f"**Quick answer: for {len(approved)} of these {len(results)} drugs an FDA label names {condition}.**"
    elif studied:
        head = f"**Quick answer: no FDA label of these drugs names {condition}; papers that study some of them for it were found.**"
    elif unknown and not none:
        head = f"**Quick answer: this could not be checked for {', '.join(r['drug'] for r in unknown)}.**"
    else:
        head = f"**Quick answer: no. None of these drugs has an FDA label that names {condition}, and no paper was found that studies them for it.**"
    lines = []
    for r in results:
        if r["label"]:
            lines.append(f"- **{r['drug']}:** yes, its FDA label names {condition}: \"{r['label']}\"")
        elif r["papers"]:
            refs = ", ".join(f"[PMID {p}](https://pubmed.ncbi.nlm.nih.gov/{p}/)" for p in r["pmids"][:3])
            lines.append(f"- **{r['drug']}:** not FDA-labelled for {condition}; {r['papers']} paper(s) mention it with {condition}: {refs}. Read the research below before drawing a conclusion.")
        elif r["error"]:
            lines.append(f"- **{r['drug']}:** could not be checked ({', '.join(r['error'])} lookup failed).")
        else:
            lines.append(f"- **{r['drug']}:** no FDA label names {condition}, and no paper was found that studies it for {condition}.")
    return (head + "\n\n" + "\n".join(lines) + "\n\n*A quick check of each drug's FDA label and of PubMed titles and abstracts only; it does not prove that a drug is never used for "
            f"{condition}. The full research follows.*\n\n---\n\n")


def run(drugs: list[str], condition: str, fda_lookup, pubmed_lookup, deadline: float = DEADLINE_S) -> str:
    """The quick answer as Markdown ('' when there is nothing to check). The lookups run side by side and a drug whose lookups do not finish in time is reported as not checked."""
    names = [d for d in dict.fromkeys(drugs or []) if d][:MAX_DRUGS]
    if not names or not cond_groups.stems_of(condition):
        return ""
    pool = ThreadPoolExecutor(max_workers=min(8, len(names)))
    try:
        futures = {pool.submit(check_drug, d, condition, fda_lookup, pubmed_lookup): d for d in names}
        done, _ = wait(list(futures), timeout=deadline)
        results = [f.result() if f in done else {"drug": futures[f], "label": "", "papers": 0, "pmids": [], "error": ["lookup timed out"]} for f in futures]
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return markdown(results, condition)


def default_fda_lookup(drug: str) -> str:
    from clinical_tools import search_fda_labels
    return search_fda_labels([], [drug], 1)


def default_pubmed_lookup(drug: str, condition: str) -> str:
    from clinical_tools import search_pubmed_quick
    words = " OR ".join(f'"{w}"[tiab]' for w in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", condition) if w.lower() not in cond_groups.FILLER) or condition
    return search_pubmed_quick(f'"{drug}"[tiab] AND ({words})', 6, 0, "relevance")
