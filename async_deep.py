"""Async twins of the deep-search job builders of qa_stream.py: deep_jobs, plan_pubmed_jobs, mesh_jobs, retry_jobs, search_fda_labels_wide, best_of_recent and run_jobs. A job is an async function with no
arguments; the query text, the cache keys and the search arguments are exactly those of the threaded version (the planning helpers are imported from qa_stream, not copied)."""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from datetime import datetime

import async_core
import async_support as asu
import async_tools as at
import async_web
import candidate_drugs
import mesh_expand
import observability
import qa_stream as qs
from async_pipeline import acached, awith_retry


async def abest_of_recent(query: str, max_results: int, years_back: int = 2) -> str:
    """Twin of qa_stream.best_of_recent: PubMed's best matches within the last years, listed newest first."""
    raw = await at.asearch_pubmed(query, max_results=max_results, years_back=years_back, sort="relevance")
    data = qs._json(raw)
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list) or not results:
        return raw
    year = lambda p: int(p["year"]) if str(p.get("year") or "").isdigit() else 0
    data["results"] = sorted(results, key=year, reverse=True)
    return json.dumps(data, ensure_ascii=False)


async def arouted_pubmed(query: str, max_results: int = 8, years_back: int = 2, sort: str = "relevance") -> str:
    """Twin of qa_stream.routed_pubmed."""
    return await abest_of_recent(query, max_results, years_back) if sort == "date" else await at.asearch_pubmed(query, max_results=max_results, years_back=years_back, sort=sort)


async def asearch_fda_labels_wide(fda_terms: list[str], drugs: list[str], question: str, broad: bool = True) -> str:
    """Twin of qa_stream.search_fda_labels_wide: the main label search, then one search for each associated word, all of those at the same time."""
    if broad:
        drugs = list(dict.fromkeys(list(drugs or []) + candidate_drugs.label_names(question)))
    main = await at.asearch_fda_labels(fda_terms, drugs, 16 if broad else 8)
    try:
        merged = json.loads(main)
    except ValueError:
        return main
    merged_fail = merged if "results" not in merged else None
    seen = {(r.get("generic") or "").lower() for r in merged.get("results", [])}
    searched = set(merged.get("searched_terms") or [])
    singles = [t for t in dict.fromkeys((list(qs.comorbid_terms(question)) if broad else []) + list(qs.associated_terms(question, fda_terms, 6 if broad else 3)) + [str(x) for x in reversed(fda_terms)]) if t and t not in searched and len(t) >= 4][:10 if broad else 4]

    async def one_term(term: str):
        try:
            return term, json.loads(await at.asearch_fda_labels([term], None, 3))
        except Exception:
            return term, None
    for term, extra in await asyncio.gather(*[one_term(t) for t in singles]):
        if not extra:
            continue
        for r in extra.get("results", []):
            key = (r.get("generic") or "").lower()
            if key and key not in seen:
                seen.add(key)
                merged.setdefault("results", []).append(r)
                merged["searched_terms"] = list(dict.fromkeys(list(merged.get("searched_terms") or []) + list(extra.get("searched_terms") or [term])))
    if merged_fail is not None and not merged.get("results"):
        return main
    merged.pop("error", None) if merged.get("results") else None
    return json.dumps(merged, ensure_ascii=False)


def amesh_jobs(plan: dict, query: str, subject: str) -> dict:
    """Twin of qa_stream.mesh_jobs: two more searches with the condition written in the MeSH vocabulary's own words."""
    terms = qs.mesh_terms(plan, subject)
    intent = qs.planned_intent_part(query)
    if not (mesh_expand.enabled() and terms):
        return {}
    asu.aprefetch(terms)
    looked_up: dict = {}
    lock = asyncio.Lock()

    async def written(target: str) -> str | None:
        async with lock:      # the two searches share one set of lookups
            if "records" not in looked_up:
                looked_up["records"] = await asu.arecords(terms, qs.MESH_LEGACY_TIMEOUT_S)
        extra = list(dict.fromkeys(c for t, record in looked_up["records"].items() for c in mesh_expand.clauses(t, record, target)))
        if not extra:
            return None
        groups = qs.subject_groups(subject)
        widened = f"({groups[0]} OR " + " OR ".join(extra) + ")"
        return widened + "".join(f" AND ({g})" for g in groups[1:]) + (f" AND {intent}" if intent else "")

    async def run(target: str, search):
        text = await written(target)
        return await search(text) if text else json.dumps({"results": [], "note": "MeSH has no heading for the condition words; nothing was added"})

    async def mesh_pubmed():
        return await run("pubmed", lambda q: acached("pm", f"{q}|mesh", 900, lambda: at.asearch_pubmed(q, max_results=10, years_back=5, sort="relevance")))

    async def mesh_epmc():
        return await run("europepmc", lambda q: acached("pm", f"{q}|emesh", 900, lambda: at.asearch_europepmc_quick(q, max_results=8, years_back=5, sort="relevance")))
    return {"pubmed_plan_mesh": mesh_pubmed, "pubmed_plan_epmc_mesh": mesh_epmc}


def aplan_pubmed_jobs(plan: dict) -> dict:
    """Twin of qa_stream.plan_pubmed_jobs: PubMed searches built from the plan."""
    if plan.get("plan_status") not in ("ok", "retried"):
        return {}
    built = qs.planned_pubmed_query(plan)
    if not built:
        return {}
    query, subject = built

    def pm(key: str, search, ttl: int = 900):
        return lambda: acached("pm", key, ttl, search)
    jobs = {"pubmed_plan_main": pm(f"{query}|pm", lambda: at.asearch_pubmed(query, max_results=12, years_back=5, sort="relevance")),
            "pubmed_plan_recent": pm(f"{subject}|prb", lambda: abest_of_recent(subject, 6))}
    if os.getenv("ENABLE_FULLTEXT_SEARCH", "true").strip().lower() not in qs.OFF_WORDS:
        jobs["pubmed_plan_epmc_fulltext"] = pm(f"{subject}|eft", lambda: at.asearch_europepmc_quick(subject, max_results=10, years_back=5, sort="relevance", fulltext=True))
    variants = plan.get("semantic_variants") or []
    if variants:
        spoken = " OR ".join(f'"{v}"' for v in variants)
        jobs["pubmed_plan_variants"] = pm(f"{spoken}|v", lambda: at.asearch_pubmed(spoken, max_results=8, years_back=5, sort="relevance"))
        if "pubmed_plan_epmc_fulltext" in jobs:
            jobs["pubmed_plan_epmc_variants"] = pm(f"{spoken}|evf", lambda: at.asearch_europepmc_quick(spoken, max_results=8, years_back=5, sort="relevance", fulltext=True))
    groups = qs.subject_groups(subject)
    scope = " AND ".join(f"({g})" for g in groups[1:]) or f"({groups[0]})"
    for number, topic in enumerate((plan.get("related_topics") or [])[:4], 1):
        text = f"({topic}) AND {scope}"
        jobs[f"pubmed_plan_topic_{number}"] = pm(f"{text}|tp", lambda t=text: at.asearch_pubmed(t, max_results=6, years_back=5, sort="relevance"))
    needs = [n for n in ((plan.get("understanding") or {}).get("info_needed") or []) if n in qs.NEED_TERMS][:3]
    for need in needs:
        text = f"{subject} AND ({qs.NEED_TERMS[need]})"
        jobs[f"pubmed_plan_{need}"] = pm(f"{text}|pn", lambda t=text: at.asearch_pubmed(t, max_results=8, years_back=5, sort="relevance"))
    jobs.update(amesh_jobs(plan, query, subject))
    return jobs


def adeep_jobs(plan: dict, question: str) -> dict:
    """Twin of qa_stream.deep_jobs: {job name: async function}."""
    year, cond = datetime.now().year, plan["condition"]
    built = qs.planned_pubmed_query(plan) if plan.get("plan_status") in ("ok", "retried") else None
    pq = built[1] if built else qs.pubmed_query_from_question(question)[0]
    quality = "(randomized controlled trial[pt] OR meta-analysis[pt] OR systematic review[pt] OR guideline[pt])"
    negative = '("no significant"[tiab] OR failed[tiab] OR negative[tiab] OR hepatotoxicity[tiab] OR "liver injury"[tiab] OR futility[tiab] OR "safety signal"[tiab])'
    nondrug = '(behavioral[tiab] OR psychosocial[tiab] OR "early intervention"[tiab] OR "parent-mediated"[tiab] OR rehabilitation[tiab] OR exercise[tiab] OR surgery[tiab] OR diet[tiab])'
    planned_terms = qs.keywords(qs.remove_age_expressions(cond))[0] if plan.get("plan_status") in ("ok", "retried") else []
    fda_terms = planned_terms or qs.keywords(qs.remove_age_expressions(question))[0] or re.findall(r"[A-Za-z]{4,}", cond)
    acronyms = [m.lower().rstrip("s") if m[-1:] == "s" and m[:-1].isupper() else m.lower() for m in re.findall(r"\b[A-Z]{3,}s?\b", question)]
    fda_terms = list(dict.fromkeys(acronyms + list(fda_terms)))[:5]

    def pm(key, search, ttl=900):
        return lambda: acached("pm", key, ttl, search)

    def web(key, query, **kw):
        return awith_retry(lambda: acached("web", key, 300, lambda: async_web.aweb_search(query, **kw)))
    jobs = {
        "pubmed_recent": pm(f"{pq}|rb", lambda: abest_of_recent(pq, 8)),
        "pubmed_quality": pm(f"{pq}|q", lambda: at.asearch_pubmed(f"({pq}) AND {quality}", max_results=5, years_back=5, sort="relevance")),
        "pubmed_bestmatch": pm(f"{question}|bm", lambda: at.asearch_pubmed(re.sub(r"[?\"]", " ", question).strip(), max_results=6, years_back=5, sort="relevance")),
        "pubmed_freefull": pm(f"{pq}|ff", lambda: at.asearch_pubmed(f"({pq}) AND pubmed pmc[sb] AND {quality}", max_results=5, years_back=5, sort="relevance")),
        "pubmed_negative": pm(f"{pq}|n", lambda: at.asearch_pubmed(f"({pq}) AND {negative}", max_results=4, years_back=5, sort="relevance")),
        "pubmed_drugs": pm(f"{pq}|dr", lambda: at.asearch_pubmed(f"({pq}) AND (pharmacological[tiab] OR medication[tiab] OR \"drug therapy\"[tiab] OR drug[tiab]) AND {quality}", max_results=5, years_back=5, sort="relevance")),
        "pubmed_drugs_review": pm(f"{pq}|drr", lambda: at.asearch_pubmed(f"({pq}) AND (treatment[ti] OR therapy[ti] OR management[ti]) AND (review[pt] OR guideline[pt] OR \"clinical trial\"[pt])", max_results=5, years_back=5, sort="relevance")),
        "trials_drugs": awith_retry(lambda: acached("ct", f"{plan['trial_condition']}|drug-only", 300, lambda: at.asearch_clinical_trials(plan["trial_condition"], "", "", 12, drug_only=True))),
        "trials_deep": awith_retry(lambda: acached("ct", f"{plan['trial_condition']}|{plan['trial_intervention']}", 300, lambda: at.asearch_clinical_trials(plan["trial_condition"], plan["trial_intervention"], "", 10))),
        "fda_deep": awith_retry(lambda: acached("fda", f"{' '.join(fda_terms)}|{','.join(plan.get('drugs') or [])}|wide{'' if not plan.get('focus') else '|f'}", 3600,
                                                lambda: asearch_fda_labels_wide(fda_terms, plan.get("drugs") or [], question, broad=not plan.get("focus")))),
        "web_approvals": web(f"{cond}|a", f"{cond} newly approved drug FDA EMA approval {year - 1} {year}"),
        "web_medications": web(f"{cond}|m", f"medications commonly prescribed for {cond}: which drugs are used for its symptoms and related conditions", snippet_chars=1500),
        "web_named_drugs": web(f"{cond}|nd", f"{cond} treatment: the specific drugs by generic name, including anti-infective, immune-directed, anti-inflammatory and symptom-directed medications", snippet_chars=1500),
        "web_symptom_drugs": web(f"{cond}|sd", f"{cond} co-occurring conditions and associated symptoms: which medications are prescribed for each and in which patients", snippet_chars=1500),
        "pubmed_drugs_symptoms": pm(f"{pq}|dsy", lambda: at.asearch_pubmed(f"({pq}) AND (comorbid*[tiab] OR symptom*[tiab] OR \"co-occurring\"[tiab]) AND (pharmacotherapy[tiab] OR medication*[tiab] OR drug*[tiab])", max_results=5, years_back=5, sort="relevance")),
        "web_regulatory": awith_retry(lambda: acached("web", f"{cond}|reg", 300, lambda: async_web.aregulator_search_many([f"FDA press announcement action treatment {cond}", f"{cond} drug approval label expansion safety communication FDA"]))),
        "web_standard": web(f"{cond}|s", f"{cond} treatment guideline first-line drugs dosage"),
    }
    seeds = candidate_drugs.other_names(question + " " + cond)
    if seeds and not plan.get("focus"):
        jobs["web_seed_supportive"] = web(f"{cond}|seed", f"{cond}: use of {', '.join(seeds[:6])} in children, with indication, dose and evidence", snippet_chars=1500)
    gl_words = [w for w in qs.keywords(question)[0] if w not in qs.GENERIC] or qs.keywords(question)[0]

    async def guidelines():
        async def search():
            return json.dumps(await async_web.afind_guidelines(question, gl_words, async_web.aguideline_search), ensure_ascii=False)
        return await acached("gl", f"{' '.join(gl_words)}|{[t['id'] for t in qs.topics_for(question)]}", 3600, search)
    jobs["guidelines_deep"] = awith_retry(guidelines)
    if not plan.get("drugs"):
        jobs["pubmed_nondrug"] = pm(f"{pq}|d", lambda: at.asearch_pubmed(f"({pq}) AND {nondrug} AND {quality}", max_results=4, years_back=5, sort="relevance"))
    jobs.update(aplan_pubmed_jobs(plan))
    return jobs


def aretry_jobs(question: str, plan: dict) -> dict:
    """Twin of qa_stream.retry_jobs: retries of the fast sources that failed, built from the planner's condition."""
    kw, _, _ = qs.keywords(qs.remove_age_expressions(str(plan.get("condition") or "")))
    if not kw:
        return {}
    cond = " ".join(kw)
    return {
        "trials_fast": lambda: acached("ct", cond, 300, lambda: at.asearch_clinical_trials(cond, "", "", 4)),
        "trials_drugs_fast": lambda: acached("ct", f"{cond}|drug-fast", 300, lambda: at.asearch_clinical_trials(cond, "", "", 4, drug_only=True)),
        "fda_fast": lambda: acached("fda", cond, 3600, lambda: at.asearch_fda_labels(kw, None, 4)),
        "web_fast": lambda: acached("web", cond, 300, lambda: async_web.aweb_search(f"{cond} FDA approved treatment drugs")),
    }


async def arun_jobs(jobs: dict, timeout: float, essential: list[str] | None = None, extra_until: float = 0.6) -> tuple[dict, dict]:
    """Twin of qa_stream.run_jobs: the jobs run at the same time; the essential ones are waited for up to `timeout`, the others get only a short grace; a job still running is marked partial (and cancelled)."""
    started = time.time()

    async def timed(name, afn):
        t0 = time.time()
        try:
            with observability.tool(name):
                return await afn(), time.time() - t0
        except Exception as error:
            return json.dumps({"error": str(error)}), time.time() - t0
    tasks = {name: asyncio.ensure_future(timed(name, fn)) for name, fn in jobs.items()}
    if tasks:
        if essential:
            must = [tasks[n] for n in essential if n in tasks]
            if must:
                await asyncio.wait(must, timeout=timeout)
            await asyncio.wait(list(tasks.values()), timeout=max(0.0, min(extra_until, timeout) - (time.time() - started)))
        else:
            await asyncio.wait(list(tasks.values()), timeout=timeout)
    raw, secs = {}, {}
    for name, task in tasks.items():
        if task.done():
            raw[name], secs[name] = task.result()[0], round(task.result()[1], 1)
        else:
            raw[name], secs[name] = json.dumps({"error": f"timed out after {timeout:.0f}s (partial)"}), timeout
            task.cancel()
    return raw, secs
