"""The drug agent: the drug side of a new question, run beside the main pipeline from the moment the plan exists.

What it does (all of it in this file, nothing in it knows a disease or a drug):
  1. QUERIES  one model call with the drug-intelligence SKILL.md as its prompt turns the plan into expanded searches (PubMed queries and web queries aimed at the specific medicines, their classes, brands,
              comorbid symptoms, guidelines and approvals). When the call fails, plain queries built from the plan's condition are used.
  2. SEARCH   the searches run at the same time (PubMed through its own API, the web through the configured providers) under one time limit; what is not back in time is dropped.
  3. RANK     the results are put in a PRIVATE paper pool (never the shared one the main pipeline ranks in another thread) and ranked: the hybrid ranking (semantic + keyword + source order) and then the local
              cross-encoder, which reads the question together with each paper and scores how well the paper answers it.
  4. HAND-OFF the ranked pool is left in `run.drug_agent_box` and the drug stage (async_stages_drugs) starts at once, reading these sources while the main pipeline is still searching and ranking.
              The sources the main pipeline finds later are read afterwards as a top-up, so nothing the old drug stage read is left out.

The agent never raises: when anything fails the box says so and the drug stage runs exactly as it did before (started after the ranking of the main pipeline).
Switches: DRUG_AGENT=on (default off), DRUG_AGENT_BUDGET_S (28), DRUG_AGENT_QUERY_S (10), DRUG_AGENT_SEARCH_S (12), DRUG_AGENT_CROSS (on), DRUG_AGENT_CROSS_TOP (44)."""
from __future__ import annotations

import asyncio
import json
import os
import re
import time

import async_core
import async_support as asu
import async_web
import qa_stream as qs


def enabled() -> bool:
    return os.getenv("DRUG_AGENT", "off").strip().lower() in ("on", "1", "true", "yes")


def _num(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return float(default)


def _on(name: str, default: str = "on") -> bool:
    return os.getenv(name, default).strip().lower() not in ("off", "0", "false", "no")


QUERY_TASK = (
    "You are the drug-search agent for this question (Phase A of this skill: scope, then the two-pass search plan). Write the searches that will find the SPECIFIC MEDICINES used for the condition, "
    "its symptoms and its associated problems: generic and brand names, drug classes together with their members, off-label and paediatric use when the question is about children, guidelines, "
    "and newly approved drugs. Answer with JSON: {\"pubmed\": [up to 5 PubMed queries, Boolean with [tiab] or [mh] field tags, each aimed at a DIFFERENT angle: the drug therapy of the condition; "
    "each named class or drug with the condition; the co-occurring symptoms treated with drugs; reviews and guidelines of the pharmacological treatment; newly approved drugs], "
    "\"web\": [up to 4 plain-language web queries, each a different angle: the names of the specific drugs; the drug options of the guidelines; each drug class with examples of its members; new approvals]}. "
    "Use only the condition, population and drugs given in the data. A search may name a drug; a drug is listed later only when a retrieved sentence says it is used."
)


def clean_queries(answer: dict, plan: dict, question: str) -> dict:
    """{'pubmed': [...], 'web': [...]} from the model's answer: strings only, trimmed, no repeats, at most 5 and 4. When nothing usable came back the plain queries of the condition are used."""
    def pick(value, limit):
        out = []
        for x in value if isinstance(value, list) else []:
            text = " ".join(str(x).split())[:300]
            if len(text) >= 6 and text.lower() not in {y.lower() for y in out}:
                out.append(text)
        return out[:limit]
    queries = {"pubmed": pick((answer or {}).get("pubmed"), 5), "web": pick((answer or {}).get("web"), 4)}
    cond = " ".join(str(plan.get("condition") or question).split())[:120]
    if not queries["pubmed"]:
        queries["pubmed"] = [f"({cond}) AND (pharmacotherapy OR \"drug therapy\" OR medication*)"]
    if not queries["web"]:
        queries["web"] = [f"{cond} treatment: the names of the specific drugs", f"{cond} guideline medications options"]
    return queries


def _direct(name: str):
    """The source's own API function (the fast pass's choice): the agent has a few seconds, and its queries are Boolean."""
    import async_pipeline
    return async_pipeline._fast_source(name)


async def _expand(run, plan: dict) -> dict:
    payload = {"condition": plan.get("condition"), "drugs_named_in_question": plan.get("drugs") or [], "population": plan.get("population") or plan.get("age_group") or ""}
    try:
        answer = await asyncio.wait_for(async_core.aask_json(run.question, QUERY_TASK, payload, 700, skill="drug-intelligence", purpose="queries"), timeout=_num("DRUG_AGENT_QUERY_S", 10))
    except Exception as error:
        print(f"[drug-agent] query expansion fell back to plain queries: {type(error).__name__}", flush=True)
        answer = {}
    return clean_queries(answer, plan, run.question)


async def _search(queries: dict) -> dict:
    """{job name: raw JSON text} of the searches that came back in time."""
    pubmed = _direct("asearch_pubmed_quick")
    jobs = {}
    for i, q in enumerate(queries["pubmed"]):
        jobs[f"pubmed_drugs_agent_{i}"] = (lambda q=q: pubmed(q, 8, 5, "relevance"))
    for i, q in enumerate(queries["web"]):
        jobs[f"web_drug_agent_{i}"] = (lambda q=q: async_web.aweb_search(q, 1500))
    tasks = {name: asyncio.ensure_future(fn()) for name, fn in jobs.items()}
    if tasks:
        await asyncio.wait(list(tasks.values()), timeout=_num("DRUG_AGENT_SEARCH_S", 12))
    raw = {}
    for name, task in tasks.items():
        if task.done() and not task.cancelled() and task.exception() is None:
            raw[name] = task.result()
        else:
            task.cancel()
    return raw


async def _rank(run, papers: dict, queries: dict, trace: dict) -> None:
    """Rank the PRIVATE pool: hybrid (semantic + keyword + source order), then the cross-encoder over the best of them."""
    if len(papers) < 3:
        trace["note"] = "too few papers to rank"
        return
    question = run.question
    began = time.time()
    try:
        await asu.arerank(question, papers, trace, queries=list(queries["web"])[:4])
    except Exception as error:
        trace["hybrid_error"] = f"{type(error).__name__}: {str(error)[:80]}"
    trace["hybrid_s"] = round(time.time() - began, 1)
    if _on("DRUG_AGENT_CROSS"):
        mark = time.time()
        try:
            await asu.across_rerank(f"{question} (the medications and drug treatments used)", papers, trace, top=int(_num("DRUG_AGENT_CROSS_TOP", 44)), wait=3.0)
        except Exception as error:
            trace["cross_error"] = f"{type(error).__name__}: {str(error)[:80]}"
        trace["cross_s"] = round(time.time() - mark, 1)


async def _work(run, box: dict) -> None:
    clock = run.clock
    try:
        await asyncio.wait_for(run.plan_ready.wait(), timeout=40)
    except asyncio.TimeoutError:
        pass
    plan = run.plan_box.get("plan")
    if not plan:
        box["error"] = "no plan"
        return
    box["plan_at_s"] = round(clock.now(), 1)
    queries = await _expand(run, plan)
    box["queries"], box["queries_at_s"] = queries, round(clock.now(), 1)
    raw = await _search(queries)
    box["searched_at_s"] = round(clock.now(), 1)
    papers: dict = {}
    web: list = []
    errors = qs.ingest(papers, {}, web, raw)
    own = len(papers) + len(web)      # what the agent's own searches found (the fast pass's papers below are only a bonus)
    try:
        for pmid, rec in dict(run.papers).items():      # the fast pass's papers: a copy, the shared pool itself is never read while it can be re-ranked
            papers.setdefault(pmid, dict(rec))
    except Exception:
        pass
    trace: dict = {}
    await _rank(run, papers, queries, trace)
    box["ranked_at_s"] = round(clock.now(), 1)
    if not own:
        box["error"] = "the searches returned nothing" + (f" ({'; '.join(f'{k}: {str(v)[:40]}' for k, v in list(errors.items())[:3])})" if errors else "")
        return
    box.update({"papers": papers, "web": web, "rank_trace": trace, "errors": errors, "ok": True})


async def run_agent(run) -> None:
    box = run.drug_agent_box
    try:
        await asyncio.wait_for(_work(run, box), timeout=_num("DRUG_AGENT_BUDGET_S", 28))
    except asyncio.TimeoutError:
        box.pop("ok", None)
        box["error"] = f"the agent did not finish within {_num('DRUG_AGENT_BUDGET_S', 28):.0f} s"
    except Exception as error:
        box.pop("ok", None)
        box["error"] = f"{type(error).__name__}: {str(error)[:120]}"
    finally:
        box["done_s"] = round(run.clock.now(), 1)
        run.state["drug_agent"] = {k: (v if k not in ("papers", "web") else len(v)) for k, v in box.items()}
        run.drug_agent_ready.set()
        if box.get("ok") and not run.focused:
            run.start_landscape()      # the drug stage reads the agent's sources now, not after the main pipeline's ranking


def start(run) -> None:
    """Called by the fast-search stage beside the plan and the deep search."""
    if getattr(run, "drug_agent_task", None) is not None:
        return
    run.drug_agent_box = {}
    run.drug_agent_ready = asyncio.Event()
    run.drug_agent_task = asyncio.ensure_future(run_agent(run))
