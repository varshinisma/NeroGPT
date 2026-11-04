"""What to do when a stage comes back empty: the orchestrator agent decides.

After a retrieval group is merged, code looks for an empty result (too few papers, no trial, no drug label). Only then is the agent asked, with the facts of the failure and the skill's failure rules,
and it can either retry a search with a query it writes (at most two retries in all, as the skill allows; the query must use the question's own words) or proceed and say what is missing. Code runs the retries
through the same search functions and merge as the first pass. When nothing is wrong the agent is not called, so a normal question pays nothing. Nothing here names a disease or a drug."""
from __future__ import annotations

import os
import re

MAX_RETRIES = 2
MIN_PAPERS = 3
OPTIONAL_STAGES = ("full_text",)      # the stages a plan may leave out
GROUP_SOURCES = {"evidence": ("pubmed",), "trials": ("trials",), "drugs": ("fda", "web")}      # the searches a retry may use for each group


def enabled() -> bool:
    return os.getenv("RECOVERY_AGENT", "on").strip().lower() not in {"off", "0", "false", "no"}


def clean_skip(value) -> list[str]:
    """The optional stages the plan leaves out (unknown names are dropped)."""
    return [s for s in dict.fromkeys(str(v).strip().lower().replace(" ", "_") for v in (value if isinstance(value, (list, tuple)) else [])) if s in OPTIONAL_STAGES]


def problems(state: dict, group: str) -> list[str]:
    """Empty results of one retrieval group, as plain sentences (nothing for a group that has content)."""
    found = []
    if group == "evidence" and len(state.get("papers") or {}) < MIN_PAPERS:
        found.append(f"only {len(state.get('papers') or {})} papers were found for the question")
    if group == "trials" and not state.get("trials_by_id"):
        found.append("no clinical trial was found")
    if group == "drugs" and state.get("medicines_needed") is not False and not state.get("regulatory"):
        found.append("no FDA drug label was found")
    return found


def report(state: dict, group: str, question: str, plan: dict, found: list[str]) -> str:
    tried = {k: (v or {}).get("retrieval_status") for k, v in (state.get("source_status") or {}).items()}
    return (f"STAGE: {group} retrieval\nQUESTION: {question}\nPLAN CONDITION: {plan.get('condition') or ''}\nPLAN DRUGS: {', '.join(map(str, plan.get('drugs') or []))}\n"
            f"PROBLEM: {'; '.join(found)}\nSEARCHES ALREADY RUN (status): {tried}\nALLOWED RETRY SOURCES: {', '.join(GROUP_SOURCES[group])}")


def words(text: str) -> set[str]:
    return {w[:5] for w in re.findall(r"[a-z]{4,}", str(text).lower())}


def clean_retries(raw: list, group: str, question: str, plan: dict) -> list[dict]:
    """The agent's retries that may run: an allowed source, a query of 3 to 200 characters that shares a word with the question or the plan's condition, no repeats, at most MAX_RETRIES."""
    allowed, known = GROUP_SOURCES.get(group, ()), words(question) | words(plan.get("condition") or "") | words(" ".join(map(str, plan.get("drugs") or [])))
    kept, seen = [], set()
    for item in raw or []:
        source, query = str(item.get("source") or "").strip().lower(), " ".join(str(item.get("query") or "").split())
        key = (source, query.lower())
        if source in allowed and 3 <= len(query) <= 200 and words(query) & known and key not in seen:
            seen.add(key)
            kept.append({"source": source, "query": query})
    return kept[:MAX_RETRIES]
