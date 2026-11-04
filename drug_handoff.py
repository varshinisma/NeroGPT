"""Step 6 of the clinical-research-orchestrator skill: drug names found in the retrieved sources go to drug-intelligence.

The label search of the first pass knows only the question's words, so it finds labels that happen to match those words and misses the medicines the papers and pages actually describe. Here one model
call reads the titles, abstracts and page snippets already retrieved and lists the medicines they describe as used for the condition, its symptoms or its co-occurring conditions. CODE then keeps a name only
if a source really writes it (no name from the model's memory reaches a search), and the labels of those names are looked up. Nothing here names a disease or a drug."""
from __future__ import annotations

import os
import re

MAX_NAMES = 18
MAX_PAPERS = 25
MAX_WEB = 10

TASK = ("From the RETRIEVED DATA sources (titles, abstracts and web page snippets that were already retrieved), list the medicines that a source describes as used to treat the question's condition, or its symptoms, "
        "or its co-occurring conditions, in any setting (approved, recommended by a guideline, off-label or used in practice). Copy each medicine's name EXACTLY as a source writes it. Never write a name that no source "
        "contains and add no medicine from your own knowledge. Leave out vaccines, devices, supplements without a drug name and non-drug therapies. Reply with ONE JSON object: "
        "{\"drugs\": [{\"name\": the name as written, \"use\": \"condition\" | \"symptom_or_comorbidity\"}]} with at most 24 names.")


def enabled() -> bool:
    return os.getenv("DRUG_HANDOFF", "on").strip().lower() not in {"off", "0", "false", "no"}


def offered(state: dict) -> dict:
    """The retrieved sources the model reads (best-ranked papers first) and the same text joined, against which every name is checked."""
    sources = []
    for p in [p for p in (state.get("papers") or {}).values() if not p.get("retracted")][:MAX_PAPERS]:
        sources.append(f"{p.get('title') or ''}. {str(p.get('abstract') or '')[:450]}".strip())
    for w in (state.get("web") or [])[:MAX_WEB]:
        sources.append(f"{w.get('title') or ''}. {str(w.get('snippet') or '')[:300]}".strip())
    sources = [s for s in sources if len(s) > 12]
    return {"sources": sources, "text": "\n".join(sources).lower()}


def verified(items, text: str, known: set[str]) -> list[dict]:
    """The listed names that a source writes word for word and that no label search has covered yet: [{'name', 'use'}], at most MAX_NAMES."""
    kept, seen = [], set(known)
    for item in items if isinstance(items, list) else []:
        name = " ".join(str((item or {}).get("name") if isinstance(item, dict) else item or "").split()).strip(" .,;")
        key = name.lower()
        if not 3 <= len(name) <= 40 or key in seen or not re.search(r"(?<![a-z])" + re.escape(key) + r"(?![a-z])", text):
            continue
        seen.add(key)
        use = (item or {}).get("use") if isinstance(item, dict) else ""
        kept.append({"name": name, "use": use if use in ("condition", "symptom_or_comorbidity") else "condition"})
    return kept[:MAX_NAMES]
