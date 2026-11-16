"""Search seeds for the drug stage: medicines clinicians commonly consider for the associated problems, treatable causes and complications of a condition
(skills/drug-intelligence/references/candidate_drugs.json). A seed says what to LOOK UP; it is never evidence. A drug is reported only when a retrieved source supports it."""
import json
import re
from pathlib import Path

_FILE = Path(__file__).parent / "skills" / "drug-intelligence" / "references" / "candidate_drugs.json"


def _groups() -> list[dict]:
    try:
        return json.loads(_FILE.read_text(encoding="utf-8")).get("groups", [])
    except (OSError, ValueError):
        return []


def matching_items(text: str) -> list[dict]:
    out, seen = [], set()
    for g in _groups():
        try:
            hit = re.search(g.get("trigger", "(?!x)x"), text or "", re.I)
        except re.error:
            hit = None
        if hit:
            for it in g.get("items", []):
                if it["drug"].lower() not in seen:
                    seen.add(it["drug"].lower())
                    out.append(it)
    return out


def label_names(text: str) -> list[str]:
    """Seed drugs that have an FDA label, to be looked up by name."""
    return [i["drug"] for i in matching_items(text) if i.get("fda_label")]


def other_names(text: str) -> list[str]:
    """Seed drugs/supplements without a US drug label: looked up in the literature and the web."""
    return [i["drug"] for i in matching_items(text) if not i.get("fda_label")]
