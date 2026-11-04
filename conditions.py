"""Questions that name two or more separate conditions.

The planner may list the conditions a question asks about. Code keeps only a condition whose words are in the question, and then uses its word stems to say, for every drug, label, announcement and
trial, which condition it is for. Nothing here names a disease or a drug: the conditions come from the question. A question with fewer than two conditions gets an empty list, and every caller
then behaves exactly as it does for a one-condition question."""
from __future__ import annotations

import re

MAX_CONDITIONS = 4
FILLER = {"disorder", "disorders", "disease", "diseases", "syndrome", "syndromes", "spectrum", "condition", "conditions", "chronic", "acute", "symptom", "symptoms", "children", "child",
          "pediatric", "paediatric", "adolescent", "adolescents", "adult", "adults", "infant", "infants", "patients", "patient", "treatment", "therapy", "treat", "management", "drug", "drugs",
          "with", "and", "the", "for", "from", "that", "this", "in", "of", "or", "type", "related", "associated", "primary", "secondary", "severe", "mild", "moderate", "major"}


def _words(text) -> list[str]:
    return re.findall(r"[a-z][a-z0-9\-]{1,}", str(text or "").lower())


def stems_of(text) -> list[str]:
    """The first five letters of each content word (a short word such as an abbreviation is kept whole), so 'depression' and 'depressive' meet."""
    out: list[str] = []
    for w in _words(text):
        if w in FILLER or len(w) < 3:
            continue
        s = w[:5]
        if s not in out:
            out.append(s)
    return out


def _matches(stem: str, words: set[str]) -> bool:
    return any(w == stem or (len(stem) >= 5 and w.startswith(stem)) for w in words)


def clean_conditions(raw, question: str, limit: int = MAX_CONDITIONS) -> list[dict]:
    """The planner's list -> [{'name', 'stems'}], or [] when the question does not name at least two different conditions. A condition is kept only when one of its own words (or one of its
    abbreviations) is written in the question; two entries with the same words count once."""
    if not isinstance(raw, list):
        return []
    q_words = set(_words(question))
    out: list[dict] = []
    for item in raw:
        if isinstance(item, dict):
            name, aliases = str(item.get("name") or item.get("condition") or ""), [str(a) for a in (item.get("aliases") or item.get("also_called") or []) if isinstance(a, str)]
        else:
            name, aliases = str(item or ""), []
        name = " ".join(name.split())[:80]
        stems = stems_of(name)
        if not name or not stems:
            continue
        if not any(_matches(s, q_words) for s in stems):
            continue      # the question never writes this condition: the model made it up
        for alias in aliases[:6]:
            for s in stems_of(alias)[:3]:
                if s not in stems and len(s) >= 3:
                    stems.append(s)
        if any(set(stems[:1]) & set(g["stems"][:1]) for g in out):
            continue      # the same condition written twice
        out.append({"name": name, "stems": stems})
    return out[:limit] if len(out) >= 2 else []


def groups_of(holder: dict | None) -> list[dict]:
    """The cleaned condition list stored on a plan or on the state (empty for a one-condition question)."""
    holder = holder or {}
    found = holder.get("conditions") or (holder.get("entities") or {}).get("conditions") or []
    return [g for g in found if isinstance(g, dict) and g.get("name") and g.get("stems")] if isinstance(found, list) else []


def names(groups: list[dict]) -> list[str]:
    return [g["name"] for g in groups]


def tag_indexes(text, groups: list[dict]) -> list[int]:
    """Positions (in the condition list) of the conditions a text is about: one whose stem starts a word of the text."""
    words = set(_words(text))
    return [i for i, g in enumerate(groups) if any(_matches(s, words) for s in g["stems"])]


def tag_label(text, groups: list[dict], empty: str = "Not specified") -> str:
    found = tag_indexes(text, groups)
    return " / ".join(groups[i]["name"] for i in found) if found else empty


def sort_index(text, groups: list[dict]) -> int:
    found = tag_indexes(text, groups)
    return found[0] if found else len(groups)


def interleave(items: list, text_of, groups: list[dict]) -> list:
    """One item per condition in turn (the first of each condition, then the second of each, ...), items of no condition last: a cap on the list then keeps every condition."""
    if not groups:
        return list(items)
    buckets: list[list] = [[] for _ in groups] + [[]]
    for it in items:
        buckets[sort_index(text_of(it), groups)].append(it)
    out: list = []
    depth = max((len(b) for b in buckets), default=0)
    for n in range(depth):
        for b in buckets:
            if n < len(b):
                out.append(b[n])
    return out


def search_terms(group: dict) -> list[str]:
    """The words of a condition's name that a label search can use."""
    return [w for w in _words(group.get("name")) if w not in FILLER and len(w) >= 4][:3]
