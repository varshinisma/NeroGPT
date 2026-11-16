"""The population a question names (children), applied to the drug section.

A drug belongs in the Treatment Drug Landscape when it is used for the question's condition IN THE QUESTION'S PATIENT GROUP. FDA approval is not required. For a question about children this module
  * says whether the question is about children (the words, or a stated age under 18);
  * keeps an FDA label only when the label itself supports use in children: its Pediatric Use section does not say 'not established' / 'not recommended', or the label states a paediatric age range or
    names children in the indication. A label that is silent about children, or says their safety is not established, is not a drug used for this patient group (a drug that is used in children appears
    through the retrieved sources instead, each with the sentence that says so);
  * gives the drug reader the same rule for the sources: a sentence about adults only does not count.
"""
from __future__ import annotations

import re

import landscape

PEDIATRIC_WORDS = re.compile(r"\b(child|children|childhood|pediatric|paediatric|paediatrics|pediatrics|infant|infants|toddler|toddlers|adolescent|adolescents|teen|teens|teenager|teenagers|kid|kids|neonat\w*|newborn\w*|preschool\w*|school[- ]age\w*|juvenile)\b", re.I)
NOT_FOR_CHILDREN = re.compile(r"not\s+(?:been\s+)?(?:established|recommended)|should\s+not\s+be\s+(?:used|given)|(?:contraindicated|not\s+indicated)\s+(?:in|for)\s+(?:pediatric|paediatric|children|patients\s+(?:under|younger|less))|"
                              r"has\s+not\s+been\s+studied|have\s+not\s+been\s+(?:studied|evaluated)|no\s+data", re.I)
FOR_CHILDREN = re.compile(r"(?:indicated|approved|used|recommended)[^.]{0,80}(?:pediatric|paediatric|children|adolescent|infant)|(?:pediatric|paediatric|children|adolescent|infants?)[^.]{0,80}(?:indicated|dose|dosage|mg\b)|"
                          r"(?:ages?|aged)\s+\d{1,2}\s*(?:to|-|–|through|and)\s*\d{1,2}", re.I)


def is_pediatric(question: str, state: dict | None = None) -> bool:
    age = (state or {}).get("patient_age_years")
    if age is None:
        age = landscape.patient_age_years(question)
    return bool(PEDIATRIC_WORDS.search(question or "")) or (age is not None and age < 18)


def label_supports_children(rec: dict) -> bool:
    extras = rec.get("label_extras") if isinstance(rec.get("label_extras"), dict) else {}
    pediatric_use = str(extras.get("pediatric_use") or "")
    indication = " ".join(str(rec.get(k) or "") for k in ("indication_excerpt", "indication_statement", "dose_statement"))
    if pediatric_use:
        if NOT_FOR_CHILDREN.search(pediatric_use) and not re.search(r"(?<!not )(?<!not been )(?:been|is|are|was|were)\s+(?:established|demonstrated)", pediatric_use, re.I):
            return False
        return True
    span = landscape.label_age_range(indication)
    if span and span[0] < 18:
        return True
    return bool(FOR_CHILDREN.search(indication))


def apply_population(state: dict, question: str) -> list[str]:
    """Marks the FDA label records that do not support the question's patient group as not shown. Returns the names removed (also kept in state['population_excluded'])."""
    if not is_pediatric(question, state):
        return []
    removed = []
    for rec in state.get("regulatory", []):
        if rec.get("model_relevant") and not label_supports_children(rec):
            rec["model_relevant"] = False
            rec["population_excluded"] = "the label does not support use in children"
            removed.append(rec.get("generic"))
    state["population_excluded"] = sorted(set(removed + list(state.get("population_excluded") or [])))
    return removed


def reader_note(question: str, state: dict | None = None) -> str:
    """The extra instruction for the drug reader when the question is about children."""
    if not is_pediatric(question, state):
        return ""
    return (" POPULATION: the question is about CHILDREN. List a drug only when its sentence says it is used, prescribed, recommended or studied in children, adolescents or infants (or in a group that includes them); "
            "a sentence about adults only, or about no stated age group, does not count. Put the age group the sentence names in `population`.")
