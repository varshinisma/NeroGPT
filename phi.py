"""Identifier scrubbing: remove personal identifiers BEFORE any text leaves this machine (model API, PubMed, web search, FDA, registry).

REMOVED:  names (patient, relative, clinician), IDs / MRNs / SSN-like numbers, dates of birth, phone numbers, e-mail addresses, street addresses.
KEPT:     age ("5-year-old"), sex, conditions, medications, doses, lab values, ordinary dates (e.g. a study year) - everything a literature search needs.

Names are found by a LOCAL spaCy model (nothing is sent anywhere) plus marker rules ("Mr. X", "named X", "X, 6 years old").
The result stores COUNTS of what was removed, never the removed values.

Known limits are listed in tests/phi_misses.py (run it to see which names the detector misses).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_nlp = None
_nlp_tried = False

# Eponymous diseases, scales and tests contain person names that must NOT be removed from a clinical question.
EPONYMS = {"parkinson", "alzheimer", "down", "wilson", "crohn", "huntington", "asperger", "rett", "tourette", "hodgkin", "graves", "cushing", "addison",
           "guillain", "barre", "duchenne", "becker", "marfan", "kawasaki", "lyme", "sjogren", "sjögren", "raynaud", "bell", "fabry", "gaucher", "pompe", "tay", "sachs",
           "niemann", "pick", "lewy", "angelman", "prader", "willi", "williams", "noonan", "turner", "klinefelter", "edwards", "patau", "apgar", "glasgow", "wechsler",
           "bayley", "vineland", "conners", "ados", "hashimoto", "behcet", "behçet", "paget", "meniere", "ménière", "wernicke", "korsakoff", "lennox", "gastaut", "dravet",
           "landau", "kleffner", "rasmussen", "sturge", "weber", "von", "willebrand", "hirschsprung", "burkitt", "kaposi", "ewing", "wilms", "bartter", "gitelman", "cohen"}
EPONYM_NEXT = {"disease", "syndrome", "palsy", "lymphoma", "anemia", "anaemia", "scale", "score", "test", "criteria", "index", "method", "questionnaire", "inventory",
               "checklist", "schedule", "trial", "study", "protocol", "sarcoma", "tumor", "tumour", "reflex", "sign", "phenomenon", "encephalopathy", "ataxia"}
MONTHS = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
DATE = rf"(?:\d{{1,2}}[/.\-]\d{{1,2}}[/.\-]\d{{2,4}}|\d{{4}}[/.\-]\d{{1,2}}[/.\-]\d{{1,2}}|(?:\d{{1,2}}(?:st|nd|rd|th)?\s+)?{MONTHS}\.?\s+(?:\d{{1,2}}(?:st|nd|rd|th)?,?\s+)?\d{{4}})"
AGE_AFTER = r"\d{1,3}\s?[- ]?(?:year|yr|y/o|yo|month|mo|week|day)s?[- ]?(?:old)?"
NAME = r"[A-Z][a-zA-Z'’\-]+"
NAME2 = r"[A-Z][a-z][a-zA-Z'’\-]*"

NOT_NAMES = (r"(?:Autism|Epilepsy|Asthma|Diabetes|ADHD|Cancer|Stroke|Migraine|Depression|Anxiety|Schizophrenia|Sepsis|Covid|COVID|Hypertension|What|How|Which|Who|When|Why|Does|Do|Is|Are|Can|Should|"
             r"The|This|That|These|Those|It|He|She|They|Patient|Child|Children|Infant|Adult|Parkinson|Alzheimer|Crohn|Down|Wilson|Huntington|FDA|NICE|WHO|CDC|MS|Lupus|Obesity|Pneumonia|Arthritis|Hepatitis|Malaria|Dengue|Tuberculosis|HIV|Anemia|Anaemia)")
PATTERNS: list[tuple[str, re.Pattern, str]] = [
    ("email", re.compile(r"[\w.+\-]+@[\w\-]+\.[\w.\-]+"), ""),
    ("dob", re.compile(rf"\b(?:DOB|D\.O\.B\.?|date of birth|birth ?date|born(?: on)?)\s*(?:is|was|:|-)?\s*{DATE}", re.I), ""),
    ("id", re.compile(r"\b(?:MRN|MR\s?(?:no|number|#)\.?|medical record(?: number| no\.?)?|patient (?:id|number|no\.?)|pt\.? ?id|UHID|NHS number|hospital (?:id|number)|"
                      r"social security(?: number)?|SSN|aadhaar|aadhar|passport(?: no\.?| number)?|insurance (?:id|number)|policy (?:no\.?|number)|ID)\s*(?:is|:|#|-)?\s*[A-Za-z]*\d[A-Za-z0-9\-/]{2,}", re.I), ""),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), ""),
    ("id", re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b"), ""),                                    # 12-digit national ID written in groups of four
    ("address", re.compile(r"\b(?:address|resides at|residing at|lives at|living at|home address)\s*(?:is|:|-)?\s*[^.\n;]{6,90}", re.I), ""),
    ("address", re.compile(rf"\b\d{{1,5}}[A-Za-z]?\s+(?:{NAME}\s+){{1,3}}(?:Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Boulevard|Blvd|Nagar|Colony|Marg|Way|Court|Ct)\b\.?(?:,?\s*{NAME}(?:\s{NAME})?)?(?:,?\s*\d{{5,6}})?"), ""),
    ("title_name", re.compile(rf"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof|Master|Baby|Smt|Shri|Sri|Kumari)\.?\s+{NAME}(?:\s+{NAME2})?(?:['’]s)?"), "the patient"),
    ("name_verb", re.compile(rf"\b(?!{NOT_NAMES}\b)({NAME})(?:['’]s(?=\s+[A-Za-z0-9])|(?=\s+(?:has|had|presents|presented|complains|suffers|developed|was diagnosed|is diagnosed)\b))"), "the patient"),
    ("marked_name", re.compile(rf"\b(?:patient name|name|named|called)\s*(?:is|:|-)?\s+(?!the\b|of\b)({NAME}(?:\s+{NAME}){{0,2}})"), "the patient"),
    ("name_age", re.compile(rf"\b(?!(?:Male|Female|Man|Woman|Boy|Girl|Child|Infant|Patient|Baby|Adult|Adolescent|Toddler|Age|Aged)\b){NAME}(?=\s*,\s*{AGE_AFTER})"), "the patient"),
]
PHONE_CAND = re.compile(r"(?<![\w.])\+?\(?\d[\d\s().\-]{8,16}\d(?![\w.])")
PHONE_WORD = re.compile(r"\b(?:phone|tel|telephone|mobile|cell|contact|call|ph|whatsapp)\b", re.I)


@dataclass
class ScrubResult:
    text: str
    removed: dict[str, int] = field(default_factory=dict)      # counts only: the removed values are never stored
    detector: str = "regex+spacy"

    @property
    def changed(self) -> bool:
        return bool(self.removed)


def _load_nlp():
    global _nlp, _nlp_tried
    if not _nlp_tried:
        _nlp_tried = True
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm", disable=["lemmatizer", "parser"])
        except Exception:
            _nlp = None
    return _nlp


def warm_up() -> str:
    """Load the local name detector once (about 1 s). Returns the detector in use."""
    return "regex+spacy" if _load_nlp() else "regex-only"


def _is_eponym(name: str, next_word: str) -> bool:
    words = [w.lower().strip("'’s") for w in re.findall(r"[A-Za-z'’\-]+", name)]
    return any(w in EPONYMS for w in words) or next_word.lower() in EPONYM_NEXT or next_word.lower().startswith(("'s", "’s"))


def _names_from_spacy(text: str) -> list[tuple[int, int]]:
    nlp = _load_nlp()
    if nlp is None:
        return []
    spans = []
    for ent in nlp(text).ents:
        if ent.label_ != "PERSON":
            continue
        after = text[ent.end_char:ent.end_char + 40]
        nxt = (re.match(r"\s*([A-Za-z'’]+)", after) or [None, ""])[1]
        if _is_eponym(ent.text, nxt):
            continue
        if re.match(r"\s*\d+(?:\.\d+)?\s?(?:mg|mcg|g|ml|iu|units?)\b", after, re.I):          # "Risperdal 0.5 mg": a brand name, not a person
            continue
        spans.append((ent.start_char, ent.end_char))
    return spans


def scrub(text: str) -> ScrubResult:
    """Remove identifiers; keep clinical content. Safe to call on any text (also on already-scrubbed text)."""
    removed: dict[str, int] = {}
    out = text or ""

    def bump(kind: str, n: int = 1):
        removed[kind] = removed.get(kind, 0) + n

    for kind, pattern, replacement in PATTERNS:
        def repl(m, kind=kind, replacement=replacement):
            if kind in ("title_name", "marked_name", "name_age", "name_verb") and _is_eponym(m.group(0), (re.match(r"\s*([A-Za-z]+)", out[m.end():m.end() + 20]) or [None, ""])[1]):
                return m.group(0)
            bump("name" if kind in ("title_name", "marked_name", "name_age", "name_verb") else kind)
            return replacement
        out = pattern.sub(repl, out)

    def phone_repl(m):
        digits = re.sub(r"\D", "", m.group(0))
        near = PHONE_WORD.search(out[max(0, m.start() - 25):m.start()])
        separated = bool(re.search(r"[\s().\-]", m.group(0).strip()))
        if 10 <= len(digits) <= 13 and (near or separated or m.group(0).lstrip().startswith(("+", "0"))) and not re.search(r"\d\.\d", m.group(0)):
            bump("phone")
            return ""
        return m.group(0)
    out = PHONE_CAND.sub(phone_repl, out)
    out = re.sub(r"\b(?:phone|tel|telephone|mobile|cell|contact|call|whatsapp)(?:\s+(?:no|number)\.?)?\s*[:\-]\s*(?=\s|$|[,;.])", "", out, flags=re.I)   # label left behind by a removed number

    spans = _names_from_spacy(out)
    for start, end in sorted(spans, reverse=True):          # right to left keeps earlier offsets valid
        out = out[:start] + "the patient" + out[end:]
        bump("name")

    # tidy what is left behind without touching clinical content
    out = re.sub(r"\b(?:phone|tel|telephone|mobile|contact|call|email|e-mail|dob|mrn|id)\b\s*[:,]?\s*(?=$|[,;.]|\s(?:or|and)\b)", "", out, flags=re.I)
    out = re.sub(r"\(\s*\)|\[\s*\]", "", out)
    out = re.sub(r"(the patient)(?:['’]s)?(?:\s*,?\s*(?:the patient))+", r"\1", out)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    out = re.sub(r"([,;:])\s*(?=[,;:.!?])", "", out)
    out = re.sub(r"^\s*[,;:\-]+\s*", "", out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return ScrubResult(out, removed, "regex+spacy" if _nlp else "regex-only")


def notice(result: ScrubResult) -> str:
    kinds = ", ".join(sorted(result.removed))
    return f"Personal identifiers were removed from your question before searching ({kinds}). Age, sex, conditions, medicines and lab values were kept."
