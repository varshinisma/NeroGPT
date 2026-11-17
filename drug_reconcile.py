"""Drug identity, use type and approval consistency for the drug tables. Nothing here knows a drug: names, brands and approvals come from what the run retrieved (FDA label records, regulator pages, sources).

- identity: one drug, however it is written (generic, brand, salt, biologic suffix, dosage form): `key`, `Aliases`, `merge_records`
- use type: is a label sentence or a source sentence about PREVENTING the condition or about treating an ACUTE episode: `use_kind`, `asks_intent`
- status by regulator: `region_status` (FDA from the label; EMA and CDSCO only from a retrieved regulator page, else "not verified")
- evidence: `evidence_kind` (study design), `population_tags`
- checks before the report is written: `validate`
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

# words that are the form or the salt of a product, not its identity
_NOISE = frozenset("""tablet tablets capsule capsules injection injectable solution suspension oral extended release extended-release delayed delayed-release usp for kit prefilled pen syringe autoinjector film coated
chewable orally disintegrating nasal spray powder intravenous subcutaneous topical cream gel patch and sodium potassium calcium magnesium hydrochloride hcl succinate maleate tartrate mesylate besylate sulfate
phosphate acetate citrate fumarate bromide chloride dihydrate monohydrate anhydrous hemihydrate in with""".split())
_BIOLOGIC_SUFFIX = re.compile(r"^([a-z]{6,})-[a-z]{4}$")      # an FDA biologic's four-letter suffix: 'eptinezumab-jjmr'


def _ingredient(text: str) -> str:
    words = []
    for w in re.sub(r"[^a-z0-9\- ]+", " ", str(text or "").lower()).split():
        found = _BIOLOGIC_SUFFIX.match(w)
        w = found.group(1) if found else w.strip("-")
        if w and w not in _NOISE and not w.isdigit() and not re.fullmatch(r"\d+(?:mg|mcg|ml|g)", w):
            words.append(w)
    return " ".join(words)


def key(name: str) -> str:
    """The identity of one written name ('Emgality', 'galcanezumab-gnlm', 'Divalproex Sodium Delayed-Release Tablets'): its ingredients, forms and salts removed, a combination's ingredients sorted."""
    outside = re.sub(r"\([^)]*\)", " ", str(name or ""))
    parts = sorted({k for k in (_ingredient(p) for p in re.split(r"[,+/;]| and ", outside)) if k})
    return "+".join(parts)


def search_names(name: str) -> list[str]:
    """The names to look a drug up by, from a name as a source writes it: 'AJOVY® (fremanezumab-vfrm)' -> ['fremanezumab-vfrm', 'AJOVY'] (marks and brackets would make the label API refuse the query)."""
    text = re.sub(r"[®™©*]", "", str(name or ""))
    parts = [re.sub(r"[^A-Za-z0-9 \-]", " ", p).strip() for p in re.findall(r"\(([^)]*)\)", text)] + [re.sub(r"[^A-Za-z0-9 \-]", " ", re.sub(r"\([^)]*\)", " ", text)).strip()]
    out = []
    for p in parts:
        p = " ".join(p.split())
        if len(p) >= 4 and p.lower() not in {o.lower() for o in out}:
            out.append(p)
    return out


def candidates(name: str) -> list[str]:
    """Every identity a written name carries: the name itself and each parenthesised name ('Emgality (galcanezumab-gnlm)' names two)."""
    inside = re.findall(r"\(([^)]*)\)", str(name or ""))
    return [k for k in [key(name)] + [key(i) for i in inside] if k]


class Aliases:
    """Brand and generic names that the retrieved records show to be one drug. A name no record connects stays its own drug: nothing is guessed."""

    def __init__(self) -> None:
        self.canon: dict[str, str] = {}

    def link(self, *names: str) -> None:
        keys = [k for n in names for k in candidates(n)]
        if not keys:
            return
        root = self.canon.get(keys[0], keys[0])
        for k in keys:
            old = self.canon.get(k, k)
            if old != root:      # an earlier group of this name joins this one
                for other, value in list(self.canon.items()):
                    if value == old:
                        self.canon[other] = root
            self.canon[k] = root

    def of(self, name: str) -> str:
        found = candidates(name)
        for k in found:
            if k in self.canon:
                return self.canon[k]
        return found[0] if found else ""

    @classmethod
    def from_records(cls, records, labels=()) -> "Aliases":
        """records: FDA label records ({'generic', 'brands'}); labels: results of a label lookup by name ({'generic', 'generic_name', 'brand'})."""
        out = cls()
        for r in records or []:
            out.link(str(r.get("generic") or ""))
            for b in r.get("brands") or []:
                out.link(str(r.get("generic") or ""), str(b))
        for found in labels or []:
            if found:
                out.link(str(found.get("generic_name") or found.get("generic") or ""), str(found.get("brand") or ""), str(found.get("generic") or ""))
                for g in found.get("generic_names") or []:      # the label's own generic and brand names: one drug
                    out.link(str(found.get("generic_name") or found.get("generic") or ""), str(g))
                for b in found.get("brands") or []:
                    out.link(str(found.get("generic_name") or found.get("generic") or ""), str(b))
        return out


def brand_of(name: str, found: dict | None = None) -> str | None:
    """The brand in a written name, when the name or its label lookup shows it is a brand: 'Aimovig (erenumab-aooe)' -> 'Aimovig'."""
    outside = re.sub(r"\s*\([^)]*\)", "", str(name or "")).strip()
    brands = {str(b).lower() for b in ((found or {}).get("brands") or [])} | {str((found or {}).get("brand") or "").lower()}
    return outside if outside and ("(" in str(name) or outside.lower() in brands) else None


def label_facts(label: dict) -> dict:
    """What an openFDA label says about the drug itself, for a lookup by name: its names, its class and the text of its indications section (the label's own words)."""
    openfda = (label or {}).get("openfda") or {}
    text = " ".join(" ".join((label or {}).get("indications_and_usage") or []).split())
    return {"generic_names": list(openfda.get("generic_name") or [])[:4], "brands": list(openfda.get("brand_name") or [])[:4], "indications": text[:3000],
            "drug_class": "; ".join(re.sub(r"\s*\[(?:EPC|MoA|PE|CS)\]\s*$", "", c) for c in (openfda.get("pharm_class_epc") or [])[:2]) or None}


_NEGATED = re.compile(r"(?i)\b(?:not|no|never)\b[^.;]{0,50}\b(?:indicated|recommended|approved|established)\b|\blimitations? of use\b")


def label_names_condition(indications: str, stems) -> str | None:
    """The sentence of a label's indications text that names the condition (and does not say the drug is not indicated for it), or None. The label's own words decide: no list of drugs."""
    for piece in re.split(r"(?<=[.;])\s+|\s*[•●]\s*", " ".join(str(indications or "").split())):
        if any(s and s.lower() in piece.lower() for s in stems or ()) and not _NEGATED.search(piece):
            return piece[:400]
    return None


# ------------------------------------------------------------------ merging the label records of one drug
def merge_records(records: list[dict], aliases: Aliases) -> list[dict]:
    """One record for each drug: the first record (the search's own order) keeps its fields; brand names are joined and the other labels of the same drug are kept in `other_labels`
    (a drug with several manufacturers' or forms' labels is one drug)."""
    merged: dict[str, dict] = {}
    order: list[str] = []
    for r in records:
        k = aliases.of(str(r.get("generic") or "")) or str(r.get("generic") or "")
        if k not in merged:
            merged[k] = dict(r, brands=list(r.get("brands") or []), other_labels=[])
            order.append(k)
            continue
        first = merged[k]
        for b in r.get("brands") or []:
            if b.lower() not in {x.lower() for x in first["brands"]}:
                first["brands"].append(b)
        if r.get("url") and r["url"] != first.get("url"):
            first["other_labels"].append({"url": r["url"], "label_date": r.get("label_date"), "name": r.get("generic")})
        for field in ("use_kind", "drug_class", "indication_statement", "dose_statement", "dose_general", "route"):
            if not first.get(field) and r.get(field):
                first[field] = r[field]
        if r.get("use_kind") and first.get("use_kind") not in (None, "", r["use_kind"]) and r["use_kind"] != "not stated":
            first["use_kind"] = "both" if first.get("use_kind") != "not stated" else r["use_kind"]
    return [merged[k] for k in order]


# ------------------------------------------------------------------ what a label or a source sentence is for
PREVENT = re.compile(r"(?i)\b(?:prevent\w*|prophyla\w*|reduc\w*\s+(?:the\s+)?(?:frequency|number|incidence)\b|decreas\w*\s+(?:the\s+)?(?:frequency|number)\b)")
ACUTE = re.compile(r"(?i)\b(?:acute\s+(?:treatment|therapy|use|attacks?|episodes?|migraine)|abortive|(?:treatment|relief)\s+of\s+acute|at\s+the\s+(?:onset|first sign)|as[\s-]needed|on[\s-]demand|rescue)\b")


def use_kind(text: str, stems=()) -> str:
    """'preventive', 'acute', 'both' or 'not stated': read from the sentences of an indication text that name the condition (all sentences when no stem is given)."""
    pieces = [p for p in re.split(r"(?<=[.;])\s+|\s*[•●]\s*", " ".join(str(text or "").split())) if p]
    named = [p for p in pieces if not stems or any(s and s.lower() in p.lower() for s in stems)] or pieces
    blob = " ".join(named)
    prevent, acute = bool(PREVENT.search(blob)), bool(ACUTE.search(blob))
    return "both" if prevent and acute else "preventive" if prevent else "acute" if acute else "not stated"


def asks_intent(question: str) -> str | None:
    """What the question asks for, when it says: 'preventive' or 'acute'; None when it asks about neither (no separation is made)."""
    prevent, acute = bool(PREVENT.search(question or "")), bool(re.search(r"(?i)\b(?:acute|abortive|rescue|attack|relief)\b", question or ""))
    return "preventive" if prevent and not acute else "acute" if acute and not prevent else None


def in_scope(kind: str, intent: str | None) -> bool:
    """Is a drug of this use kind part of the main list for what the question asks? A drug whose kind is not stated stays (it is marked), a drug used for both stays."""
    if intent is None or kind in ("both", "not stated", "", None):
        return True
    return kind == intent


def kind_label(kind: str) -> str:
    return {"preventive": "Preventive", "acute": "Acute", "both": "Preventive and acute"}.get(kind or "", "Type not stated")


# ------------------------------------------------------------------ approval status by regulator
_REGULATORS = (("FDA (US)", ("fda.gov",)), ("EMA (EU)", ("ema.europa.eu",)), ("CDSCO (India)", ("cdsco.gov.in", "cdsco.nic.in")))


def region_status(canon: str, aliases: Aliases, fda_record: dict | None, regulator_items: list[dict] | None, fda_note: str | None = None) -> str:
    """'FDA (US): approved for the use shown (label date); EMA (EU): not verified; CDSCO (India): not verified'. FDA comes from a retrieved label; another regulator only from a retrieved page of its own
    that names the drug. An approval by one regulator is never carried over to another."""
    parts = []
    for name, hosts in _REGULATORS:
        if name.startswith("FDA") and fda_record is None and fda_note:
            parts.append(f"{name}: {fda_note}")
            continue
        if name.startswith("FDA") and fda_record is not None:
            parts.append(f"{name}: approved for the use shown (label {fda_record.get('label_date') or 'date not retrieved'})")
            continue
        hit = None
        for item in regulator_items or []:
            host = (urlparse(str(item.get("url") or "")).hostname or "").lower()
            if any(host == h or host.endswith("." + h) for h in hosts) and aliases.of(str(item.get("drug") or "")) == canon:
                hit = item
                break
        if hit:
            parts.append(f"{name}: {str(hit.get('classification') or 'mentioned').replace('_', ' ')} ({hit.get('date') or 'date not stated'})")
        else:
            parts.append(f"{name}: " + ("no label found for this use" if name.startswith("FDA") else "not verified"))
    return "; ".join(parts)


# ------------------------------------------------------------------ evidence
_DESIGNS = (("Meta-Analysis", "meta-analysis"), ("Systematic Review", "systematic review"), ("Randomized Controlled Trial", "randomized controlled trial"), ("Practice Guideline", "guideline"),
            ("Guideline", "guideline"), ("Clinical Trial", "clinical trial"), ("Observational Study", "observational study"), ("Review", "review"), ("Case Reports", "case report"))


def evidence_kind(source_kind: str, study_types) -> str:
    """The source's design in words (from PubMed's publication types), or the kind of page when it is not a paper; 'design not stated' otherwise: never inferred."""
    if source_kind == "guideline":
        return "guideline"
    types = [str(t) for t in (study_types or [])]
    for needle, word in _DESIGNS:
        if any(needle.lower() == t.lower() or needle.lower() in t.lower() for t in types):
            return word
    return {"paper": "published paper (design not stated)", "web": "web page"}.get(source_kind, "source (design not stated)")


_POP = re.compile(r"(?i)\b(episodic|chronic|refractory|resistant|adults?|children|pediatric|paediatric|adolescents?|elderly|infants?|pregnan\w+)\b")


def population_tags(*texts: str) -> str:
    """The patient groups the text itself names ('episodic; adults'), or 'not stated'."""
    found = []
    for t in texts:
        for m in _POP.finditer(str(t or "")):
            w = m.group(1).lower()
            w = "adults" if w == "adult" else "adolescents" if w == "adolescent" else w
            if w not in found:
                found.append(w)
    return "; ".join(found) if found else "not stated"


# ------------------------------------------------------------------ checks
def validate(rows: list[dict], aliases: Aliases, intent: str | None, report_text: str = "", known_names: list[str] | None = None) -> list[str]:
    """Problems left in the drug tables. rows: dicts with drug, category, kind, source (the listed rows, FDA and others). Returns plain sentences; an empty list means every check passed."""
    problems = []
    seen: dict[str, dict] = {}
    for r in rows:
        k = aliases.of(str(r.get("drug") or ""))
        if not k:
            continue
        if k in seen:
            problems.append(f"{r['drug']} is listed more than once (also as {seen[k]['drug']}).")
            continue
        seen[k] = r
    approved = {k for k, r in seen.items() if str(r.get("category") or "").startswith("FDA-approved")}
    for k, r in seen.items():
        cat = str(r.get("category") or "")
        if k in approved and cat.startswith(("Off-label", "Investigational", "Studied")):
            problems.append(f"{r['drug']} is both FDA-approved and shown as {cat.split('(')[0].strip().lower()}.")
        if intent and r.get("kind") == "acute" and intent == "preventive" and not r.get("separate"):
            problems.append(f"{r['drug']} is an acute treatment but is listed among the preventive drugs.")
        if not (r.get("source") or r.get("url")):
            problems.append(f"{r['drug']} has a classification but no supporting source.")
    text = " ".join((report_text or "").lower().split())
    for name in known_names or []:
        k = aliases.of(name)
        if k and k not in seen and re.search(rf"\b{re.escape(_ingredient(name).split(' ')[0] if _ingredient(name) else name.lower())}\b", text) and len(_ingredient(name)) >= 5:
            problems.append(f"{name} is discussed in the report but is not in the drug table.")
            seen[k] = {"drug": name}      # said once
    return problems
