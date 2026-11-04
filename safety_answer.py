"""The answer to a follow-up about the side effects (and interactions) of the drugs of the earlier answers.

Not the drug landscape again: one table, a row per drug, built from what was retrieved FOR THAT QUESTION: the FDA label's own adverse-reaction, warning, contraindication and interaction text, and the papers and pages found for each
drug's side effects. A model writes the cells from that text only; code checks every cited paper, every link and every number against what was offered, and a cell that fails the check is replaced by the label's own sentence.
A drug the sources say nothing about gets a row that says so: it is never filled from memory."""
from __future__ import annotations

import re

SAFETY_WORDS = re.compile(r"(?i)adverse|side[- ]effect|safety|tolerab|toxicit|risk|warning|contraindicat")
NUMBER = re.compile(r"\d+(?:\.\d+)?")
MAX_PAPERS = 3
MAX_WEB = 2
CELLS = ("common", "serious", "avoid_or_monitor", "interactions")

TASK = ("Answer the question for each drug from ITS OWN sources only (its FDA label text, its papers, its pages). Reply with ONE JSON object: {\"rows\": [{\"drug\": name exactly as given, \"common\": the most common side effects, \"serious\": the serious risks and warnings, "
        "\"avoid_or_monitor\": who should not take it and what to monitor, \"interactions\": the drug interactions, \"sources\": [\"label\" when the label text was used, \"PMID <n>\" for each paper used, or the page's address for a web page]}]}. "
        "One row per drug, in the order given. Each cell is a short list of plain phrases (under 45 words) taken from that drug's sources; use the facts for children when the question or the source says so. Never add a side effect, a number or a "
        "warning that the sources do not state. When a source does not say, the cell is exactly \"not stated in the retrieved sources\". Only cite what you used.")


def _key(name) -> str:
    return re.sub(r"[^a-z]+", " ", str(name or "").lower()).strip().split(" ")[0] if str(name or "").strip() else ""


def label_views(state: dict, drug: str, question: str = "") -> list[dict]:
    """The FDA labels of one drug: the label's own text for each safety part (nothing written by a model). Homeopathic and similar unsuitable labels are left out."""
    try:
        from landscape import unsuitable_label
    except Exception:
        unsuitable_label = lambda q, r: None
    views = []
    for rec in state.get("regulatory") or []:
        names = [rec.get("generic")] + list(rec.get("brands") or [])
        if not any(_key(n) == _key(drug) for n in names if n) or unsuitable_label(question, rec) == "homeopathic product":
            continue
        text, extra = rec.get("safety_text") or {}, rec.get("label_extras") or {}
        view = {"label": rec.get("url") or "", "brand": ", ".join(rec.get("brands") or [])[:60], "boxed_warning": text.get("boxed"), "adverse_reactions": text.get("adverse") or extra.get("adverse_effects"),
                "warnings": text.get("warnings") or extra.get("precautions"), "contraindications": text.get("contraindications") or extra.get("contraindications"), "interactions": text.get("interactions") or extra.get("interactions")}
        if any(view[k] for k in ("boxed_warning", "adverse_reactions", "warnings", "contraindications", "interactions")):
            views.append(view)
    return views[:2]


def source_views(state: dict, drug: str) -> tuple[list[dict], list[dict]]:
    """The papers and the web pages that name this drug, those that talk about safety first."""
    pattern = re.compile(rf"(?<![A-Za-z]){re.escape(drug)}", re.I)
    papers = [p for p in (state.get("papers") or {}).values() if not p.get("retracted") and pattern.search(f"{p.get('title') or ''} {p.get('abstract') or ''}")]
    papers.sort(key=lambda p: 0 if SAFETY_WORDS.search(f"{p.get('title') or ''} {p.get('abstract') or ''}") else 1)
    pages = [w for w in state.get("web") or [] if w.get("url") and pattern.search(f"{w.get('title') or ''} {w.get('snippet') or ''}")]
    pages.sort(key=lambda w: 0 if SAFETY_WORDS.search(f"{w.get('title') or ''} {w.get('snippet') or ''}") else 1)
    return ([{"pmid": str(p.get("pmid")), "year": p.get("year"), "cite": str(p.get("cite") or ""), "title": str(p.get("title") or "")[:140], "text": str(p.get("abstract") or "")[:800]} for p in papers[:MAX_PAPERS]],
            [{"url": w["url"], "title": str(w.get("title") or "")[:100], "text": str(w.get("snippet") or "")[:500]} for w in pages[:MAX_WEB]])


def offered(state: dict, drugs: list[str], question: str = "") -> list[dict]:
    out = []
    for drug in drugs:
        papers, pages = source_views(state, drug)
        out.append({"drug": drug, "labels": label_views(state, drug, question), "papers": papers, "pages": pages})
    return out


def _text_of(item: dict) -> str:
    parts = []
    for label in item["labels"]:
        parts += [str(v) for k, v in label.items() if k not in ("label", "brand") and v]
    parts += [p["text"] + " " + p["title"] for p in item["papers"]] + [w["text"] + " " + w["title"] for w in item["pages"]]
    return " ".join(parts)


def _clean(cell) -> str:
    text = " ".join(str(cell or "").replace("|", "/").split())
    return text[:420]


def _first_sentences(text, limit: int = 260) -> str:
    text = " ".join(str(text or "").split())
    return (text[:limit].rsplit(" ", 1)[0] + "…") if len(text) > limit else text


def fallback_cells(item: dict) -> dict:
    """The label's own words for each cell (or 'not stated'), used when the model's cell failed the check or the model did not answer."""
    label = item["labels"][0] if item["labels"] else {}
    unsaid = "not stated in the retrieved sources"
    return {"common": _first_sentences(label.get("adverse_reactions")) or unsaid, "serious": _first_sentences(label.get("boxed_warning") or label.get("warnings")) or unsaid,
            "avoid_or_monitor": _first_sentences(label.get("contraindications"), 200) or unsaid, "interactions": _first_sentences(label.get("interactions"), 200) or unsaid}


def cite_of(paper: dict) -> str:
    return f"({paper['cite']})" if paper.get("cite") else f"[PMID {paper['pmid']}]"


def checked_row(item: dict, row: dict | None) -> tuple[dict, list[str]]:
    """One row of the table: the model's cells where each passes the check against this drug's own sources, the label's words where it does not. Returns (cells + sources, the cells that were replaced)."""
    replaced, base, source_text = [], fallback_cells(item), _text_of(item)
    known = set(NUMBER.findall(source_text))
    nothing = not (item["labels"] or item["papers"] or item["pages"])      # no source at all: nothing the model writes can be checked, so nothing is shown
    cells = {}
    for key in CELLS:
        value = "" if nothing else _clean((row or {}).get(key))
        if not value or any(n not in known for n in NUMBER.findall(value)):      # a number that is in none of this drug's sources is not shown
            cells[key] = base[key]
            if value and value != base[key]:
                replaced.append(key)
        else:
            cells[key] = value
    allowed = {f"PMID {p['pmid']}": p for p in item["papers"]}      # a paper is cited the way the whole report cites one: (Author, Journal, Year [PMID n]), which the citation check reads
    urls = {w["url"] for w in item["pages"]}
    sources = []
    for s in (row or {}).get("sources") or []:
        s = " ".join(str(s).split())
        if s.lower() == "label" and item["labels"]:
            sources.append(f"[FDA label]({item['labels'][0]['label']})" if item["labels"][0]["label"] else "FDA label")
        elif s in allowed:
            sources.append(cite_of(allowed[s]))
        elif s in urls:
            domain = re.sub(r"^https?://(?:www\.)?", "", s).split("/")[0]
            sources.append(f"[{domain}]({s})")
    if not sources:      # the sources of a fallback cell are the label (or the best paper)
        if item["labels"] and item["labels"][0]["label"]:
            sources.append(f"[FDA label]({item['labels'][0]['label']})")
        sources += [cite_of(p) for p in item["papers"][:1]]
    cells["sources"] = "; ".join(dict.fromkeys(sources)) or "none retrieved"
    return cells, replaced


FACTS_TASK = ("Answer the question for each drug from ITS OWN sources only (the parts of its FDA label, its papers, its pages). The question asks for these facts about each drug: {attributes}. "
              "Reply with ONE JSON object: {{\"rows\": [{{\"drug\": name exactly as given, \"cells\": {{<each fact exactly as written above>: a short list of plain phrases (under 45 words) taken from that drug's sources}}, "
              "\"sources\": [\"label\" when label text was used, \"PMID <n>\" for each paper used, or the page's address for a web page]}}]}}. One row per drug, in the order given. Use the facts for the patients the question names when the source says so. "
              "Never add a fact, a number or a warning that the sources do not state. When a source does not say, the cell is exactly \"not stated in the retrieved sources\". Only cite what you used.")


def facts_label_views(state: dict, drug: str, question: str = "") -> list[dict]:
    """The label parts of one drug, all of them (the question decides which one matters; the model reads only what answers it)."""
    try:
        from landscape import unsuitable_label
    except Exception:
        unsuitable_label = lambda q, r: None
    views = []
    for rec in state.get("regulatory") or []:
        names = [rec.get("generic")] + list(rec.get("brands") or [])
        if not any(_key(n) == _key(drug) for n in names if n) or unsuitable_label(question, rec) == "homeopathic product":
            continue
        parts = dict(rec.get("label_sections") or {})
        if not parts:      # a record made before the parts were kept: the safety parts it has
            text = rec.get("safety_text") or {}
            parts = {k: v for k, v in (("adverse_reactions", text.get("adverse")), ("warnings", text.get("warnings")), ("contraindications", text.get("contraindications")), ("interactions", text.get("interactions"))) if v}
        if rec.get("dose_statement") and "dosage" not in parts:
            parts["dosage"] = rec["dose_statement"]
        if parts:
            views.append({"label": rec.get("url") or "", "brand": ", ".join(rec.get("brands") or [])[:60], **parts})
    return views[:2]


def facts_markdown(question: str, drugs: list[str], state: dict, ask, attributes: list[str]) -> str:
    """A table of exactly the facts the follow-up asked for (the agent named them), one row per drug of the earlier answers: each cell from that drug's label parts, papers and pages only, checked like the side-effects table."""
    attributes = [" ".join(str(a).split())[:50] for a in attributes if str(a).strip()][:6]
    items = []
    for drug in drugs:
        papers, pages = source_views(state, drug)
        items.append({"drug": drug, "labels": facts_label_views(state, drug, question), "papers": papers, "pages": pages})
    rows_by_drug: dict = {}
    try:
        answer = ask(FACTS_TASK.format(attributes=", ".join(f'"{a}"' for a in attributes)),
                     {"question": question, "drugs": [{"drug": i["drug"], "labels": i["labels"], "papers": i["papers"], "pages": i["pages"]} for i in items]})
        for row in (answer or {}).get("rows") or []:
            if isinstance(row, dict):
                rows_by_drug.setdefault(_key(row.get("drug")), row)
    except Exception:
        answer = None
    unsaid = "not stated in the retrieved sources"
    header = ["Drug"] + [a[:1].upper() + a[1:] for a in attributes] + ["Source"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    replaced_all = {}
    for item in items:
        row = rows_by_drug.get(_key(item["drug"])) or {}
        nothing = not (item["labels"] or item["papers"] or item["pages"])
        known = set(NUMBER.findall(_text_of(item)))
        given = row.get("cells") if isinstance(row.get("cells"), dict) else {}
        values, replaced = [], []
        for a in attributes:
            value = "" if nothing else _clean(next((v for k, v in given.items() if " ".join(str(k).lower().split()) == a.lower()), ""))
            if not value or any(n not in known for n in NUMBER.findall(value)):
                if value:
                    replaced.append(a)
                value = unsaid
            values.append(value)
        if replaced:
            replaced_all[item["drug"]] = replaced
        src = checked_row({**item, "labels": [{"label": item["labels"][0]["label"]}] if item["labels"] else []}, {"sources": row.get("sources")})[0]["sources"]
        lines.append("| " + " | ".join([item["drug"]] + values + [src]) + " |")
    state["safety_answer"] = {"drugs": drugs, "attributes": attributes, "label_drugs": [i["drug"] for i in items if i["labels"]], "paper_ids": sorted({p["pmid"] for i in items for p in i["papers"]}),
                              "model_answered": bool(rows_by_drug), "cells_replaced": replaced_all}
    title = ", ".join(attributes)
    return (f"## {title[:1].upper() + title[1:]} of the drugs from the earlier answers\n\n"
            "From each drug's FDA label text and the papers and pages retrieved for this question; a cell no source states says so. Read the full label before any use.\n\n" + "\n".join(lines) + "\n")


def markdown(question: str, drugs: list[str], state: dict, ask, want_interactions: bool = False) -> str:
    """The side-effects table. `ask(task, payload) -> dict` is the model call (a JSON object back); a failed call leaves every cell as the label's own words."""
    items = offered(state, drugs, question)
    rows_by_drug: dict = {}
    try:
        answer = ask(TASK + (" The question also asks about interactions: fill \"interactions\" for every drug." if want_interactions else ""),
                     {"question": question, "drugs": [{"drug": i["drug"], "labels": i["labels"], "papers": i["papers"], "pages": i["pages"]} for i in items]})
        for row in (answer or {}).get("rows") or []:
            if isinstance(row, dict):
                rows_by_drug.setdefault(_key(row.get("drug")), row)
    except Exception:
        answer = None
    header = ["Drug", "Common side effects", "Serious risks and warnings", "Who should avoid it, and what to monitor"] + (["Interactions"] if want_interactions else []) + ["Source"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    replaced_all = {}
    for item in items:
        cells, replaced = checked_row(item, rows_by_drug.get(_key(item["drug"])))
        if replaced:
            replaced_all[item["drug"]] = replaced
        values = [item["drug"], cells["common"], cells["serious"], cells["avoid_or_monitor"]] + ([cells["interactions"]] if want_interactions else []) + [cells["sources"]]
        lines.append("| " + " | ".join(_clean(v) if i < len(values) - 1 else v for i, v in enumerate(values)) + " |")
    state["safety_answer"] = {"drugs": drugs, "label_drugs": [i["drug"] for i in items if i["labels"]], "paper_ids": sorted({p["pmid"] for i in items for p in i["papers"]}),
                              "model_answered": bool(rows_by_drug), "cells_replaced_by_label_text": replaced_all}
    return ("## Side effects of the drugs from the earlier answers\n\n"
            "From each drug's FDA label text and the papers and pages retrieved for its side effects; a cell no source states says so. This is not a complete safety review: read the full label.\n\n" + "\n".join(lines) + "\n")
