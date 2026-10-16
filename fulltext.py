"""Key sections (methods, results, conclusion) of the best papers that have a free full text.

PubMed and Europe PMC searches match titles and abstracts only, so the body of a paper is never searched; this module reads it AFTER the papers are ranked, for the few papers the report leans on most.
Only open-access papers (a PMC id, in the PubMed Central Open Access subset) have a free full text; every other paper stays abstract-only. FULL_TEXT=off turns it off, FULL_TEXT_PAPERS and
FULL_TEXT_WAIT_S set how many papers and how long. The text is only read, never rewritten: the writer is shown it, and verification counts it as part of the source."""
import os
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, wait

LIMITS = {"methods": 350, "results": 900, "conclusion": 450}      # characters of each section shown to the writer
TITLES = {"methods": re.compile(r"\b(methods?|materials|study design|participants|procedures?)\b", re.I),
          "results": re.compile(r"\b(results?|findings|outcomes?)\b", re.I),
          "conclusion": re.compile(r"\b(conclusions?|interpretation|summary)\b", re.I)}


def enabled() -> bool:
    return os.getenv("FULL_TEXT", "on").strip().lower() not in {"off", "0", "false", "no"}


def _text(node) -> str:
    return " ".join(" ".join(node.itertext()).split())


def _paragraphs(section) -> str:
    """The running text of a section: its own paragraphs and its sub-sections', without tables, figures or reference lists."""
    parts = []
    for node in section.iter():
        if node.tag == "p":
            parts.append(_text(node))
    return " ".join(p for p in parts if p)


def sections_from_xml(xml_text: str) -> dict:
    """{'methods','results','conclusion'} -> text (empty sections left out), from a JATS article. A body without recognisable section titles gives {}."""
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return {}
    body = root.find(".//body")
    found: dict[str, str] = {}
    if body is not None:
        for sec in body.findall("sec"):
            title = _text(sec.find("title")) if sec.find("title") is not None else ""
            for name, pattern in TITLES.items():
                if name not in found and pattern.search(title):
                    text = _paragraphs(sec)
                    if text:
                        found[name] = text
                    break
    if "conclusion" not in found:
        for abstract in root.iter("abstract"):      # the structured abstract's own conclusion, when the body has none
            for sec in abstract.findall("sec"):
                title = _text(sec.find("title")) if sec.find("title") is not None else ""
                if TITLES["conclusion"].search(title) and _paragraphs(sec):
                    found["conclusion"] = _paragraphs(sec)
    return found


def shorten(sections: dict) -> dict:
    """Each section cut to its limit at a sentence end where possible. Results keep their start (the main outcome comes first); the conclusion keeps its end."""
    out = {}
    for name, text in sections.items():
        limit = LIMITS.get(name, 400)
        if len(text) <= limit:
            out[name] = text
            continue
        cut = text[:limit] if name != "conclusion" else text[-limit:]
        stop = max(cut.rfind(". "), cut.rfind("; ")) if name != "conclusion" else cut.find(". ")
        if name != "conclusion" and stop > limit // 2:
            cut = cut[:stop + 1]
        elif name == "conclusion" and 0 <= stop < limit // 2:
            cut = cut[stop + 2:]
        out[name] = cut.strip()
    return out


def attach(papers: list[dict], fetch, cap_s: float = 6.0, workers: int = 4) -> dict:
    """Fetch the sections of `papers` side by side for at most `cap_s` seconds and put them on each paper as `body` (shortened) and `body_text` (everything shown, for verification).
    `fetch(pmcid) -> (xml_text, keys) | None` is clinical_tools._article. Returns a record {'asked','read','seconds_cap'}; a paper that fails or is late keeps only its abstract."""
    chosen = [p for p in papers if p.get("pmcid") and not p.get("retracted") and not p.get("body")]
    record = {"asked": len(chosen), "read": 0, "cap_s": cap_s}
    if not chosen:
        return record
    pool = ThreadPoolExecutor(max_workers=min(len(chosen), workers))
    futures = {pool.submit(fetch, p["pmcid"]): p for p in chosen}
    try:
        done, _ = wait(list(futures), timeout=cap_s)
        for future in done:
            paper = futures[future]
            try:
                article = future.result()
                sections = shorten(sections_from_xml(article[0])) if article else {}
            except Exception:
                continue
            if sections:
                paper["body"] = sections
                paper["body_text"] = " ".join(sections.values())
                record["read"] += 1
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return record
