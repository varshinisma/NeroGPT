"""'Established care' section: filled ONLY from retrieved authoritative guideline pages. No model writes it.

Each topic row shows the guideline title, the organisation, the year, the link and a short VERBATIM excerpt of the retrieved text;
if nothing from an allow-listed organisation was retrieved the row says exactly 'No guideline retrieved' (or that the search was unavailable).
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

CONFIG_PATH = Path(__file__).parent / "config" / "care_topics.json"
NONE_TEXT = "No guideline retrieved"
UNAVAILABLE_TEXT = "Guideline search unavailable (the source failed); this does not mean no guideline exists"


def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def topics_for(question: str, config: dict | None = None) -> list[dict]:
    """The neurodevelopmental override when the question matches it, otherwise the generic default."""
    config = config or load_config()
    q = question.lower()
    for override in config.get("overrides", {}).values():
        if any(word in q for word in override["match"]):
            return override["topics"]
    return config["default"]["topics"]


def organisation_for(url: str, config: dict) -> str | None:
    host = (urlparse(url).hostname or "").lower()
    if any(host == h or host.endswith("." + h) for h in config.get("excluded_hosts", [])):
        return None
    allowed = config["authoritative_domains"]
    for domain in sorted(allowed, key=len, reverse=True):           # longest (most specific) domain first
        if host == domain or host.endswith("." + domain):
            return allowed[domain]
    return None


def _year(*texts: str) -> str | None:
    for t in texts:
        m = re.search(r"(?<!\d)((?:19|20)\d\d)(?!\d)", t or "")
        if m:
            return m.group(1)
    return None


def _excerpt(snippet: str, limit: int = 280) -> str | None:
    text = re.sub(r"\s+", " ", snippet or "").strip()
    text = re.sub(r"^(?:[A-Z][a-z]{2,8}\.? \d{1,2}, \d{4}|\d{1,2} [A-Z][a-z]{2,8} \d{4})\s*\W{0,3}\s*", "", text)      # a leading page date such as 'Aug 28, 2013 -'
    text = text.replace("�", "-")
    text = re.sub(r"^[#*\-\s]+", "", text)
    if len(text) < 40:
        return None
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    return cut[:end + 1] if end > 80 else cut.rsplit(" ", 1)[0] + " ..."


def find_guidelines(question: str, condition_words: list[str], searcher, config: dict | None = None, max_workers: int = 3) -> dict:
    """searcher(query) -> JSON string of [{title, url, snippet}] or an error string. Returns {'results': [...], 'failed': n, 'topics': n} or {'error': ...}."""
    config = config or load_config()
    topics = topics_for(question, config)
    stems = [w[:5] for w in condition_words if len(w) >= 4]
    condition = " ".join(condition_words)

    def one(topic: dict) -> dict:
        query = f"{condition} {topic['query']} guideline recommendations"
        raw = searcher(query)
        if not str(raw).lstrip().startswith("["):          # rate-limited or failed: ONE retry after a short pause
            time.sleep(1.0)
            raw = searcher(query)
        try:
            items = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            return {"topic": topic["id"], "label": topic["label"], "error": str(raw)[:120]}
        if not isinstance(items, list):
            return {"topic": topic["id"], "label": topic["label"], "error": str(raw)[:120]}
        for item in items:
            url = item.get("url") or ""
            org = organisation_for(url, config)
            excerpt = _excerpt(item.get("snippet") or "")
            title = re.sub(r"\s+", " ", item.get("title") or "").strip()
            blob = f"{title} {item.get('snippet') or ''}".lower()
            if org and excerpt and title and (not stems or any(s in blob for s in stems)):
                return {"topic": topic["id"], "label": topic["label"], "title": title, "organisation": org, "year": _year(title, item.get("snippet"), url),
                        "url": url, "excerpt": excerpt, "text": re.sub(r"\s+", " ", item.get("snippet") or "")[:1500]}
        return {"topic": topic["id"], "label": topic["label"], "none": True}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        rows = list(pool.map(one, topics))
    failed = sum(1 for r in rows if r.get("error"))
    if failed == len(rows):
        return {"error": f"guideline search failed for all {len(rows)} topics: {rows[0].get('error')}"}
    return {"results": rows, "failed": failed, "topics": len(rows)}


def _cell(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).replace("|", "/").strip()


def established_care_markdown(rows: list[dict], search_ok: bool) -> str:
    """Table of retrieved guidelines per topic. search_ok False = the guideline source failed entirely."""
    head = ("### Established care\n\n*Only guideline pages retrieved from recognised organisations are shown, with a short verbatim excerpt. "
            "Nothing in this section is model-written. A topic without a retrieved guideline is not evidence that none exists.*\n\n")
    lines = ["| Topic | Guideline | Organisation | Year | Excerpt |", "|---|---|---|---|---|"]
    for r in rows:
        if r.get("title"):
            lines.append(f"| {_cell(r['label'])} | [{_cell(r['title'])[:110]}]({r['url']}) | {_cell(r['organisation'])} | {r.get('year') or 'year not stated'} | “{_cell(r['excerpt'])}” |")
        elif r.get("error") or not search_ok:
            lines.append(f"| {_cell(r.get('label', ''))} | {UNAVAILABLE_TEXT} | | | |")
        else:
            lines.append(f"| {_cell(r['label'])} | {NONE_TEXT} | | | |")
    if not rows:
        lines.append(f"| All topics | {UNAVAILABLE_TEXT} | | | |")
    return head + "\n".join(lines)


REC_WORDS = re.compile(r"\b(recommend\w*|should|first-line|evidence|effective|therap\w+|intervention\w*|treatment\w*|support\w*|screen\w*|diagnos\w+|services|programme|program)\b", re.I)
VERB_WORDS = re.compile(r"\b(is|are|was|were|can|may|should|must|will|recommend\w*|include\w*|help\w*|improve\w*|provide\w*|use[sd]?|seek|offer\w*|ensure\w*|aim\w*)\b", re.I)


def _usable_sentence(sentence: str) -> bool:
    """A real sentence of a page: starts with a capital letter, ends with a full stop, has a verb, is not a title or a navigation line."""
    return bool(60 <= len(sentence) <= 320 and re.match(r"^[A-Z]", sentence) and re.search(r"[.!?]$", sentence) and VERB_WORDS.search(sentence)
                and not re.search(r"cookie|skip to|menu|subscribe|sign in|\bclinical practice guideline\b|\breport of recommendations\b|\bcopyright\b", sentence, re.I))


def _short_title(title: str, limit: int = 140) -> str:
    title = re.sub(r"\s+", " ", title or "").strip()
    return title if len(title) <= limit else title[:limit].rsplit(" ", 1)[0]


def guideline_snapshot_markdown(state: dict, limit: int = 5) -> str:
    """'Standard of care: what guideline organisations say': one complete sentence per retrieved guideline page, in the page's own words, with organisation and link.
    A sentence that states a recommendation, treatment or service is preferred. Empty string when no guideline page was retrieved (nothing is printed in that case)."""
    from landscape import _clean_source, complete_sentence
    from verification import _sentences
    picks = []
    for g in state.get("guidelines", []):
        sentences = [x.strip() for x in _sentences(_clean_source(g.get("text") or g.get("excerpt") or ""))]
        good = [x for x in sentences if _usable_sentence(x)]
        best = next((x for x in good if REC_WORDS.search(x)), None)
        if best:
            picks.append((g, best))
    rows, seen = [], set()
    for g, sentence in picks:
        if g["url"] in seen or sentence in {r[1] for r in rows}:
            continue
        seen.add(g["url"])
        rows.append((g, sentence))
        if len(rows) >= limit:
            break
    if not rows:
        return ""
    lines = []
    for g, sentence in rows:
        year = f", {g['year']}" if g.get("year") else ""
        lines.append(f"- **{g['organisation']}{year}**: “{complete_sentence(sentence)}” ([{_short_title(g['title'])}]({g['url']}))")
    return ("### Standard of care: what guideline organisations say\n\n" + "\n".join(lines) +
            "\n\n*One sentence from each retrieved guideline page, in the page's own words; read the page for the full recommendation.*")
