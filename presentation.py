"""Final presentation of a VERIFIED report: numbered citations [1], [2] with one reference list, and a plain closing note.

Verification runs on the long citation form (author, journal, year, PMID) so every identifier can be checked; only AFTER it passes does this module turn the
citations into numbers. The mapping number -> PMID is kept in the saved state, so nothing is lost.
"""
from __future__ import annotations

import re

DISCLAIMER = "Research support, not a substitute for clinical judgement."
PMID = r"\d{6,9}"
GROUP = re.compile(r"\(([^()]*?\[PMID\s*" + PMID + r"\][^()]*?)\)")           # (Wang et al., Journal, 2026 [PMID 123]; Lee et al., 2025 [PMID 456])
BRACKET = re.compile(r"\[PMID\s*(" + PMID + r")\]")
PAREN = re.compile(r"\(\s*PMID\s*(" + PMID + r")(?:\s*[;,]\s*PMID\s*(" + PMID + r"))*\s*\)")
BARE = re.compile(r"PMID\s*(" + PMID + r")(?:\s*\([^()]*\))?")


def _authors(p: dict) -> str:
    names = [str(a).strip() for a in (p.get("authors") or []) if str(a).strip()]
    return (", ".join(names[:3]) + (", et al." if len(names) >= 3 else "")) if names else "Authors not listed"


def numberize(text: str, papers: dict) -> tuple[str, list[str]]:
    """Returns (text with numbered citations and a References section, the PMIDs in citation order). The old '### Sources' list is replaced."""
    cut = text.find("\n### Sources")
    body, sources = (text[:cut], text[cut:]) if cut >= 0 else (text, "")
    order: list[str] = []

    def num(pmid: str) -> str | None:
        p = papers.get(pmid)
        if not p or p.get("retracted"):
            return None      # a retracted paper is only named (as excluded), never numbered as a reference
        if pmid not in order:
            order.append(pmid)
        return str(order.index(pmid) + 1)

    def group(m):
        ids = re.findall(PMID, m.group(1))
        nums = [n for n in (num(i) for i in ids) if n]
        return "[" + ", ".join(dict.fromkeys(nums)) + "]" if nums else m.group(0)

    def single(m):
        n = num(m.group(1))
        return f"[{n}]" if n else m.group(0)

    body = re.sub(r"PMID\s*\[(" + PMID + r")\]\(https?://[^)\s]*\)", r"PMID \1", body)      # 'PMID [123](https://...)' (a link the model sometimes writes) is the same citation
    body = re.sub(r"\[PMID\s*(" + PMID + r")\]\(https?://[^)\s]*\)", r"[PMID \1]", body)
    body = GROUP.sub(group, body)
    body = BRACKET.sub(single, body)
    def paren(m):
        nums = list(dict.fromkeys(n for n in (num(i) for i in re.findall(PMID, m.group(0))) if n))
        return "[" + ", ".join(nums) + "]" if nums else m.group(0)      # nothing numbered (a retracted paper): the text stays as it was
    body = PAREN.sub(paren, body)
    body = BARE.sub(single, body)
    body = re.sub(r"\[(\d+)\](?:\s*\[(\d+)\])+", lambda m: "[" + ", ".join(dict.fromkeys(re.findall(r"\d+", m.group(0)))) + "]", body)      # [1] [2] -> [1, 2]

    refs = []
    for i, pmid in enumerate(order, 1):
        p = papers[pmid]
        doi = f" DOI {p['doi']}." if p.get("doi") else ""
        who = _authors(p)
        refs.append(f"{i}. {who}{'' if who.endswith('.') else '.'} {p.get('title') or 'Title not retrieved'} *{p.get('journal') or ''}*. {p.get('date') or p.get('year') or ''}. "
                    f"[PMID {pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/).{doi}")
    other = []
    for line in sources.splitlines():
        m = re.match(r"^\d+\.\s+(.*)$", line)
        if m and not re.search(r"PMID\s*" + PMID, m.group(1)):
            link = re.match(r"^(.*?)\.?\s+(https?://\S+)$", m.group(1))
            other.append(f"- [{link.group(1)}]({link.group(2)})" if link else f"- {m.group(1)}")
    out = body.rstrip()
    if refs:
        out += "\n\n## Sources\n\n" + "\n".join(refs)
    if other:
        out += "\n\n**Other sources (regulators, guideline organisations, web pages, FDA labels)**\n\n" + "\n".join(other)
    return out, order


def closing_note(status: str) -> str:
    check = ("All citations, trial IDs, links and numbers in this report were checked against the retrieved sources."
             if status == "PASS" else "Some items could not be fully verified; the detail is in the saved verification record.")
    return f"\n\n---\n*{DISCLAIMER}* {check}"
