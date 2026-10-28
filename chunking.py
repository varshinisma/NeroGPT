"""Passage (chunk) ranking: the writer is shown the passages of each paper that bear on the question, not the whole abstract.

A paper's abstract and, for open-access papers, the methods / results / conclusion read by fulltext.py are cut into short passages of a few sentences. The cross-encoder (the same local model as the
paper re-ranking, hybrid_search.cross_scorer) scores every passage against the question, and the best passages of each paper are kept, back in their original order. Nothing is rewritten: a passage
is a piece of the paper's own text, so the verification still finds every number in it. CHUNKING=off turns this off; CHUNKS_PER_PAPER sets how many passages a paper keeps. If the model is not ready,
the writer gets the abstract as before."""
import os
import re
import time

SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def enabled() -> bool:
    return os.getenv("CHUNKING", "on").strip().lower() not in {"off", "0", "false", "no"}


def per_paper() -> int:
    try:
        return max(1, int(os.getenv("CHUNKS_PER_PAPER", "3")))
    except ValueError:
        return 3


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END.split(" ".join(str(text or "").split())) if s.strip()]


def passages(text: str, size: int = 2, chars: int = 420) -> list[str]:
    """Windows of `size` sentences (one sentence step), none longer than `chars`. A short text is one passage."""
    parts = sentences(text)
    if not parts:
        return []
    if len(parts) <= size:
        return [" ".join(parts)[:chars]]
    out = []
    for i in range(0, len(parts), size):
        window = " ".join(parts[i:i + size]).strip()
        if window:
            out.append(window[:chars])
    return out


def paper_chunks(paper: dict) -> list[dict]:
    """[{'section','text'}] for one paper: its abstract passages, then its full-text sections' passages (when they were read)."""
    chunks = [{"section": "abstract", "text": t} for t in passages(paper.get("abstract") or "")[:5]]
    for name in ("methods", "results", "conclusion"):
        text = (paper.get("body") or {}).get(name)
        if text:
            chunks += [{"section": name, "text": t} for t in passages(text, size=2, chars=480)[:3]]
    return chunks


def select(question: str, papers: list[dict], scorer, keep: int | None = None) -> dict:
    """Score the passages of `papers` against the question and set `paper['passages']` ([{'section','text','score'}], original order) on each paper that has any.
    `scorer(question, texts) -> scores`. Returns a trace {'papers','chunks','kept','seconds'}; a scorer failure leaves every paper without passages."""
    began = time.time()
    keep = keep or per_paper()
    work = [(p, paper_chunks(p)) for p in papers]
    work = [(p, c) for p, c in work if c]
    texts = [f"{p.get('title') or ''}. {c['text']}" for p, chunks in work for c in chunks]
    trace = {"papers": len(work), "chunks": len(texts), "kept": 0}
    if not texts:
        trace["seconds"] = round(time.time() - began, 2)
        return trace
    try:
        scores = [float(s) for s in scorer(question, texts)]
        if len(scores) != len(texts):
            raise ValueError("score count")
    except Exception as error:
        trace.update(skipped=f"{type(error).__name__}", seconds=round(time.time() - began, 2))
        return trace
    at = 0
    for paper, chunks in work:
        scored = [{**c, "score": round(scores[at + i], 3)} for i, c in enumerate(chunks)]
        at += len(chunks)
        best = sorted(range(len(scored)), key=lambda i: -scored[i]["score"])[:keep]
        paper["passages"] = [scored[i] for i in sorted(best)]      # the best passages, back in reading order
        trace["kept"] += len(paper["passages"])
    trace["seconds"] = round(time.time() - began, 2)
    return trace
