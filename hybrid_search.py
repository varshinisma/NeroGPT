"""Hybrid search: the keyword search finds the papers, then a semantic ranking and a lexical ranking are fused with the order the sources returned them.

1. keyword retrieval (PubMed / Europe PMC queries, plus PubMed's own "Best Match" ranking of the plain question) pools the candidates; nothing here searches by itself;
2. every candidate (its title and the start of its abstract) and the question are turned into embedding vectors by one call to the embedding model; the cosine similarity of a paper to the
   question is its SEMANTIC score, so a paper that uses other words for the same idea still ranks high;
3. a BM25 score of the same text against the question's own words is its LEXICAL score;
4. Reciprocal Rank Fusion (score = sum of 1 / (60 + rank) over the three rankings: the sources' own order, BM25, semantic) gives the final order. RRF needs no tuning and no score scale.

Nothing in here knows a disease, a drug or a question type. If the embedding call is off, fails or is slow, the fusion uses the other two rankings only (the result is then lexical + source order) and says so
in the trace. Settings: HYBRID_SEARCH=off turns it off, EMBED_MODEL overrides the embedding model, EMBED_TIMEOUT_S the time allowed for the embedding call."""
from __future__ import annotations

import hashlib
import math
import os
import re
import threading

import httpx

from tracing import traced
EMBED_URL = "https://api.mistral.ai/v1/embeddings"
DEFAULT_EMBED_MODEL = "mistral-embed"
RRF_K = 60
TEXT_CHARS = 700
_cache: dict = {}
_lock = threading.Lock()
_STOP = set("a an and are as at be by for from how in is it of on or that the this to was what which with do does can".split())


def enabled() -> bool:
    return os.getenv("HYBRID_SEARCH", "on").strip().lower() not in {"off", "0", "false", "no"}


def _terms(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if w not in _STOP and len(w) > 1]


def bm25_scores(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """BM25 of every document against the query's words (the documents are the pool itself, so the idf is the pool's own)."""
    tokens = [_terms(d) for d in docs]
    n = len(tokens)
    if not n:
        return []
    average = (sum(len(t) for t in tokens) / n) or 1.0
    frequency: dict = {}
    for t in tokens:
        for w in set(t):
            frequency[w] = frequency.get(w, 0) + 1
    scores = []
    for t in tokens:
        counts: dict = {}
        for w in t:
            counts[w] = counts.get(w, 0) + 1
        s = 0.0
        for w in set(_terms(query)):
            if w in counts:
                idf = math.log(1 + (n - frequency[w] + 0.5) / (frequency[w] + 0.5))
                s += idf * counts[w] * (k1 + 1) / (counts[w] + k1 * (1 - b + b * len(t) / average))
        scores.append(s)
    return scores


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


@traced("embed", "chain")
def embed(texts: list[str], timeout: float | None = None) -> list[list[float]] | None:
    """One embedding vector per text (the embedding model, one request for all texts not seen before); None when the key is missing or the call fails."""
    key = os.getenv("MISTRAL_API_KEY")
    if not key or not texts:
        return None
    model = os.getenv("EMBED_MODEL", "").strip() or DEFAULT_EMBED_MODEL
    wait = timeout if timeout is not None else float(os.getenv("EMBED_TIMEOUT_S", "3"))
    keys = [hashlib.sha1((model + "|" + t).encode("utf-8")).hexdigest() for t in texts]
    with _lock:
        missing = [(k, t) for k, t in zip(keys, texts) if k not in _cache]
    if missing:
        unique = list(dict.fromkeys(missing))
        try:
            response = httpx.post(EMBED_URL, headers={"Authorization": f"Bearer {key}"}, json={"model": model, "input": [t for _, t in unique]}, timeout=wait)
            response.raise_for_status()
            data = sorted(response.json().get("data") or [], key=lambda d: d.get("index", 0))
            if len(data) != len(unique):
                return None
        except Exception:
            return None
        with _lock:
            for (k, _), item in zip(unique, data):
                _cache[k] = item["embedding"]
    with _lock:
        return [_cache[k] for k in keys]


def paper_text(paper: dict) -> str:
    return f"{paper.get('title') or ''}. {(paper.get('abstract') or '')[:TEXT_CHARS]}".strip()


def _ranks(values: list[float]) -> list[int]:
    order = sorted(range(len(values)), key=lambda i: -values[i])
    ranks = [0] * len(values)
    for position, i in enumerate(order):
        ranks[i] = position + 1
    return ranks


@traced("hybrid_rerank", "chain")
def rerank(question: str, papers: dict, trace: dict | None = None) -> None:
    """Reorder `papers` (pmid -> record) IN PLACE by the fused ranking, and store each paper's `hybrid` scores. The sources' own order is one of the three rankings, so a paper the sources put first
    is never buried by a weak text match. Does nothing when switched off or when there are fewer than 3 papers."""
    trace = trace if trace is not None else {}
    if not enabled() or len(papers) < 3:
        trace["hybrid"] = "off" if not enabled() else "too few papers"
        return
    ids = list(papers)
    docs = [paper_text(papers[i]) for i in ids]
    lexical = bm25_scores(question, docs)
    vectors = embed([question] + docs)
    semantic = [cosine(vectors[0], v) for v in vectors[1:]] if vectors else None
    source_rank = list(range(1, len(ids) + 1))
    rankings = [source_rank, _ranks(lexical)] + ([_ranks(semantic)] if semantic else [])
    fused = [sum(1.0 / (RRF_K + r[i]) for r in rankings) for i in range(len(ids))]
    for i, pmid in enumerate(ids):
        papers[pmid]["hybrid"] = {"fused": round(fused[i], 5), "lexical": round(lexical[i], 3), "semantic": round(semantic[i], 4) if semantic else None}
    order = sorted(range(len(ids)), key=lambda i: -fused[i])
    reordered = {ids[i]: papers[ids[i]] for i in order}
    papers.clear()
    papers.update(reordered)
    trace.update({"hybrid": "semantic + lexical + source order" if semantic else "lexical + source order (embedding call unavailable)", "papers_ranked": len(ids)})


# ------------------------------------------------------------------ second stage: a cross-encoder reads the question and each paper TOGETHER
CROSS_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"      # a small cross-encoder run locally (ONNX, no GPU); CROSS_ENCODER_MODEL overrides it
CROSS_WEIGHT = 2.0      # the cross-encoder judges a (question, paper) pair jointly, so its verdict counts twice as much as the first stage's position in the final fusion
_cross: dict = {"model": None, "failed": False, "thread": None}


def cross_enabled() -> bool:
    return os.getenv("CROSS_ENCODER", "on").strip().lower() not in {"off", "0", "false", "no"}


def load_cross_encoder(timeout: float = 0.0):
    """The cross-encoder model, loaded once in a background thread (the first run downloads about 80 MB). With `timeout` the caller waits that long for it; returns None while it is not ready or if it cannot load."""
    with _lock:
        if _cross["model"] is None and not _cross["failed"] and _cross["thread"] is None:
            def load():
                try:
                    from fastembed.rerank.cross_encoder import TextCrossEncoder
                    _cross["model"] = TextCrossEncoder(os.getenv("CROSS_ENCODER_MODEL", "").strip() or CROSS_MODEL)
                except Exception:
                    _cross["failed"] = True
            _cross["thread"] = threading.Thread(target=load, daemon=True)
            _cross["thread"].start()
        thread = _cross["thread"]
    if _cross["model"] is None and thread is not None and timeout > 0:
        thread.join(timeout=timeout)
    return _cross["model"]


@traced("cross_encoder_rerank", "chain")
def cross_rerank(question: str, papers: dict, trace: dict | None = None, top: int = 36, wait: float = 3.0, scorer=None) -> None:
    """Second stage, on the `top` papers of the first-stage order: a cross-encoder scores each (question, title + abstract start) pair TOGETHER, which separates close papers better than comparing two
    separate embeddings. The final order fuses the first-stage position with the cross-encoder's score (a sigmoid of it, weight CROSS_WEIGHT), so a clear verdict moves a paper and a near-tie does not. Papers beyond `top` keep their order after them.
    Skips (and says why in the trace) when switched off, when there are fewer than 3 papers, or when the model is not ready within `wait` seconds. `scorer(question, texts) -> scores` replaces the model (tests)."""
    trace = trace if trace is not None else {}
    if not cross_enabled() or len(papers) < 3:
        trace["cross_encoder"] = "off" if not cross_enabled() else "too few papers"
        return
    if scorer is None:
        model = load_cross_encoder(timeout=wait)
        if model is None:
            trace["cross_encoder"] = "skipped: the model is not available" + (" (could not load)" if _cross["failed"] else " (still loading)")
            return
        scorer = lambda q, texts: [float(s) for s in model.rerank(q, texts)]
    ids = list(papers)[:top]
    try:
        scores = scorer(question, [paper_text(papers[i]) for i in ids])
    except Exception as error:
        trace["cross_encoder"] = f"skipped: {type(error).__name__}"
        return
    n = len(ids)
    fused = [(1.0 - position / n) + CROSS_WEIGHT / (1.0 + math.exp(-float(scores[position]))) for position in range(n)]      # first-stage position (1 for the first, falling) + the cross-encoder's probability-like score: a clear verdict reorders, a near-tie keeps the first-stage order
    for position, pmid in enumerate(ids):
        papers[pmid].setdefault("hybrid", {})["cross"] = round(float(scores[position]), 3)
    order = sorted(range(len(ids)), key=lambda p: -fused[p])
    rest = {k: papers[k] for k in list(papers)[top:]}
    reordered = {ids[p]: papers[ids[p]] for p in order}
    papers.clear()
    papers.update(reordered)
    papers.update(rest)
    trace["cross_encoder"] = f"reranked the first {len(ids)} papers"


def cross_scorer(wait: float = 0.0):
    """scorer(query, texts) -> scores with the cross-encoder, or None when the model is not ready (the screen then leaves everything to the model-based assessment)."""
    model = load_cross_encoder(timeout=wait)
    if model is None:
        return None
    return lambda query, texts: [float(s) for s in model.rerank(query, texts)]
