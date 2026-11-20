"""EMBED_PROVIDER=local: the embeddings are made on this computer (fastembed, ONNX, no GPU, no API key, no rate limit) instead of by Mistral or OpenAI.

The model is EMBED_LOCAL_MODEL (default BAAI/bge-small-en-v1.5, 384 numbers per text; the first use downloads it, about 130 MB). The vectors go into the same cache as the others (hybrid_search._cache, keyed by model name and text),
so the Tavily page selection and the hybrid ranking use them unchanged. Returns None when the model cannot be loaded or fails: the callers then keep the plain order, as they do when an embedding call fails."""
from __future__ import annotations

import hashlib
import os
import threading

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
_state: dict = {"model": None, "name": None}
_load_lock = threading.Lock()


def enabled() -> bool:
    return os.getenv("EMBED_PROVIDER", "").strip().lower() == "local"


def model_name() -> str:
    return os.getenv("EMBED_LOCAL_MODEL", "").strip() or DEFAULT_MODEL


def load():
    """The model, loaded once (a second call waits for the first)."""
    with _load_lock:
        name = model_name()
        if _state["model"] is None or _state["name"] != name:
            from fastembed import TextEmbedding
            _state["model"], _state["name"] = TextEmbedding(model_name=name), name
        return _state["model"]


def load_in_background() -> None:
    threading.Thread(target=lambda: _safe_load(), daemon=True).start()


def _safe_load() -> None:
    try:
        load()
    except Exception:
        pass


def embed(texts: list[str]):
    """One vector per text, from the cache or from the local model; None when there is nothing to embed or the model fails."""
    import hybrid_search as hs
    if not texts:
        return None
    name = "local:" + model_name()
    keys = [hashlib.sha1((name + "|" + t).encode("utf-8")).hexdigest() for t in texts]
    with hs._lock:
        missing = [(k, t) for k, t in zip(keys, texts) if k not in hs._cache]
    if missing:
        unique = list(dict.fromkeys(missing))
        try:
            vectors = list(load().embed([t for _, t in unique]))
        except Exception:
            return None
        if len(vectors) != len(unique):
            return None
        with hs._lock:
            for (k, _), v in zip(unique, vectors):
                hs._cache[k] = [float(x) for x in v]
    with hs._lock:
        return [hs._cache[k] for k in keys]
