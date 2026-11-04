"""Tracing: LangSmith when it is set up, and a local fallback that never depends on it.

LangSmith (optional, off unless asked for): set LANGSMITH_TRACING=true and LANGSMITH_API_KEY (and optionally LANGSMITH_PROJECT) in .env and every decorated step is recorded in LangSmith with its inputs,
outputs, timing and errors: the question breakdown, each retrieval tool, the hybrid ranking, each model call (skill instructions + data) and the figure selection.

When LangSmith says the usage quota is used up, the program notices (once), stops sending to LangSmith for the rest of the run, stops the repeated error messages, and keeps working exactly as before.
It then writes a light LOCAL trace instead: one JSON line per step (name, seconds, shortened inputs and output, error) in demo_output/traces/<date>.jsonl, on this computer only.
Set LOCAL_TRACING=true to write the local trace always (also without LangSmith). Without LangSmith and without LOCAL_TRACING, `traced` returns the function unchanged: no overhead, nothing is written, no data leaves the machine.

Everything a traced step receives or returns is sent to LangSmith when it is on (the question, retrieved abstracts, model prompts); the API key itself is never part of any input.
The local trace keeps only the first LOCAL_TRACE_CHARS characters (default 1500) of each input and output."""
from __future__ import annotations

import functools
import inspect
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()      # the settings must be known when the decorators are applied (at import time)
except ImportError:
    pass

TRACE_DIR = Path(__file__).parent / "demo_output" / "traces"
_lock = threading.Lock()
_state = {"quota_hit": False}
QUOTA_WORDS = ("usage limit", "usage limits", "ratelimiterror", "rate limit exceeded for https://api.smith.langchain.com", "monthly unique traces")


def enabled() -> bool:
    flag = (os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2") or "").strip().lower()
    return flag in {"1", "true", "yes", "on"} and bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"))


def local_always() -> bool:
    return (os.getenv("LOCAL_TRACING") or "").strip().lower() in {"1", "true", "yes", "on"}


def quota_hit() -> bool:
    return _state["quota_hit"]


def note_quota_message(message: str) -> bool:
    """True when a LangSmith log message says the quota is used up. The first one switches LangSmith off for this run and prints one short notice."""
    text = str(message).lower()
    if not any(word in text for word in QUOTA_WORDS):
        return False
    if not _state["quota_hit"]:
        _state["quota_hit"] = True
        print("[tracing] LangSmith quota is used up: tracing to LangSmith is switched off for this run; the run continues and a local trace is written to demo_output/traces/.", file=sys.stderr)
    return True


class _QuotaFilter(logging.Filter):
    """Watches LangSmith's own log lines. Once the quota is known to be over, its repeated 429 messages are not printed again."""
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        if note_quota_message(message):
            return False
        return not (_state["quota_hit"] and "langsmith" in message.lower() and "429" in message)


def _watch_langsmith_logs() -> None:
    flt = _QuotaFilter()
    for name in ("langsmith", "langsmith.client", "langsmith._internal._background_thread", "langsmith._internal._operations"):
        logging.getLogger(name).addFilter(flt)


def _short(value) -> str:
    limit = int(os.getenv("LOCAL_TRACE_CHARS", "1500") or 1500)
    try:
        text = repr(value)
    except Exception:
        text = "<unprintable>"
    return text if len(text) <= limit else text[:limit] + f"... [{len(text) - limit} more characters]"


def write_local(record: dict) -> None:
    """One JSON line per traced step. Never raises: tracing must not break a run."""
    try:
        with _lock:
            TRACE_DIR.mkdir(parents=True, exist_ok=True)
            with open(TRACE_DIR / f"{datetime.now():%Y-%m-%d}.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def make_wrapper(fn, name: str, run_type: str, langsmith_fn=None):
    """The function to run instead of `fn`: through LangSmith while it works, plain once its quota is over, and with a local trace when asked for or when LangSmith is over quota."""
    @functools.wraps(fn)
    def inner(*args, **kwargs):
        call = langsmith_fn if (langsmith_fn is not None and not _state["quota_hit"]) else fn
        if not (local_always() or (langsmith_fn is not None and _state["quota_hit"])):
            return call(*args, **kwargs)
        started = time.time()
        base = {"ts": datetime.now().isoformat(timespec="seconds"), "name": name, "run_type": run_type, "inputs": _short((args, kwargs))}
        if inspect.isgeneratorfunction(fn):
            def watched():
                items, error = 0, None
                try:
                    for item in call(*args, **kwargs):
                        items += 1
                        yield item
                except BaseException as exc:      # the record is written whatever ends the stream
                    error = f"{type(exc).__name__}: {str(exc)[:200]}"
                    raise
                finally:
                    write_local({**base, "seconds": round(time.time() - started, 2), "output": f"<stream of {items} items>", "error": error})
            return watched()
        try:
            result = call(*args, **kwargs)
        except Exception as exc:
            write_local({**base, "seconds": round(time.time() - started, 2), "output": None, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
            raise
        write_local({**base, "seconds": round(time.time() - started, 2), "output": _short(result), "error": None})
        return result
    return inner


def traced(name: str, run_type: str = "chain"):
    """Decorator: record the call as a run called `name` (run_type: chain, tool, llm or retriever). Returns the function itself when no tracing is set up."""
    def wrap(fn):
        langsmith_fn = None
        if enabled():
            try:
                from langsmith import traceable
                langsmith_fn = traceable(name=name, run_type=run_type)(fn)
            except ImportError:
                langsmith_fn = None
        if langsmith_fn is None and not local_always():
            return fn
        return make_wrapper(fn, name, run_type, langsmith_fn)
    return wrap


_watch_langsmith_logs()
