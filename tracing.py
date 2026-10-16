"""LangSmith tracing, optional and off unless asked for.

Set LANGSMITH_TRACING=true and LANGSMITH_API_KEY (and optionally LANGSMITH_PROJECT) in .env and every decorated step is recorded in LangSmith with its inputs, outputs, timing and errors:
the question breakdown, each retrieval tool, the hybrid ranking, each model call (skill instructions + data) and the figure selection. Without the setting, or without the `langsmith` package,
`traced` returns the function unchanged, so there is no overhead and no data leaves the machine.

Everything a traced step receives or returns is sent to LangSmith when tracing is on (the question, retrieved abstracts, model prompts); the API key itself is never part of any input."""
from __future__ import annotations

import os

try:
    from dotenv import load_dotenv
    load_dotenv()      # the settings must be known when the decorators are applied (at import time)
except ImportError:
    pass


def enabled() -> bool:
    flag = (os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2") or "").strip().lower()
    return flag in {"1", "true", "yes", "on"} and bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"))


def traced(name: str, run_type: str = "chain"):
    """Decorator: record the call in LangSmith as a run called `name` (run_type: chain, tool, llm or retriever). Returns the function itself when tracing is off or unavailable."""
    def wrap(fn):
        if not enabled():
            return fn
        try:
            from langsmith import traceable
        except ImportError:
            return fn
        return traceable(name=name, run_type=run_type)(fn)
    return wrap
