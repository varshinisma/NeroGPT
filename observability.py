"""Latency observability for the research agent: one run is a chain of model calls, retrievals, tool executions and orchestration stages, and each of them is recorded as a span.

Nothing here needs an outside service (no LangSmith quota): spans are kept in memory during a run and one JSON line per run is appended to observability/runs.jsonl.
`python latency_report.py` turns that file into percentile tables (p50 / p95 / p99) and a waterfall (observability/latency_report.html); the API serves the same page at /latency.

What is measured
- MODEL time: every model call (the writers, the JSON decision calls, the orchestrator agent): time to first token (TTFT), generation speed after the first token, total, characters in and out
  (characters / 4 is the token estimate), errors and retries.
- TOOL time: every retrieval or tool execution (PubMed, trials, FDA labels, web, guidelines, label lookups): seconds and whether it failed.
- ORCHESTRATION & WAIT time: the rest of each stage's wall time (planning, context assembly, waiting for threads, joins). It is the stage's seconds minus the time a model call or a tool was running in it.
- DIAGNOSTIC signals next to the time: number of model calls and tool calls, errors, retries, and how the model input grows over the run.

One run is recorded at a time per process (the app serves one person); two runs at once would be mixed into one record."""
from __future__ import annotations

import contextlib
import functools
import json
import math
import os
import threading
import time
from datetime import datetime
from pathlib import Path

OBS_DIR = Path(__file__).resolve().parent / "observability"
RUNS_FILE = OBS_DIR / "runs.jsonl"
_DEFAULT_RUNS_FILE = RUNS_FILE
MAX_SPANS = 600
_lock = threading.Lock()
_run: dict = {}


def enabled() -> bool:
    return os.getenv("OBSERVABILITY", "on").strip().lower() not in {"0", "off", "false", "no"}


def now() -> float:
    return time.time()


# ------------------------------------------------------------------ the run
def begin_run(question: str = "", run_id: str | None = None) -> None:
    """Start recording a run. A run that is already open (the API opens it before the routing decision) is kept, so the decision call is part of the same run."""
    if not enabled():
        return
    with _lock:
        if _run.get("open") and now() - _run.get("t0", 0) < 900:
            if question and not _run.get("question"):
                _run["question"] = question[:200]
            return
        _run.clear()
        _run.update(open=True, t0=now(), id=run_id or datetime.now().strftime("%Y%m%d-%H%M%S-%f"), question=question[:200], spans=[], counts={}, at=datetime.now().isoformat(timespec="seconds"))


def count(name: str, n: int = 1) -> None:
    if _run.get("open"):
        with _lock:
            _run["counts"][name] = _run["counts"].get(name, 0) + n


def record(kind: str, name: str, start: float, end: float, **attrs) -> None:
    """One finished span (kind: model, tool or stage). `start` and `end` are absolute times from now()."""
    if not _run.get("open"):
        return
    row = {"kind": kind, "name": name, "start": round(start - _run["t0"], 3), "end": round(end - _run["t0"], 3), **{k: v for k, v in attrs.items() if v is not None}}
    with _lock:
        if len(_run["spans"]) < MAX_SPANS:
            _run["spans"].append(row)
    live_line(row)


def live_line(row: dict) -> None:
    """One line in the backend terminal for every finished step (set OBSERVABILITY_STEPS=off to silence). Plain ASCII."""
    if os.getenv("OBSERVABILITY_STEPS", "on").strip().lower() in {"0", "off", "false", "no"}:
        return
    try:
        took = round(row["end"] - row["start"], 2)
        extra = ""
        if row["kind"] == "model":
            gen, out = row.get("gen_s"), row.get("out_chars")
            extra = f"  first token {row.get('ttft', '-')}s"
            if gen and out:
                extra += f", {round(out / 4 / max(gen, 0.05), 1)} tokens/s"
            extra += f", input ~{int((row.get('in_chars') or 0) / 4)} tokens, output ~{int((out or 0) / 4)} tokens"
        if row.get("error"):
            extra += f"  FAILED: {row['error']}"
        print(f"[step] {row['kind']:<5} {str(row['name']):<22}{took:>7}s  (at {row['start']}s -> {row['end']}s){extra}", flush=True)
    except Exception:
        pass      # printing never breaks a run


@contextlib.contextmanager
def span(kind: str, name: str, **attrs):
    began = now()
    info: dict = dict(attrs)
    try:
        yield info
    except BaseException as error:
        info["error"] = f"{type(error).__name__}: {str(error)[:120]}"
        count(f"{kind}_errors")
        raise
    finally:
        record(kind, name, began, now(), **info)
        count(f"{kind}_calls")


def model_call(fn):
    """Decorator for a generator that streams model text (qa_stream.stream_llm): records time to first token, generation speed, size in and out, and the purpose of the call."""
    @functools.wraps(fn)
    def inner(*args, **kwargs):
        if not enabled():
            yield from fn(*args, **kwargs)
            return
        began, first, out_chars, chunks, error = now(), None, 0, 0, None
        in_chars = sum(len(str(a)) for a in args[:2])
        purpose = kwargs.get("purpose") or "writer"
        try:
            for chunk in fn(*args, **kwargs):
                if first is None:
                    first = now()
                chunks += 1
                out_chars += len(chunk)
                yield chunk
        except BaseException as exc:
            error = f"{type(exc).__name__}: {str(exc)[:120]}"
            count("model_errors")
            raise
        finally:
            end = now()
            record("model", purpose, began, end, ttft=round(first - began, 3) if first else None, gen_s=round(end - first, 3) if first else None, in_chars=in_chars, out_chars=out_chars, chunks=chunks, error=error)
            count("model_calls")
    return inner


def tool(name: str):
    """Context manager for one tool or retrieval execution."""
    return span("tool", name)


def wrap_tools(jobs: dict) -> dict:
    """The same jobs ({name: function}), each recorded as a tool span when it runs. (run_jobs itself is left untouched: the fast path is held byte for byte.)"""
    def wrapped(name, fn):
        @functools.wraps(fn)
        def inner(*a, **k):
            with tool(name):
                return fn(*a, **k)
        return inner
    return {name: wrapped(name, fn) for name, fn in jobs.items()}


# ------------------------------------------------------------------ numbers
def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile (q in 0..100); None for no values."""
    data = sorted(v for v in values if isinstance(v, (int, float)))
    if not data:
        return None
    if len(data) == 1:
        return round(data[0], 2)
    pos = (len(data) - 1) * q / 100.0
    low, high = math.floor(pos), math.ceil(pos)
    return round(data[low] + (data[high] - data[low]) * (pos - low), 2)


def _union(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for s, e in sorted(i for i in intervals if i[1] > i[0]):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(a, b) for a, b in merged]


def _length(intervals) -> float:
    return sum(e - s for s, e in intervals)


def _clip(intervals, lo: float, hi: float) -> list[tuple[float, float]]:
    return [(max(s, lo), min(e, hi)) for s, e in intervals if min(e, hi) > max(s, lo)]


def _minus(intervals, cut) -> list[tuple[float, float]]:
    out = []
    for s, e in intervals:
        parts = [(s, e)]
        for cs, ce in cut:
            nxt = []
            for ps, pe in parts:
                if ce <= ps or cs >= pe:
                    nxt.append((ps, pe))
                else:
                    if cs > ps:
                        nxt.append((ps, cs))
                    if ce < pe:
                        nxt.append((ce, pe))
            parts = nxt
        out += parts
    return out


def summarize(spans: list[dict], total_s: float, counts: dict | None = None) -> dict:
    """The breakdown of one run: model, tool and orchestration & wait time (wall time, so overlapping parallel calls are not counted twice), per stage and for the whole run."""
    model = [(s["start"], s["end"]) for s in spans if s["kind"] == "model"]
    tools = [(s["start"], s["end"]) for s in spans if s["kind"] == "tool"]
    model_u, tool_u = _union(model), _union(tools)
    stages = {}
    for s in spans:
        if s["kind"] != "stage":
            continue
        lo, hi = s["start"], s["end"]
        m = _union(_clip(model_u, lo, hi))
        t = _union(_minus(_clip(tool_u, lo, hi), m))      # a tool running while a model call also runs is counted under the model
        wall = hi - lo
        stages[s["name"]] = {"seconds": round(wall, 2), "model_s": round(_length(m), 2), "tool_s": round(_length(t), 2), "orchestration_wait_s": round(max(0.0, wall - _length(m) - _length(t)), 2)}
    model_wall, tool_only = _length(model_u), _length(_minus(tool_u, model_u))
    by_purpose: dict[str, dict] = {}
    for s in spans:
        if s["kind"] == "model":
            row = by_purpose.setdefault(s["name"], {"calls": 0, "seconds": [], "ttft": [], "tok_per_s": [], "in_chars": []})
            row["calls"] += 1
            row["seconds"].append(s["end"] - s["start"])
            if s.get("ttft") is not None:
                row["ttft"].append(s["ttft"])
            if s.get("gen_s") and s.get("out_chars"):
                row["tok_per_s"].append(round(s["out_chars"] / 4 / max(s["gen_s"], 0.05), 1))
            row["in_chars"].append(s.get("in_chars") or 0)
    model_view = {k: {"calls": v["calls"], "total_s": round(sum(v["seconds"]), 2), "ttft_p50": percentile(v["ttft"], 50), "ttft_max": round(max(v["ttft"]), 2) if v["ttft"] else None,
                      "tokens_per_s_p50": percentile(v["tok_per_s"], 50), "in_tokens_max": int(max(v["in_chars"]) / 4) if v["in_chars"] else 0} for k, v in by_purpose.items()}
    tool_names: dict[str, dict] = {}
    for s in spans:
        if s["kind"] == "tool":
            row = tool_names.setdefault(s["name"], {"calls": 0, "seconds": 0.0, "errors": 0})
            row["calls"] += 1
            row["seconds"] += s["end"] - s["start"]
            row["errors"] += 1 if s.get("error") else 0
    growth = [int((s.get("in_chars") or 0) / 4) for s in sorted((s for s in spans if s["kind"] == "model"), key=lambda x: x["start"])]
    counts = dict(counts or {})
    return {"total_s": round(total_s, 2), "model_wall_s": round(model_wall, 2), "tool_wall_s": round(tool_only, 2), "orchestration_wait_s": round(max(0.0, total_s - model_wall - tool_only), 2),
            "model_calls": sum(1 for s in spans if s["kind"] == "model"), "tool_calls": sum(1 for s in spans if s["kind"] == "tool"), "errors": sum(1 for s in spans if s.get("error")),
            "retries": counts.get("model_retries", 0) + counts.get("rate_limit_retries", 0), "model_by_purpose": model_view,
            "tools": {k: {"calls": v["calls"], "total_s": round(v["seconds"], 2), "errors": v["errors"]} for k, v in tool_names.items()}, "stages": stages,
            "input_tokens_first": growth[0] if growth else 0, "input_tokens_max": max(growth) if growth else 0, "input_tokens_total": sum(growth)}


def end_run(extra: dict | None = None) -> dict:
    """Close the run, append its record to runs.jsonl and return the summary (without the spans) for the run's state."""
    if not _run.get("open"):
        return {}
    with _lock:
        _run["open"] = False
        spans, counts, total = list(_run["spans"]), dict(_run["counts"]), now() - _run["t0"]
        head = {"id": _run["id"], "at": _run["at"], "question": _run["question"]}
    summary = summarize(spans, total, counts)
    summary.update(extra or {})
    console_report({**head, **summary})
    if "unittest" in __import__("sys").modules and RUNS_FILE == _DEFAULT_RUNS_FILE:      # a test run never writes into the real log
        return {**head, **summary}
    try:
        OBS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RUNS_FILE, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({**head, **summary, "spans": spans}, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass      # observing a run never breaks it
    return {**head, **summary}


def console_report(run: dict) -> None:
    """Print the finished run to the backend terminal (set OBSERVABILITY_PRINT=off to silence it). Plain ASCII, so any console prints it."""
    if os.getenv("OBSERVABILITY_PRINT", "on").strip().lower() in {"0", "off", "false", "no"}:
        return
    try:
        stages = run.get("stages") or {}
        lines = [f"[latency] {str(run.get('question') or '')[:70]}",
                 f"[latency] total {run.get('total_s')}s = model {run.get('model_wall_s')}s + tools {run.get('tool_wall_s')}s + orchestration/wait {run.get('orchestration_wait_s')}s"
                 f" | first token {run.get('first_token_s', '-')}s | first useful {run.get('first_useful_s', '-')}s",
                 f"[latency] {run.get('model_calls')} model calls, {run.get('tool_calls')} tool calls, {run.get('errors')} errors, {run.get('retries')} retries"
                 f" | input tokens: first {run.get('input_tokens_first')}, largest {run.get('input_tokens_max')}, total {run.get('input_tokens_total')}"]
        for name, row in sorted(stages.items(), key=lambda kv: -kv[1].get("seconds", 0)):
            if row.get("seconds", 0) >= 0.5:
                lines.append(f"[latency]   stage {name:<14}{row['seconds']:>7}s  (model {row['model_s']}s, tools {row['tool_s']}s, wait {row['orchestration_wait_s']}s)")
        for name, row in sorted((run.get("model_by_purpose") or {}).items(), key=lambda kv: -kv[1].get("total_s", 0)):
            lines.append(f"[latency]   model {name:<20}{row['calls']:>3} calls, {row['total_s']}s total, first token p50 {'-' if row.get('ttft_p50') is None else str(row['ttft_p50']) + 's'}, {'-' if row.get('tokens_per_s_p50') is None else row['tokens_per_s_p50']} tokens/s")
        for name, row in sorted((run.get("tools") or {}).items(), key=lambda kv: -kv[1].get("total_s", 0))[:8]:
            lines.append(f"[latency]   tool  {name:<20}{row['calls']:>3} calls, {row['total_s']}s total, {row['errors']} failed")
        print("\n".join(lines), flush=True)
    except Exception:
        pass      # printing never breaks a run


def load_runs(path: Path | None = None) -> list[dict]:
    runs = []
    try:
        for text in (path or RUNS_FILE).read_text(encoding="utf-8").splitlines():
            try:
                runs.append(json.loads(text))
            except ValueError:
                continue
    except OSError:
        pass
    return runs
