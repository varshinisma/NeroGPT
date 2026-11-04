"""NeuroGPT backend (FastAPI). The research pipeline runs here; a front end (the Streamlit page, or any web page) only sends the question and draws what comes back.

Run (from this folder):   uvicorn api:app --host 127.0.0.1 --port 8000
Endpoints
    GET  /health                 {"status": "ok", "ready": bool}: is the server up, and are the pipeline's libraries loaded
    POST /ask                    body {"question", "history": [{"role","content"}], "conversation_id"}; the answer streams back as NDJSON, one JSON event per line
                                 (scrubbed, stage, token, warning, milestone, verification, final, figures, pdf, error); the full research state stays on the server
    GET  /pdf/{name}             a saved report PDF
    GET  /file?path=...          an image the report uses (a figure under demo_output/figures)
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
load_dotenv(ROOT / ".env")

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

import conversation_memory
import observability
import orchestrator
import quick_check

OUTPUT_DIR = ROOT / "demo_output"
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
READY = threading.Event()
MAX_QUESTION_CHARS = 4000


def pipeline():
    """The pipeline's entry points (loaded on first use, so importing this module is quick, and tests can replace this function)."""
    from qa_stream import deliver_outputs, stream_answer
    return stream_answer, deliver_outputs


def default_ask_model(instructions: str, prompt: str) -> str:
    """One short, deterministic model call (used to read a follow-up question)."""
    from qa_stream import stream_llm
    return "".join(stream_llm(instructions, prompt, 300, temperature=0.0, purpose="triage"))


def warm() -> None:
    """Load the pipeline's libraries and open the model and data-source connections once, in the background, so the first question does not pay for it."""
    try:
        import hybrid_search
        import qa_stream
        import server  # noqa: F401
        hybrid_search.load_cross_encoder()
        qa_stream.warm_up()
    except Exception:
        pass
    finally:
        READY.set()


class AskBody(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS)
    history: list[dict] = Field(default_factory=list)      # earlier turns, already scrubbed when they were first asked
    conversation_id: str = ""


def line(event: dict) -> bytes:
    return (json.dumps(event, ensure_ascii=False, default=str) + "\n").encode("utf-8")


def public(event: dict) -> dict:
    """What a front end may see of an event: the final event loses the full research state, and keeps only the verification status a page shows."""
    if event.get("type") != "final":
        return event
    state = event.get("state") or {}
    out = {k: v for k, v in event.items() if k != "state"}
    out["verification_status"] = (state.get("verification") or {}).get("status", "INCOMPLETE")
    return out


def default_chat_stream(instructions: str, prompt: str):
    """The writing model, streaming a conversational answer."""
    from qa_stream import stream_llm
    yield from stream_llm(instructions, prompt, 1100, temperature=0.2, purpose="writer")


def conversational_events(question: str, memory: list[dict], chosen: list[int], chat_stream):
    """An answer written from the earlier answers of this conversation, yielded as token events while it is written. The last event is {'type': '_done', 'answer': text}, or
    {'type': '_done', 'answer': None} when the earlier answers do not contain what is needed (the model said so): the caller then researches it. Numbers that the earlier answers do
    not contain are flagged under the answer."""
    context = conversation_memory.build_context(memory, chosen)
    marker = conversation_memory.NEEDS_RESEARCH
    parts: list[str] = []
    decided = False
    for chunk in chat_stream(conversation_memory.CHAT_INSTRUCTIONS, conversation_memory.chat_prompt(question, memory, chosen)):
        parts.append(chunk)
        head = "".join(parts)
        if not decided:
            if len(head.strip()) < len(marker):
                continue      # too little to tell yet whether the model is asking for research
            decided = True
            if head.strip().startswith(marker):
                yield {"type": "_done", "answer": None}
                return
            yield {"type": "token", "stage": "FINAL", "text": head}
        else:
            yield {"type": "token", "stage": "FINAL", "text": chunk}
    head = "".join(parts)
    if not decided:
        if head.strip().startswith(marker) or not head.strip():
            yield {"type": "_done", "answer": None}
            return
        yield {"type": "token", "stage": "FINAL", "text": head}
    answer = head.strip()
    missing = conversation_memory.unmatched_numbers(answer, context)
    if missing:
        note = "\n\n*These numbers are not in the earlier answers and could not be checked against them: " + ", ".join(missing) + ". Check them before use.*"
        yield {"type": "token", "stage": "FINAL", "text": note}
        answer += note
    yield {"type": "_done", "answer": answer}


def ask_events(body: AskBody, get_pipeline=None, ask_model=None, chat_stream=None, quick_lookups=None, run_agent=None):
    """The events of one question, as NDJSON lines. Personal details are removed from the question first (the scrubbed text is the first event, so the page can keep it). Then the question is
    read against what the conversation has already answered: a follow-up that only needs explaining is answered from the earlier answers at once; one that needs new evidence is rewritten as one
    complete question and researched (the drugs and conditions copied by code from the earlier answers); anything else is researched as asked. The pipeline's events follow, then the figures and
    the PDF; the conversation is saved when the PDF is ready."""
    import time
    import server
    stream_answer, deliver_outputs = (get_pipeline or pipeline)()
    clean = server.scrub_messages([{"role": "user", "content": body.question.strip()}])[0]["content"]
    yield line({"type": "scrubbed", "text": clean})
    observability.begin_run(clean)      # the routing decision belongs to the run
    memory = conversation_memory.load(body.conversation_id) or conversation_memory.turns_from_history(body.history)      # what this conversation has already answered
    if run_agent is None and ask_model is not None:      # a caller that supplies its own JSON reader (tests): no agent
        resolved = conversation_memory.resolve(clean, memory, ask_model)
    elif orchestrator.AGENT_ENABLED:      # the orchestrator agent chooses the path; its fallback is the JSON router
        resolved = orchestrator.decide(clean, memory, run_agent, ask_model or default_ask_model)
    else:
        resolved = conversation_memory.resolve(clean, memory, ask_model or default_ask_model)
    asked = resolved["question"]
    earlier = [{"role": m.get("role", "user"), "content": str(m.get("content", ""))} for m in body.history if isinstance(m, dict)]
    if resolved["mode"] == "clarify":      # the agent could not tell what is meant: one short question back, no search
        text = resolved["clarification"]
        yield line({"type": "understood", "text": "Asking for clarification", "mode": "chat", "turns": []})
        yield line({"type": "token", "stage": "FINAL", "text": text})
        observability.end_run({"mode": "clarify"})      # the run ends here: its timings are printed and recorded
        yield line({"type": "final", "mode": "chat", "markdown": text, "timings": {"final_s": 0}})
        try:
            conversation_memory.save(body.conversation_id, memory + [conversation_memory.chat_turn(clean, text)])
            server.save_conversation(earlier + [{"role": "user", "content": clean}], text, body.conversation_id)
        except Exception:
            pass
        return
    if resolved["mode"] == "conversation":
        began, answer, announced = time.time(), None, False
        for event in conversational_events(clean, memory, resolved["turns"], chat_stream or default_chat_stream):
            if event["type"] == "_done":
                answer = event["answer"]
                break
            if not announced:
                announced = True
                yield line({"type": "understood", "text": "Answering from this conversation", "mode": "chat", "turns": resolved["turns"]})
            yield line(event)      # the text reaches the page while it is being written
        if answer is not None:
            observability.end_run({"mode": "conversation"})      # answered from the conversation: no search, but its timings are recorded too
            yield line({"type": "final", "mode": "chat", "markdown": answer, "timings": {"final_s": round(time.time() - began, 1)}})
            try:
                conversation_memory.save(body.conversation_id, memory + [conversation_memory.chat_turn(clean, answer)])
                server.save_conversation(earlier + [{"role": "user", "content": clean}], answer, body.conversation_id)
            except Exception:
                pass      # remembering is a convenience: a failure never stops the answer
            return
        asked = resolved.get("research_question") or clean      # the earlier answers do not have it: research it, as a complete question
        resolved = {**resolved, "follow_up": asked != clean}
    if resolved["follow_up"]:
        yield line({"type": "understood", "text": asked, "turns": resolved["turns"]})
    request, preface = resolved.get("quick_check"), ""
    if request:      # 'are those drugs also used for <another condition>?': a short grounded answer first, then the research
        fda_lookup, pubmed_lookup = quick_lookups or (quick_check.default_fda_lookup, quick_check.default_pubmed_lookup)
        try:
            preface = quick_check.run(request["drugs"], request["condition"], fda_lookup, pubmed_lookup)
        except Exception:
            preface = ""
        if preface:
            yield line({"type": "preface", "text": preface})
    final, saved = None, False
    try:
        plan = resolved.get("plan")
        if plan and resolved.get("follow_up"):
            try:      # a follow-up starts from what the earlier questions retrieved and searches only for the gap
                import source_store
                stored = source_store.load(body.conversation_id)
                plan = {**plan, "prior_sources": stored} if stored else plan
            except Exception:
                pass
        for event in (stream_answer(asked, plan=plan) if plan else stream_answer(asked)):      # the plan the agent wrote in its decision goes with the question
            if event.get("type") == "final":
                final = event
            yield line(public(event))
        if final:
            try:
                remembered = conversation_memory.turn_from_final(clean, asked, final, follow_up=bool(resolved.get("follow_up")))
                remembered["markdown"] = (preface + remembered["markdown"])[:conversation_memory.MAX_REPORT_CHARS]      # the quick answer is part of what was said
                conversation_memory.save(body.conversation_id, memory + [remembered])
                import source_store
                source_store.save(body.conversation_id, final.get("state") or {})      # what this run retrieved stays for the conversation's follow-ups
            except Exception:
                pass      # remembering is a convenience: a failure never stops the report
            for event in deliver_outputs(asked, final):
                yield line(event)
                if event.get("type") == "pdf" and not saved:
                    server.save_conversation(earlier + [{"role": "user", "content": clean}], final["markdown"], body.conversation_id)
                    saved = True
    except GeneratorExit:      # the page was closed: stop quietly
        raise
    except Exception as error:
        yield line({"type": "error", "message": str(error)[:300]})


app = FastAPI(title="NeuroGPT API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"], allow_methods=["GET", "POST"], allow_headers=["*"])


@app.on_event("startup")
def start_warm_up() -> None:
    threading.Thread(target=warm, daemon=True).start()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "ready": READY.is_set()}


@app.get("/latency", response_class=HTMLResponse)
def latency_page(run: str | None = Query(None, max_length=80), last: int = Query(0, ge=0, le=1000)) -> HTMLResponse:
    """Where the agent's time goes: percentiles over the recorded runs and the waterfall of one run (observability.py, latency_report.py)."""
    import latency_report
    runs = observability.load_runs()
    return HTMLResponse(latency_report.html_report(runs[-last:] if last else runs, run))


@app.get("/latency/summary")
def latency_summary(last: int = Query(0, ge=0, le=1000)) -> dict:
    import latency_report
    runs = observability.load_runs()
    return latency_report.collect(runs[-last:] if last else runs)


@app.post("/ask")
def ask(body: AskBody) -> StreamingResponse:
    return StreamingResponse(ask_events(body), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/pdf/{name}")
def get_pdf(name: str) -> FileResponse:
    if not re.fullmatch(r"[A-Za-z0-9._\-]+\.pdf", name):
        raise HTTPException(status_code=400, detail="Not a report file name.")
    path = (OUTPUT_DIR / name).resolve()
    if OUTPUT_DIR.resolve() not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="No such report.")
    return FileResponse(path, media_type="application/pdf", filename=name)


@app.get("/file")
def get_file(path: str = Query(..., max_length=300)) -> FileResponse:
    target = (ROOT / path).resolve()
    figures = (OUTPUT_DIR / "figures").resolve()
    if figures not in target.parents or target.suffix.lower() not in IMAGE_TYPES or not target.is_file():
        raise HTTPException(status_code=404, detail="No such image.")
    return FileResponse(target)
