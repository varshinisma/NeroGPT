"""Agno web-search chat agent. Run with: python server.py"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from dotenv import load_dotenv

from clinical_tools import search_clinical_trials, search_pubmed

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")
HISTORY_DIR = ROOT / "conversation_history"


def search_live_web(query: str) -> str:
    """Search the live web for a specific query and return up to five relevant results.

    Args:
        query: The exact words or question to search for on the web.
    """
    try:
        from ddgs import DDGS
        results, error = [], None
        for backend in ("yahoo", "duckduckgo", "auto"):      # the engines that answer quickly first; the default 'auto' tries slow ones and can take 20 s
            try:
                results = list(DDGS(timeout=8).text(query, max_results=5, backend=backend))
            except Exception as failure:
                error = failure
                continue
            if results:
                break
        if not results and error:
            raise error
    except Exception as error:
        return f"Web search failed: {error}"
    if not results:
        return "No web results were found for that query."
    return json.dumps(
        [
            {
                "title": item.get("title", "Untitled"),
                "url": item.get("href", item.get("url", "")),
                "snippet": item.get("body", item.get("snippet", "")),
            }
            for item in results
        ],
        ensure_ascii=False,
    )


SKILLS_DIR = ROOT / "skills"
SKILL_ORDER = [
    "clinical-research-orchestrator",
    "latest-evidence",
    "drug-intelligence",
    "clinical-trials",
    "evidence-synthesis",
    "clinical-answer-writer",
    "citation-verification",
    "quality-control",
]

SINGLE_AGENT_PREAMBLE = f"""You are NeuroGPT, a clinical research assistant for doctors. Today's date is {datetime.now():%Y-%m-%d}.

You work through the skills listed at the bottom as stages of ONE workflow. You are the orchestrator.
SKILLS ARE LOADED ON DEMAND: only each skill's name and description are shown below. Before you perform a stage, call load_skill("<skill-name>") to read its full rules, follow them, and load the next skill only when you reach that stage. Never load a skill you do not need. Your first step for a clinical question is load_skill("clinical-research-orchestrator").
Stages:
- Retrieval stages (latest-evidence, drug-intelligence, clinical-trials): call tools. Make independent tool calls together in the same step, not one by one.
  * search_pubmed -> papers (use years_back=1 for "latest"; add "randomized controlled trial", "meta-analysis", "guideline" to target study types; search drug names found in papers too).
  * search_clinical_trials -> trial registry data (current status, NCT IDs).
  * search_live_web / Tavily -> approvals, FDA/EMA labels, dosing, guidelines and anything PubMed or the registry cannot answer. Open or search for the real source so you can state its date.
- Synthesis, writing, citation-verification and quality-control stages: do these yourself on the retrieved results before you send the final answer. Check every claim, year, dose, drug category and trial status against the tool results, and remove or qualify anything unsupported.
- Only cite what the tools returned (title, year, PMID/DOI/NCT ID/URL). If a value was not returned, write "Not retrieved". Never fill gaps from memory.
- If a tool fails or returns nothing, say what is missing in the answer's limitations.
- You send one complete answer per request (no streaming here), so apply the Phase 1 content first (bottom line, recent finding, top drugs), then the Phase 2 sections.
- If the user's message is not a clinical or medical research question, answer it briefly and normally; skip the pipeline.

AVAILABLE SKILLS (name: when to use):
"""


def load_skills(names: list[str] | None = None) -> str:
    """Read the SKILL.md bodies as agent instructions. Pass skill names to load only those; otherwise
    NEUROGPT_SKILLS decides (all, off, or a comma-separated list)."""
    if names is None:
        setting = os.getenv("NEUROGPT_SKILLS", "all").strip().lower()
        if setting in {"off", "none", "0", "false"}:
            return ""
        names = SKILL_ORDER if setting in {"", "all"} else [n.strip() for n in setting.split(",") if n.strip()]
    sections = []
    for name in names:
        path = SKILLS_DIR / name / "SKILL.md"
        if not path.is_file():
            continue
        body = path.read_text(encoding="utf-8").split("---", 2)[-1]
        # Keep the rules and steps; drop worked examples and file pointers to save tokens.
        parts = re.split(r"(?m)^(?=## )", body)
        body = "".join(p for p in parts if not re.match(r"## (Examples?|References)\b", p))
        sections.append(f"=== SKILL: {name} ===\n{body.strip()}")
    return "\n\n".join(sections)


def skill_catalog() -> str:
    """Level 1 of progressive loading: only each skill's name and description (about 100 tokens each)."""
    lines = []
    for name in SKILL_ORDER:
        path = SKILLS_DIR / name / "SKILL.md"
        if not path.is_file():
            continue
        match = re.search(r"(?ms)^description:\s*(.+?)\n---", path.read_text(encoding="utf-8"))
        lines.append(f"- {name}: {match.group(1).strip() if match else ''}")
    return "\n".join(lines)


_loaded_skills: set[str] = set()


def load_skill(skill_name: str) -> str:
    """Load the full rules of ONE skill so you can follow them. Call this when you reach that stage,
    not before. Valid names: clinical-research-orchestrator, latest-evidence, drug-intelligence,
    clinical-trials, evidence-synthesis, clinical-answer-writer, citation-verification, quality-control.

    Args:
        skill_name: Exact skill name from the available skills list.
    """
    name = skill_name.strip()
    if name not in SKILL_ORDER:
        return f"Unknown skill '{name}'. Valid names: {', '.join(SKILL_ORDER)}"
    text = load_skills([name])
    print(f"[skills] loaded on demand: {name}", flush=True)
    return text


def build_model(max_tokens: int | None = None):
    """Create the chat model selected by MODEL_PROVIDER (groq or mistral). max_tokens caps the output size."""
    provider = os.getenv("MODEL_PROVIDER", "groq").lower()
    if provider == "groq":
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("Set GROQ_API_KEY in .env before starting the server.")
        from agno.models.groq import Groq
        return Groq(id=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"), max_tokens=max_tokens)
    if provider == "mistral":
        if not os.getenv("MISTRAL_API_KEY"):
            raise RuntimeError("Set MISTRAL_API_KEY in .env before starting the server.")
        from agno.models.mistral import MistralChat
        return MistralChat(id=os.getenv("MISTRAL_MODEL", "mistral-small-latest"), max_tokens=max_tokens)
    raise RuntimeError("MODEL_PROVIDER must be groq or mistral.")


def build_agent():
    """Create an Agno agent that uses live web search, not local files."""
    try:
        from agno.agent import Agent
        from agno.tools.tavily import TavilyTools
    except ImportError as error:
        raise RuntimeError("Dependencies are missing. Run: python -m pip install -r requirements.txt") from error

    model = build_model()

    tools = [search_pubmed, search_clinical_trials, search_live_web]
    if os.getenv("TAVILY_API_KEY"):
        tools.append(TavilyTools(search_depth="advanced", include_answer=True))

    # "agent" mode: skills are loaded one at a time through the load_skill tool (never all at once).
    # The default "pipeline" mode (pipeline.py) handles clinical questions in code instead.
    if os.getenv("NEUROGPT_MODE", "pipeline").lower() != "agent":
        return Agent(
            name="Web Search Assistant",
            model=model,
            tools=tools,
            instructions=[
                "You are a helpful research and coding assistant.",
                "For factual, current, or research questions, call a web-search tool with a clear query before answering.",
                "Base research answers on the search results. State when results are insufficient.",
                "Do not claim you searched the web if you did not call a web-search tool.",
            ],
            markdown=True,
        )

    return Agent(
        name="NeuroGPT",
        model=model,
        tools=tools + [load_skill],
        instructions=[SINGLE_AGENT_PREAMBLE + skill_catalog()],
        markdown=True,
    )


def call_agent(messages: list[dict]) -> str:
    try:
        if os.getenv("NEUROGPT_MODE", "pipeline").lower() == "pipeline":
            from pipeline import run_pipeline
            question = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
            result = run_pipeline(str(question))
            if result is not None:  # None means "not a clinical question": use the normal agent below
                return result["answer"]
        agent = build_agent()
        history = "\n".join(
            f"{item.get('role', 'user').upper()}: {item.get('content', '')}"
            for item in messages[-12:]
        )
        result = agent.run(history)
        return str(result.content or "I couldn't produce an answer.")
    except RuntimeError as error:
        return str(error)
    except Exception as error:
        return f"Agent error: {error}"


def run_streaming_pipeline(messages: list[dict]) -> str:
    """Run the streaming pipeline to the end and return the final report (non-streaming clients). Also saves the md/state/PDF like a normal run."""
    from qa_stream import save_outputs, stream_answer
    question = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "").strip()
    final = None
    for event in stream_answer(question):
        if event["type"] == "final":
            final = event
        elif event["type"] == "error":
            return f"Error: {event.get('message')}"
    if not final:
        return "No answer was produced."
    save_outputs(question, final)
    return final["markdown"]


def scrub_messages(messages: list[dict]) -> list[dict]:
    """Remove personal identifiers from every user message before it is searched, sent to a model, or saved."""
    import phi
    return [{**m, "content": phi.scrub(str(m.get("content", ""))).text} if m.get("role") == "user" else m for m in messages]


def save_conversation(messages: list[dict], answer: str, conversation_id: str) -> Path:
    """Update one Markdown file with the complete conversation so far."""
    HISTORY_DIR.mkdir(exist_ok=True)
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", conversation_id)[:64]
    if not safe_id:
        safe_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = HISTORY_DIR / f"conversation_{safe_id}.md"
    transcript = []
    for message in messages:
        role = str(message.get("role", "user")).title()
        content = str(message.get("content", ""))
        transcript.append(f"## {role}\n\n{content}")
    transcript.append(f"## Assistant\n\n{answer}")
    filename.write_text(
        f"# Agent conversation\n\n"
        f"**Last updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        + "\n\n---\n\n".join(transcript) + "\n",
        encoding="utf-8",
    )
    return filename


class AgentHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")      # the browser must never reuse an old page or script after the code changes
        super().end_headers()

    def stream_chat(self) -> None:
        """Server-sent progress as NDJSON: INITIAL answer within seconds, then the enrichment stages (see qa_stream.py)."""
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            messages = scrub_messages(body.get("messages", []))   # history stores the SCRUBBED text, never the original
            question = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "").strip()
            if not question:
                raise ValueError("A message is required.")
        except (json.JSONDecodeError, ValueError) as error:
            self.send_error(HTTPStatus.BAD_REQUEST, str(error))
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        def send(event: dict) -> None:
            self.wfile.write((json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8"))
            self.wfile.flush()

        from qa_stream import save_outputs, stream_answer
        final = None
        try:
            for event in stream_answer(question):
                if event["type"] == "final":
                    final = event
                    send({k: v for k, v in event.items() if k != "state"})  # the full ResearchState stays on the server
                else:
                    send(event)
            if final:
                paths = save_outputs(question, final)  # every run also saves the answer and a PDF
                send({"type": "pdf", "name": paths["pdf"].name if paths["pdf"] else None})
                save_conversation(messages, final["markdown"], str(body.get("conversation_id", "")))
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as error:
            try:
                send({"type": "error", "message": str(error)[:300]})
            except OSError:
                pass

    def do_POST(self) -> None:
        if self.path == "/api/chat-stream":
            self.stream_chat()
            return
        if self.path != "/api/chat":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            messages = body.get("messages", [])
            if not isinstance(messages, list) or not messages:
                raise ValueError("A message is required.")
            messages = scrub_messages(messages)
            answer = run_streaming_pipeline(messages)    # an old browser tab that still calls /api/chat gets the SAME pipeline and report as /api/chat-stream
            save_conversation(messages, answer, str(body.get("conversation_id", "")))
            response = json.dumps({"answer": answer}).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)
        except (json.JSONDecodeError, ValueError) as error:
            self.send_error(HTTPStatus.BAD_REQUEST, str(error))


def make_server() -> ThreadingHTTPServer:
    """Listen on IPv4 AND IPv6: 'localhost' resolves to ::1 first on Windows, and an IPv4-only server made every request wait ~2 s."""
    import socket

    class DualStackServer(ThreadingHTTPServer):
        address_family = socket.AF_INET6
        allow_reuse_address = False      # on Windows address reuse lets a SECOND server bind the same port silently (old code kept answering)

        def server_bind(self) -> None:
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            super().server_bind()

    def port_busy(error: OSError) -> bool:
        return error.errno in (98, 10048, 10013) or "address" in str(error).lower() and "use" in str(error).lower()

    try:
        return DualStackServer(("::", 8000), AgentHandler)
    except OSError as error:
        if port_busy(error):
            raise SystemExit("Port 8000 is already in use: another server is running. Stop it first (Ctrl+C in its terminal) so only ONE server answers.")
        class V4Server(ThreadingHTTPServer):      # no IPv6 on this machine
            allow_reuse_address = False
        return V4Server(("127.0.0.1", 8000), AgentHandler)


if __name__ == "__main__":
    os.chdir(ROOT)
    import threading
    from qa_stream import warm_up
    threading.Thread(target=warm_up, daemon=True).start()  # open model + data-source connections before the first question
    httpd = make_server()                                   # refuses to start if another server already holds the port
    print("NeuroGPT running at http://localhost:8000  (streaming Q/A at /api/chat-stream)")
    if not os.environ.get("NEUROGPT_NO_BROWSER"):      # opens your default browser by itself; set NEUROGPT_NO_BROWSER=1 to stop that
        import webbrowser
        threading.Timer(1.0, lambda: webbrowser.open("http://localhost:8000")).start()
    httpd.serve_forever()
