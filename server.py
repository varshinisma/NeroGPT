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
        results = list(DDGS().text(query, max_results=5))
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


def build_agent():
    """Create an Agno agent that uses live web search, not local files."""
    try:
        from agno.agent import Agent
        from agno.tools.tavily import TavilyTools
    except ImportError as error:
        raise RuntimeError("Dependencies are missing. Run: python -m pip install -r requirements.txt") from error

    provider = os.getenv("MODEL_PROVIDER", "groq").lower()
    if provider == "groq":
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("Set GROQ_API_KEY in .env before starting the server.")
        from agno.models.groq import Groq
        model = Groq(id=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
    elif provider == "mistral":
        if not os.getenv("MISTRAL_API_KEY"):
            raise RuntimeError("Set MISTRAL_API_KEY in .env before starting the server.")
        from agno.models.mistral import MistralChat
        model = MistralChat(id=os.getenv("MISTRAL_MODEL", "mistral-small-latest"))
    else:
        raise RuntimeError("MODEL_PROVIDER must be groq or mistral.")

    tools = [search_live_web]
    if os.getenv("TAVILY_API_KEY"):
        tools.append(TavilyTools(search_depth="advanced", include_answer=True))

    return Agent(
        name="Web Search Assistant",
        model=model,
        tools=tools,
        instructions=[
            "You are a helpful research and coding assistant.",
            "For factual, current, or research questions, call a web-search tool with a clear query before answering.",
            "Use the Tavily search tool for in-depth research when it is available; otherwise use search_live_web.",
            "Base research answers on the search results. State when results are insufficient.",
            "Do not claim you searched the web if you did not call a web-search tool.",
            "Answer clearly and concisely.",
        ],
        markdown=True,
    )


def call_agent(messages: list[dict]) -> str:
    try:
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
    def do_POST(self) -> None:
        if self.path != "/api/chat":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            messages = body.get("messages", [])
            if not isinstance(messages, list) or not messages:
                raise ValueError("A message is required.")
            answer = call_agent(messages)
            save_conversation(messages, answer, str(body.get("conversation_id", "")))
            response = json.dumps({"answer": answer}).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)
        except (json.JSONDecodeError, ValueError) as error:
            self.send_error(HTTPStatus.BAD_REQUEST, str(error))


if __name__ == "__main__":
    os.chdir(ROOT)
    print("Agno web-search agent running at http://localhost:8000")
    ThreadingHTTPServer(("127.0.0.1", 8000), AgentHandler).serve_forever()
