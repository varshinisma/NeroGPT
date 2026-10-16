"""NeuroGPT chat in Streamlit: the same research pipeline as before (`stream_answer` + `deliver_outputs`), in a light page.

Run (from this folder):   streamlit run streamlit_app.py --server.port 8501
Stages show as they happen and the answer streams in; when the papers' original figures arrive they are placed INSIDE the report section that cites their paper (the same records the PDF uses),
with their explanation lines, caption and a credits list; the PDF can be downloaded."""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)



@st.cache_resource
def warm_in_background() -> bool:
    """Load the pipeline (its libraries and connections) in a background thread, ONCE per server start, so the page itself appears at once instead of waiting about two seconds for the imports."""
    import threading

    def load():
        import server  # noqa: F401
        import hybrid_search
        import qa_stream
        hybrid_search.load_cross_encoder()      # the cross-encoder model is ready before the first question
        try:
            qa_stream.warm_up()
        except Exception:
            pass
    threading.Thread(target=load, daemon=True).start()
    return True

STAGE_TEXT = {"INITIAL": "Finding the first evidence", "ENRICHING": "Reading the evidence", "EVIDENCE_ENRICHING": "Reading the evidence", "FINAL": "Final verified answer"}
FIGURE_NOTE = "Each figure shows one study's result; it does not by itself establish that a treatment works or justify a recommendation."
STYLE = """
<style>
:root { --navy: #0f2a43; --blue: #1d4ed8; --teal: #0f766e; --line: #e2e8f0; --muted: #64748b; --soft: #f8fafc; }
html, body, [class*="css"], [data-testid="stAppViewContainer"] { font-family: Inter, "Segoe UI", system-ui, -apple-system, sans-serif; }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 980px; padding-top: 1.2rem; padding-bottom: 7rem; }

/* sidebar */
[data-testid="stSidebar"] { background: var(--soft); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] .block-container, [data-testid="stSidebarUserContent"] { padding-top: 1.4rem; }
.brand { display: flex; align-items: center; gap: .65rem; margin-bottom: .2rem; }
.brand .mark { width: 38px; height: 38px; border-radius: 10px; background: linear-gradient(135deg, #0f2a43, #1d4ed8); display: flex; align-items: center; justify-content: center; }
.brand .name { font-size: 1.18rem; font-weight: 800; color: var(--navy); line-height: 1.1; }
.brand .sub { font-size: .72rem; color: var(--muted); letter-spacing: .04em; text-transform: uppercase; }
.side-label { font-size: .72rem; font-weight: 700; color: var(--muted); letter-spacing: .08em; text-transform: uppercase; margin: 1.1rem 0 .35rem; }
[data-testid="stSidebar"] button { text-align: left; justify-content: flex-start; border-radius: 9px; font-size: .88rem; border: 1px solid var(--line); background: #fff; color: #1e293b; }
[data-testid="stSidebar"] button:hover { border-color: var(--blue); color: var(--blue); }
[data-testid="stSidebar"] button[kind="primary"] { background: #eaf1ff; border-color: #bcd2ff; color: var(--navy); font-weight: 600; }
[data-testid="stSidebar"] [data-testid="stButton"]:first-of-type button { background: var(--navy); color: #fff; border-color: var(--navy); justify-content: center; font-weight: 600; }

/* page header and the empty state */
.hero { text-align: center; padding: 7vh 1rem 1rem; }
.hero .mark { width: 64px; height: 64px; border-radius: 18px; margin: 0 auto 1rem; background: linear-gradient(135deg, #0f2a43, #1d4ed8); display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 24px rgba(29, 78, 216, .25); }
.hero h1 { font-size: 2.1rem; margin: 0; color: var(--navy); letter-spacing: -.01em; }
.hero p { color: var(--muted); font-size: 1.02rem; margin: .5rem 0 1.1rem; }
.chips { display: flex; flex-wrap: wrap; gap: .45rem; justify-content: center; }
.chips span { font-size: .78rem; color: #334155; background: #eef2f7; border: 1px solid var(--line); border-radius: 999px; padding: .22rem .7rem; }
.topbar { display: flex; align-items: center; gap: .6rem; padding-bottom: .7rem; margin-bottom: .6rem; border-bottom: 1px solid var(--line); }
.topbar .t { font-weight: 800; color: var(--navy); font-size: 1.05rem; }
.topbar .s { color: var(--muted); font-size: .85rem; }

/* conversation */
[data-testid="stChatMessage"] { border-radius: 14px; padding: 1rem 1.15rem; margin-bottom: .6rem; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background: #eef4ff; border: 1px solid #d5e3ff; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) { background: #ffffff; border: 1px solid var(--line); box-shadow: 0 1px 3px rgba(15, 42, 67, .06); }

/* report text */
h1, h2, h3, h4 { font-weight: 700 !important; letter-spacing: -.005em; }
h2 { color: var(--navy); border-bottom: 2px solid #c7dafd; padding-bottom: .3rem; margin-top: 1.7rem !important; font-size: 1.3rem !important; }
h3 { color: var(--teal); margin-top: 1.2rem !important; font-size: 1.08rem !important; }
h4 { color: #4338ca; font-size: 1rem !important; }
[data-testid="stMarkdownContainer"] p > strong:only-child { color: #0b4f6c; font-weight: 800; font-size: 1.03rem; }
[data-testid="stMarkdownContainer"] table { border-collapse: collapse; width: 100%; font-size: .86rem; margin: .5rem 0 1rem; }
[data-testid="stMarkdownContainer"] th { background: var(--navy); color: #fff; text-align: left; padding: .5rem .6rem; font-weight: 600; }
[data-testid="stMarkdownContainer"] td { border: 1px solid var(--line); padding: .45rem .6rem; vertical-align: top; }
[data-testid="stMarkdownContainer"] tr:nth-child(even) td { background: var(--soft); }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 12px; }

/* the question box */
[data-testid="stChatInput"] { border: 1.5px solid #b6c6dd !important; border-radius: 16px !important; background: #ffffff !important; box-shadow: 0 4px 18px rgba(15, 42, 67, .10); }
[data-testid="stChatInput"]:focus-within { border-color: var(--blue) !important; box-shadow: 0 4px 18px rgba(29, 78, 216, .18); }
[data-testid="stChatInput"] textarea { color: #0f172a !important; font-size: 1rem !important; }
[data-testid="stBottom"] > div { background: #ffffff !important; padding-top: .6rem; }
</style>
"""


def figure_markdown(fig: dict) -> str:
    """The text under one figure: the lines under it on the page (what it shows, how it helps with treatment options, how to read it)."""
    described = fig.get("shows_for_question") or (fig.get("image_description") if fig.get("image_description") and not any(c.isdigit() for c in fig["image_description"]) else "")
    lines = [f"**{title}:** {text}" for title, text in (("What this figure shows", described), ("How it helps with treatment options", fig.get("helps_treatment")),
                                                          ("How to read this figure", fig.get("how_to_read"))) if text]
    return "\n\n".join(lines)


def caption_text(fig: dict) -> str:
    caption = fig.get("caption") or ""
    return caption if caption[:6].lower().startswith(("figure", "fig")) else f"{fig.get('label') or 'Figure'}. {caption}"


def credit_markdown(fig: dict) -> str:
    """One line of the 'Figure credits' list: the paper, its identifiers, whether the report cites it, and the licence."""
    parts = [f"{fig.get('label') or 'Figure'}: " + (fig["title"].rstrip(".") + ". " if fig.get("title") else "") + (fig["design"] + ". " if fig.get("design") else "")]
    links = []
    if fig.get("pmid"):
        links.append(f"[PMID {fig['pmid']}](https://pubmed.ncbi.nlm.nih.gov/{fig['pmid']}/)")
    if fig.get("doi"):
        links.append(f"[DOI {fig['doi']}](https://doi.org/{fig['doi']})")
    if fig.get("source_url"):
        links.append(f"[PubMed Central ({fig.get('pmcid')})]({fig['source_url']})")
    parts.append(", ".join(links))
    parts.append(f". Cited in this report as [{fig['reference']}]." if fig.get("reference") else ". Not discussed in the report text.")
    if fig.get("licence_label"):
        parts.append(f" Licence: [{fig['licence_label']}]({fig['licence_url']}); original image, unaltered." if fig.get("licence_url") else f" Licence: {fig['licence_label']}; original image, unaltered.")
    if fig.get("relation") == "indirect":
        parts.append(" Indirect evidence: a related condition or population.")
    return "- " + "".join(parts)


def sections_with_figures(markdown: str, figures: list[dict]) -> list[tuple[str, list[dict]]]:
    """The report cut into its `##` sections, each with the figures whose paper that section cites (a figure whose section is not found comes last, under its own heading)."""
    parts = [p for p in re.split(r"(?m)^(?=## )", markdown) if p.strip()]
    heading = lambda part: part.splitlines()[0].lstrip("# ").strip().lower() if part.startswith("## ") else ""
    placed, out = set(), []
    for part in parts:
        mine = [f for f in figures if heading(part) and (f.get("section") or "").strip().lower() == heading(part)]
        placed.update(id(f) for f in mine)
        out.append((part, mine))
    rest = [f for f in figures if id(f) not in placed]
    if rest:
        out.append(("## Figures from the published papers", rest))
    return out


def show_figure(fig: dict) -> None:
    path = ROOT / str(fig.get("url") or "")
    with st.container(border=True):
        if path.is_file():
            st.image(str(path), use_container_width=True)
        else:
            st.warning(f"The image file could not be found at {fig.get('url')}.")
        text = figure_markdown(fig)
        if text:
            st.markdown(text)
        st.caption(caption_text(fig))


def render_report(markdown: str, figures: list[dict]) -> None:
    for part, mine in sections_with_figures(markdown, figures):
        st.markdown(part)
        for fig in mine:
            show_figure(fig)
    if figures:
        st.markdown("#### Figure credits")
        st.markdown("\n".join(credit_markdown(f) for f in figures))
        st.caption(FIGURE_NOTE)


def pdf_button(name: str | None, key: str) -> None:
    path = ROOT / "demo_output" / name if name else None
    if path and path.is_file():
        st.download_button("Download the report as PDF", data=path.read_bytes(), file_name=name, mime="application/pdf", key=key)


def render_turn(turn: dict, index: int) -> None:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        for note in turn.get("warnings", []):
            st.warning(note)
        if turn.get("status"):
            st.caption(turn["status"])
        render_report(turn["markdown"], turn.get("figures") or [])
        if turn.get("figures_note"):
            st.caption(turn["figures_note"])
        pdf_button(turn.get("pdf"), f"pdf-{index}-{turn.get('pdf')}")


def run_question(question: str, index: int) -> dict:
    """Run the pipeline for one question, updating the page as events arrive. Returns the finished turn (kept in the session so the page can be redrawn)."""
    import server      # imported here, not at the top: the page must not wait for the pipeline's libraries
    from qa_stream import deliver_outputs, stream_answer
    clean = server.scrub_messages([{"role": "user", "content": question}])[0]["content"]      # personal identifiers are removed before anything is searched, sent to a model or saved
    turn = {"question": question, "saved_question": clean, "markdown": "", "figures": [], "pdf": None, "warnings": [], "status": "", "figures_note": ""}
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        status = st.status("Starting…", expanded=True)
        warning_box = st.container()
        report_box = st.empty()
        figures_note = st.empty()
        pdf_box = st.empty()
        final = None
        text = ""
        saved = False

        def draw():
            with report_box.container():
                render_report(turn["markdown"] or text, turn["figures"])

        def events():
            nonlocal final
            for event in stream_answer(clean):
                if event["type"] == "final":
                    final = event
                yield event
            if final:
                yield from deliver_outputs(clean, final)

        for event in events():
            kind = event.get("type")
            if kind == "stage":
                status.update(label=STAGE_TEXT.get(event["stage"], str(event["stage"]).title()) + "…")
            elif kind == "token":
                text += event["text"]
                report_box.markdown(text)
            elif kind == "warning":
                turn["warnings"].append(event.get("message", ""))
                warning_box.warning(event.get("message", ""))
            elif kind == "milestone" and event.get("name") == "first_useful_answer":
                status.update(label=f"Preliminary answer ready ({event.get('seconds')} s); the verified report will replace it…")
            elif kind == "verification":
                status.update(label=f"Verification: {event.get('status')} ({event['checked']['pmids']} PMIDs, {event['checked']['trials']} trials checked)")
            elif kind == "final":
                turn["markdown"] = event["markdown"]
                tm = event.get("timings") or {}
                turn["status"] = (f"Final verified answer · first token {tm.get('first_token_s')} s · first useful answer {tm.get('first_useful_s')} s · retrieval {tm.get('retrieval_deep_s')} s"
                                  f" · verified {tm.get('verification_done_s')} s · final {tm.get('final_s')} s")
                status.update(label="Final verified answer. Looking for figures from the papers…", state="running")
                draw()
            elif kind == "figures":
                turn["figures"] = event.get("items") or []
                turn["figures_note"] = "" if turn["figures"] else ("No figure passed the checks for this question" + (" (or they were not ready in time)." if event.get("late") else "."))
                draw()
                figures_note.caption(turn["figures_note"])
            elif kind == "pdf" and event.get("name"):
                turn["pdf"] = event["name"]
                with pdf_box.container():
                    pdf_button(event["name"], f"pdf-{index}-{event['name']}-{time.time()}")
                if not saved:
                    earlier = [m for past in st.session_state.get("turns", []) for m in ({"role": "user", "content": past.get("saved_question") or past.get("question", "")}, {"role": "assistant", "content": past.get("markdown", "")})]
                    server.save_conversation(earlier + [{"role": "user", "content": clean}], turn["markdown"], str(st.session_state.get("conversation_id", "")))
                    saved = True
            elif kind == "error":
                turn["warnings"].append(f"Stopped: {event.get('message')}")
                warning_box.error(f"Stopped: {event.get('message')}")
        status.update(label=turn["status"] or "Done", state="complete", expanded=False)
    return turn


CHAT_DIR = ROOT / "conversation_history" / "chats"      # one JSON file per conversation: every question asked in it, its report, figures and PDF name
TITLE_CHARS = 48


def chat_title(turns: list[dict]) -> str:
    first = " ".join(str((turns[0].get("saved_question") or turns[0].get("question") or "") if turns else "").split())
    return (first[:TITLE_CHARS].rstrip() + ("…" if len(first) > TITLE_CHARS else "")) or "New chat"


def save_chat(chat_id: str, turns: list[dict]) -> None:
    """Keep the conversation on disk so it is still there after a restart. Only the identifier-scrubbed question is saved (the one the pipeline actually used)."""
    if not turns:
        return
    CHAT_DIR.mkdir(parents=True, exist_ok=True)
    kept = [{**t, "question": t.get("saved_question") or t.get("question")} for t in turns]
    data = {"id": chat_id, "title": chat_title(turns), "updated": time.time(), "turns": kept}
    (CHAT_DIR / f"{re.sub(r'[^a-zA-Z0-9_-]', '', chat_id)}.json").write_text(json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")


def list_chats() -> list[dict]:
    """The saved conversations, most recently used first: [{'id', 'title', 'updated'}]."""
    found = []
    for path in CHAT_DIR.glob("*.json") if CHAT_DIR.is_dir() else []:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            found.append({"id": data["id"], "title": data.get("title") or "New chat", "updated": data.get("updated", 0)})
        except (OSError, ValueError, KeyError):
            continue
    return sorted(found, key=lambda c: -c["updated"])


def load_chat(chat_id: str) -> list[dict]:
    try:
        return json.loads((CHAT_DIR / f"{re.sub(r'[^a-zA-Z0-9_-]', '', chat_id)}.json").read_text(encoding="utf-8")).get("turns") or []
    except (OSError, ValueError):
        return []


def delete_chat(chat_id: str) -> None:
    (CHAT_DIR / f"{re.sub(r'[^a-zA-Z0-9_-]', '', chat_id)}.json").unlink(missing_ok=True)


def new_chat() -> None:
    st.session_state["turns"] = []
    st.session_state["conversation_id"] = f"chat-{int(time.time() * 1000)}"


def open_chat(chat_id: str) -> None:
    st.session_state["turns"] = load_chat(chat_id)
    st.session_state["conversation_id"] = chat_id


def remove_chat(chat_id: str) -> None:
    delete_chat(chat_id)
    if st.session_state.get("conversation_id") == chat_id:
        new_chat()


MARK = '''<svg width="{n}" height="{n}" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="2.2"/><circle cx="5" cy="6" r="1.6"/><circle cx="19" cy="6" r="1.6"/><circle cx="5" cy="18" r="1.6"/><circle cx="19" cy="18" r="1.6"/><path d="M10.2 10.6 6.2 7.2M13.8 10.6l4-3.4M10.2 13.4l-4 3.4M13.8 13.4l4 3.4"/></svg>'''


def chat_list(box, round_no: int = 0) -> None:
    """The conversations in the sidebar, drawn into `box` (an empty slot) so it can be redrawn after a new question WITHOUT rerunning the page (a rerun would throw the reader to the bottom)."""
    with box.container():
        chats = list_chats()
        if chats:
            st.markdown('<div class="side-label">Conversations</div>', unsafe_allow_html=True)
        for chat in chats:
            open_col, delete_col = st.columns([6, 1])
            current = chat["id"] == st.session_state.get("conversation_id")
            open_col.button(chat["title"], key=f"open-{chat['id']}-{round_no}", on_click=open_chat, args=(chat["id"],), use_container_width=True, type="primary" if current else "secondary")
            delete_col.button("✕", key=f"del-{chat['id']}-{round_no}", on_click=remove_chat, args=(chat["id"],), help="Delete this conversation")


def sidebar():
    with st.sidebar:
        st.markdown(f'<div class="brand"><div class="mark">{MARK.format(n=22)}</div><div><div class="name">NeuroGPT</div><div class="sub">Clinical research</div></div></div>', unsafe_allow_html=True)
        st.write("")
        st.button("＋  New chat", use_container_width=True, on_click=new_chat, key="new-chat")
        box = st.empty()
    chat_list(box)
    return box


def main() -> None:
    st.set_page_config(page_title="NeuroGPT", page_icon="🧠", layout="centered", initial_sidebar_state="expanded")
    st.markdown(STYLE, unsafe_allow_html=True)
    warm_in_background()
    st.session_state.setdefault("turns", [])
    st.session_state.setdefault("conversation_id", f"chat-{int(time.time() * 1000)}")
    list_box = sidebar()
    question = st.chat_input("Ask a clinical research question")
    if not st.session_state["turns"] and not question:
        st.markdown(f'<div class="hero"><div class="mark">{MARK.format(n=34)}</div><h1>Ask a clinical research question</h1>'
                    '<p>Evidence-checked answers with sources, treatment options, trials and figures.</p>'
                    '<div class="chips"><span>PubMed</span><span>Europe PMC</span><span>ClinicalTrials.gov</span><span>FDA labels</span><span>Guidelines</span><span>Original figures</span><span>PDF report</span></div></div>', unsafe_allow_html=True)
    elif st.session_state["turns"]:
        st.markdown('<div class="topbar"><span class="t">NeuroGPT</span><span class="s">Clinical research assistant</span></div>', unsafe_allow_html=True)
    for i, past in enumerate(st.session_state["turns"]):
        render_turn(past, i)
    if question:
        st.session_state["turns"].append(run_question(question, len(st.session_state["turns"])))
        save_chat(st.session_state["conversation_id"], st.session_state["turns"])
        chat_list(list_box, 1)      # the new conversation appears in the sidebar; the page is NOT rerun, so what is on screen stays exactly where the reader is


if __name__ == "__main__":      # Streamlit runs the script as __main__; importing the module (tests) draws nothing
    main()
