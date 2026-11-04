"""NeuroGPT front end (Streamlit): a light page that sends the question to the backend (api.py, FastAPI) over HTTP and draws the events it streams back.

Run, in two terminals (from this folder):
    uvicorn api:app --host 127.0.0.1 --port 8000
    streamlit run streamlit_app.py --server.port 8501
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



API_URL = os.getenv("NEUROGPT_API_URL", "http://127.0.0.1:8000").rstrip("/")      # the backend (uvicorn api:app): the research pipeline runs there, this page only draws what it sends


def backend_status() -> dict | None:
    """{'status', 'ready'} from the backend's /health, or None when it cannot be reached."""
    import httpx
    try:
        reply = httpx.get(f"{API_URL}/health", timeout=3.0)
        return reply.json() if reply.status_code == 200 else None
    except Exception:
        return None


def api_events(question: str, history: list[dict], conversation_id: str):
    """The backend's answer to one question, as events (dicts), read line by line while it streams. A backend that cannot be reached or that fails gives one `error` event."""
    import httpx
    body = {"question": question, "history": history, "conversation_id": conversation_id}
    try:
        with httpx.stream("POST", f"{API_URL}/ask", json=body, timeout=httpx.Timeout(10.0, read=600.0)) as reply:
            if reply.status_code != 200:
                reply.read()
                yield {"type": "error", "message": f"The backend answered {reply.status_code}: {reply.text[:200]}"}
                return
            for raw in reply.iter_lines():
                if raw.strip():
                    try:
                        yield json.loads(raw)
                    except ValueError:
                        continue
    except httpx.HTTPError as error:
        yield {"type": "error", "message": f"The backend at {API_URL} could not be reached ({type(error).__name__}). Start it with: uvicorn api:app --port 8000"}


@st.cache_data(show_spinner=False, max_entries=64)
def fetch_bytes(url: str, params: tuple = ()) -> bytes | None:
    import httpx
    try:
        reply = httpx.get(url, params=dict(params), timeout=30.0)
        return reply.content if reply.status_code == 200 else None
    except Exception:
        return None


STAGE_TEXT = {"INITIAL": "Finding the first evidence", "ENRICHING": "Reading the evidence", "EVIDENCE_ENRICHING": "Reading the evidence", "FINAL": "Final verified answer"}
FIGURE_NOTE = "Each figure shows one study's result; it does not by itself establish that a treatment works or justify a recommendation."
STYLE = """
<style>
:root { --navy: #0f2a43; --blue: #1d4ed8; --teal: #0f766e; --line: #e2e8f0; --muted: #64748b; --soft: #f8fafc; }
html, body, [class*="css"], [data-testid="stAppViewContainer"] { font-family: Inter, "Segoe UI", system-ui, -apple-system, sans-serif; }
[data-testid="stHeader"] { background: transparent; }
[data-stale="true"], .stApp [data-stale="true"] { opacity: 1 !important; transition: none !important; }
[data-testid="stMain"] [data-stale="true"]:not(:has([data-testid="stChatInput"])), section.main [data-stale="true"]:not(:has([data-testid="stChatInput"])) { display: none !important; }      /* what the PREVIOUS run left on the page and the new run has not replaced yet (the first answer, with its frozen "Reading the evidence" box) is not shown while the next question is answered; the earlier turns are drawn again at once as new elements */      /* Streamlit greys out the earlier content while a new run starts: it looked like the page had frozen */
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
    image = fetch_bytes(f"{API_URL}/file", (("path", str(fig.get("url") or "")),))
    with st.container(border=True):
        if image:
            st.image(image, use_container_width=True)
        else:
            st.warning(f"The image file could not be found at {fig.get('url')}.")
        text = figure_markdown(fig)
        if text:
            st.markdown(text)
        st.caption(caption_text(fig))


FIGURES_INLINE = os.getenv("FIGURES_INLINE", "on").strip().lower() != "off"      # on (default): each figure sits inside the report section that cites its paper, as it always did; off: all figures come after the report


def render_figures(figures: list[dict]) -> None:
    """All the figures, one after another, then their credits. Drawn BELOW the report, so adding them never moves the text the reader is looking at."""
    if not figures:
        return
    st.markdown("## Figures from the published papers")
    for fig in figures:
        show_figure(fig)
    st.markdown("#### Figure credits")
    st.markdown("\n".join(credit_markdown(f) for f in figures))
    st.caption(FIGURE_NOTE)


def render_credits(figures: list[dict]) -> None:
    if figures:
        st.markdown("#### Figure credits")
        st.markdown("\n".join(credit_markdown(f) for f in figures))
        st.caption(FIGURE_NOTE)


def render_text(markdown: str) -> str:
    from presentation import hide_unneeded_sections
    return hide_unneeded_sections(markdown)


def render_report(markdown: str, figures: list[dict]) -> None:
    markdown = render_text(markdown)
    if not FIGURES_INLINE:
        st.markdown(markdown)
        render_figures(figures)
        return
    for part, mine in sections_with_figures(markdown, figures):
        st.markdown(part)
        for fig in mine:
            show_figure(fig)
    if figures:
        st.markdown("#### Figure credits")
        st.markdown("\n".join(credit_markdown(f) for f in figures))
        st.caption(FIGURE_NOTE)


def pdf_button(name: str | None, key: str) -> None:
    data = fetch_bytes(f"{API_URL}/pdf/{name}") if name else None
    if data:
        st.download_button("Download the report as PDF", data=data, file_name=name, mime="application/pdf", key=key)


def render_turn(turn: dict, index: int) -> None:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        if turn.get("understood"):
            st.info(turn["understood"] if turn["understood"] == "Answering from this conversation" else f"Understood as: {turn['understood']}")
        if turn.get("status"):
            st.caption(turn["status"])
        render_report(turn["markdown"], turn.get("figures") or [])
        for note in turn.get("warnings", []):      # after the report, as in the live view
            st.warning(note)
        if turn.get("figures_note"):
            st.caption(turn["figures_note"])
        pdf_button(turn.get("pdf"), f"pdf-{index}-{turn.get('pdf')}")


START_SCROLL_JS = """<script>/*%s*/(function(){try{const w=window.parent,d=w.document;const el=d.querySelector('[data-testid="stMain"]')||d.querySelector('section.main');if(!el)return;
const S=w.__ngScroll=w.__ngScroll||{};
if(S.el!==el||!S.bound){S.el=el;S.bound=true;S.user=0;S.drag=false;
const mark=()=>{S.user=Date.now()};
['wheel','touchstart','touchmove','keydown'].forEach(ev=>el.addEventListener(ev,mark,{passive:true}));
d.addEventListener('keydown',mark,{passive:true});
el.addEventListener('mousedown',()=>{S.drag=true;S.user=Date.now()},{passive:true});
d.addEventListener('mouseup',()=>{S.drag=false},{passive:true});
el.addEventListener('scroll',()=>{if(S.drag||Date.now()-S.user<400){const r=el.getBoundingClientRect();const n=d.elementFromPoint(r.left+r.width/2,r.top+90);S.saved=el.scrollTop;if(n){S.ref=n;S.refOff=n.getBoundingClientRect().top-r.top;}}},{passive:true});}
S.saved=null;S.ref=null;S.start=Date.now();
const id='%s';
const go=()=>{const a=d.getElementById(id);if(a){const r=el.getBoundingClientRect();const delta=a.getBoundingClientRect().top-r.top-70;if(Math.abs(delta)>2)el.scrollTop+=delta;}};
go();let n=0;const t=setInterval(()=>{if((S.user>S.start+700)||++n>90){clearInterval(t);return;}go();},100);}catch(x){}})();</script>"""
KEEP_SCROLL_JS = """<script>/*%s*/(function(){try{const w=window.parent,d=w.document,S=w.__ngScroll||{};const el=S.el||d.querySelector('[data-testid="stMain"]')||d.querySelector('section.main');if(!el)return;
const id='%s';
const apply=()=>{if(Date.now()-(S.user||0)<300)return;const r=el.getBoundingClientRect();
if(S.ref&&S.ref.isConnected){el.scrollTop+=(S.ref.getBoundingClientRect().top-r.top)-S.refOff;}
else if(S.saved!=null){el.scrollTop=S.saved;}
else{const a=d.getElementById(id);if(a){el.scrollTop+=a.getBoundingClientRect().top-r.top-70;}}};
const t0=Date.now();apply();const iv=setInterval(()=>{if((S.user||0)>t0||Date.now()-t0>9000){clearInterval(iv);return;}apply();},150);}catch(x){}})();</script>"""


def scroll_script(box, template: str, *args) -> None:
    """Zero-height script in the slot `box` at the very bottom of the answer. START scrolls to the question that was just asked (and keeps it at the top while the answer grows, until the reader scrolls themselves)
    and remembers what the READER is looking at (their own wheel, keys or scrollbar only); KEEP puts the page back on that content when the answer finishes, because Streamlit otherwise throws a follow-up's page to
    the bottom, even if images were inserted above it. A reader who did not scroll stays on their question."""
    try:
        import streamlit.components.v1 as components
        with box.container():
            components.html(template % ((str(time.time()),) + args), height=0)
    except Exception:
        pass


STEP_TEXT = {"fast_search": "Searched PubMed, trials and FDA labels", "fast_answer": "Fast answer written", "deep_search": "Deep search finished", "rank": "Papers ranked", "read_evidence": "Evidence read",
             "drugs": "Drug landscape started", "escalate": "Gaps checked", "write_start": "Writing the report", "figures": "Looking for figures", "write_stream": "Report written",
             "assemble": "Report assembled", "limitations": "Limitations written", "verify": "Verified", "finalize": "Finished"}
STREAM_EVERY_S = 0.15      # the report text is redrawn at most this often: one redraw per token made the page flicker and shift


def run_question(question: str, index: int) -> dict:
    """Run the pipeline for one question, updating the page as events arrive. Returns the finished turn (kept in the session so the page can be redrawn).
    The layout is fixed from the first moment (question, status, report, notes, figures, PDF), and everything that arrives later is added BELOW what is already there, so the text the reader is on never moves."""
    anchor = f"q-{index}"
    start_box = st.empty()      # the START scroll script is drawn FIRST so it loads at once: the reader is taken to the new question before the old answer can be read
    scroll_script(start_box, START_SCROLL_JS, anchor)
    with st.chat_message("user"):      # drawn FIRST: the reader sees their question and a status at once, before the pipeline's libraries are loaded
        st.markdown(f'<div id="{anchor}"></div>', unsafe_allow_html=True)      # the page scrolls to here when the question is sent
        st.markdown(question)
    with st.chat_message("assistant"):
        understood_box = st.empty()
        status = st.status("Starting…", expanded=False)
        turn = {"question": question, "saved_question": "[question not processed: the backend did not answer]", "markdown": "", "figures": [], "pdf": None, "warnings": [], "status": "", "figures_note": ""}
        report_area = st.container()      # the report: one slot per `##` section, each with a slot after it for its figures
        slots: list[dict] = []
        warning_box = st.container()
        figures_box = st.empty()
        figures_note = st.empty()
        pdf_box = st.empty()
        script_box = st.empty()      # the KEEP scroll script lives here, at the very bottom: it takes no room above the text
        text = ""
        last_draw = [0.0]

        def show_text(force: bool = False) -> None:
            """Each section of the report has its own slot and is redrawn only when its own text changed, so a finished section never moves and figures can be added after it later."""
            if not (force or time.time() - last_draw[0] >= STREAM_EVERY_S):
                return
            last_draw[0] = time.time()
            parts = [x for x in re.split(r"(?m)^(?=## )", text) if x.strip()]
            for i, part in enumerate(parts):
                if i >= len(slots):
                    with report_area:
                        slots.append({"text": st.empty(), "figs": st.empty(), "shown": None, "heading": ""})
                slot = slots[i]
                if part != slot["shown"]:
                    slot["text"].markdown(part)
                    slot["shown"] = part
                    slot["heading"] = part.splitlines()[0].lstrip("# ").strip().lower() if part.startswith("## ") else ""
            for slot in slots[len(parts):]:
                if slot["shown"]:
                    slot["text"].empty()
                    slot["shown"], slot["heading"] = "", ""

        history = [m for past in st.session_state.get("turns", []) for m in ({"role": "user", "content": past.get("saved_question") or past.get("question", "")}, {"role": "assistant", "content": past.get("markdown", "")})]

        for event in api_events(question, history, str(st.session_state.get("conversation_id", ""))):
            kind = event.get("type")
            if kind == "scrubbed":
                turn["saved_question"] = event.get("text") or question      # the question as the backend stored it, personal details removed
            elif kind == "understood":
                turn["understood"] = event.get("text") or ""      # a follow-up, rewritten as one complete question from the conversation so far
                understood_box.info(turn["understood"] if event.get("mode") == "chat" else f"Understood as: {turn['understood']}")
            elif kind == "preface":
                turn["preface"] = event.get("text") or ""      # the quick answer, shown before the research text
                text += turn["preface"]
                show_text(True)
            elif kind == "stage":
                status.update(label=STAGE_TEXT.get(event["stage"], str(event["stage"]).title()) + "…")
            elif kind == "progress":
                done = STEP_TEXT.get(event.get("stage_name"), "")
                if done:
                    status.update(label=f"{done} · {event.get('papers', 0)} papers, {event.get('trials', 0)} trials so far · {event.get('t')} s")
            elif kind == "token":
                text += event["text"]
                show_text()
            elif kind == "warning":
                turn["warnings"].append(event.get("message", ""))
                warning_box.warning(event.get("message", ""))      # below the report: it never pushes the text down
            elif kind == "milestone" and event.get("name") == "first_useful_answer":
                show_text(True)
                status.update(label=f"Fast answer ready ({event.get('seconds')} s); researching and adding evidence...")
            elif kind == "verification":
                status.update(label=f"Verification: {event.get('status')} ({event['checked']['pmids']} PMIDs, {event['checked']['trials']} trials checked)")
            elif kind == "final" and event.get("mode") == "chat":
                turn["markdown"] = event["markdown"]      # answered from the conversation: no search, no figures, no PDF
                turn["status"] = f"Answered from this conversation · {(event.get('timings') or {}).get('final_s')} s"
                text = turn["markdown"]
                show_text(True)
            elif kind == "final":
                show_text(True)
                turn["markdown"] = (turn.get("preface") or "") + event["markdown"]
                tm = event.get("timings") or {}
                from presentation import verification_label
                verification_status = event.get("verification_status") or "INCOMPLETE"
                verification_text = verification_label(verification_status)
                turn["status"] = (f"Report ready · {verification_text} · first token {tm.get('first_token_s')} s · first useful answer {tm.get('first_useful_s')} s · retrieval {tm.get('retrieval_deep_s')} s"
                                  f" · verification {tm.get('verification_done_s')} s · final {tm.get('final_s')} s")
                status.update(label=f"Report complete · {verification_text}. Looking for figures from the papers…", state="running")
            elif kind == "figures":
                turn["figures"] = event.get("items") or []
                turn["figures_note"] = "" if turn["figures"] else ("No figure passed the checks for this question" + (" (or they were not ready in time)." if event.get("late") else "."))
                if FIGURES_INLINE:
                    placed: set = set()
                    for slot in slots:      # each figure goes into the slot after the section that cites its paper; the text itself is not redrawn
                        mine = [f for f in turn["figures"] if slot["heading"] and (f.get("section") or "").strip().lower() == slot["heading"]]
                        if mine:
                            placed.update(id(f) for f in mine)
                            with slot["figs"].container():
                                for fig in mine:
                                    show_figure(fig)
                    rest = [f for f in turn["figures"] if id(f) not in placed]
                    with figures_box.container():      # a figure whose section was not found, and the credits, come last
                        if rest:
                            st.markdown("## Figures from the published papers")
                            for fig in rest:
                                show_figure(fig)
                        render_credits(turn["figures"])
                else:
                    with figures_box.container():
                        render_figures(turn["figures"])
                figures_note.caption(turn["figures_note"])
            elif kind == "pdf" and event.get("name"):
                turn["pdf"] = event["name"]
                with pdf_box.container():
                    pdf_button(event["name"], f"pdf-{index}-{event['name']}-{time.time()}")
            elif kind == "error":
                turn["warnings"].append(f"Stopped: {event.get('message')}")
                warning_box.error(f"Stopped: {event.get('message')}")
        show_text(True)
        status.update(label=turn["status"] or "Done", state="complete", expanded=False)
        scroll_script(script_box, KEEP_SCROLL_JS, anchor)
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
    health = backend_status()
    if health is None:
        st.error(f"The backend at {API_URL} is not running. Start it in another terminal:  uvicorn api:app --host 127.0.0.1 --port 8000")
    st.session_state.setdefault("turns", [])
    st.session_state.setdefault("conversation_id", f"chat-{int(time.time() * 1000)}")
    list_box = sidebar()
    question = st.chat_input("Ask a clinical research question")
    # The page is made of THREE fixed slots, always in this order: header, earlier turns, the turn being answered. Every rerun replaces a slot as a whole, so the old page and the new one
    # never mix (before this, the first question showed again under the report while the follow-up was being answered, and the page jumped).
    header_box, history_box, live_box = st.empty(), st.empty(), st.empty()
    if question:
        live_box.empty()      # what the last run left in the live slot is now part of the earlier turns
    if not st.session_state["turns"] and not question:
        header_box.markdown(f'<div class="hero"><div class="mark">{MARK.format(n=34)}</div><h1>Ask a clinical research question</h1>'
                            '<p>Evidence-checked answers with sources, treatment options, trials and figures.</p>'
                            '<div class="chips"><span>PubMed</span><span>Europe PMC</span><span>ClinicalTrials.gov</span><span>FDA labels</span><span>Guidelines</span><span>Original figures</span><span>PDF report</span></div></div>', unsafe_allow_html=True)
    else:
        header_box.markdown('<div class="topbar"><span class="t">NeuroGPT</span><span class="s">Clinical research assistant</span></div>', unsafe_allow_html=True)
    with history_box.container():
        for i, past in enumerate(st.session_state["turns"]):
            render_turn(past, i)
    if question:
        with live_box.container():
            finished = run_question(question, len(st.session_state["turns"]))
        st.session_state["turns"].append(finished)
        save_chat(st.session_state["conversation_id"], st.session_state["turns"])
        chat_list(list_box, 1)      # the new conversation appears in the sidebar; the page is NOT rerun, so what is on screen stays exactly where the reader is
    else:
        live_box.empty()

if __name__ == "__main__":      # Streamlit runs the script as __main__; importing the module (tests) draws nothing
    main()
