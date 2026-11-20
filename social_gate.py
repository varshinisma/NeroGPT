"""A message that is not a clinical question ('hi', 'how are you', 'thanks', 'who are you') must not start a research run, and its reply is written by the model like any assistant's, not by a fixed sentence.

One tiny model call reads the message: it answers CLINICAL (the message is about health, medicine or research, a patient case, a fragment naming a condition, a drug or a symptom, or a request about the earlier
answer), or it writes the short friendly reply itself. A very short first message is read BEFORE the run starts (no plan, no search); a longer one is read while the run is already starting, and the run's events are
held back until the answer is in. A failure, silence or a long message means CLINICAL: a real question is never turned away by a failure. Nothing here names a disease or a drug."""
from __future__ import annotations

import os
import queue
import threading

MAX_WORDS = 12      # a longer message is taken as a real question without asking
EARLY_WORDS = 3      # a first message this short is read before the run starts, so no search or plan is made for 'hi'
WAIT_S = float(os.getenv("CLINICAL_GATE_WAIT_S", "4"))
FALLBACK = "I'm NeuroGPT, a clinical research assistant. What would you like to look up?"
INSTRUCTIONS = ("You are NeuroGPT, a friendly clinical research assistant that searches PubMed, ClinicalTrials.gov, FDA labels and the web. Read ONE message and do exactly one of these.\n"
                "1. If it asks for or gives medical, clinical, drug, disease, treatment, trial or health-research information (including a patient case, a short fragment that names a condition, a drug or a symptom, "
                "or a request about an earlier answer), reply with the single word CLINICAL and nothing else. When unsure, reply CLINICAL.\n"
                "2. Otherwise (a greeting, how are you, thanks, goodbye, a question about you or what you can do, small talk), reply in one or two warm, natural sentences in the person's language, "
                "answering what they said like a normal assistant would, and, when it fits, mention that you can look into clinical and medical questions. Do not give any medical facts and do not use markdown.")


def enabled() -> bool:
    return os.getenv("CLINICAL_GATE", "on").strip().lower() not in {"off", "0", "false", "no"}


def needs_gate(question: str) -> bool:
    words = (question or "").split()
    return enabled() and 0 < len(words) <= MAX_WORDS


def needs_early(question: str) -> bool:
    words = (question or "").split()
    return enabled() and 0 < len(words) <= EARLY_WORDS


def converse(question: str, ask=None) -> str | None:
    """The model's own short reply when the message is not a clinical question; None when it is clinical, or when anything failed (the message is then researched)."""
    try:
        if ask is None:
            import qa_stream
            ask = lambda system, prompt: qa_stream.run_llm(system, prompt, 400, temperature=0.0, purpose="triage")      # room for a reasoning model to think before its short reply
        answer = " ".join(str(ask(INSTRUCTIONS, f"MESSAGE: {question}") or "").split())
        print(f"[gate] model answer for {question[:40]!r}: {answer[:60]!r}", flush=True)
        if not answer or answer.upper().strip(" .!") == "CLINICAL" or answer.upper().startswith("CLINICAL"):
            return None
        return answer[:600]
    except Exception as error:
        print(f"[gate] the check failed ({type(error).__name__}: {str(error)[:120]}): the message is treated as clinical", flush=True)
        return None


class Rejected:
    """Stands in for a run that is never started (the plan already said the message is not clinical). `reply` is the text to give."""
    rejected = True

    def __init__(self, reply: str = ""):
        self.reply = reply or FALLBACK

    def __iter__(self):
        return iter(())


class Gated:
    """The events of a run, released only when the gate says the message is clinical. When it wrote a reply instead, `rejected` is True, `reply` holds the text, no event is yielded and the run is closed."""

    def __init__(self, events, gate, wait_s: float | None = None):
        self.events, self.gate, self.wait_s = events, gate, WAIT_S if wait_s is None else wait_s
        self.rejected, self.reply = False, ""

    def __iter__(self):
        box: queue.Queue = queue.Queue()
        stop = threading.Event()
        verdict: dict = {}

        def decide() -> None:
            try:
                verdict["reply"] = self.gate() or ""
            except Exception:
                verdict["reply"] = ""
            verdict["done"] = True

        def pump() -> None:      # the run goes on beside the question: its first seconds are not spent waiting for the gate
            try:
                for event in self.events:
                    if stop.is_set():
                        break
                    box.put(("event", event))
                box.put(("done", None))
            except BaseException as error:
                box.put(("error", error))
            finally:
                close = getattr(self.events, "close", None)
                if close:
                    try:
                        close()
                    except Exception:
                        pass

        gate_thread = threading.Thread(target=decide, daemon=True)
        gate_thread.start()
        threading.Thread(target=pump, daemon=True).start()
        gate_thread.join(timeout=self.wait_s)
        try:
            if verdict.get("done") and verdict.get("reply"):
                self.rejected, self.reply = True, verdict["reply"]
                return
            while True:
                kind, value = box.get()
                if kind == "event":
                    yield value
                elif kind == "error":
                    raise value
                else:
                    return
        finally:
            stop.set()      # the run behind it stops after its current event (or is closed at once when nothing more is wanted)
