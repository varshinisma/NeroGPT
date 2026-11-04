"""The stages of a research run and the rules for their order.

The clinical-research-orchestrator SKILL.md describes the run as steps (Step 5 the fast path, Step 6 the deep path, Step 7 verify and repair, Step 8 finalize). The program offers those steps as the
named stages below; the ORCHESTRATOR AGENT chooses their order (in the same call that writes the plan), and code only checks the choice: a stage never runs before the stages whose results it reads,
every required stage runs, and an unknown name is dropped. A missing or unusable choice means DEFAULT_ORDER, the order the skill describes. Nothing here names a disease or a drug."""
from __future__ import annotations

# name: (SKILL.md step, stages whose results it reads, required, what it does)
STAGES: dict[str, tuple[str, tuple[str, ...], bool, str]] = {
    "fast_search": ("Step 5", (), True, "quick searches of the literature, trials, FDA labels and the web from the question's own words"),
    "fast_answer": ("Step 5", ("fast_search",), True, "the first answer, streamed, and the top-drugs box"),
    "deep_search": ("Step 6", ("fast_answer",), True, "wait for the planned deep searches and merge the evidence group"),
    "rank": ("Step 6", ("deep_search",), True, "rank the papers (keyword, meaning, cross-encoder)"),
    "read_evidence": ("Step 6", ("rank",), True, "start the relevance reading of papers and trials (evidence-synthesis) and the full-text fetch"),
    "drugs": ("Step 6", ("rank",), True, "start the drug stage (drug-intelligence): labels, doses, landscape, safety and approvals, trials table"),
    "escalate": ("Step 6", ("read_evidence",), True, "finish the relevance reading and run the follow-up searches the reading asks for"),
    "write_start": ("Step 6", ("escalate",), True, "start the section writers (clinical-answer-writer, evidence-synthesis) in parallel"),
    "figures": ("Step 6", ("write_start",), True, "find the figures of the relevant open-access papers beside the writing"),
    "write_stream": ("Step 6", ("write_start",), True, "stream the written sections in the report's order"),
    "assemble": ("Step 6", ("write_stream", "drugs", "fast_answer"), True, "wait for the drug stage, then put the sections into the report's order"),
    "limitations": ("Step 6", ("assemble",), True, "list what could not be retrieved, screened or verified"),
    "verify": ("Step 7", ("limitations",), True, "citation-verification, repair of failed claims, quality-control"),
    "finalize": ("Step 8", ("verify",), True, "set the status to FINAL, number the references, log the times, send the report"),
}
DEFAULT_ORDER = ("fast_search", "fast_answer", "deep_search", "rank", "read_evidence", "drugs", "escalate", "write_start", "figures", "write_stream", "assemble", "limitations", "verify", "finalize")
FIRST = ("fast_search", "fast_answer")      # the skill's fast path: it runs before any plan exists, so the program starts it itself


def clean_order(requested) -> list[str]:
    """The agent's order, made valid: unknown or repeated names dropped, required stages the agent left out added, and every stage placed after the stages it reads (the agent's order is kept wherever that is allowed)."""
    if not isinstance(requested, (list, tuple)) or not requested:
        return list(DEFAULT_ORDER)      # no choice made: the order the skill describes
    wanted = [n for n in dict.fromkeys(str(x).strip().lower().replace(" ", "_") for x in (requested if isinstance(requested, (list, tuple)) else [])) if n in STAGES]
    for name in DEFAULT_ORDER:
        if STAGES[name][2] and name not in wanted:
            wanted.append(name)
    placed: list[str] = []
    pending = list(wanted)
    while pending:
        ready = next((n for n in pending if all(r in placed for r in STAGES[n][1])), None)
        if ready is None:      # cannot happen with this table; the default order is the safe answer
            return list(DEFAULT_ORDER)
        placed.append(ready)
        pending.remove(ready)
    return placed


def describe() -> str:
    """The stages as the agent reads them: name, skill step, what it reads, what it does."""
    return "\n".join(f"- {n} ({STAGES[n][0]}{'' if STAGES[n][2] else ', optional'}; after: {', '.join(STAGES[n][1]) or 'nothing'}): {STAGES[n][3]}" for n in DEFAULT_ORDER)
