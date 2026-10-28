"""Source-aware retrieval routing for the DEEP search: validated operations, execution with retry and budget, need coverage, conditional escalation and section states.

The model writes the queries and chooses, for each, a `target` from a fixed allowlist; Python validates, executes, tracks and reports. Nothing here knows a disease, a drug or a question type, and nothing here
imports the pipeline: the search functions are passed in as `adapters` (so everything can be tested offline with fakes).

The fast search is NOT part of this module and is never touched by it.

Vocabulary
- operation: one search = target + query/args + the information NEED it serves (`need_id`, `priority`) + an optional `alternate` (a second source for the same need).
- result status: completed_found | completed_empty | failed | timed_out | not_run   (a successful search with nothing found is NOT a failure)
- need state: covered | no_relevant_evidence | failed | timed_out | not_retrieved | not_generated | not_applicable
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

TARGETS = ("pubmed", "europepmc", "guideline_sites", "regulator", "fda_label", "web", "trials")
TEXT_TARGETS = ("pubmed", "europepmc", "guideline_sites", "regulator", "web")      # targets whose operation is a natural-language query
STUDY_TYPES = ("rct", "meta_analysis", "systematic_review", "guideline", "review", "negative_results")
TRIAL_STATUSES = ("recruiting", "completed", "any")
PRIORITIES = ("essential", "supporting")
INFO_NEEDED = ("treatment_options", "efficacy", "safety", "dosing", "diagnosis", "prognosis", "mechanism", "guideline_recommendation", "trial_status", "regulatory_status", "drug_interaction")
SAFE_FALLBACK_SKILLS = ("latest-evidence", "clinical-trials", "drug-intelligence")
MAX_OPERATIONS = int(os.getenv("ROUTING_MAX_QUERIES", "14"))      # the most routed searches one question may run (the coverage check may add a few more, see MAX_COVERAGE_ADDITIONS)
MAX_ESCALATIONS = 4
STOP_WORDS = set("a an the of in on for to and or with without about from by at as is are was were be been do does did can could should would how what which who why when where this that these those".split())
STATE_TEXT = {      # the one wording of each state (the report's sections show it; landscape.py imports it)
    "covered": "",
    "not_applicable": "Not applicable to this question: it does not call for this information, so no search was made for it.",
    "not_retrieved": "Not retrieved: the information is relevant, but the search did not run or did not finish, so this does not show that none exists.",
    "no_relevant_evidence": "No evidence found: a search of the appropriate sources completed but returned nothing relevant to this question.",
    "failed": "Retrieval failed: the source could not be searched, so this does not show that no evidence exists.",
    "timed_out": "Retrieval timed out: the source did not answer in time, so this does not show that no evidence exists.",
    "not_generated": "Not retrieved: no search was written for this information need, so this does not show that none exists.",
}


# ---------------------------------------------------------------------------------------------- operations
@dataclass
class Operation:
    id: str
    need_id: str
    need: str
    target: str
    query: str = ""
    args: dict = field(default_factory=dict)
    priority: str = "supporting"
    sub_intent: str = ""
    why: str = ""
    skill: str = ""
    alternate: dict | None = None
    origin: str = "generated"      # generated | baseline | escalation
    escalation: dict | None = None
    also_serves: list = field(default_factory=list)      # other need ids an identical search was also written for
    serves: str = ""      # which of the listed information needs (info_needed) this search answers; "" when the skill did not say

    @property
    def key(self) -> str:
        return op_key(self.target, self.query, self.args)

    def needs(self) -> list[str]:
        return [self.need_id] + list(self.also_serves)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9\-]*", str(text).lower())


def normalize_query(text: str) -> str:
    return " ".join(w for w in _words(text) if w not in STOP_WORDS)


def op_key(target: str, query: str, args: dict | None) -> str:
    """Two operations with the same target, the same words and the same arguments are the same search."""
    return json.dumps([target, normalize_query(query), args or {}], sort_keys=True, ensure_ascii=False)


def _stem(word: str) -> str:
    """A word reduced so inflections match: the first five letters of a long word ('medication' and 'medications'), the singular of a short one ('tics' and 'tic')."""
    return word[:5] if len(word) > 5 else (word[:-1] if word.endswith("s") and len(word) > 3 else word)


WEAK_ANCHORS = {"used", "using", "uses", "give", "given", "which", "there", "their", "about", "within", "between", "among", "other", "currently", "commonly", "common", "list", "tell", "please", "above", "below"}      # words of a question that say nothing about its topic


def stem_set(words) -> set[str]:
    return {_stem(w) for w in words if w}


def stem_overlap(text: str, anchors: set[str]) -> int:
    """How many of the question's words (by stem) a text keeps."""
    return len(stem_set(re.findall(r"[a-z][a-z0-9\-]{2,}", str(text).lower())) & stem_set(anchors))


def anchor_words(question: str, condition: str = "", case_facts: list | None = None) -> set[str]:
    """The content words of the question (and its condition and case facts): a search must keep some of them, so it stays about THIS question."""
    text = " ".join([question or "", condition or ""] + [str(f) for f in (case_facts or [])])
    return {w for w in re.findall(r"[a-z][a-z0-9\-]{3,}", text.lower()) if w not in STOP_WORDS and w not in WEAK_ANCHORS}


def _clean_list(value, limit: int, maxlen: int = 60) -> list[str]:
    items = value if isinstance(value, list) else []
    return list(dict.fromkeys(" ".join(str(x).split())[:maxlen] for x in items if isinstance(x, (str, int)) and str(x).strip()))[:limit]


def validate_args(target: str, args) -> dict:
    """Only the arguments each target understands, with values from fixed sets or clamped ranges. Anything else the model wrote is dropped."""
    args = args if isinstance(args, dict) else {}
    out: dict = {}
    if target in ("pubmed", "europepmc"):
        out["sort"] = args.get("sort") if args.get("sort") in ("relevance", "date") else "relevance"
        try:
            out["years_back"] = min(15, max(1, int(args.get("years_back", 5))))
        except (TypeError, ValueError):
            out["years_back"] = 5
        if target == "europepmc" and args.get("fulltext") is True:
            out["fulltext"] = True      # the words are searched in the full text of open-access articles
        if target == "pubmed":
            types = [t for t in (args.get("study_types") if isinstance(args.get("study_types"), list) else []) if t in STUDY_TYPES]
            if types:
                out["study_types"] = sorted(set(types))
            if args.get("open_access") is True:
                out["open_access"] = True
    elif target == "trials":
        plain = lambda text: " ".join(re.sub(r"\b(?:AND|OR|NOT)\b|[()\"]", " ", str(text or "")).split())      # the registry's condition field is not a boolean query: words like AND made it return nothing
        out["condition"] = plain(args.get("condition"))[:160]
        out["intervention"] = plain(args.get("intervention"))[:100]
        out["status"] = args.get("status") if args.get("status") in TRIAL_STATUSES else "any"
        out["drug_only"] = args.get("drug_only") is True
    elif target == "fda_label":
        out["condition_terms"] = _clean_list(args.get("condition_terms"), 6, 40)
        out["drug_names"] = _clean_list(args.get("drug_names"), 6, 40)
    return out


def _parse_one(item, skill: str, question: str, anchors: set[str], n: int, *, need_floor: int = 1) -> tuple[Operation | None, str]:
    """One model-written search -> an Operation, or (None, why it was rejected)."""
    if not isinstance(item, dict):
        return None, "not an object"
    target = str(item.get("target") or "").strip().lower()
    if target not in TARGETS:
        return None, f"invalid target {target!r}"
    need = " ".join(str(item.get("need") or item.get("sub_intent") or "").split())[:60]
    if not need:
        return None, "no information need named"
    args = validate_args(target, item.get("args"))
    query = " ".join(str(item.get("query") or "").split())[:300]
    squashed_question = " ".join(_words(question))
    if target in TEXT_TARGETS:
        if not 4 <= len(query.split()) <= 40:
            return None, "query is not a phrase of 4 to 40 words"
        if stem_overlap(query, anchors) < need_floor:
            return None, "query does not keep the question's terms"
    elif target == "trials":
        if not args["condition"] or not stem_overlap(args["condition"], anchors):
            return None, "trial condition does not keep the question's terms"
        query = args["condition"]
    elif target == "fda_label":
        terms = [t for t in args["condition_terms"] if stem_overlap(t, anchors)]
        drugs = [d for d in args["drug_names"] if " ".join(_words(d)) and " ".join(_words(d)) in squashed_question]      # a drug may only be named if the question names it
        if not terms and not drugs:
            return None, "fda_label needs condition terms from the question or a drug the question names"
        args = {"condition_terms": terms, "drug_names": drugs}
        query = " ".join(terms + drugs)
    priority = item.get("priority") if item.get("priority") in PRIORITIES else "supporting"
    need_id = re.sub(r"[^a-z0-9_\-]", "", str(item.get("need_id") or "").lower())[:24] or "n" + re.sub(r"[^a-z0-9]", "", need.lower())[:20]
    serves = str(item.get("serves") or "").strip().lower()
    op = Operation(id=f"{skill}:{n}", need_id=need_id, need=need, target=target, query=query, args=args, priority=priority, sub_intent=" ".join(str(item.get("sub_intent") or "").split())[:48],
                   why=" ".join(str(item.get("why") or "").split())[:120], skill=skill, serves=serves if serves in INFO_NEEDED else "")
    alt = item.get("alternate")
    if isinstance(alt, dict):
        alt_op, why = _parse_one({**alt, "need": need, "need_id": need_id, "priority": priority}, skill, question, anchors, n, need_floor=need_floor)
        if alt_op is not None and alt_op.key != op.key:
            op.alternate = {"target": alt_op.target, "query": alt_op.query, "args": alt_op.args}
    return op, ""


def declined(reply) -> bool:
    """True when the skill's own raw answer was an empty JSON object (it had nothing to add). A reply that could not be read at all is NOT a decline."""
    raw = reply.get("_raw") if isinstance(reply, dict) else None
    if not isinstance(raw, str) or isinstance(reply.get("queries"), list):
        return False
    text = re.sub(r"```(?:json)?", "", raw).strip()
    try:
        return json.loads(text) == {}
    except ValueError:
        return False


def parse_operations(skill: str, reply, question: str, anchors: set[str], *, max_ops: int = 8) -> tuple[list[Operation], list[dict]]:
    """The searches a skill wrote -> (valid operations, [{item, reason}] for every rejected one). A malformed reply gives no operations and one rejection; it never raises."""
    items = reply.get("queries") if isinstance(reply, dict) else None
    if not isinstance(items, list):
        if declined(reply):
            return [], []      # the skill said its source is not needed for this question: that is an answer, not an error
        return [], [{"item": None, "reason": "the reply has no list of queries"}]
    need_floor = 2 if len(anchors) >= 4 else 1
    ops, rejected, seen = [], [], set()
    for n, item in enumerate(items):
        op, why = _parse_one(item, skill, question, anchors, n, need_floor=need_floor)
        if op is None:
            rejected.append({"item": item if isinstance(item, dict) else None, "reason": why})
        elif op.key in seen:
            rejected.append({"item": item, "reason": "duplicate search"})
        else:
            seen.add(op.key)
            ops.append(op)
    return ops[:max_ops], rejected


NEAR_DUPLICATE = 0.8      # two searches of the same source whose words overlap this much (Jaccard) ask the same thing


def near_duplicate(a: Operation, b: Operation) -> bool:
    if a.target != b.target or a.args != b.args:
        return False
    wa, wb = stem_set(normalize_query(a.query).split()), stem_set(normalize_query(b.query).split())      # by stem: "trial" and "trials" are the same word
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= NEAR_DUPLICATE


def merge_operations(*lists: list[Operation], cap: int = MAX_OPERATIONS) -> tuple[list[Operation], list[Operation]]:
    """Operations from several skills in one list. An identical (or near-identical) search is run once; it remembers every need it serves. Returns (operations, duplicates removed)."""
    by_key: dict[str, Operation] = {}
    dropped: list[Operation] = []
    for ops in lists:
        for op in ops:
            kept = by_key.get(op.key) or next((k for k in by_key.values() if near_duplicate(k, op)), None)
            if kept is None:
                by_key[op.key] = op
            else:
                dropped.append(op)
                if op.need_id != kept.need_id and op.need_id not in kept.also_serves:
                    kept.also_serves.append(op.need_id)
                if op.priority == "essential":
                    kept.priority = "essential"
                kept.alternate = kept.alternate or op.alternate
    ordered = list(by_key.values())
    return ordered[:cap], dropped + ordered[cap:]


# ---------------------------------------------------------------------------------------------- skills
def load_skill_catalog(skills_dir: Path) -> dict[str, dict]:
    """{skill: {'description', 'covers'}} from each SKILL.md's front matter. `covers` lists the information needs the skill can serve (declared in the skill file, not in Python)."""
    catalog: dict[str, dict] = {}
    for path in sorted(Path(skills_dir).glob("*/SKILL.md")):
        head = path.read_text(encoding="utf-8").split("---")[1] if path.read_text(encoding="utf-8").startswith("---") else ""
        name = re.search(r"(?m)^name:\s*(\S+)", head)
        covers = re.search(r"(?m)^covers:\s*(.+)$", head)
        description = re.search(r"(?ms)^description:\s*(.+?)(?=\n\w+:|\Z)", head)
        if name:
            catalog[name.group(1)] = {"description": " ".join(description.group(1).split()) if description else "",
                                      "covers": [c.strip() for c in covers.group(1).split(",") if c.strip()] if covers else []}
    return catalog


def select_skills(proposed, info_needed, catalog: dict[str, dict], retrieval_skills: tuple[str, ...] = SAFE_FALLBACK_SKILLS) -> tuple[list[str], dict]:
    """The model's proposed skills, checked: only skills in the catalog; and every information need the model listed must be covered by a selected skill that DECLARES it (`covers`). A need nobody selected
    adds the first skill that declares it. If the plan gave neither usable skills nor needs, the safe wide strategy runs all retrieval skills. Skills stay independently selectable."""
    allowed = [s for s in retrieval_skills if s in catalog]
    chosen = [s for s in (proposed if isinstance(proposed, list) else []) if s in allowed]
    needs = [n for n in (info_needed if isinstance(info_needed, list) else []) if n in INFO_NEEDED]
    trace: dict = {"proposed": list(proposed) if isinstance(proposed, list) else [], "added": [], "dropped": [s for s in (proposed if isinstance(proposed, list) else []) if s not in allowed]}
    if not chosen and not needs:
        return list(allowed), {**trace, "fallback": "plan_unusable"}
    for need in needs:
        if not any(need in catalog[s]["covers"] for s in chosen):
            extra = next((s for s in allowed if need in catalog[s]["covers"]), None)
            if extra and extra not in chosen:
                chosen.append(extra)
                trace["added"].append({"skill": extra, "because": need})
    if not chosen:
        chosen = [s for s in allowed if s == "latest-evidence"] or allowed[:1]
        trace["fallback"] = "no skill covers the needs"
    return [s for s in allowed if s in chosen], trace


def baseline_operations(question: str, skills: list[str], *, fallback: bool = False) -> list[Operation]:
    """The original question as a PubMed search (relevance, and newest first): only when literature is relevant to the question or a safe fallback requires it."""
    if "latest-evidence" not in skills and not fallback:
        return []
    clean = re.sub(r"[?\"]", " ", question).strip()
    return [Operation(id="baseline:0", need_id="baseline", need="the question as asked", target="pubmed", query=clean, args=validate_args("pubmed", {"sort": "relevance", "years_back": 5}),
                      priority="essential", origin="baseline", skill="baseline"),
            Operation(id="baseline:1", need_id="baseline", need="the question as asked", target="pubmed", query=clean, args=validate_args("pubmed", {"sort": "date", "years_back": 2}),
                      priority="supporting", origin="baseline", skill="baseline")]


# ---------------------------------------------------------------------------------------------- execution
@dataclass
class Result:
    status: str = "not_run"      # completed_found | completed_empty | failed | timed_out | not_run
    n: int = 0
    error: str = ""
    seconds: float = 0.0
    attempts: int = 0
    raw: str = ""
    late: bool = False      # finished after the budget ended


TRANSIENT = re.compile(r"timed? ?out|timeout|429|rate.?limit|too many requests|5\d\d|temporar|unavailable|connection|reset|overloaded", re.I)


def classify_raw(raw) -> tuple[str, int, str]:
    """(status, number of results, error) of what a search function returned. A tool error is a failure; a normal reply with no results is a valid empty result."""
    text = raw if isinstance(raw, str) else json.dumps(raw)
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        if re.search(r"\bno (?:web )?results?\b", text, re.I) and not re.search(r"time.?out|429|rate|5\d\d|connection|unavailable", text, re.I):
            return "completed_empty", 0, ""      # the search engine answered that it found nothing: a valid empty result, not a failed source
        return "failed", 0, text[:160]
    if isinstance(data, dict) and data.get("error"):
        return "failed", 0, str(data["error"])[:160]
    n = len(data) if isinstance(data, list) else len(data.get("results", [])) if isinstance(data, dict) else 0
    return ("completed_found" if n else "completed_empty"), n, ""


def is_transient(error: str) -> bool:
    return bool(TRANSIENT.search(error or ""))


class Execution:
    """Operations run in parallel under a time budget. Unfinished work is recorded (never dropped silently): searches that never started are `not_run`, searches that were running are `timed_out`,
    and any that finish later are kept in `late` with their results so the caller can say that they arrived too late to be used."""

    def __init__(self, ops: list[Operation], adapters: dict, budget_s: float, workers: int = 6, clock=time.monotonic, retry: bool = True, hold: bool = False):
        """`hold=True`: the searches start as soon as they are submitted (`submit`) and the budget starts when `finish()` is called, so searches written early run while the rest are still being written."""
        self.ops, self.adapters, self.budget_s, self.clock, self.retry = list(ops), adapters, budget_s, clock, retry
        self.results: dict[str, Result] = {op.id: Result() for op in ops}
        self.late: dict[str, Result] = {}
        self._lock = threading.Lock()
        self._deadline = float("inf") if hold else clock() + budget_s
        self._closed = False
        self._pool = ThreadPoolExecutor(max_workers=max(1, workers if hold else min(workers, len(ops) or 1)))
        self._futures = {op.id: self._pool.submit(self._run, op) for op in ops}
        if not hold:
            self.finish()

    def submit(self, ops: list[Operation]) -> None:
        if self._closed:      # the budget has ended: a search written after that is not started
            return
        for op in ops:
            self.ops.append(op)
            self.results[op.id] = Result()
            self._futures[op.id] = self._pool.submit(self._run, op)

    def finish(self) -> None:
        """Wait for the searches up to the budget (counted from now) and record what did not finish."""
        self._closed = True
        self._deadline = self.clock() + self.budget_s
        done, pending = wait(list(self._futures.values()), timeout=self.budget_s)
        by_future = {f: op_id for op_id, f in self._futures.items()}
        for future in done:
            self.results[by_future[future]] = future.result()
        for future in pending:
            op_id = by_future[future]
            if future.cancel():
                self.results[op_id] = Result(status="not_run", error="not started before the budget ended")
            else:
                self.results[op_id] = Result(status="timed_out", error=f"no answer within {self.budget_s:.0f} s")
                future.add_done_callback(lambda f, op_id=op_id: self._arrived_late(op_id, f))
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _arrived_late(self, op_id: str, future) -> None:
        try:
            result = future.result()
        except Exception as error:
            result = Result(status="failed", error=f"{type(error).__name__}: {str(error)[:100]}")
        result.late = True
        with self._lock:
            self.late[op_id] = result

    def _call(self, op: Operation) -> Result:
        began = self.clock()
        try:
            raw = self.adapters[op.target](op)
        except KeyError:
            return Result(status="failed", error=f"no adapter for target {op.target}", seconds=0.0, attempts=1)
        except Exception as error:
            return Result(status="failed", error=f"{type(error).__name__}: {str(error)[:120]}", seconds=self.clock() - began, attempts=1)
        status, n, error = classify_raw(raw)
        return Result(status=status, n=n, error=error, seconds=self.clock() - began, attempts=1, raw=raw if isinstance(raw, str) else json.dumps(raw))

    def _run(self, op: Operation) -> Result:
        result = self._call(op)
        if result.status == "failed" and self.retry and is_transient(result.error) and self.clock() < self._deadline:      # a transient failure is retried ONCE on the same source
            second = self._call(op)
            second.attempts = 2
            return second
        return result

    def late_results(self) -> dict[str, Result]:
        with self._lock:
            return dict(self.late)


def screen_results(query: str, items: list[dict], scorer, threshold: float, margin: float = 1.0) -> dict:
    """First relevance screen for web, guideline and regulator results with a cheap local scorer (the cross-encoder). Scores clearly above the threshold are relevant, clearly below are not, and the band in
    between is `borderline`: those, and everything when the scorer fails, are left to the model-based evidence assessment, never decided by the score alone."""
    texts = [f"{i.get('title') or ''}. {i.get('snippet') or i.get('text') or ''}".strip() for i in items]
    try:
        scores = [float(s) for s in scorer(query, texts)]
        if len(scores) != len(items):
            raise ValueError("score count")
    except Exception:
        return {"relevant": [], "borderline": list(items), "irrelevant": [], "scorer": "failed"}
    out = {"relevant": [], "borderline": [], "irrelevant": [], "scorer": "ok"}
    for item, score in zip(items, scores):
        item = {**item, "screen_score": round(score, 3)}
        out["relevant" if score >= threshold + margin else "borderline" if score >= threshold - margin else "irrelevant"].append(item)
    return out


# ---------------------------------------------------------------------------------------------- needs, escalation, section states
@dataclass
class NeedStatus:
    need_id: str
    need: str
    priority: str
    state: str
    reason: str = ""
    op_ids: list = field(default_factory=list)
    relevant: int = 0


def evaluate_needs(ops: list[Operation], results: dict[str, Result], relevant_count) -> dict[str, NeedStatus]:
    """Per information need: was it covered by relevant evidence, or why not. `relevant_count(op, result)` says how many records of that search were judged relevant (the caller decides how: the
    evidence triage for papers and trials, the label decision for FDA labels, the screen for web pages). A search that failed, timed out or never ran is never counted as 'no evidence'."""
    needs: dict[str, NeedStatus] = {}
    for op in ops:
        for need_id in op.needs():
            st = needs.setdefault(need_id, NeedStatus(need_id, op.need, op.priority, "not_generated"))
            if op.priority == "essential":
                st.priority = "essential"
            st.op_ids.append(op.id)
    for st in needs.values():
        rows = [(op, results.get(op.id, Result())) for op in ops if op.id in st.op_ids]
        relevant = sum(relevant_count(op, r) for op, r in rows if r.status == "completed_found")
        st.relevant = relevant
        statuses = {r.status for _, r in rows}
        if relevant > 0:
            st.state = "covered"
        elif "failed" in statuses:
            st.state, st.reason = "failed", next(r.error for _, r in rows if r.status == "failed")
        elif "timed_out" in statuses:
            st.state, st.reason = "timed_out", next(r.error for _, r in rows if r.status == "timed_out")
        elif "not_run" in statuses:
            st.state, st.reason = "not_retrieved", "the search did not run"
        elif statuses <= {"completed_found", "completed_empty"} and statuses:
            st.state = "no_relevant_evidence"
            st.reason = "empty" if statuses == {"completed_empty"} else "no_relevant"
    return needs


def plan_escalations(needs: dict[str, NeedStatus], ops: list[Operation], executed_keys: set[str], *, max_escalations: int = MAX_ESCALATIONS, already: set[str] | None = None) -> tuple[list[Operation], list[str], list[dict]]:
    """At most ONE alternate-source search per essential need whose searches completed without relevant evidence (or failed for a reason other than the budget). Supporting needs never escalate. Returns
    (escalation operations, need ids that have no alternate yet, records of the needs that were skipped and why)."""
    already = already or set()
    out: list[Operation] = []
    missing: list[str] = []
    skipped: list[dict] = []
    seen = set(executed_keys)
    for need_id, st in needs.items():
        if need_id in already or need_id == "baseline":
            continue
        if st.priority != "essential":
            if st.state not in ("covered",):
                skipped.append({"need_id": need_id, "reason": "supporting need: not escalated"})
            continue
        if st.state not in ("failed", "no_relevant_evidence"):
            if st.state in ("timed_out", "not_retrieved"):
                skipped.append({"need_id": need_id, "reason": f"{st.state}: the budget ended, no time to escalate"})
            continue
        if len(out) >= max_escalations:
            skipped.append({"need_id": need_id, "reason": "escalation cap reached"})
            continue
        source = next((op for op in ops if op.id in st.op_ids and op.alternate), None)
        if source is None:
            missing.append(need_id)
            continue
        alt = source.alternate
        op = Operation(id=f"escalation:{need_id}", need_id=need_id, need=st.need, target=alt["target"], query=alt.get("query", ""), args=validate_args(alt["target"], alt.get("args")), priority="essential",
                       skill=source.skill, origin="escalation",
                       escalation={"from": source.id, "from_target": source.target, "reason": "failed" if st.state == "failed" else st.reason or "no_relevant", "round": 1})
        if op.key in seen:
            skipped.append({"need_id": need_id, "reason": "the alternate search was already run"})
            continue
        seen.add(op.key)
        out.append(op)
    return out, missing, skipped


def need_section_state(need_states: list[str], applicable: bool = True) -> str:
    """One section's state from its needs' states. 'covered' if any is; otherwise the most serious reason it is not, so an absence is never claimed after a failure."""
    if not applicable:
        return "not_applicable"
    if not need_states:
        return "not_generated"
    if "covered" in need_states:
        return "covered"
    for state in ("failed", "timed_out", "not_retrieved", "not_generated", "no_relevant_evidence"):
        if state in need_states:
            return state
    return "not_generated"


def state_text(state: str) -> str:
    return STATE_TEXT.get(state, "")


# ---------------------------------------------------------------------------------------------- adapters: one per target, wrapping the EXISTING search functions
NEGATIVE_FILTER = '("no significant"[tiab] OR failed[tiab] OR negative[tiab] OR hepatotoxicity[tiab] OR "liver injury"[tiab] OR futility[tiab] OR "safety signal"[tiab])'
STUDY_FILTERS = {"rct": "randomized controlled trial[pt]", "meta_analysis": "meta-analysis[pt]", "systematic_review": "systematic review[pt]", "guideline": "guideline[pt]", "review": "review[pt]",
                 "negative_results": NEGATIVE_FILTER}
OPEN_ACCESS_FILTER = "pubmed pmc[sb]"
TRIAL_STATUS_FILTER = {"recruiting": "RECRUITING,NOT_YET_RECRUITING,ACTIVE_NOT_RECRUITING", "completed": "COMPLETED", "any": ""}


def boolean_form(text: str, anchors: set[str], required: int = 3, optional: int = 10) -> str:
    """A long natural-language search as a boolean query that returns papers: PubMed and Europe PMC require EVERY word of a long phrase, which finds nothing. The first few words of the phrase that belong to the
    question (its condition and patient details) are required; the other concept words are alternatives (OR), so the source's own relevance ranking can weigh them. Years and filler words are dropped."""
    words = [w for w in re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", text) if w.lower() not in STOP_WORDS and not re.fullmatch(r"20\d\d|\d+", w)]
    words = list(dict.fromkeys(w.lower() for w in words))
    must = [w for w in words if w in anchors][:required]
    rest = [w for w in words if w not in must][:optional]
    if not must:
        return " ".join(words[:8])
    return "(" + " AND ".join(must) + ")" + (" AND (" + " OR ".join(rest) + ")" if rest else "")


def make_adapters(*, search_pubmed, search_europepmc_quick, search_clinical_trials, search_fda_labels, guideline_search, regulator_search, web_search, anchors: set[str],
                  guideline_filter=None, cache=None) -> dict:
    """target -> function(operation) -> the search tool's own JSON string. The search functions are the existing ones, passed in. `cache(kind, key, ttl_seconds, fn)` is the pipeline's result cache when given.
    `guideline_filter(raw_json) -> raw_json` keeps only pages from the authoritative guideline organisations."""
    def cached(kind, key, ttl, fn):
        return cache(kind, key, ttl, fn) if cache else fn()

    def pubmed(op):
        form = boolean_form(op.query, anchors)
        types = op.args.get("study_types") or []
        if types:
            form = f"({form}) AND (" + " OR ".join(STUDY_FILTERS[t] for t in types) + ")"
        if op.args.get("open_access"):
            form = f"({form}) AND {OPEN_ACCESS_FILTER}"
        sort, years = op.args.get("sort", "relevance"), op.args.get("years_back", 5)
        return cached("pm", f"{form}|{sort}|{years}|routing", 900, lambda: search_pubmed(form, max_results=8 if sort == "date" else 6, years_back=years, sort=sort))

    def europepmc(op):
        form = boolean_form(op.query, anchors)
        sort, years = op.args.get("sort", "relevance"), op.args.get("years_back", 5)
        full = bool(op.args.get("fulltext"))
        return cached("pm", f"{form}|{sort}|{years}|epmc|{full}", 900, lambda: search_europepmc_quick(form, max_results=8 if full else 6, years_back=years, sort=sort, **({"fulltext": True} if full else {})))

    def trials(op):
        condition, intervention = op.args.get("condition", ""), op.args.get("intervention", "")
        status, drug_only = TRIAL_STATUS_FILTER.get(op.args.get("status", "any"), ""), bool(op.args.get("drug_only"))

        def search(with_intervention: str):
            return cached("ct", f"{condition}|{with_intervention}|{status}|{drug_only}|routing", 300, lambda: search_clinical_trials(condition, with_intervention, status, 12 if drug_only else 10, drug_only))      # as many records as the legacy trial searches returned
        raw = search(intervention)
        if intervention and classify_raw(raw)[0] == "completed_empty":      # the same source, a wider question: the condition alone (a narrower registry search often returns nothing)
            wider = search("")
            if classify_raw(wider)[0] == "completed_found":
                return wider
        words = condition.split()
        if classify_raw(raw)[0] == "completed_empty" and len(words) > 3 and " OR " not in condition:      # a compound condition (several findings in one phrase) matches no single record: each concept on its own
            chunks = [" ".join(words[i:i + 3]) for i in range(0, len(words), 3)]
            alternatives = cached("ct", f"{condition}|any-concept|{status}|{drug_only}|routing", 300, lambda: search_clinical_trials(" OR ".join(chunks), "", status, 10, drug_only))
            if classify_raw(alternatives)[0] == "completed_found":
                return alternatives
        return raw

    def fda_label(op):
        terms, drugs = op.args.get("condition_terms") or [], op.args.get("drug_names") or []
        return cached("fda", f"{' '.join(terms)}|{','.join(drugs)}|routing", 3600, lambda: search_fda_labels(terms, drugs or None, 8))

    def guideline_sites(op):
        raw = cached("web", f"{op.query}|guideline", 300, lambda: guideline_search(op.query))
        return guideline_filter(raw) if guideline_filter else raw

    def regulator(op):
        return cached("web", f"{op.query}|regulator", 300, lambda: regulator_search(op.query))

    def web(op):
        return cached("web", f"{op.query}|web", 300, lambda: web_search(op.query))

    return {"pubmed": pubmed, "europepmc": europepmc, "trials": trials, "fda_label": fda_label, "guideline_sites": guideline_sites, "regulator": regulator, "web": web}


# ---------------------------------------------------------------------------------------------- integration helpers (pure)
STATUS_KEY_PREFIX = {"pubmed": "pubmed", "europepmc": "pubmed_epmc", "trials": "trials", "fda_label": "fda", "guideline_sites": "web_guideline", "regulator": "web_regulator", "web": "web"}


def status_key(op: Operation, n: int) -> str:
    """The key under which an operation's result is ingested and its status recorded. Its prefix is what the existing ingest and the source families read (pubmed*, trials*, fda*, web*)."""
    return f"{STATUS_KEY_PREFIX[op.target]}_op_{n}"


def raw_for_ingest(ops: list[Operation], results: dict[str, Result], index_in: list[Operation] | None = None) -> dict[str, str]:
    """{status key: the search tool's own JSON}, so the existing ingest and `record_source_status` handle routed operations exactly like the old jobs. A failed, timed-out or never-run operation
    becomes an error reply (it is never an empty result): the words 'timed out' make it a partial, anything else a failure."""
    out: dict[str, str] = {}
    for position, op in enumerate(ops):
        n = index_in.index(op) if index_in is not None else position      # an escalation search continues the numbering of the whole operation list, so no key is used twice
        r = results.get(op.id, Result())
        if r.status in ("completed_found", "completed_empty"):
            out[status_key(op, n)] = r.raw
        elif r.status == "timed_out":
            out[status_key(op, n)] = json.dumps({"error": f"timed out: {r.error}"})
        elif r.status == "not_run":
            out[status_key(op, n)] = json.dumps({"error": f"not run: {r.error}"})
        else:
            out[status_key(op, n)] = json.dumps({"error": r.error or "failed"})
    return out


def baseline_for(skill: str, question: str, anchors: set[str], condition: str = "", drugs: list | None = None) -> list[Operation]:
    """The minimal search for a skill whose queries could not be generated (the call failed or its reply could not be read, twice): built only from the question's own words and labelled as a baseline.
    It exists for a query-generation failure, never for a skill that declined or whose search came back empty."""
    words = [w for w in re.findall(r"[a-z][a-z0-9\-]{3,}", f"{condition} {question}".lower()) if w in anchors and w not in STOP_WORDS]
    words = list(dict.fromkeys(words))[:6]
    squashed = " ".join(_words(question))
    named = [d for d in (drugs or []) if isinstance(d, str) and " ".join(_words(d)) and " ".join(_words(d)) in squashed][:4]
    if skill == "clinical-trials" and words:
        return [Operation(id=f"baseline:{skill}", need_id=f"baseline-{skill}", need="trials (baseline: the search could not be written)", target="trials", args=validate_args("trials", {"condition": " ".join(words[:5])}),
                          query=" ".join(words[:5]), priority="essential", origin="baseline", skill=skill)]
    if skill == "drug-intelligence" and (words or named):
        return [Operation(id=f"baseline:{skill}", need_id=f"baseline-{skill}", need="drug labels (baseline: the search could not be written)", target="fda_label",
                          args=validate_args("fda_label", {"condition_terms": words[:4], "drug_names": named}), query=" ".join(words[:4] + named), priority="essential", origin="baseline", skill=skill)]
    return []


def routing_record(ops: list[Operation], results: dict[str, Result], *, rejected=None, trace=None, late: dict | None = None, escalations=None) -> dict:
    """What was planned and what happened, for the saved state and the trace: every operation with its target, query, need, priority, status and timing."""
    rows = []
    for n, op in enumerate(ops):
        r = results.get(op.id, Result())
        rows.append({"key": status_key(op, n), "id": op.id, "skill": op.skill, "origin": op.origin, "need": op.need, "need_id": op.need_id, "serves": op.serves, "priority": op.priority, "target": op.target,
                     "query": op.query[:200], "args": op.args, "status": r.status, "n": r.n, "seconds": round(r.seconds, 1), "attempts": r.attempts, "error": r.error[:120], "escalation": op.escalation})
    return {"operations": rows, "rejected": list(rejected or [])[:20], "trace": trace or {}, "late": {k: {"status": v.status, "n": v.n} for k, v in (late or {}).items()}, "escalations": list(escalations or [])}


# ---------------------------------------------------------------------------------------------- coverage check before retrieval
def load_capabilities(path: Path | None = None) -> dict:
    return json.loads((path or Path(__file__).parent / "config" / "source_capabilities.json").read_text(encoding="utf-8"))


def coverage_gaps(ops: list[Operation], info_needed: list, skills: list[str], capabilities: dict) -> list[dict]:
    """Gaps in the written searches, noticed BEFORE anything is retrieved. (1) an information need the model listed that no search serves on a source that can answer it; (2) when the question concerns
    treatment and the drug skill is in play, fewer web-discovery searches than the minimum. Only a gap detector: the model decides what to add and everything added is validated. A search with no `serves`
    tag cannot be matched to a need, so the first check is skipped when no search carries one."""
    gaps: list[dict] = []
    needs = [n for n in (info_needed if isinstance(info_needed, list) else []) if n in INFO_NEEDED]
    sources = capabilities.get("sources", {})
    if needs and any(op.serves for op in ops):
        for need in needs:
            suitable = [t for t, spec in sources.items() if need in spec.get("suits", [])]
            if suitable and not any(op.serves == need and op.target in suitable for op in ops):
                gaps.append({"kind": "need_without_suitable_source", "need": need, "suitable": suitable})
    by_need: dict[str, list[Operation]] = {}
    for op in ops:
        if op.priority == "essential" and op.serves and op.need_id != "baseline" and not op.need_id.startswith("baseline-"):
            by_need.setdefault(op.need_id, []).append(op)
    for need_id, group in by_need.items():      # an essential need answered from ONE source only, while other sources that suit it were not asked: the evidence may sit in one of them
        served = group[0].serves
        used = {op.target for op in group}
        others = [t for t, spec in sources.items() if served in spec.get("suits", []) and t not in used]
        if len(used) == 1 and others:
            gaps.append({"kind": "single_source_need", "need_id": need_id, "need": group[0].need, "serves": served, "used": sorted(used), "other_sources": others,
                         "offers": {t: sources[t].get("offers", "") for t in others}})
    rules = capabilities.get("authority") or []
    rules = [rules] if isinstance(rules, dict) else rules      # one rule or a list of rules
    missing: list[str] = []
    for rule in rules:      # the guideline organisations' and the regulators' own pages give evidence no literature search gives; each rule says when it applies
        if not any(n in needs for n in rule.get("applies_when_info_needed_has", [])):
            continue
        if rule.get("when_skill") and rule["when_skill"] not in skills:
            continue
        for target, least in (rule.get("min_searches") or {}).items():
            if sum(1 for op in ops if op.target == target) < int(least) and target not in missing:
                missing.append(target)
    if missing:
        gaps.append({"kind": "authority_source_missing", "missing_targets": missing})
    discovery = capabilities.get("discovery") or {}
    if discovery.get("skill") in skills and any(n in needs for n in discovery.get("applies_when_info_needed_has", [])):
        have = [op for op in ops if op.skill == discovery["skill"] and op.target == "web"]
        if len(have) < int(discovery.get("min_web_queries", 3)):
            gaps.append({"kind": "discovery_shortfall", "have": [op.query for op in have], "missing": int(discovery["min_web_queries"]) - len(have), "purposes": discovery.get("purposes", [])})
    return gaps
