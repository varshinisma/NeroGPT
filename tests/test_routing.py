"""Offline tests for the deep-search routing (retrieval_routing.py): validated multi-source operations, retries, the time budget, need coverage, conditional escalation, relevance screening and section
states, plus a snapshot that proves the FAST search is unchanged. Fake model replies and fake search tools only: no network, no model.
"""
import hashlib
import inspect
import json
import os
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import qa_stream  # noqa: E402
import retrieval_routing as rr  # noqa: E402

SKILLS = Path(__file__).resolve().parent.parent / "skills"
CATALOG = rr.load_skill_catalog(SKILLS)


def found(n=2):
    return json.dumps({"results": [{"pmid": str(i)} for i in range(n)]})


EMPTY = json.dumps({"results": []})
ERROR = json.dumps({"error": "boom (bad request)"})
TRANSIENT = json.dumps({"error": "429 too many requests"})


class FakeAdapters(dict):
    """target -> function(op) that records every call and answers from a script (a list per target, the last answer repeats)."""

    def __init__(self, script):
        super().__init__()
        self.calls, self.script, self._lock = [], {t: list(v) for t, v in script.items()}, threading.Lock()
        for target in rr.TARGETS:
            self[target] = self._make(target)

    def _make(self, target):
        def run(op):
            with self._lock:
                self.calls.append((target, op.query, op.id))
                answers = self.script.get(target, [EMPTY])
                answer = answers.pop(0) if len(answers) > 1 else answers[0]
            if callable(answer):
                return answer(op)
            return answer
        return run

    def targets_called(self):
        return [c[0] for c in self.calls]


def op(target="pubmed", query="alpha beta gamma delta", need="n1", priority="essential", alternate=None, op_id=None, **args):
    o = rr.Operation(id=op_id or f"t:{need}:{target}", need_id=need, need=need, target=target, query=query, args=rr.validate_args(target, args), priority=priority, alternate=alternate)
    return o


class FastPathIsUnchanged(unittest.TestCase):
    """The fast search, its timeouts, its retry and the preliminary-answer flow must stay exactly as they are."""
    SNAPSHOT = {"_fast_jobs": "07a323aaf2c254ad", "fast_jobs": "012e722131d9edce", "with_retry": "4332567971e488d5", "primary_with_fallback": "d5f081677afb90cc", "run_jobs": "97aa21d88e41abed"}

    def test_the_fast_functions_are_byte_for_byte_the_same(self):
        for name, digest in self.SNAPSHOT.items():
            self.assertEqual(hashlib.sha256(inspect.getsource(getattr(qa_stream, name)).encode()).hexdigest()[:16], digest, f"{name} changed")

    def test_the_six_fast_jobs_and_their_timeouts(self):
        self.assertEqual(sorted(qa_stream._fast_jobs("condition X in children")), sorted(["pubmed_fast", "pubmed_fast_q", "trials_fast", "trials_drugs_fast", "fda_fast", "web_fast"]))
        self.assertEqual(qa_stream.FAST_TIMEOUT_S, 4.5)
        source = inspect.getsource(qa_stream.stream_answer)
        self.assertIn('essential=["pubmed_fast", "trials_fast", "fda_fast"], extra_until=2.5', source)

    def test_the_plan_still_starts_only_after_the_first_token(self):
        source = inspect.getsource(qa_stream.stream_answer)
        self.assertLess(source.index('clock.mark("first_token_s")'), source.index("threading.Thread(target=plan_worker"))
        self.assertLess(source.index("threading.Thread(target=plan_worker"), source.index('yield event("token", stage="INITIAL", text=chunk)'))

    def test_the_first_answer_does_not_wait_for_the_relevance_reading(self):
        source = inspect.getsource(qa_stream.stream_answer)
        self.assertLess(source.index('yield event("token", stage="INITIAL", text=chunk)'), source.index("triage_evidence(state, question, skipped)"))

    def test_routing_does_not_import_or_change_the_fast_path(self):
        source = inspect.getsource(rr)
        self.assertNotIn("qa_stream", source.replace("qa_stream.py", ""))
        self.assertNotIn("_fast_jobs", source)


class OperationParsing(unittest.TestCase):
    QUESTION = "Compare sertraline and fluoxetine for depression in adolescents"
    ANCHORS = rr.anchor_words(QUESTION)

    def parse(self, items, skill="latest-evidence"):
        return rr.parse_operations(skill, {"queries": items}, self.QUESTION, self.ANCHORS)

    def test_one_skill_can_target_several_sources(self):
        ops, rejected = self.parse([
            {"need_id": "n1", "need": "head to head efficacy", "target": "pubmed", "query": "sertraline versus fluoxetine efficacy depression adolescents randomized", "priority": "essential"},
            {"need_id": "n2", "need": "label safety warnings", "target": "fda_label", "args": {"condition_terms": ["depression"], "drug_names": ["sertraline", "fluoxetine"]}, "priority": "essential"},
            {"need_id": "n3", "need": "recent approvals", "target": "regulator", "query": "fluoxetine sertraline adolescent depression approval safety communication"}])
        self.assertEqual([o.target for o in ops], ["pubmed", "fda_label", "regulator"])
        self.assertEqual(rejected, [])
        self.assertEqual(ops[1].args["drug_names"], ["sertraline", "fluoxetine"])

    def test_two_queries_may_target_one_source_for_different_needs(self):
        ops, _ = self.parse([{"need_id": "a", "need": "efficacy", "target": "pubmed", "query": "sertraline fluoxetine adolescents depression efficacy trial"},
                             {"need_id": "b", "need": "safety", "target": "pubmed", "query": "sertraline fluoxetine adolescents depression suicidality safety"}])
        self.assertEqual(len(ops), 2)

    def test_invalid_or_missing_targets_are_rejected_and_never_executed(self):
        ops, rejected = self.parse([{"need": "x", "target": "run_shell", "query": "sertraline fluoxetine adolescents depression"}, {"need": "y", "query": "sertraline fluoxetine adolescents depression efficacy"},
                                    {"need": "z", "target": "web", "query": "sertraline fluoxetine adolescents depression guideline"}])
        self.assertEqual([o.target for o in ops], ["web"])
        self.assertEqual(len(rejected), 2)
        self.assertIn("invalid target", rejected[0]["reason"])

    def test_a_query_must_keep_the_question(self):
        ops, rejected = self.parse([{"need": "x", "target": "pubmed", "query": "football league results this season for champions"}])
        self.assertEqual(ops, [])
        self.assertIn("question's terms", rejected[0]["reason"])

    def test_a_drug_may_only_be_named_if_the_question_names_it(self):
        ops, _ = self.parse([{"need": "label", "target": "fda_label", "args": {"condition_terms": ["depression"], "drug_names": ["sertraline", "madeupdrug"]}}])
        self.assertEqual(ops[0].args["drug_names"], ["sertraline"])

    def test_arguments_are_clamped_and_unknown_keys_dropped(self):
        ops, _ = self.parse([{"need": "x", "target": "pubmed", "query": "sertraline fluoxetine adolescents depression efficacy", "args": {"years_back": 99, "sort": "chaos", "shell": "rm -rf", "study_types": ["rct", "magic"], "open_access": True}}])
        self.assertEqual(ops[0].args, {"sort": "relevance", "years_back": 15, "study_types": ["rct"], "open_access": True})

    def test_malformed_replies_never_raise(self):
        for reply in (None, "text", {}, {"queries": "no"}, {"queries": [None, 3, "x"]}):
            ops, rejected = rr.parse_operations("latest-evidence", reply, self.QUESTION, self.ANCHORS)
            self.assertEqual(ops, [])
            self.assertTrue(rejected)

    def test_an_alternate_is_validated_like_a_query(self):
        ops, _ = self.parse([{"need": "x", "target": "pubmed", "query": "sertraline fluoxetine adolescents depression efficacy", "priority": "essential",
                              "alternate": {"target": "europepmc", "query": "sertraline fluoxetine adolescents depression efficacy comparison"}},
                             {"need": "y", "target": "pubmed", "query": "sertraline fluoxetine adolescents depression safety", "alternate": {"target": "nonsense", "query": "sertraline fluoxetine"}}])
        self.assertEqual(ops[0].alternate["target"], "europepmc")
        self.assertIsNone(ops[1].alternate)

    def test_duplicates_inside_a_reply_and_across_skills_are_one_search(self):
        a, _ = self.parse([{"need_id": "n1", "need": "efficacy", "target": "pubmed", "query": "sertraline fluoxetine adolescents depression efficacy trial"},
                           {"need_id": "n1", "need": "efficacy", "target": "pubmed", "query": "Sertraline, fluoxetine: adolescents depression efficacy trial"}])
        self.assertEqual(len(a), 1)
        b, _ = self.parse([{"need_id": "n9", "need": "same thing", "target": "pubmed", "query": "sertraline fluoxetine adolescents depression efficacy trial"}], skill="drug-intelligence")
        merged, dropped = rr.merge_operations(a, b)
        self.assertEqual(len(merged), 1)
        self.assertEqual(len(dropped), 1)
        self.assertEqual(merged[0].needs(), ["n1", "n9"])      # the one search still counts for both needs


class SkillSelection(unittest.TestCase):
    def test_skill_files_declare_what_they_cover(self):
        for name in ("latest-evidence", "clinical-trials", "drug-intelligence"):
            self.assertTrue(CATALOG[name]["covers"], name)
        self.assertIn("regulatory_status", CATALOG["drug-intelligence"]["covers"])
        self.assertIn("trial_status", CATALOG["clinical-trials"]["covers"])

    def test_skills_are_independently_selectable(self):
        skills, trace = rr.select_skills(["latest-evidence"], ["mechanism"], CATALOG)
        self.assertEqual(skills, ["latest-evidence"])      # a mechanism question does not need trials or drugs
        skills, _ = rr.select_skills(["drug-intelligence"], ["dosing", "safety"], CATALOG)
        self.assertEqual(skills, ["drug-intelligence"])

    def test_a_need_nobody_covers_adds_the_skill_that_declares_it(self):
        skills, trace = rr.select_skills(["latest-evidence"], ["efficacy", "regulatory_status", "trial_status"], CATALOG)
        self.assertEqual(skills, ["latest-evidence", "clinical-trials", "drug-intelligence"])
        self.assertEqual({a["because"] for a in trace["added"]}, {"regulatory_status", "trial_status"})

    def test_unknown_skills_are_dropped_not_run(self):
        skills, trace = rr.select_skills(["latest-evidence", "delete-everything"], ["efficacy"], CATALOG)
        self.assertEqual(skills, ["latest-evidence"])
        self.assertEqual(trace["dropped"], ["delete-everything"])

    def test_an_unusable_plan_runs_the_safe_wide_strategy(self):
        skills, trace = rr.select_skills(None, None, CATALOG)
        self.assertEqual(skills, ["latest-evidence", "clinical-trials", "drug-intelligence"])
        self.assertEqual(trace["fallback"], "plan_unusable")

    def test_the_baseline_search_only_when_literature_is_relevant_or_a_fallback_needs_it(self):
        self.assertEqual(rr.baseline_operations("How is condition X treated?", ["drug-intelligence"]), [])
        self.assertEqual([o.args["sort"] for o in rr.baseline_operations("How is condition X treated?", ["latest-evidence"])], ["relevance", "date"])
        self.assertTrue(rr.baseline_operations("How is condition X treated?", [], fallback=True))


class HeldExecution(unittest.TestCase):
    def test_searches_start_when_submitted_and_a_search_submitted_after_the_budget_is_ignored(self):
        adapters = FakeAdapters({"pubmed": [found(1)], "trials": [found(1)]})
        ex = rr.Execution([], adapters, 5, hold=True)
        ex.submit([op("pubmed", op_id="a")])
        deadline = time.time() + 2
        while not adapters.calls and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(len(adapters.calls), 1)      # it ran before finish() was called
        ex.budget_s = 0.0
        ex.finish()
        ex.submit([op("trials", "x y z w", op_id="b", condition="alpha")])
        self.assertEqual([c[2] for c in adapters.calls], ["a"])
        self.assertIn("a", ex.results)
        self.assertNotIn("b", ex.results)


class Execution(unittest.TestCase):
    def test_found_empty_and_failed_are_different_statuses(self):
        adapters = FakeAdapters({"pubmed": [found(3)], "trials": [EMPTY], "web": [ERROR]})
        ex = rr.Execution([op("pubmed", op_id="a"), op("trials", "x y z w", op_id="b", condition="alpha"), op("web", "alpha beta gamma delta", op_id="c")], adapters, 5)
        self.assertEqual([ex.results[i].status for i in "abc"], ["completed_found", "completed_empty", "failed"])
        self.assertEqual(ex.results["a"].n, 3)

    def test_a_transient_failure_is_retried_once_on_the_same_source(self):
        adapters = FakeAdapters({"pubmed": [TRANSIENT, found(2)]})
        ex = rr.Execution([op("pubmed", op_id="a")], adapters, 5)
        self.assertEqual(ex.results["a"].status, "completed_found")
        self.assertEqual(ex.results["a"].attempts, 2)
        self.assertEqual(adapters.targets_called(), ["pubmed", "pubmed"])

    def test_a_transient_failure_that_keeps_failing_is_retried_only_once(self):
        adapters = FakeAdapters({"pubmed": [TRANSIENT]})
        ex = rr.Execution([op("pubmed", op_id="a")], adapters, 5)
        self.assertEqual(ex.results["a"].status, "failed")
        self.assertEqual(len(adapters.calls), 2)

    def test_a_permanent_failure_is_not_retried(self):
        adapters = FakeAdapters({"pubmed": [ERROR]})
        ex = rr.Execution([op("pubmed", op_id="a")], adapters, 5)
        self.assertEqual(ex.results["a"].status, "failed")
        self.assertEqual(len(adapters.calls), 1)

    def test_an_adapter_that_raises_is_a_failure_not_a_crash(self):
        adapters = FakeAdapters({"pubmed": [lambda o: 1 / 0]})
        ex = rr.Execution([op("pubmed", op_id="a")], adapters, 5)
        self.assertEqual(ex.results["a"].status, "failed")
        self.assertIn("ZeroDivisionError", ex.results["a"].error)

    def test_budget_expiry_records_timed_out_and_not_run_and_keeps_late_results(self):
        def slow(o):
            time.sleep(0.5)
            return found(4)
        adapters = FakeAdapters({"pubmed": [found(1)], "web": [slow], "trials": [slow], "regulator": [slow]})
        ops = [op("pubmed", op_id="fast"), op("web", op_id="slow1"), op("trials", op_id="slow2", condition="alpha"), op("regulator", op_id="queued")]
        began = time.monotonic()
        ex = rr.Execution(ops, adapters, 0.15, workers=2)
        self.assertLess(time.monotonic() - began, 0.45)      # the caller is released at the budget, not when the slow work ends
        self.assertEqual(ex.results["fast"].status, "completed_found")
        self.assertEqual(ex.results["slow1"].status, "timed_out")
        self.assertEqual(ex.results["slow2"].status, "timed_out")
        self.assertEqual(ex.results["queued"].status, "not_run")      # it never started: unfinished work is cancelled and recorded
        time.sleep(0.7)
        late = ex.late_results()      # late results are recorded, not silently dropped (the caller says they arrived too late to be used)
        self.assertEqual(sorted(late), ["slow1", "slow2"])
        self.assertTrue(all(r.late and r.status == "completed_found" for r in late.values()))

    def test_classify_raw(self):
        self.assertEqual(rr.classify_raw(found(2))[:2], ("completed_found", 2))
        self.assertEqual(rr.classify_raw(json.dumps([{"url": "u"}]))[:2], ("completed_found", 1))
        self.assertEqual(rr.classify_raw(EMPTY)[0], "completed_empty")
        self.assertEqual(rr.classify_raw("Web search failed: x")[0], "failed")
        self.assertEqual(rr.classify_raw(ERROR)[0], "failed")


class NeedsAndEscalation(unittest.TestCase):
    ALT = {"target": "europepmc", "query": "alpha beta gamma delta other index", "args": {}}

    def evaluate(self, ops, statuses, relevant=None):
        results = {o.id: rr.Result(status=s, n=2 if s == "completed_found" else 0, error="429" if s == "failed" else "") for o, s in zip(ops, statuses)}
        relevant = relevant or {}
        return rr.evaluate_needs(ops, results, lambda o, r: relevant.get(o.id, 0)), results

    def test_need_states_never_call_a_failure_an_absence(self):
        ops = [op(op_id="a", need="n1"), op("trials", op_id="b", need="n2", condition="alpha"), op("web", op_id="c", need="n3"), op("regulator", op_id="d", need="n4")]
        needs, _ = self.evaluate(ops, ["completed_found", "completed_empty", "failed", "completed_found"], relevant={"a": 2})
        self.assertEqual({k: v.state for k, v in needs.items()}, {"n1": "covered", "n2": "no_relevant_evidence", "n3": "failed", "n4": "no_relevant_evidence"})
        self.assertEqual(needs["n2"].reason, "empty")
        self.assertEqual(needs["n4"].reason, "no_relevant")

    def test_a_need_with_one_failed_and_one_empty_search_is_failed_not_absent(self):
        ops = [op(op_id="a", need="n1"), op("web", op_id="b", need="n1")]
        needs, _ = self.evaluate(ops, ["completed_empty", "failed"])
        self.assertEqual(needs["n1"].state, "failed")

    def test_escalation_goes_to_one_alternate_for_an_essential_need_with_nothing_relevant(self):
        ops = [op(op_id="a", need="n1", alternate=self.ALT)]
        needs, _ = self.evaluate(ops, ["completed_found"], relevant={"a": 0})
        esc, missing, skipped = rr.plan_escalations(needs, ops, {o.key for o in ops})
        self.assertEqual([(e.target, e.escalation["reason"], e.escalation["from_target"]) for e in esc], [("europepmc", "no_relevant", "pubmed")])
        self.assertEqual(esc[0].escalation["round"], 1)
        self.assertEqual(missing, [])

    def test_empty_failed_and_irrelevant_reasons_are_recorded(self):
        for status, relevant, reason in (("completed_empty", 0, "empty"), ("failed", 0, "failed"), ("completed_found", 0, "no_relevant")):
            ops = [op(op_id="a", need="n1", alternate=self.ALT)]
            needs, _ = self.evaluate(ops, [status], relevant={"a": relevant})
            esc, _, _ = rr.plan_escalations(needs, ops, {o.key for o in ops})
            self.assertEqual(esc[0].escalation["reason"], reason)

    def test_covered_needs_add_no_search(self):
        ops = [op(op_id="a", need="n1", alternate=self.ALT)]
        needs, _ = self.evaluate(ops, ["completed_found"], relevant={"a": 3})
        self.assertEqual(rr.plan_escalations(needs, ops, set())[0], [])

    def test_supporting_needs_never_escalate(self):
        ops = [op(op_id="a", need="n1", priority="supporting", alternate=self.ALT)]
        needs, _ = self.evaluate(ops, ["completed_empty"])
        esc, _, skipped = rr.plan_escalations(needs, ops, set())
        self.assertEqual(esc, [])
        self.assertIn("supporting", skipped[0]["reason"])

    def test_a_need_without_an_alternate_is_reported_for_one_batched_request(self):
        ops = [op(op_id="a", need="n1")]
        needs, _ = self.evaluate(ops, ["completed_empty"])
        esc, missing, _ = rr.plan_escalations(needs, ops, set())
        self.assertEqual((esc, missing), ([], ["n1"]))

    def test_budget_ended_needs_are_not_escalated_and_say_so(self):
        ops = [op(op_id="a", need="n1", alternate=self.ALT), op("web", op_id="b", need="n2", alternate=self.ALT)]
        needs, _ = self.evaluate(ops, ["timed_out", "not_run"])
        esc, _, skipped = rr.plan_escalations(needs, ops, set())
        self.assertEqual(esc, [])
        self.assertEqual({s["need_id"] for s in skipped}, {"n1", "n2"})

    def test_the_cap_and_one_round_only(self):
        ops = [op(op_id=f"o{i}", need=f"n{i}", query=f"alpha beta gamma delta {i}", alternate={"target": "web", "query": f"alpha beta gamma delta alt {i}"}) for i in range(7)]
        needs, _ = self.evaluate(ops, ["completed_empty"] * 7)
        esc, _, skipped = rr.plan_escalations(needs, ops, {o.key for o in ops}, max_escalations=4)
        self.assertEqual(len(esc), 4)
        self.assertEqual(sum("cap" in s["reason"] for s in skipped), 3)
        again, _, _ = rr.plan_escalations(needs, ops, {o.key for o in ops} | {e.key for e in esc}, already={e.need_id for e in esc})
        # a need that already escalated is never escalated again
        self.assertFalse({e.need_id for e in again} & {e.need_id for e in esc})

    def test_a_duplicate_alternate_is_skipped(self):
        existing = op("europepmc", "alpha beta gamma delta other index", need="n9", op_id="e")
        ops = [op(op_id="a", need="n1", alternate=self.ALT), existing]
        needs, _ = self.evaluate(ops, ["completed_empty", "completed_found"], relevant={"e": 1})
        esc, _, skipped = rr.plan_escalations(needs, ops, {o.key for o in ops})
        self.assertEqual(esc, [])
        self.assertIn("already", skipped[0]["reason"])

    def test_escalation_improves_coverage_without_searching_every_source(self):
        ops = [op(op_id="a", need="n1", alternate={"target": "regulator", "query": "alpha beta gamma delta approval notice"})]
        adapters = FakeAdapters({"pubmed": [found(2)], "regulator": [found(1)]})
        first = rr.Execution(ops, adapters, 5)
        needs = rr.evaluate_needs(ops, first.results, lambda o, r: 0)      # PubMed answered but nothing was judged relevant
        self.assertEqual(needs["n1"].state, "no_relevant_evidence")
        esc, _, _ = rr.plan_escalations(needs, ops, {o.key for o in ops})
        second = rr.Execution(esc, adapters, 5)
        results = {**first.results, **second.results}
        needs = rr.evaluate_needs(ops + esc, results, lambda o, r: 1 if o.origin == "escalation" else 0)
        self.assertEqual(needs["n1"].state, "covered")
        self.assertEqual(adapters.targets_called(), ["pubmed", "regulator"])      # one extra search, to one suitable source only


class RelevanceScreening(unittest.TestCase):
    ITEMS = [{"title": "a", "snippet": "good"}, {"title": "b", "snippet": "middle"}, {"title": "c", "snippet": "bad"}]

    def test_clear_scores_decide_and_the_middle_band_goes_to_the_model(self):
        out = rr.screen_results("q", self.ITEMS, lambda q, texts: [5.0, 0.2, -6.0], threshold=0.0, margin=1.0)
        self.assertEqual([i["title"] for i in out["relevant"]], ["a"])
        self.assertEqual([i["title"] for i in out["borderline"]], ["b"])
        self.assertEqual([i["title"] for i in out["irrelevant"]], ["c"])

    def test_a_failed_scorer_decides_nothing(self):
        out = rr.screen_results("q", self.ITEMS, lambda q, texts: 1 / 0, threshold=0.0)
        self.assertEqual((out["relevant"], out["irrelevant"], len(out["borderline"]), out["scorer"]), ([], [], 3, "failed"))
        out = rr.screen_results("q", self.ITEMS, lambda q, texts: [1.0], threshold=0.0)      # wrong number of scores
        self.assertEqual(out["scorer"], "failed")


class SectionStates(unittest.TestCase):
    def test_the_four_states_are_distinct_and_failures_win_over_absence(self):
        self.assertEqual(rr.need_section_state(["covered", "failed"]), "covered")
        self.assertEqual(rr.need_section_state(["no_relevant_evidence", "failed"]), "failed")
        self.assertEqual(rr.need_section_state(["no_relevant_evidence", "timed_out"]), "timed_out")
        self.assertEqual(rr.need_section_state(["no_relevant_evidence"]), "no_relevant_evidence")
        self.assertEqual(rr.need_section_state(["not_retrieved"]), "not_retrieved")
        self.assertEqual(rr.need_section_state([], applicable=False), "not_applicable")
        self.assertEqual(rr.need_section_state([]), "not_generated")
        texts = {rr.state_text(s) for s in ("not_applicable", "not_retrieved", "no_relevant_evidence", "failed", "timed_out", "not_generated")}
        self.assertEqual(len(texts), 6)      # every state reads differently

    def test_a_failed_search_is_never_worded_as_no_evidence(self):
        self.assertNotIn("no relevant evidence", rr.state_text("failed").lower())
        self.assertIn("does not show that no evidence exists", rr.state_text("failed"))


class FourQuestions(unittest.TestCase):
    """Canned model replies (what a skill might write) run through the real validators, dispatcher and fake sources."""

    def run_question(self, question, understanding, replies, script):
        anchors = rr.anchor_words(question, understanding.get("condition", ""))
        skills, trace = rr.select_skills(understanding["skills"], understanding["info_needed"], CATALOG)
        lists = [rr.parse_operations(skill, replies.get(skill, {}), question, anchors)[0] for skill in skills]
        ops, _ = rr.merge_operations(*lists)
        ops = rr.baseline_operations(question, skills) + ops
        adapters = FakeAdapters(script)
        execution = rr.Execution(ops, adapters, 5)
        return skills, ops, adapters, execution

    def test_sertraline_vs_fluoxetine_in_adolescents(self):
        q = "Compare sertraline and fluoxetine for depression in adolescents"
        skills, ops, adapters, ex = self.run_question(q, {"skills": ["latest-evidence"], "info_needed": ["efficacy", "safety", "dosing"], "condition": "depression"}, {
            "latest-evidence": {"queries": [{"need_id": "n1", "need": "head to head efficacy", "target": "pubmed", "priority": "essential", "query": "sertraline versus fluoxetine efficacy depression adolescents randomized"},
                                            {"need_id": "n2", "need": "safety in adolescents", "target": "pubmed", "query": "sertraline fluoxetine adolescent depression suicidality adverse effects"}]},
            "drug-intelligence": {"queries": [{"need_id": "n3", "need": "label dosing and warnings", "target": "fda_label", "priority": "essential", "args": {"condition_terms": ["depression"], "drug_names": ["sertraline", "fluoxetine"]}}]}},
            {"pubmed": [found(3)], "fda_label": [found(2)]})
        self.assertEqual(skills, ["latest-evidence", "drug-intelligence"])      # dosing needs the drug skill, which the model had not chosen
        self.assertNotIn("trials", adapters.targets_called())
        self.assertIn("fda_label", adapters.targets_called())
        self.assertTrue(all("adolescent" in o.query or o.target == "fda_label" for o in ops if o.origin == "generated"))      # the patient group stays in every written search

    def test_metformin_mechanism_launches_no_unrelated_searches(self):
        q = "How does metformin lower blood glucose?"
        skills, ops, adapters, ex = self.run_question(q, {"skills": ["latest-evidence", "clinical-trials", "drug-intelligence"], "info_needed": ["mechanism"], "condition": "blood glucose"}, {
            "latest-evidence": {"queries": [{"need_id": "n1", "need": "mechanism of action", "target": "pubmed", "priority": "essential", "query": "metformin mechanism lowering blood glucose hepatic gluconeogenesis AMPK"}]}},
            {"pubmed": [found(3)]})
        self.assertEqual(skills, ["latest-evidence", "clinical-trials", "drug-intelligence"])      # the model's own choice stands: Python only ADDS skills for uncovered needs, it does not remove
        self.assertEqual(set(adapters.targets_called()), {"pubmed"})      # but only the searches that were written ran: no trial, label, regulator or web search

    def test_alzheimers_what_is_new_routes_each_need_to_its_source(self):
        q = "What is new in Alzheimer's treatment this year?"
        skills, ops, adapters, ex = self.run_question(q, {"skills": ["latest-evidence", "clinical-trials", "drug-intelligence"], "info_needed": ["treatment_options", "regulatory_status", "trial_status"], "condition": "Alzheimer's disease"}, {
            "latest-evidence": {"queries": [{"need_id": "n1", "need": "recent treatment evidence", "target": "pubmed", "priority": "essential", "query": "Alzheimer's disease new treatments clinical studies systematic reviews this year"}]},
            "drug-intelligence": {"queries": [{"need_id": "n2", "need": "new approvals", "target": "regulator", "priority": "essential", "query": "Alzheimer's disease treatment new approvals indications this year"},
                                              {"need_id": "n4", "need": "label changes", "target": "fda_label", "args": {"condition_terms": ["Alzheimer's"], "drug_names": []}}]},
            "clinical-trials": {"queries": [{"need_id": "n3", "need": "ongoing research", "target": "trials", "priority": "essential", "args": {"condition": "Alzheimer's disease", "intervention": "", "status": "recruiting"}}]}},
            {"pubmed": [found(2)], "regulator": [found(1)], "trials": [found(2)], "fda_label": [found(1)]})
        self.assertEqual(set(adapters.targets_called()), {"pubmed", "regulator", "trials", "fda_label"})
        self.assertEqual(len([c for c in adapters.calls if c[0] == "pubmed"]), 3)      # the model's search plus the two baseline searches

    def test_three_year_old_with_developmental_delay_keeps_the_patient(self):
        q = "Three year old child has global developmental delay, regression of milestones and autistic tendencies. What could be the work up and management?"
        replies = {"latest-evidence": {"queries": [
            {"need_id": "n1", "need": "diagnostic work up", "target": "pubmed", "priority": "essential", "query": "three year old child global developmental delay regression milestones autism diagnostic work up"},
            {"need_id": "n2", "need": "early intervention", "target": "pubmed", "priority": "essential", "query": "early intervention global developmental delay autistic features young children therapies"},
            {"need_id": "n3", "need": "guideline recommendations", "target": "guideline_sites", "query": "developmental regression global developmental delay autism evaluation guideline recommendations"}]},
            "clinical-trials": {"queries": [{"need_id": "n4", "need": "intervention trials", "target": "trials", "args": {"condition": "global developmental delay autism", "intervention": "early intervention"}}]}}
        skills, ops, adapters, ex = self.run_question(q, {"skills": ["latest-evidence", "clinical-trials"], "info_needed": ["diagnosis", "treatment_options"], "condition": "global developmental delay"}, replies,
                                                      {"pubmed": [found(2)], "guideline_sites": [found(1)], "trials": [found(1)]})
        generated = [o for o in ops if o.origin == "generated"]
        anchors = rr.anchor_words(q, "global developmental delay")
        self.assertTrue(all(len(set(rr.normalize_query(o.query).split()) & anchors) >= 2 for o in generated if o.target in rr.TEXT_TARGETS))      # the child and the clinical picture stay in every query
        self.assertIn("trials", adapters.targets_called())
        self.assertIn("guideline_sites", adapters.targets_called())
        self.assertNotIn("fda_label", adapters.targets_called())      # no drug search without a drug need or a drug skill


if __name__ == "__main__":
    unittest.main()


class SkillFilesAllowNoSearches(unittest.TestCase):
    """The retrieval skills must not force a search for every question, and must describe the schema the validators accept."""
    FILES = {name: (SKILLS / name / "SKILL.md").read_text(encoding="utf-8") for name in ("clinical-research-orchestrator", "latest-evidence", "clinical-trials", "drug-intelligence")}

    def test_no_instruction_forces_every_skill_for_every_question(self):
        for name, text in self.FILES.items():
            for phrase in ("Always run `drug-intelligence`", "Always include: `latest-evidence`", "for **every** query", "for every query", "any clinical query at all", "or for any clinical query", "The intent changes emphasis only, never whether drugs and trials run"):
                self.assertNotIn(phrase, text, f"{name}: {phrase}")
        self.assertNotIn("Write 1-2 queries in FAST mode, 4-6 in DEEP mode", self.FILES["latest-evidence"])

    def test_each_retrieval_skill_may_return_an_empty_list(self):
        for name in ("latest-evidence", "clinical-trials", "drug-intelligence"):
            self.assertIn('reply `{"queries": []}`', self.FILES[name], name)

    def test_the_schema_in_the_skill_files_matches_the_validators(self):
        for name in ("latest-evidence", "clinical-trials", "drug-intelligence"):
            text = self.FILES[name]
            for target in rr.TARGETS:
                self.assertIn(f"`{target}`", text, f"{name} does not describe the target {target}")
            for field in ("need_id", "priority", "alternate", "target"):
                self.assertIn(field, text, f"{name}: {field}")
            for study_type in rr.STUDY_TYPES:
                if name == "latest-evidence":
                    self.assertIn(study_type, text)

    def test_the_orchestrator_lets_the_model_choose_skills_by_need(self):
        text = self.FILES["clinical-research-orchestrator"]
        self.assertIn("Choose a skill only for an information need the question really has", text)
        self.assertIn("none of them is required for every question", text)

    def test_the_catalog_the_plan_call_reads_no_longer_says_every_query(self):
        for name in ("clinical-trials", "drug-intelligence", "latest-evidence"):
            self.assertNotIn("every query", CATALOG[name]["description"].lower())
            self.assertNotIn("always collected", CATALOG[name]["description"].lower())


class NoResultsIsNotAFailure(unittest.TestCase):
    def test_a_search_engine_that_found_nothing_is_an_empty_result_not_a_failed_source(self):
        for text in ("Web search failed: No results found.", "No web results found for this query"):
            self.assertEqual(rr.classify_raw(text)[0], "completed_empty", text)

    def test_real_failures_are_still_failures(self):
        for text in ("Web search failed: connection reset", "Web search failed: timed out", "Web search failed: 429 too many requests", "Web search failed: boom"):
            self.assertEqual(rr.classify_raw(text)[0], "failed", text)
