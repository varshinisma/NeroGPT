"""Offline tests of the deep routing stage wired into qa_stream: fake model replies (what each skill writes) and fake search tools. Covers skill selection by need, empty lists, query-generation
failures, duplicates across skills, provenance, ingestion of routed results, and that the fast search is untouched."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import landscape  # noqa: E402
import qa_stream  # noqa: E402
import retrieval_routing as rr  # noqa: E402
from tests.test_routing import FakeAdapters, found, EMPTY, ERROR  # noqa: E402

QUESTION = "Compare sertraline and fluoxetine for depression in adolescents"


def plan(skills=("latest-evidence", "drug-intelligence"), info=("efficacy", "safety"), drugs=("sertraline", "fluoxetine")):
    return {"intent": "comparison", "condition": "depression", "drugs": list(drugs), "skills": list(skills),
            "understanding": {"info_needed": list(info), "case_facts": [], "asked": "compare", "answer_format": "options_comparison"}}


def pubmed_query(need, query, priority="essential", alternate=None, need_id=None):
    item = {"need_id": need_id or need[:4], "need": need, "target": "pubmed", "query": query, "priority": priority}
    if alternate:
        item["alternate"] = alternate
    return item


class FakeModel:
    """Stands in for ask_json: answers per skill (a list per skill is consumed one reply per call)."""

    def __init__(self, replies):
        self.replies, self.calls = {k: list(v) for k, v in replies.items()}, []

    def __call__(self, question, task, payload, max_tokens, skill="x"):
        self.calls.append(skill)
        answers = self.replies.get(skill, [{"queries": []}])
        answer = answers.pop(0) if len(answers) > 1 else answers[0]
        if isinstance(answer, Exception):
            raise answer
        return answer


LATEST = {"queries": [pubmed_query("head to head efficacy", "sertraline versus fluoxetine efficacy depression adolescents randomized"),
                      {"need_id": "g1", "need": "guideline recommendations", "target": "guideline_sites", "query": "adolescent depression sertraline fluoxetine guideline recommendations", "priority": "supporting"}]}
DRUGS = {"queries": [{"need_id": "l1", "need": "label dosing and warnings", "target": "fda_label", "priority": "essential", "args": {"condition_terms": ["depression"], "drug_names": ["sertraline", "fluoxetine"]}},
                     {"need_id": "r1", "need": "safety communications", "target": "regulator", "query": "sertraline fluoxetine adolescent depression safety communication", "priority": "supporting"}]}


class RouteDeep(unittest.TestCase):
    def test_skills_write_queries_for_several_targets_in_parallel(self):
        ask = FakeModel({"latest-evidence": [LATEST], "drug-intelligence": [DRUGS]})
        routed = qa_stream.route_deep(QUESTION, plan(), ask=ask)
        self.assertEqual(sorted(ask.calls), ["drug-intelligence", "latest-evidence"])
        self.assertEqual({o.target for o in routed["ops"] if o.origin == "generated"}, {"pubmed", "guideline_sites", "fda_label", "regulator"})
        self.assertEqual({o.id for o in routed["ops"] if o.origin == "baseline"}, {"baseline:0", "baseline:1"})      # literature is relevant: the original question is searched too

    def test_a_skill_that_is_not_chosen_is_never_asked(self):
        ask = FakeModel({"latest-evidence": [{"queries": [pubmed_query("mechanism of action", "metformin mechanism lowering blood glucose hepatic gluconeogenesis")]}]})
        routed = qa_stream.route_deep("How does metformin lower blood glucose?", plan(skills=["latest-evidence"], info=["mechanism"], drugs=["metformin"]), ask=ask)
        self.assertEqual(ask.calls, ["latest-evidence"])
        self.assertEqual({o.target for o in routed["ops"]}, {"pubmed"})

    def test_a_skill_that_declines_adds_nothing_and_is_not_retried_or_replaced_by_a_baseline(self):
        ask = FakeModel({"latest-evidence": [LATEST], "clinical-trials": [{"queries": []}], "drug-intelligence": [{"_raw": "```json\n{}\n```"}]})
        routed = qa_stream.route_deep(QUESTION, plan(skills=["latest-evidence", "clinical-trials", "drug-intelligence"]), ask=ask)
        self.assertEqual(ask.calls.count("clinical-trials"), 1)
        self.assertEqual(ask.calls.count("drug-intelligence"), 1)
        self.assertEqual(routed["trace"]["skills"]["clinical-trials"]["outcome"], "declined")
        self.assertEqual(routed["trace"]["skills"]["drug-intelligence"]["outcome"], "declined")
        self.assertFalse([o for o in routed["ops"] if o.target in ("trials", "fda_label")])

    def test_a_query_generation_failure_is_retried_once_then_gets_a_labelled_minimal_baseline(self):
        ask = FakeModel({"latest-evidence": [LATEST], "drug-intelligence": [{"queries": "garbage"}, {"_raw": "{\"queries\": [{\"need\": \"x\", \"tar"}]})
        routed = qa_stream.route_deep(QUESTION, plan(), ask=ask)
        self.assertEqual(ask.calls.count("drug-intelligence"), 2)
        self.assertEqual(routed["trace"]["skills"]["drug-intelligence"]["outcome"], "malformed")
        base = [o for o in routed["ops"] if o.skill == "drug-intelligence"]
        self.assertEqual([(o.target, o.origin) for o in base], [("fda_label", "baseline")])
        self.assertEqual(base[0].args["drug_names"], ["sertraline", "fluoxetine"])      # only drugs the question names

    def test_a_model_call_that_raises_is_a_generation_failure_not_a_crash(self):
        ask = FakeModel({"latest-evidence": [RuntimeError("model down"), RuntimeError("model down")]})
        routed = qa_stream.route_deep(QUESTION, plan(skills=["latest-evidence"], info=["efficacy"]), ask=ask)
        self.assertEqual(routed["trace"]["skills"]["latest-evidence"]["outcome"], "error")
        self.assertEqual([o.origin for o in routed["ops"]], ["baseline", "baseline"])      # the literature baseline still covers it

    def test_the_same_search_written_by_two_skills_runs_once_and_serves_both_needs(self):
        same = {"queries": [pubmed_query("efficacy comparison", "sertraline fluoxetine adolescent depression efficacy comparison trial", need_id="aaa")]}
        other = {"queries": [pubmed_query("effectiveness", "Sertraline, fluoxetine: adolescent depression efficacy comparison trial", need_id="bbb")]}
        routed = qa_stream.route_deep(QUESTION, plan(), ask=FakeModel({"latest-evidence": [same], "drug-intelligence": [other]}))
        generated = [o for o in routed["ops"] if o.origin == "generated"]
        self.assertEqual(len(generated), 1)
        self.assertEqual(generated[0].needs(), ["aaa", "bbb"])
        self.assertEqual(routed["trace"]["duplicates_removed"], 1)

    def test_an_unusable_plan_runs_the_safe_wide_strategy_with_the_literature_baseline(self):
        ask = FakeModel({})
        routed = qa_stream.route_deep(QUESTION, {"intent": "x", "condition": "depression", "skills": [], "understanding": {"info_needed": []}}, ask=ask)
        self.assertEqual(sorted(ask.calls), ["clinical-trials", "drug-intelligence", "latest-evidence"])
        self.assertEqual(routed["trace"]["select_skills"]["fallback"], "plan_unusable")
        self.assertTrue([o for o in routed["ops"] if o.origin == "baseline" and o.target == "pubmed"])

    def test_the_baseline_is_not_added_when_literature_is_not_relevant(self):
        ask = FakeModel({"drug-intelligence": [DRUGS]})
        routed = qa_stream.route_deep("What is the label dose of sertraline?", plan(skills=["drug-intelligence"], info=["dosing"], drugs=["sertraline"]), ask=ask)
        self.assertEqual(ask.calls, ["drug-intelligence"])
        self.assertFalse([o for o in routed["ops"] if o.target == "pubmed"])


class RunStage(unittest.TestCase):
    def stage(self, script, ask, budget=5, p=None):
        adapters = FakeAdapters(script)
        result = qa_stream.run_routing_stage(QUESTION, p or plan(), make_adapters=lambda anchors: adapters, budget_s=budget, ask=ask)
        return result, adapters

    def test_each_target_reaches_its_own_search_function_and_results_become_ingestable(self):
        result, adapters = self.stage({"pubmed": [found(2)], "guideline_sites": [json.dumps([{"title": "g", "url": "https://x.org", "snippet": "s"}])], "fda_label": [found(1)], "regulator": [json.dumps([{"title": "r", "url": "https://fda.gov/x", "snippet": "s"}])]},
                                      FakeModel({"latest-evidence": [LATEST], "drug-intelligence": [DRUGS]}))
        self.assertEqual(set(adapters.targets_called()), {"pubmed", "guideline_sites", "fda_label", "regulator"})
        self.assertTrue(all(k.split("_")[0] in ("pubmed", "web", "fda", "trials") for k in result["raw"]))
        self.assertEqual({landscape.source_family(k) for k in result["raw"]}, {"literature", "guideline", "fda_label", "regulator"})

    def test_failed_and_empty_searches_are_recorded_differently(self):
        result, _ = self.stage({"pubmed": [EMPTY], "guideline_sites": [ERROR], "fda_label": [found(1)], "regulator": [found(1)]}, FakeModel({"latest-evidence": [LATEST], "drug-intelligence": [DRUGS]}))
        state = {"source_status": {}}
        landscape.record_source_status(state, result["raw"])
        statuses = {k: (v["retrieval_status"], v["evidence_status"]) for k, v in state["source_status"].items()}
        self.assertIn(("success", "not_found_after_search"), statuses.values())      # a valid empty result
        self.assertIn(("failed", "unavailable"), statuses.values())      # a failed source: never an absence
        self.assertEqual(landscape.family_status(state, "guideline")["retrieval_status"], "failed")

    def test_a_search_past_the_budget_is_partial_and_its_late_result_is_kept(self):
        import time

        def slow(op):
            time.sleep(0.4)
            return found(3)
        result, adapters = self.stage({"pubmed": [found(1)], "guideline_sites": [slow], "fda_label": [found(1)], "regulator": [found(1)]}, FakeModel({"latest-evidence": [LATEST], "drug-intelligence": [DRUGS]}), budget=0.15)
        state = {"source_status": {}}
        landscape.record_source_status(state, result["raw"])
        self.assertEqual(landscape.family_status(state, "guideline")["retrieval_status"], "partial")
        time.sleep(0.6)
        self.assertEqual(len(result["execution"].late_results()), 1)

    def test_the_record_for_the_saved_state_lists_every_operation_with_its_outcome(self):
        result, _ = self.stage({"pubmed": [found(2)], "guideline_sites": [ERROR], "fda_label": [found(1)], "regulator": [EMPTY]}, FakeModel({"latest-evidence": [LATEST], "drug-intelligence": [DRUGS]}))
        record = rr.routing_record(result["ops"], result["execution"].results, rejected=result["rejected"], trace=result["trace"], late=result["execution"].late_results())
        by_target = {(o["target"], o["origin"]): o["status"] for o in record["operations"]}
        self.assertEqual(by_target[("guideline_sites", "generated")], "failed")
        self.assertEqual(by_target[("regulator", "generated")], "completed_empty")
        self.assertEqual(by_target[("fda_label", "generated")], "completed_found")
        json.dumps(record)      # it is saved as JSON


class IngestKeepsProvenance(unittest.TestCase):
    def test_a_paper_found_by_the_fast_search_and_a_routed_search_keeps_both_sources(self):
        paper = {"pmid": "111", "title": "t", "year": 2024, "authors": ["A B"], "journal": "J", "abstract_excerpt": "x", "study_type": ["Randomized Controlled Trial"]}
        raw_fast = {"pubmed_fast": json.dumps({"results": [paper]})}
        raw_routed = {"pubmed_op_0": json.dumps({"results": [paper, {**paper, "pmid": "222"}]})}
        papers, trials, web = {}, {}, []
        qa_stream.ingest(papers, trials, web, raw_fast)
        qa_stream.ingest(papers, trials, web, raw_routed)
        self.assertEqual(papers["111"]["origin"], "pubmed_fast")
        self.assertEqual(papers["111"]["found_by"], ["pubmed_fast", "pubmed_op_0"])
        self.assertEqual(papers["222"]["found_by"], ["pubmed_op_0"])
        self.assertEqual(len(papers), 2)      # merged by PMID, not duplicated

    def test_a_trial_found_twice_is_one_record_with_both_sources(self):
        trial = {"nct_id": "NCT00000001", "title": "t", "status": "RECRUITING"}
        papers, trials, web = {}, {}, []
        qa_stream.ingest(papers, trials, web, {"trials_fast": json.dumps({"results": [trial]}), "trials_op_1": json.dumps({"results": [trial]})})
        self.assertEqual(len(trials), 1)
        self.assertEqual(trials["NCT00000001"]["found_by"], ["trials_fast", "trials_op_1"])


class SearchesStartAsSoonAsTheyAreWritten(unittest.TestCase):
    def test_a_fast_skills_searches_run_before_a_slow_skill_has_answered(self):
        import time
        order = []

        class Slow(FakeModel):
            def __call__(self, question, task, payload, max_tokens, skill="x", **kw):
                if skill == "drug-intelligence":
                    time.sleep(0.6)      # the slow skill
                    order.append("slow skill answered")
                return super().__call__(question, task, payload, max_tokens, skill)

        adapters = FakeAdapters({"pubmed": [found(2)], "guideline_sites": [found(1)], "fda_label": [found(1)], "regulator": [found(1)]})
        wrapped = {target: (lambda op, original=original: (order.append("search started: " + op.target), original(op))[1]) for target, original in adapters.items()}
        stage = qa_stream.run_routing_stage(QUESTION, plan(), make_adapters=lambda anchors: wrapped, budget_s=5, ask=Slow({"latest-evidence": [LATEST], "drug-intelligence": [DRUGS]}))
        self.assertLess(order.index("search started: pubmed"), order.index("slow skill answered"))
        self.assertEqual(len({o.id for o in stage["ops"]}), len(stage["ops"]))
        self.assertEqual(set(stage["execution"].results), {o.id for o in stage["ops"]})


if __name__ == "__main__":
    unittest.main()
