"""Source coverage of the routed deep search: the validator keeps good queries (stem matching), each information need is checked for a search on a source that can answer it BEFORE retrieval, the medicine-discovery
set is completed for treatment questions only, near-identical searches are one search, and the one bounded model call that fills a gap is validated and budgeted. Offline: fake model replies and fake search tools."""
import json
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import qa_stream  # noqa: E402
import retrieval_routing as rr  # noqa: E402
from tests.test_routing_stage import FakeModel  # noqa: E402

CAPS = rr.load_capabilities()
QUESTION = "What medication is used for insomnia in children?"
ANCHORS = rr.anchor_words(QUESTION, "insomnia")


def op(target="web", query="medicines used for insomnia in children by generic name", need="n1", serves="", skill="drug-intelligence", priority="essential", **args):
    return rr.Operation(id=f"{skill}:{target}:{need}:{query[:12]}", need_id=need, need=need, target=target, query=query, args=rr.validate_args(target, args), priority=priority, skill=skill, serves=serves)


def plan(info=("treatment_options", "dosing"), skills=("latest-evidence", "drug-intelligence")):
    return {"intent": "drug_treatment", "condition": "insomnia", "drugs": [], "skills": list(skills), "understanding": {"info_needed": list(info), "case_facts": [], "asked": "x", "answer_format": "summary"}}


class StemMatching(unittest.TestCase):
    def test_inflections_and_plurals_match(self):
        self.assertEqual(rr.stem_overlap("medications for insomnia in kids", ANCHORS), 2)      # medication / medications, insomnia
        self.assertEqual(rr.stem_overlap("tic disorders", {"tics"}), 1)
        self.assertEqual(rr.stem_overlap("football results", ANCHORS), 0)

    def test_a_query_that_the_exact_word_rule_used_to_reject_is_kept(self):
        item = {"need": "medicines", "target": "web", "query": "medications that treat sleeplessness insomnia", "priority": "essential"}
        ops, rejected = rr.parse_operations("drug-intelligence", {"queries": [item]}, QUESTION, ANCHORS)
        self.assertEqual((len(ops), rejected), (1, []))
        trial = {"need": "t", "target": "trials", "args": {"condition": "Insomnias in childhood"}}
        self.assertEqual(len(rr.parse_operations("clinical-trials", {"queries": [trial]}, QUESTION, ANCHORS)[0]), 1)

    def test_off_topic_queries_are_still_rejected(self):
        ops, rejected = rr.parse_operations("drug-intelligence", {"queries": [{"need": "x", "target": "web", "query": "football league results this season champions"}]}, QUESTION, ANCHORS)
        self.assertEqual(ops, [])
        self.assertTrue(rejected)


class NearDuplicates(unittest.TestCase):
    def test_a_search_that_asks_the_same_thing_is_dropped_and_its_need_is_kept(self):
        a = op("pubmed", "insomnia children medications efficacy safety randomized trials", need="a", skill="latest-evidence")
        b = op("pubmed", "insomnia children medications efficacy safety randomized trial", need="b", skill="drug-intelligence")
        merged, dropped = rr.merge_operations([a], [b])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].needs(), ["a", "b"])

    def test_different_questions_of_one_source_and_the_same_words_in_two_sources_are_kept(self):
        a = op("pubmed", "insomnia children melatonin efficacy randomized trials", need="a", skill="latest-evidence")
        b = op("pubmed", "insomnia children behavioural therapy outcomes systematic review", need="b", skill="latest-evidence")
        c = op("europepmc", "insomnia children melatonin efficacy randomized trials", need="c", skill="latest-evidence")
        self.assertEqual(len(rr.merge_operations([a], [b], [c])[0]), 3)


class GapDetection(unittest.TestCase):
    @staticmethod
    def only(gaps, kind):
        return [g for g in gaps if g["kind"] == kind]

    def test_an_information_need_with_no_search_on_a_source_that_can_answer_it_is_a_gap(self):
        ops = [op("pubmed", "insomnia children medication evidence review", serves="efficacy", skill="latest-evidence"), op("pubmed", "insomnia children medication adverse effects", serves="safety", skill="latest-evidence")]
        gaps = self.only(rr.coverage_gaps(ops, ["efficacy", "safety", "regulatory_status"], ["latest-evidence"], CAPS), "need_without_suitable_source")
        self.assertEqual([g["need"] for g in gaps], ["regulatory_status"])
        self.assertIn("regulator", gaps[0]["suitable"])

    def test_a_need_served_on_a_suitable_source_is_not_a_gap(self):
        ops = [op("regulator", "insomnia children medication approvals safety communications", serves="regulatory_status", skill="latest-evidence")]
        self.assertEqual(self.only(rr.coverage_gaps(ops, ["regulatory_status"], ["latest-evidence"], CAPS), "need_without_suitable_source"), [])

    def test_the_wrong_source_for_a_need_does_not_count_as_coverage(self):
        ops = [op("trials", "insomnia children", serves="regulatory_status", skill="clinical-trials", condition="insomnia children")]
        self.assertEqual([g["need"] for g in self.only(rr.coverage_gaps(ops, ["regulatory_status"], ["clinical-trials"], CAPS), "need_without_suitable_source")], ["regulatory_status"])

    def test_searches_without_a_need_tag_cannot_be_matched_so_the_need_check_is_skipped(self):
        ops = [op("pubmed", "insomnia children medication evidence review", skill="latest-evidence")]
        self.assertEqual(self.only(rr.coverage_gaps(ops, ["efficacy"], ["latest-evidence"], CAPS), "need_without_suitable_source"), [])

    def test_treatment_questions_with_too_few_discovery_searches_are_a_gap_that_names_how_many_are_missing(self):
        ops = [op("web", "medicines used for insomnia in children by generic name"), op("web", "medications for sleep problems co-occurring with childhood anxiety")]
        gaps = self.only(rr.coverage_gaps(ops, ["treatment_options"], ["drug-intelligence"], CAPS), "discovery_shortfall")
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["missing"], 1)
        self.assertEqual(len(gaps[0]["purposes"]), 3)

    def test_three_discovery_searches_are_enough(self):
        ops = [op("web", f"medicines for insomnia in children purpose {i}") for i in range(3)]
        self.assertEqual(self.only(rr.coverage_gaps(ops, ["treatment_options"], ["drug-intelligence"], CAPS), "discovery_shortfall"), [])

    def test_a_treatment_question_without_a_guideline_or_regulator_search_is_missing_its_authority_sources(self):
        ops = [op("pubmed", "insomnia children medication evidence review", skill="latest-evidence")]
        gap = self.only(rr.coverage_gaps(ops, ["treatment_options"], ["latest-evidence", "drug-intelligence"], CAPS), "authority_source_missing")
        self.assertEqual(sorted(gap[0]["missing_targets"]), ["guideline_sites", "regulator"])      # medicines bear on the question (the drug skill is in play): both authorities
        gap = self.only(rr.coverage_gaps(ops, ["treatment_options"], ["latest-evidence"], CAPS), "authority_source_missing")
        self.assertEqual(gap[0]["missing_targets"], ["guideline_sites"])      # no medicines in play: the regulators' pages are not forced
        both = ops + [op("guideline_sites", "insomnia children medication guideline recommendations", skill="latest-evidence"), op("regulator", "insomnia children medication approvals safety", skill="latest-evidence")]
        self.assertEqual(self.only(rr.coverage_gaps(both, ["treatment_options"], ["latest-evidence", "drug-intelligence"], CAPS), "authority_source_missing"), [])
        one = ops + [op("regulator", "insomnia children medication approvals safety", skill="latest-evidence")]
        self.assertEqual(self.only(rr.coverage_gaps(one, ["treatment_options"], ["latest-evidence", "drug-intelligence"], CAPS), "authority_source_missing")[0]["missing_targets"], ["guideline_sites"])

    def test_an_essential_need_answered_from_one_source_only_is_a_gap_naming_the_other_suitable_sources(self):
        ops = [op("pubmed", "insomnia children medication efficacy randomized trials", serves="efficacy", skill="latest-evidence")]
        ops[0].priority = "essential"
        gaps = self.only(rr.coverage_gaps(ops, ["efficacy"], ["latest-evidence"], CAPS), "single_source_need")
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["used"], ["pubmed"])
        self.assertIn("europepmc", gaps[0]["other_sources"])
        ops[0].priority = "supporting"
        self.assertEqual(self.only(rr.coverage_gaps(ops, ["efficacy"], ["latest-evidence"], CAPS), "single_source_need"), [])

    def test_nothing_is_forced_for_unrelated_questions(self):
        self.assertEqual(rr.coverage_gaps([], ["mechanism"], ["latest-evidence"], CAPS), [])      # a mechanism question: no discovery, no authority searches
        self.assertEqual(rr.coverage_gaps([], ["prognosis"], ["latest-evidence"], CAPS), [])
        self.assertEqual(rr.coverage_gaps([], ["diagnosis", "prognosis"], ["latest-evidence"], CAPS), [{"kind": "authority_source_missing", "missing_targets": ["guideline_sites"]}])      # a work-up question: the guideline organisations are asked, no regulator search
        self.assertEqual(self.only(rr.coverage_gaps([], ["dosing"], ["drug-intelligence"], CAPS), "discovery_shortfall"), [])      # a dosing question is not a discovery question
        self.assertEqual(rr.coverage_gaps([], [], ["drug-intelligence"], CAPS), [])


class FillTheGap(unittest.TestCase):
    GAPS = [{"kind": "discovery_shortfall", "have": ["medicines used for insomnia in children by generic name"], "missing": 2, "purposes": CAPS["discovery"]["purposes"]}]
    EXISTING = [op("web", "medicines used for insomnia in children by generic name")]

    def reply(self, *items):
        return {"queries": [{"need_id": f"x{i}", "need": f"discovery {i}", "target": t, "query": q, "priority": "essential"} for i, (t, q) in enumerate(items)]}

    def fill(self, reply, call_s=2, ask=None):
        model = ask or FakeModel({"drug-intelligence": [reply]})
        return qa_stream.fill_coverage(QUESTION, plan(), self.GAPS, self.EXISTING, ANCHORS, ask=model, call_s=call_s), model

    def test_missing_searches_are_added_validated_and_count_toward_the_drug_skill(self):
        result, model = self.fill(self.reply(("web", "medications for sleep problems co-occurring with anxiety in children insomnia"), ("web", "specific medicines of the classes named for insomnia in children")))
        self.assertEqual(result["outcome"], "added")
        self.assertEqual([(o.origin, o.skill, o.target) for o in result["ops"]], [("coverage", "drug-intelligence", "web")] * 2)
        self.assertEqual(model.calls, ["drug-intelligence"])      # ONE call

    def test_near_duplicates_of_what_was_written_and_invalid_targets_are_dropped(self):
        result, _ = self.fill(self.reply(("web", "medicines used for insomnia in children by generic names"), ("run_shell", "medicines for insomnia in children"), ("web", "medications for children with insomnia and anxiety")))
        self.assertEqual([o.query for o in result["ops"]], ["medications for children with insomnia and anxiety"])
        self.assertGreaterEqual(result["rejected"], 2)

    def test_the_number_of_additions_is_capped(self):
        many = self.reply(*[("web", f"medicines for insomnia in children topic number {i} with distinct wording {i * 7}") for i in range(9)])
        result, _ = self.fill(many)
        self.assertLessEqual(len(result["ops"]), qa_stream.MAX_COVERAGE_ADDITIONS)

    def test_an_off_topic_addition_is_rejected(self):
        result, _ = self.fill(self.reply(("web", "football league results this season champions")))
        self.assertEqual((result["ops"], result["outcome"]), ([], "nothing_added"))

    def test_a_slow_call_is_abandoned_at_its_time_limit_and_adds_nothing(self):
        def slow(*args, **kwargs):
            time.sleep(0.6)
            return self.reply(("web", "medications for children with insomnia and anxiety"))
        began = time.monotonic()
        result, _ = self.fill(None, call_s=0.15, ask=slow)
        self.assertLess(time.monotonic() - began, 0.45)
        self.assertEqual((result["ops"], result["outcome"]), ([], "timed_out"))

    def test_a_failed_call_adds_nothing_and_is_recorded(self):
        def broken(*args, **kwargs):
            raise RuntimeError("model down")
        result, _ = self.fill(None, ask=broken)
        self.assertEqual((result["ops"], result["outcome"]), ([], "error"))


class RoutingWithTheCoverageCheck(unittest.TestCase):
    LATEST = {"queries": [{"need_id": "t1", "need": "treatment evidence", "serves": "efficacy", "target": "pubmed", "priority": "supporting", "query": "insomnia children medication treatment efficacy randomized trials"}]}
    ONE_WEB = {"queries": [{"need_id": "d1", "need": "medicine names", "serves": "treatment_options", "target": "web", "priority": "essential", "query": "medicines used for insomnia in children by generic name"}]}
    FILL = {"queries": [{"need_id": "d2", "need": "co-occurring", "serves": "treatment_options", "target": "web", "priority": "essential", "query": "medications for children with insomnia and co-occurring anxiety"},
                        {"need_id": "d3", "need": "classes", "serves": "treatment_options", "target": "web", "priority": "essential", "query": "specific medicines of the classes named for insomnia in children"}]}

    def test_a_treatment_question_whose_drug_skill_wrote_one_discovery_search_gets_the_missing_two(self):
        ask = FakeModel({"latest-evidence": [self.LATEST], "drug-intelligence": [self.ONE_WEB, self.FILL]})
        routed = qa_stream.route_deep(QUESTION, plan(), ask=ask)
        web = [o for o in routed["ops"] if o.target == "web"]
        self.assertEqual(len(web), 3)
        self.assertEqual(routed["trace"]["coverage"]["outcome"], "added")
        self.assertEqual(routed["trace"]["coverage"]["added"], 2)
        self.assertEqual(ask.calls.count("drug-intelligence"), 2)      # the skill's own call plus the one bounded coverage call

    def test_no_extra_call_when_the_skill_already_covered_the_gap(self):
        three = {"queries": [{"need_id": "g1", "need": "guidelines", "serves": "treatment_options", "target": "guideline_sites", "priority": "supporting", "query": "insomnia children medication guideline recommendations first line"},
                             {"need_id": "r1", "need": "approvals", "serves": "treatment_options", "target": "regulator", "priority": "supporting", "query": "insomnia children medicines approvals safety communications"}] + [
                            {"need_id": f"d{i}", "need": f"discovery {i}", "serves": "treatment_options", "target": "web", "priority": "supporting", "query": q}
                             for i, q in enumerate(["medicines used for insomnia in children by generic name", "medications for sleep problems in children with co-occurring anxiety or ADHD",
                                                    "specific drugs in the classes named for paediatric insomnia such as melatonin agonists and antihistamines"])]}
        ask = FakeModel({"latest-evidence": [self.LATEST], "drug-intelligence": [three]})
        routed = qa_stream.route_deep(QUESTION, plan(info=["treatment_options"]), ask=ask)
        self.assertEqual(routed["trace"]["coverage"]["outcome"], "no_gaps")
        self.assertEqual(ask.calls.count("drug-intelligence"), 1)

    def test_an_unrelated_question_is_not_given_discovery_searches(self):
        ask = FakeModel({"latest-evidence": [{"queries": [{"need_id": "m1", "need": "mechanism", "serves": "mechanism", "target": "pubmed", "priority": "supporting", "query": "metformin mechanism lowering blood glucose pathway"}]}]})
        routed = qa_stream.route_deep("How does metformin lower blood glucose?", plan(info=["mechanism"], skills=["latest-evidence"]), ask=ask)
        self.assertEqual(ask.calls, ["latest-evidence"])
        self.assertEqual(routed["trace"]["coverage"]["outcome"], "no_gaps")
        self.assertFalse([o for o in routed["ops"] if o.target == "web"])

    def test_a_need_with_no_suitable_source_gets_one_search_on_a_suitable_one(self):
        fill = {"queries": [{"need_id": "r1", "need": "approvals", "serves": "regulatory_status", "target": "regulator", "priority": "supporting", "query": "insomnia children medicines approvals safety communications"}]}
        ask = FakeModel({"latest-evidence": [self.LATEST, fill]})
        routed = qa_stream.route_deep(QUESTION, plan(info=["efficacy", "regulatory_status"], skills=["latest-evidence"]), ask=ask)
        self.assertEqual([g["need"] for g in routed["trace"]["coverage"]["gaps"] if g["kind"] == "need_without_suitable_source"], ["regulatory_status"])
        self.assertIn("regulator", [o.target for o in routed["ops"]])

    def test_the_total_number_of_searches_stays_within_the_budget(self):
        many = {"queries": [{"need_id": f"p{i}", "need": f"need {i}", "serves": "treatment_options", "target": "pubmed", "priority": "supporting", "query": f"insomnia children medication distinct topic {i} wording {i * 13} extra{i}"} for i in range(8)]}
        ask = FakeModel({"latest-evidence": [many], "drug-intelligence": [self.ONE_WEB, {"queries": [dict(self.FILL["queries"][0], need_id=f"z{i}", query=f"medications for children with insomnia, variant {i} wording {i * 17}") for i in range(4)]}]})
        routed = qa_stream.route_deep(QUESTION, plan(), ask=ask)
        self.assertLessEqual(len(routed["ops"]), rr.MAX_OPERATIONS + qa_stream.MAX_COVERAGE_ADDITIONS)

    def test_the_check_can_be_switched_off(self):
        os.environ["COVERAGE_CHECK"] = "off"
        try:
            ask = FakeModel({"latest-evidence": [self.LATEST], "drug-intelligence": [self.ONE_WEB]})
            routed = qa_stream.route_deep(QUESTION, plan(), ask=ask)
        finally:
            del os.environ["COVERAGE_CHECK"]
        self.assertEqual(routed["trace"]["coverage"]["outcome"], "not_needed")
        self.assertEqual(ask.calls.count("drug-intelligence"), 1)


class FastPathUnchanged(unittest.TestCase):
    def test_the_fast_functions_are_byte_for_byte_the_same(self):
        import hashlib
        import inspect
        for name, digest in {"_fast_jobs": "07a323aaf2c254ad", "fast_jobs": "012e722131d9edce", "with_retry": "4332567971e488d5", "primary_with_fallback": "d5f081677afb90cc", "run_jobs": "97aa21d88e41abed"}.items():
            self.assertEqual(hashlib.sha256(inspect.getsource(getattr(qa_stream, name)).encode()).hexdigest()[:16], digest, name)
