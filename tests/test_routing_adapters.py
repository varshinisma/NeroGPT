"""Offline tests of the seven search adapters: each target must call the existing search function with the right arguments. Fake search functions only."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import retrieval_routing as rr  # noqa: E402

ANCHORS = {"depression", "adolescents", "sertraline", "fluoxetine", "alzheimer"}


class Recorder:
    def __init__(self):
        self.calls = []

    def fn(self, name, answer=None):
        def run(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return answer if answer is not None else json.dumps({"results": [{"x": 1}]})
        return run


def build(guideline_filter=None, cache=None):
    rec = Recorder()
    adapters = rr.make_adapters(search_pubmed=rec.fn("pubmed"), search_europepmc_quick=rec.fn("europepmc"), search_clinical_trials=rec.fn("trials"), search_fda_labels=rec.fn("fda_label"),
                                guideline_search=rec.fn("guideline_sites", json.dumps([{"title": "t", "url": "https://x.org/a", "snippet": "s"}])), regulator_search=rec.fn("regulator", json.dumps([{"url": "u"}])),
                                web_search=rec.fn("web", json.dumps([{"url": "w"}])), anchors=ANCHORS, guideline_filter=guideline_filter, cache=cache)
    return adapters, rec


def operation(target, query="sertraline fluoxetine adolescents depression efficacy comparison", **args):
    return rr.Operation(id="t", need_id="n", need="n", target=target, query=query, args=rr.validate_args(target, args))


class AdaptersCallTheExistingSearchFunctions(unittest.TestCase):
    def test_every_target_has_an_adapter(self):
        adapters, _ = build()
        self.assertEqual(sorted(adapters), sorted(rr.TARGETS))

    def test_pubmed_uses_a_boolean_form_the_sort_and_the_years(self):
        adapters, rec = build()
        adapters["pubmed"](operation("pubmed", sort="date", years_back=2))
        name, args, kwargs = rec.calls[0]
        self.assertTrue(args[0].startswith("(") and " AND (" in args[0] and " OR " in args[0])
        self.assertNotIn("2023", args[0])
        self.assertEqual((kwargs["sort"], kwargs["years_back"], kwargs["max_results"]), ("date", 2, 8))

    def test_study_types_become_the_existing_filters_and_open_access_prefers_free_full_text(self):
        adapters, rec = build()
        adapters["pubmed"](operation("pubmed", study_types=["rct", "negative_results"], open_access=True))
        query = rec.calls[0][1][0]
        self.assertIn("randomized controlled trial[pt]", query)
        self.assertIn('"no significant"[tiab]', query)
        self.assertIn("pubmed pmc[sb]", query)

    def test_europepmc_is_a_second_literature_index(self):
        adapters, rec = build()
        adapters["europepmc"](operation("europepmc", years_back=3))
        name, args, kwargs = rec.calls[0]
        self.assertEqual(name, "europepmc")
        self.assertEqual((kwargs["years_back"], kwargs["sort"]), (3, "relevance"))

    def test_trials_map_the_status_and_the_drug_only_flag(self):
        for status, expected in (("recruiting", "RECRUITING,NOT_YET_RECRUITING,ACTIVE_NOT_RECRUITING"), ("completed", "COMPLETED"), ("any", "")):
            adapters, rec = build()
            adapters["trials"](operation("trials", query="", condition="depression adolescents", intervention="sertraline", status=status, drug_only=True))
            name, args, kwargs = rec.calls[0]
            self.assertEqual(args, ("depression adolescents", "sertraline", expected, 12, True))

    def test_a_trial_search_with_an_intervention_that_finds_nothing_is_widened_to_the_condition_alone(self):
        rec = Recorder()
        answers = [json.dumps({"results": []}), json.dumps({"results": [{"nct_id": "NCT1"}]})]

        def trials(*args, **kwargs):
            rec.calls.append(("trials", args, kwargs))
            return answers.pop(0)
        adapters = rr.make_adapters(search_pubmed=rec.fn("p"), search_europepmc_quick=rec.fn("e"), search_clinical_trials=trials, search_fda_labels=rec.fn("f"), guideline_search=rec.fn("g"), regulator_search=rec.fn("r"),
                                    web_search=rec.fn("w"), anchors=ANCHORS)
        raw = adapters["trials"](operation("trials", query="", condition="depression adolescents", intervention="madeupdrug"))
        self.assertEqual([c[1][1] for c in rec.calls], ["madeupdrug", ""])      # same source, same condition, without the intervention
        self.assertIn("NCT1", raw)

    def test_boolean_words_are_removed_from_a_trial_condition(self):
        self.assertEqual(rr.validate_args("trials", {"condition": "severe autism AND (aggression OR irritability)"})["condition"], "severe autism aggression irritability")

    def test_fda_label_passes_condition_terms_and_the_drugs_the_question_names(self):
        adapters, rec = build()
        adapters["fda_label"](operation("fda_label", query="", condition_terms=["depression"], drug_names=["sertraline"]))
        self.assertEqual(rec.calls[0][1], (["depression"], ["sertraline"], 8))
        adapters["fda_label"](operation("fda_label", query="", condition_terms=["depression"], drug_names=[]))
        self.assertEqual(rec.calls[1][1][1], None)

    def test_guideline_sites_pass_through_the_authoritative_domain_filter(self):
        adapters, rec = build(guideline_filter=lambda raw: json.dumps([]))
        self.assertEqual(adapters["guideline_sites"](operation("guideline_sites")), "[]")
        adapters, rec = build()
        self.assertIn("https://x.org/a", adapters["guideline_sites"](operation("guideline_sites")))

    def test_regulator_and_web_get_the_query_as_written(self):
        adapters, rec = build()
        adapters["regulator"](operation("regulator", query="alzheimer new approvals indications this year"))
        adapters["web"](operation("web", query="alzheimer specific medicines of the class"))
        self.assertEqual([c[1][0] for c in rec.calls], ["alzheimer new approvals indications this year", "alzheimer specific medicines of the class"])

    def test_the_pipeline_cache_is_used_with_the_right_kind_and_lifetime(self):
        seen = []

        def cache(kind, key, ttl, fn):
            seen.append((kind, ttl))
            return fn()
        adapters, _ = build(cache=cache)
        for target, extra in (("pubmed", {}), ("europepmc", {}), ("trials", {"condition": "depression"}), ("fda_label", {"condition_terms": ["depression"]}), ("guideline_sites", {}), ("regulator", {}), ("web", {})):
            adapters[target](operation(target, **extra))
        self.assertEqual(seen, [("pm", 900), ("pm", 900), ("ct", 300), ("fda", 3600), ("web", 300), ("web", 300), ("web", 300)])

    def test_adapters_run_under_the_execution_engine_and_a_tool_error_is_a_failure(self):
        rec = Recorder()
        adapters = rr.make_adapters(search_pubmed=rec.fn("pubmed", json.dumps({"error": "429 too many requests"})), search_europepmc_quick=rec.fn("e"), search_clinical_trials=rec.fn("t"),
                                    search_fda_labels=rec.fn("f"), guideline_search=rec.fn("g"), regulator_search=rec.fn("r"), web_search=rec.fn("w"), anchors=ANCHORS)
        ex = rr.Execution([operation("pubmed")], adapters, 5)
        self.assertEqual(ex.results["t"].status, "failed")
        self.assertEqual(ex.results["t"].attempts, 2)      # a transient error was retried once


class SkillThatDeclines(unittest.TestCase):
    def test_an_empty_json_object_is_a_decline_not_an_error(self):
        anchors = rr.anchor_words("How does metformin lower blood glucose?")
        for reply in ({"queries": []}, {"_raw": "```json\n{}\n```"}, {"_raw": '```json\n{"queries": []}\n```', "queries": []}):
            ops, rejected = rr.parse_operations("clinical-trials", reply, "How does metformin lower blood glucose?", anchors)
            self.assertEqual((ops, rejected), ([], []))

    def test_an_unreadable_or_truncated_reply_is_still_an_error(self):
        anchors = rr.anchor_words("How does metformin lower blood glucose?")
        for reply in ({}, {"_raw": "```json\n{\"queries\": [{\"need\": \"x\", \"targ"}, {"_raw": "I am not sure"}):
            ops, rejected = rr.parse_operations("clinical-trials", reply, "How does metformin lower blood glucose?", anchors)
            self.assertEqual(ops, [])
            self.assertTrue(rejected)


if __name__ == "__main__":
    unittest.main()
