"""Source families: stable names for 'which source', so the report, the audit and the drug and trial sections no longer depend on legacy job names."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import landscape  # noqa: E402
import report  # noqa: E402


def ok():
    return {"retrieval_status": "success", "evidence_status": "found", "error": None}


def empty():
    return {"retrieval_status": "success", "evidence_status": "not_found_after_search", "error": None}


def failed(error="boom"):
    return {"retrieval_status": "failed", "evidence_status": "unavailable", "error": error}


class Families(unittest.TestCase):
    def test_every_kind_of_key_maps_to_one_family(self):
        cases = {"pubmed_fast": "literature", "pubmed_recent": "literature", "pubmed_op_3": "literature", "pubmed_epmc_op_1": "literature", "trials_fast": "trials", "trials_deep": "trials", "trials_op_0": "trials",
                 "fda_fast": "fda_label", "fda_deep": "fda_label", "fda_op_0": "fda_label", "web_fast": "web", "web_standard": "web", "web_op_2": "web", "web_regulatory": "regulator",
                 "web_regulator_op_1": "regulator", "guidelines_deep": "guideline", "web_guideline_op_0": "guideline"}
        for key, family in cases.items():
            self.assertEqual(landscape.source_family(key), family, key)

    def test_the_deep_status_wins_and_a_fast_one_is_the_fallback(self):
        state = {"source_status": {"fda_fast": ok(), "fda_op_0": failed("down")}}
        self.assertEqual(landscape.family_status(state, "fda_label")["retrieval_status"], "failed")
        self.assertTrue(landscape.family_has_deep(state, "fda_label"))
        only_fast = {"source_status": {"fda_fast": ok()}}
        self.assertEqual(landscape.family_status(only_fast, "fda_label")["retrieval_status"], "success")
        self.assertFalse(landscape.family_has_deep(only_fast, "fda_label"))
        self.assertEqual(landscape.family_status({"source_status": {}}, "fda_label"), {})

    def test_several_deep_searches_of_a_family_succeed_if_any_does(self):
        state = {"source_status": {"trials_fast": failed(), "trials_op_0": failed(), "trials_op_1": empty()}}
        status = landscape.family_status(state, "trials")
        self.assertEqual(status["retrieval_status"], "success")
        self.assertEqual(status["evidence_status"], "not_found_after_search")

    def test_deep_only_does_not_fall_back_to_the_fast_job(self):
        self.assertEqual(landscape.family_status({"source_status": {"trials_fast": ok()}}, "trials", deep_only=True), {})

    def test_a_failed_fast_job_is_not_a_problem_when_a_deep_search_of_its_family_succeeded(self):
        status = {"pubmed_fast": failed(), "pubmed_op_0": ok(), "trials_fast": failed(), "web_fast": failed(), "web_regulator_op_0": ok()}
        self.assertTrue(landscape.superseded_by_deep(status, "pubmed_fast"))
        self.assertFalse(landscape.superseded_by_deep(status, "trials_fast"))      # no deep trial search succeeded
        self.assertFalse(landscape.superseded_by_deep(status, "web_fast"))      # the regulator family is another family

    def test_the_failure_banner_groups_by_family_and_ignores_superseded_fast_jobs(self):
        state = {"source_status": {"pubmed_fast": failed("slow"), "pubmed_op_0": ok(), "trials_op_0": failed("registry down"), "web_guideline_op_0": failed("x")}}
        problems = {label: kind for label, kind, _ in report.source_problems(state)}
        self.assertNotIn("PubMed / Europe PMC", problems)
        self.assertEqual(problems["ClinicalTrials.gov"], "failed")
        self.assertEqual(problems["Guideline search"], "failed")
        self.assertNotIn("Web search", problems)      # a guideline page search is not a web-search problem
        self.assertEqual([k for k in report.source_problems(state, core_only=True)], [p for p in report.source_problems(state) if p[0] != "Guideline search"])

    def test_limitations_name_the_family_in_plain_words(self):
        self.assertEqual(landscape.source_name("fda_op_0"), "FDA drug-label search")
        self.assertEqual(landscape.source_name("web_guideline_op_0"), "guideline-organisation search")
        self.assertEqual(landscape.source_name("web_regulator_op_1"), "web search")
        self.assertEqual(landscape.source_name("pubmed_epmc_op_2"), "PubMed / Europe PMC search")
        self.assertEqual(landscape.source_name("deep_evidence"), "deep-retrieval stage")

    def test_the_drug_and_trial_sections_read_the_family_status(self):
        base = {"regulatory": [], "trials_by_id": {}, "drug_records": [], "papers": {}, "web": []}
        failing = {**base, "source_status": {"trials_op_0": failed("registry down")}}
        self.assertIn("could not be retrieved", landscape.trials_markdown(failing))
        self.assertIn("does not mean that no trials exist", landscape.trials_markdown(failing))
        fine = {**base, "source_status": {"trials_op_0": empty()}}
        self.assertIn("No trials were found", landscape.trials_markdown(fine))
        # the legacy names still work while they exist
        legacy = {**base, "source_status": {"trials_deep": failed("down")}}
        self.assertIn("could not be retrieved", landscape.trials_markdown(legacy))


if __name__ == "__main__":
    unittest.main()
