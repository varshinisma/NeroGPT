import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import qa_stream

PLAN = {"condition": "condition x", "trial_condition": "condition x", "trial_intervention": "", "drugs": [], "intent": "case_evaluation"}
QUESTION = "What work-up does a child with condition x need?"


def jobs():
    return qa_stream.deep_jobs(dict(PLAN), QUESTION)


class LegacyByIntent(unittest.TestCase):
    def test_a_question_without_medicines_or_trials_drops_both_job_groups(self):
        plan = {**PLAN, "skills": ["latest-evidence"], "understanding": {"info_needed": ["diagnosis"], "medicines_needed": False}}
        kept, trace = qa_stream.legacy_jobs_for_intent(plan, jobs())
        self.assertIn("pubmed_recent", kept)
        self.assertIn("web_standard", kept)
        self.assertIn("guidelines_deep", kept)
        for name in qa_stream.LEGACY_BY_NEED["drugs"] | qa_stream.LEGACY_BY_NEED["trials"]:
            self.assertNotIn(name, kept)
        self.assertFalse(trace["kept_all"])

    def test_the_drug_skill_keeps_the_drug_jobs_and_not_the_trial_jobs(self):
        plan = {**PLAN, "skills": ["latest-evidence", "drug-intelligence"], "understanding": {"info_needed": ["treatment_options"], "medicines_needed": True}}
        kept, _ = qa_stream.legacy_jobs_for_intent(plan, jobs())
        self.assertIn("fda_deep", kept)
        self.assertNotIn("trials_deep", kept)

    def test_the_trials_skill_keeps_the_trial_jobs(self):
        plan = {**PLAN, "skills": ["latest-evidence", "clinical-trials"], "understanding": {"info_needed": ["trial_status"], "medicines_needed": False}}
        kept, _ = qa_stream.legacy_jobs_for_intent(plan, jobs())
        self.assertIn("trials_deep", kept)
        self.assertNotIn("fda_deep", kept)

    def test_a_plan_that_failed_keeps_every_job(self):
        everything = jobs()
        kept, trace = qa_stream.legacy_jobs_for_intent({**PLAN, "plan_status": "fallback"}, everything)
        self.assertEqual(set(kept), set(everything))
        self.assertTrue(trace["kept_all"])


class PlannedPubmedJobs(unittest.TestCase):
    PLAN2 = {**PLAN, "plan_status": "ok", "pubmed_query": '("aaa bbb" OR "ccc") AND ("child" OR "pediatric") AND ("workup" OR "diagnosis" OR "management")',
             "understanding": {"info_needed": ["diagnosis", "treatment_options", "regulatory_status"]}}

    def test_the_subject_part_leaves_out_the_intent_group(self):
        query, subject = qa_stream.planned_pubmed_query(self.PLAN2)
        self.assertEqual(query, self.PLAN2["pubmed_query"])
        self.assertEqual(subject, '("aaa bbb" OR "ccc") AND ("child" OR "pediatric")')

    def test_one_search_is_written_per_information_need_that_has_wording(self):
        names = set(qa_stream.plan_pubmed_jobs(self.PLAN2))
        self.assertEqual(names, {"pubmed_plan_main", "pubmed_plan_recent", "pubmed_plan_epmc_fulltext", "pubmed_plan_diagnosis", "pubmed_plan_treatment_options"})

    def test_no_plan_searches_without_a_good_plan(self):
        self.assertEqual(qa_stream.plan_pubmed_jobs({**self.PLAN2, "plan_status": "fallback"}), {})
        self.assertEqual(qa_stream.plan_pubmed_jobs({**self.PLAN2, "pubmed_query": '("aaa" OR'}), {})
        self.assertEqual(qa_stream.plan_pubmed_jobs({k: v for k, v in self.PLAN2.items() if k != "plan_status"}), {})


if __name__ == "__main__":
    unittest.main()
