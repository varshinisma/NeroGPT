import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import clinical_tools
import qa_stream


class RegistryCondition(unittest.TestCase):
    def test_an_explicit_or_between_concepts_is_kept_and_each_concept_is_cleaned(self):
        text = clinical_tools._ctgov_condition_text("global developmental delay OR regression of milestones OR autistic tendencies")
        self.assertEqual(text.count(" OR "), 2)
        self.assertIn("global developmental delay", text)

    def test_a_plain_phrase_has_no_operator(self):
        self.assertNotIn(" OR ", clinical_tools._ctgov_condition_text("global developmental delay regression"))


class RetryOfFailedFastSources(unittest.TestCase):
    PLAN = {"condition": "global developmental delay regression of milestones autistic tendencies"}
    QUESTION = "Three year old child has global developmental delay, regression of milestones autistic tendencies. What could be the work and management for this child?"

    def test_the_retry_uses_the_planners_condition_not_the_questions_first_words(self):
        jobs = qa_stream.retry_jobs(self.QUESTION, self.PLAN)
        self.assertEqual(sorted(jobs), ["fda_fast", "trials_drugs_fast", "trials_fast", "web_fast"])
        seen = []
        original = qa_stream.search_clinical_trials
        qa_stream.search_clinical_trials = lambda cond, *a, **k: seen.append(cond) or "{}"
        try:
            qa_stream._cache.clear() if hasattr(qa_stream, "_cache") else None
            jobs["trials_fast"]()
        finally:
            qa_stream.search_clinical_trials = original
        self.assertTrue(seen)
        self.assertNotIn("three", seen[0].lower())
        self.assertIn("global", seen[0])

    def test_without_a_condition_nothing_is_retried(self):
        self.assertEqual(qa_stream.retry_jobs("x", {"condition": ""}), {})

    def test_the_frozen_fast_functions_are_not_involved(self):
        import hashlib
        import inspect
        self.assertEqual(hashlib.sha256(inspect.getsource(qa_stream._fast_jobs).encode()).hexdigest()[:16], "07a323aaf2c254ad")


if __name__ == "__main__":
    unittest.main()
