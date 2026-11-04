import unittest

import qa_stream


class ClinicalApproachTests(unittest.TestCase):
    def test_workup_questions_get_the_section(self):
        q = "What could be the work and management for this child?"
        self.assertTrue(qa_stream.wants_workup({"info_needed": ["diagnosis"]}, q))
        self.assertTrue(qa_stream.wants_workup({"answer_format": "stepwise_plan", "info_needed": []}, ""))
        self.assertTrue(qa_stream.wants_workup({"info_needed": []}, q))

    def test_other_questions_do_not(self):
        self.assertFalse(qa_stream.wants_workup({"info_needed": ["efficacy", "safety"]}, "Is sertraline effective for adolescent depression?"))
        self.assertFalse(qa_stream.wants_workup({}, ""))

    def test_sections_follow_the_question(self):
        out = qa_stream.approach_sections({"info_needed": ["treatment_options"]}, "work and management")
        self.assertIn("### Work-up", out)
        self.assertIn("### Management", out)
        self.assertIn("### Treatment options", qa_stream.approach_sections({"info_needed": ["treatment_options"]}, "What treatments help?"))
        out = qa_stream.approach_sections({"info_needed": ["diagnosis"]}, "How do you diagnose PANDAS")
        self.assertIn("### Diagnostic approach", out)
        self.assertIn("no MANAGEMENT part", out)

    def test_uncited_steps_are_dropped(self):
        out = qa_stream.drop_uncited_steps("### Management plan\n- Therapy helps (PMID 1234567).\n- Rest is good.\n")
        self.assertIn("Therapy helps", out)
        self.assertNotIn("Rest is good", out)

    def test_prompts_carry_the_new_rules(self):
        self.assertIn("Name no medicine", qa_stream.FAST_TASK)
        self.assertIn("Never write 'Not retrieved'", qa_stream.CURRENT_TASK)


if __name__ == "__main__":
    unittest.main()
