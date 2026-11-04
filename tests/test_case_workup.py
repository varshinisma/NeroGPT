import unittest

import qa_stream

Q = "Three year old child has global developmental delay, regression of milestones autistic tendencies. What could be the work and management for this child ?"
ANCHORS = {"global", "developmental", "delay", "regression", "milestones", "autistic", "tendencies", "three", "year"}
PLAN = {"understanding": {"answer_format": "stepwise_plan", "info_needed": ["diagnosis", "treatment_options"]}}


class Op:
    def __init__(self, query, target="pubmed"):
        self.query, self.target = query, target


class CaseWorkup(unittest.TestCase):
    def test_missing_standard_parts_get_a_search(self):
        names = {o.id for o in qa_stream.mandatory_case_operations(Q, PLAN, [], ANCHORS)}
        self.assertEqual(names, {"case:sensory", "case:genetic", "case:neuro", "case:labs"})

    def test_a_part_the_model_already_searched_is_not_added_again(self):
        ops = [Op("developmental delay genetic testing exome"), Op("hearing and vision screening in developmental delay")]
        names = {o.id for o in qa_stream.mandatory_case_operations(Q, PLAN, ops, ANCHORS)}
        self.assertEqual(names, {"case:neuro", "case:labs"})

    def test_age_words_stay_out_of_the_added_searches(self):
        for o in qa_stream.mandatory_case_operations(Q, PLAN, [], ANCHORS):
            self.assertNotIn("three", o.query)
            self.assertIn("developmental delay", o.query)

    def test_other_questions_get_nothing(self):
        self.assertEqual(qa_stream.mandatory_case_operations("Is sertraline effective for adolescent depression?", {"understanding": {}}, [], {"sertraline"}), [])
        self.assertEqual(qa_stream.mandatory_case_operations("How to treat autism in children", {"understanding": {"info_needed": ["treatment_options"]}}, [], {"autism"}), [])


if __name__ == "__main__":
    unittest.main()


class AssociatedFindingTerms(unittest.TestCase):
    def test_the_associated_finding_is_searched_too(self):
        import qa_stream
        terms = qa_stream.associated_terms("Three year old child has global developmental delay, regression of milestones autistic tendencies. What could be the work and management?", ["devel", "delay"])
        self.assertIn("autistic", terms)
        self.assertNotIn("developmental", terms)
