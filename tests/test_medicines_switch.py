import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import landscape
import qa_stream

LABEL = {"generic": "Drugx", "brands": [], "indication_excerpt": "Drugx is indicated for irritability associated with condition Z in children and adolescents (ages 5 to 17 years).",
         "url": "https://example.org/label", "label_date": "2026-01-01", "model_relevant": True, "model_relation": "treats_associated_symptom_or_comorbidity",
         "relevance": "treats_associated_symptom_or_comorbidity", "model_evidence": "Drugx is indicated for irritability associated with condition Z (ages 5 to 17 years)."}


def state(**extra):
    base = {"source_status": {"fda_fast": {"retrieval_status": "success", "evidence_status": "found", "error": None}}, "regulatory": [LABEL], "papers": {}, "trials_by_id": {}, "web": [],
            "drug_records": [], "fda_terms": ["condz"]}
    base.update(extra)
    return base


class MedicinesSwitch(unittest.TestCase):
    def test_a_question_that_does_not_ask_about_medicines_lists_none(self):
        s = state(medicines_needed=False)
        self.assertIn("not listed", landscape.top_drugs_markdown(s).lower())
        self.assertNotIn("Drugx", landscape.top_drugs_markdown(s))
        md = landscape.landscape_markdown(s, [], [])
        self.assertTrue(md.startswith("## Treatment Drug Landscape"))
        self.assertNotIn("Drugx", md)

    def test_an_unknown_answer_keeps_the_earlier_behaviour(self):
        for value in (None, True):
            s = state(medicines_needed=value)
            self.assertIn("Drugx", landscape.top_drugs_markdown(s))

    def test_the_understanding_carries_only_a_real_yes_or_no(self):
        plan = {"intent": "diagnosis", "condition": "x", "info_needed": ["diagnosis"], "medicines_needed": False}
        self.assertIs(qa_stream.verified_understanding(plan, "question x")["medicines_needed"], False)
        plan["medicines_needed"] = "yes"
        self.assertIsNone(qa_stream.verified_understanding(plan, "question x")["medicines_needed"])
        del plan["medicines_needed"]
        self.assertIsNone(qa_stream.verified_understanding(plan, "question x")["medicines_needed"])


class LabelAge(unittest.TestCase):
    def test_the_patients_age_is_read_from_the_question(self):
        self.assertEqual(landscape.patient_age_years("Three year old child has x"), 3.0)
        self.assertEqual(landscape.patient_age_years("A 3-year-old with x"), 3.0)
        self.assertEqual(landscape.patient_age_years("An 18 month old infant"), 1.5)
        self.assertIsNone(landscape.patient_age_years("How to treat x in children?"))

    def test_a_label_for_older_children_gets_a_note_for_a_younger_patient(self):
        self.assertIn("labelled for ages 5 to 17 years; this patient is about 3", landscape.age_note(LABEL, 3.0))
        self.assertEqual(landscape.age_note(LABEL, 10.0), "")
        self.assertEqual(landscape.age_note(LABEL, None), "")

    def test_the_note_appears_in_the_drug_table(self):
        self.assertIn("labelled for ages 5 to 17 years", landscape.top_drugs_markdown(state(patient_age_years=3.0)))
        self.assertNotIn("labelled for ages", landscape.top_drugs_markdown(state(patient_age_years=10.0)))

    def test_an_adult_label_range_is_read(self):
        self.assertEqual(landscape.label_age_range("for patients 18 years and older"), (18.0, 120.0))
        self.assertIn("18 years and older", landscape.age_note({"indication_excerpt": "for patients 18 years and older"}, 3.0))


class PlanCallModel(unittest.TestCase):
    def test_the_plan_call_can_use_its_own_provider_and_model(self):
        settings = {"MODEL_PROVIDER": "mistral", "MISTRAL_MODEL": "open-mistral-nemo", "PLANNER_PROVIDER": "groq", "PLANNER_MODEL": "some/model"}
        with mock.patch.dict(os.environ, settings, clear=False), mock.patch.object(qa_stream.server, "build_model", side_effect=lambda **kw: kw) as build:
            old = qa_stream._models
            qa_stream._models = {}
            try:
                planning = qa_stream.model_for(300, 0.0, "planning")
                writer = qa_stream.model_for(300, None, "writer")
            finally:
                qa_stream._models = old
        self.assertEqual((planning["provider"], planning["model_id"]), ("groq", "some/model"))
        self.assertEqual((writer["provider"], writer["model_id"]), (None, "open-mistral-nemo"))

    def test_without_a_planner_setting_nothing_changes(self):
        with mock.patch.dict(os.environ, {"PLANNER_PROVIDER": "", "PLANNER_MODEL": ""}, clear=False):
            self.assertEqual(qa_stream.planner_choice(), (None, None))


if __name__ == "__main__":
    unittest.main()
