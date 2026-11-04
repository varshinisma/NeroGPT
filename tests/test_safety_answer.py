import unittest

import safety_answer as sa


def state():
    return {"regulatory": [{"generic": "Topiramate", "brands": ["Topamax"], "url": "https://dailymed.example/x", "safety_text": {"boxed": None, "adverse": "The most common adverse reactions (incidence 5% or more) were paraesthesia, fatigue and anorexia.",
                                                                                                          "warnings": "Acute myopia and secondary angle closure glaucoma have been reported.", "contraindications": "None.", "interactions": None}, "label_extras": {}},
                           {"generic": "Melatonin", "brands": ["Melatonin"], "indication_excerpt": "homeopathic principles", "safety_text": {"adverse": "x" * 40}, "label_extras": {}}],
            "papers": {"1": {"pmid": "111", "cite": "Smith et al., J Neurol, 2024 [PMID 111]", "title": "Topiramate adverse effects in children", "abstract": "Cognitive slowing occurred in 12% of children.", "year": "2024"}},
            "web": []}


class SafetyAnswer(unittest.TestCase):
    def ask_with(self, rows):
        return lambda task, payload: {"rows": rows}

    def test_the_model_cells_are_kept_when_every_number_is_in_the_drugs_own_sources(self):
        rows = [{"drug": "Topiramate", "common": "paraesthesia, fatigue, anorexia (5% or more)", "serious": "angle closure glaucoma; cognitive slowing in 12% of children", "avoid_or_monitor": "not stated in the retrieved sources",
                 "interactions": "not stated in the retrieved sources", "sources": ["label", "PMID 111"]}]
        md = sa.markdown("what are the side effects?", ["Topiramate"], state(), self.ask_with(rows))
        self.assertIn("paraesthesia, fatigue, anorexia (5% or more)", md)
        self.assertIn("[FDA label](https://dailymed.example/x)", md)
        self.assertIn("(Smith et al., J Neurol, 2024 [PMID 111])", md)

    def test_a_number_no_source_states_is_replaced_by_the_labels_own_words(self):
        rows = [{"drug": "Topiramate", "common": "fatigue in 40% of patients", "serious": "glaucoma", "avoid_or_monitor": "", "interactions": "", "sources": ["label"]}]
        st = state()
        md = sa.markdown("what are the side effects?", ["Topiramate"], st, self.ask_with(rows))
        self.assertNotIn("40%", md)
        self.assertIn("most common adverse reactions", md.lower())
        self.assertEqual(st["safety_answer"]["cells_replaced_by_label_text"], {"Topiramate": ["common"]})

    def test_a_paper_the_model_cites_but_was_not_offered_is_not_shown(self):
        rows = [{"drug": "Topiramate", "common": "fatigue", "serious": "glaucoma", "avoid_or_monitor": "", "interactions": "", "sources": ["PMID 999", "label"]}]
        self.assertNotIn("999", sa.markdown("q", ["Topiramate"], state(), self.ask_with(rows)))

    def test_a_drug_with_no_source_says_so_whatever_the_model_wrote(self):
        rows = [{"drug": "Valerian", "common": "headache and dizziness", "serious": "liver damage", "avoid_or_monitor": "pregnancy", "interactions": "none", "sources": []}]
        md = sa.markdown("q", ["Valerian"], state(), self.ask_with(rows))
        self.assertNotIn("liver damage", md)
        self.assertIn("not stated in the retrieved sources", md)
        self.assertIn("none retrieved", md)

    def test_a_failed_model_call_leaves_the_labels_own_words(self):
        def broken(task, payload):
            raise RuntimeError("down")
        md = sa.markdown("q", ["Topiramate"], state(), broken)
        self.assertIn("paraesthesia", md)

    def test_a_homeopathic_label_is_not_used_as_the_drugs_label(self):
        self.assertEqual(sa.label_views(state(), "Melatonin", "q"), [])

    def test_interactions_get_a_column_only_when_asked(self):
        self.assertNotIn("| Interactions |", sa.markdown("q", ["Topiramate"], state(), self.ask_with([])))
        self.assertIn("| Interactions |", sa.markdown("q", ["Topiramate"], state(), self.ask_with([]), want_interactions=True))


if __name__ == "__main__":
    unittest.main()


class DrugFacts(unittest.TestCase):
    def state(self):
        return {"regulatory": [{"generic": "Alphadrug", "brands": ["Alphabrand"], "url": "https://dailymed.example/x", "label_sections": {"contraindications": "Contraindicated in patients with a history of pancreatitis.", "dosage": "Start 0.25 mg weekly."}}],
                "papers": {}, "web": []}

    def test_a_table_of_exactly_the_facts_asked(self):
        import safety_answer as sa
        answer = {"rows": [{"drug": "Alphadrug", "cells": {"contraindications": "history of pancreatitis"}, "sources": ["label"]}]}
        md = sa.facts_markdown("What are the contraindications?", ["Alphadrug"], self.state(), lambda task, payload: answer, ["contraindications"])
        self.assertIn("| Drug | Contraindications | Source |", md)
        self.assertIn("history of pancreatitis", md)
        self.assertNotIn("Common side effects", md)
        self.assertNotIn("Dose", md)
        self.assertIn("[FDA label](https://dailymed.example/x)", md)

    def test_a_number_that_no_source_states_is_not_shown(self):
        import safety_answer as sa
        answer = {"rows": [{"drug": "Alphadrug", "cells": {"contraindications": "not in 99 patients"}, "sources": ["label"]}]}
        md = sa.facts_markdown("q", ["Alphadrug"], self.state(), lambda task, payload: answer, ["contraindications"])
        self.assertNotIn("99", md)
        self.assertIn("not stated in the retrieved sources", md)

    def test_several_facts_become_several_columns(self):
        import safety_answer as sa
        md = sa.facts_markdown("q", ["Alphadrug"], self.state(), lambda task, payload: {"rows": []}, ["use in pregnancy", "what to monitor"])
        self.assertIn("| Drug | Use in pregnancy | What to monitor | Source |", md)
