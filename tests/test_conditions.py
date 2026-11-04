import unittest

import conditions as c
import landscape

Q = "What are the drug treatments for depression and for autism in children?"
RAW = [{"name": "depression", "aliases": ["major depressive disorder"]}, {"name": "autism", "aliases": ["ASD", "autism spectrum disorder"]}]


class CleanConditions(unittest.TestCase):
    def test_two_named_conditions_kept(self):
        g = c.clean_conditions(RAW, Q)
        self.assertEqual(c.names(g), ["depression", "autism"])

    def test_one_condition_gives_nothing(self):
        self.assertEqual(c.clean_conditions([{"name": "depression"}], Q), [])
        self.assertEqual(c.clean_conditions([], Q), [])
        self.assertEqual(c.clean_conditions(None, Q), [])

    def test_invented_condition_is_dropped(self):
        self.assertEqual(c.clean_conditions(RAW + [{"name": "epilepsy"}], Q) and c.names(c.clean_conditions(RAW + [{"name": "epilepsy"}], Q)), ["depression", "autism"])
        self.assertEqual(c.clean_conditions([{"name": "depression"}, {"name": "epilepsy"}], Q), [])

    def test_same_condition_twice_counts_once(self):
        self.assertEqual(c.clean_conditions([{"name": "depression"}, {"name": "depression in children"}], Q), [])

    def test_three_conditions(self):
        q = "treatments for depression, autism and epilepsy in children"
        g = c.clean_conditions(["depression", "autism", "epilepsy"], q)
        self.assertEqual(len(g), 3)

    def test_plain_strings_work(self):
        self.assertEqual(len(c.clean_conditions(["depression", "autism"], Q)), 2)


class Tagging(unittest.TestCase):
    def setUp(self):
        self.g = c.clean_conditions(RAW, Q)

    def test_tags(self):
        self.assertEqual(c.tag_label("irritability associated with autistic disorder", self.g), "autism")
        self.assertEqual(c.tag_label("adjunct for major depressive disorder", self.g), "depression")
        self.assertEqual(c.tag_label("schizophrenia, autism irritability, depression adjunct", self.g), "depression / autism")
        self.assertEqual(c.tag_label("ketamine for cesarean", self.g), "Not specified")

    def test_abbreviation_alias(self):
        self.assertEqual(c.tag_label("children with ASD", self.g), "autism")

    def test_interleave_keeps_every_condition_under_a_cap(self):
        items = ["autism a", "autism b", "autism c", "depression x", "depression y"]
        out = c.interleave(items, lambda s: s, self.g)[:4]
        self.assertTrue(any("depression" in s for s in out))
        self.assertEqual(out[0], "depression x")


class Rendering(unittest.TestCase):
    def state(self, groups=True):
        reg = [{"generic": "drugone", "brands": ["Brand1"], "model_relevant": True, "indication_excerpt": "treatment of irritability associated with autistic disorder", "url": "http://x/1", "label_date": "2026-01-01"},
               {"generic": "drugtwo", "brands": [], "model_relevant": True, "indication_excerpt": "treatment of major depressive disorder in adolescents", "url": "http://x/2", "label_date": "2026-01-01"}]
        return {"regulatory": reg, "conditions": c.clean_conditions(RAW, Q) if groups else [], "source_status": {}, "searched_on": "2026-11-03", "regulator_items": [],
                "trials_by_id": {}, "drug_labels": [{"drug": "drugtwo", "url": "http://x/2", "boxed_warning": "SUICIDAL THOUGHTS in children"}]}

    def test_drug_table_has_condition_column_and_both_conditions(self):
        md = landscape.landscape_markdown(self.state(), [], [])
        self.assertIn("| Condition | Drug |", md)
        self.assertIn("depression", md)
        self.assertIn("autism", md)
        self.assertLess(md.index("depression |"), md.index("autism |"))      # grouped in the order the question names them
        self.assertIn("names 2 conditions", md)

    def test_safety_drug_carries_its_condition(self):
        md = landscape.landscape_markdown(self.state(), [], [])
        self.assertIn("(depression)", md)

    def test_single_condition_is_unchanged(self):
        md = landscape.landscape_markdown(self.state(groups=False), [], [])
        self.assertNotIn("| Condition |", md)
        self.assertNotIn("names 2 conditions", md)
        self.assertIn("| Drug | Indication | Dose / route (if sourced) | Key evidence |", md)

    def test_top_drugs_box_shows_both(self):
        md = landscape.top_drugs_markdown(self.state())
        self.assertIn("| Condition | Drug |", md)
        self.assertIn("depression", md)
        self.assertIn("autism", md)

    def test_unrelated_regulator_item_hidden(self):
        st = self.state()
        st["regulator_items"] = [{"date": "June 30, 2025", "type": "Safety communication", "relation": "risk_or_safety_related", "title": "Stimulant weight loss notice", "url": "http://r/1", "quote": "weight loss in young children", "drug": "x"},
                                 {"date": "September 22, 2025", "type": "Approval", "relation": "treats_condition", "title": "Treatment for autism symptoms", "url": "http://r/2", "quote": "autism treatment approved", "drug": "y"}]
        md = landscape.regulatory_updates_markdown(st)
        self.assertIn("r/2", md)
        self.assertNotIn("r/1", md)
        self.assertIn("none of the question's conditions", md)

    def test_trials_grouped_and_unrelated_hidden(self):
        t = lambda n, conds, title: {"nct_id": n, "url": "http://t/" + n, "conditions": conds, "title": title, "interventions": ["Drug"], "phase": ["PHASE2"], "recruitment_status": "RECRUITING",
                                     "last_update": "2026-10-01", "triage": {"relevance": "direct", "kind": "treatment_trial"}}
        st = self.state()
        st["trials_by_id"] = {"NCT1": t("NCT1", ["Autism"], "autism drug"), "NCT2": t("NCT2", ["Depression"], "depression drug"), "NCT3": t("NCT3", ["Cesarean pain"], "analgesia")}
        md = landscape.trials_markdown(st)
        self.assertIn("| Condition | Trial |", md)
        self.assertNotIn("NCT3", md)
        self.assertLess(md.index("NCT2"), md.index("NCT1"))


class SearchWiring(unittest.TestCase):
    def test_each_condition_gets_its_own_searches_only_when_several(self):
        import qa_stream as q
        import retrieval_routing as rr
        g = c.clean_conditions(RAW, Q)
        ops = [rr.Operation(id="a", need_id="a", need="n", target="trials", args={"condition": "autism children"}),
               rr.Operation(id="b", need_id="b", need="n", target="fda_label", args={"condition_terms": ["autis"]})]
        new = q.condition_operations({"conditions": g}, ops)
        self.assertEqual(sorted(o.target for o in new), ["fda_label", "trials"])      # only the uncovered condition is added
        self.assertEqual(q.condition_operations({"conditions": []}, ops), [])

    def test_fda_labels_of_every_condition_are_read(self):
        import qa_stream as q
        g = c.clean_conditions(RAW, Q)
        recs = [{"generic": f"a{i}", "indication_excerpt": "autism irritability", "set_id": str(i)} for i in range(14)] + \
               [{"generic": f"d{i}", "indication_excerpt": "major depressive disorder", "set_id": f"d{i}"} for i in range(3)]
        seen = []
        original = q.ask_json
        q.ask_json = lambda question, task, payload, mt, skill="x", purpose="t": (seen.append([f["id"] for f in payload["fda_labels"]]) or {"fda_labels": []})
        try:
            q.interpret_fda_labels(Q, recs, g)
            q.interpret_fda_labels(Q, recs, None)
        finally:
            q.ask_json = original
        self.assertTrue(any(":d0:" in i for i in seen[0]))
        self.assertEqual(len(seen[-1]), 17)      # one condition: the cap is 20 (it was 12), so the labels of the medicines the sources describe are read as well
        self.assertLessEqual(len(seen[-1]), 20)


if __name__ == "__main__":
    unittest.main()
