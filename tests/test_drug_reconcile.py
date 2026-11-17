"""Offline tests of the drug identity, classification and table logic (drug_reconcile.py, landscape.py). The drug names below are test data: the code under test holds no drug list.
Run:  python -m unittest tests.test_drug_reconcile   (from the NeroGPT folder)"""
import os
import sys
import unittest
import unittest.mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import drug_reconcile as dr   # noqa: E402
import landscape               # noqa: E402


def label(generic, brands=(), indication="", kind=None, relevant=True, klass=None, date="2024-01-02", url=None):
    return {"generic": generic, "brands": list(brands), "jurisdiction": "FDA", "indication_excerpt": indication, "indication_statement": indication, "model_evidence": indication,
            "model_relevant": relevant, "dose_statement": None, "dose_general": None, "route": None, "label_extras": {}, "safety_text": None, "label_date": date,
            "use_kind": kind or dr.use_kind(indication), "drug_class": klass, "url": url or f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={generic[:6]}", "set_id": generic[:6]}


def base_state(regulatory, question="What drugs prevent migraine?"):
    return {"query": question, "regulatory": regulatory, "source_status": {}, "entities": {}, "core_stems": ["migrain"], "trials_by_id": {}, "regulator_items": [], "fda_terms": ["migraine"]}


class Identity(unittest.TestCase):
    def test_brand_generic_salt_form_and_biologic_suffix_are_one_drug(self):
        a = dr.Aliases.from_records([label("Galcanezumab-gnlm", ["Emgality"]), label("Eptinezumab-jjmr", ["Vyepti"])])
        self.assertEqual(a.of("Emgality"), a.of("galcanezumab"))
        self.assertEqual(a.of("Vyepti (eptinezumab-jjmr)"), a.of("Eptinezumab"))
        self.assertEqual(dr.key("Divalproex Sodium Delayed-Release Tablets"), dr.key("divalproex"))
        self.assertNotEqual(a.of("Emgality"), a.of("Vyepti"))
        self.assertEqual(dr.key("Valsartan, Amlodipine"), dr.key("amlodipine + valsartan"))      # a combination's ingredients, in any order

    def test_a_name_no_record_connects_stays_its_own_drug(self):
        a = dr.Aliases.from_records([label("Galcanezumab-gnlm", ["Emgality"])])
        self.assertNotEqual(a.of("Aimovig"), a.of("galcanezumab"))      # nothing is guessed

    def test_lookup_results_connect_a_brand_to_its_generic(self):
        a = dr.Aliases.from_records([], labels=[{"generic": "Aimovig", "generic_name": "erenumab-aooe", "brand": "AIMOVIG"}])
        self.assertEqual(a.of("Aimovig"), a.of("erenumab"))

    def test_several_labels_of_one_drug_merge(self):
        recs = [label("Divalproex Sodium", [], "prophylaxis of migraine headaches"), label("Divalproex Sodium Delayed-release Tablets", ["Depakote"], "prophylaxis of migraine headaches", url="https://x/2"),
                label("Galcanezumab-gnlm", ["Emgality"], "preventive treatment of migraine")]
        merged = dr.merge_records(recs, dr.Aliases.from_records(recs))
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["brands"], ["Depakote"])
        self.assertEqual(len(merged[0]["other_labels"]), 1)


class UseKind(unittest.TestCase):
    def test_kinds_are_read_from_the_sentences_naming_the_condition(self):
        self.assertEqual(dr.use_kind("indicated for the preventive treatment of migraine in adults.", ["migrain"]), "preventive")
        self.assertEqual(dr.use_kind("indicated for the acute treatment of migraine with or without aura.", ["migrain"]), "acute")
        self.assertEqual(dr.use_kind("indicated for the acute treatment of migraine and the preventive treatment of episodic migraine.", ["migrain"]), "both")
        self.assertEqual(dr.use_kind("indicated for epilepsy. Indicated for the acute treatment of cluster headache. Indicated for prophylaxis of migraine.", ["migrain"]), "preventive")      # other conditions' sentences do not count
        self.assertEqual(dr.use_kind("indicated for migraine.", ["migrain"]), "not stated")

    def test_the_question_says_what_it_asks(self):
        self.assertEqual(dr.asks_intent("What drugs are used for migraine prophylaxis?"), "preventive")
        self.assertEqual(dr.asks_intent("what stops an acute migraine attack"), "acute")
        self.assertIsNone(dr.asks_intent("how is migraine treated"))
        self.assertTrue(dr.in_scope("both", "preventive"))
        self.assertFalse(dr.in_scope("acute", "preventive"))
        self.assertTrue(dr.in_scope("not stated", "preventive"))


class Tables(unittest.TestCase):
    def regs(self):
        return [label("Eptinezumab-jjmr", ["Vyepti"], "Vyepti is indicated for the preventive treatment of migraine in adults.", klass="Calcitonin Gene-related Peptide Receptor Antagonist"),
                label("Galcanezumab-gnlm", ["Emgality"], "Emgality is indicated for the preventive treatment of migraine in adults.", klass="CGRP Antagonist"),
                label("Atogepant", ["Qulipta"], "Atogepant is indicated for the preventive treatment of migraine in adults."),
                label("Zolmitriptan", ["Zomig"], "Zolmitriptan is indicated for the acute treatment of migraine with or without aura in adults."),
                label("Rimegepant", ["Nurtec"], "Rimegepant is indicated for the acute treatment of migraine and the preventive treatment of episodic migraine in adults."),
                label("Fifth Preventive", [], "Fifth is indicated for prophylaxis of migraine headaches.")]

    def test_every_labelled_drug_is_in_the_top_list_and_an_acute_only_drug_is_not(self):
        text = landscape.top_drugs_markdown(base_state(self.regs()))
        for name in ("Eptinezumab", "Galcanezumab", "Atogepant", "Rimegepant", "Fifth Preventive"):      # the fifth was cut by a limit of four
            self.assertIn(name, text)
        self.assertNotIn("Zolmitriptan", text)

    def test_the_approved_drug_is_not_called_investigational_under_another_name(self):
        state = base_state(self.regs())
        state["trials_by_id"] = {"NCT1": {"nct_id": "NCT1", "phase": ["PHASE3"], "last_update": "2026-05-01", "triage": {"kind": "treatment_trial", "relevance": "direct"},
                                            "drug_interventions": ["Eptinezumab"], "recruitment_status": "RECRUITING", "url": "u", "conditions": ["migraine"], "title": "t"},
                                 "NCT2": {"nct_id": "NCT2", "phase": ["PHASE3"], "last_update": "2026-01-01", "triage": {"kind": "treatment_trial", "relevance": "direct"},
                                            "drug_interventions": ["Newdrug"], "recruitment_status": "RECRUITING", "url": "u", "conditions": ["migraine"], "title": "t"}}
        self.assertEqual(landscape.notable_latest_drug(state)["drug"], "Newdrug")      # the newer trial is of a drug with a label: skipped
        state["trials_by_id"].pop("NCT2")
        self.assertIsNone(landscape.notable_latest_drug(state))

    def test_main_table_has_one_row_per_drug_and_the_acute_drug_is_in_its_own_section(self):
        recs = self.regs() + [label("Emgality", [], "Emgality is indicated for the preventive treatment of migraine.")]      # the same drug under its brand name
        md = landscape.landscape_markdown(base_state(recs), [], [])
        main, _, background = md.partition("### Other use of these drugs")
        self.assertEqual(main.lower().count("galcanezumab"), 1)
        self.assertEqual(main.count("(Emgality)"), 1)      # the brand is in the one row of the drug
        self.assertIn("Zolmitriptan", background)
        self.assertNotIn("Zolmitriptan", main)
        self.assertIn("Rimegepant", main)      # labelled for both: it stays
        self.assertIn("Preventive and acute", main)
        self.assertIn("Calcitonin Gene-related Peptide Receptor Antagonist", main)      # the label's own class
        self.assertIn("EMA (EU): not verified", main)
        self.assertIn("CDSCO (India): not verified", main)
        for line in main.splitlines():      # every table row is one line with the same number of cells as its header
            if line.startswith("| Drug | Class"):
                cells = line.count("|")
            elif line.startswith("| ") and "---" not in line and "| Drug | Class" not in line and line.count("|") > 3:
                self.assertEqual(line.count("|"), cells, line[:80])

    def test_without_an_intent_nothing_is_separated(self):
        md = landscape.landscape_markdown(base_state(self.regs(), "How is migraine treated?"), [], [])
        self.assertNotIn("### Other use of these drugs", md)
        self.assertIn("Zolmitriptan", md)

    def test_regulator_status_is_separate_for_each_regulator(self):
        recs = self.regs()
        a = dr.Aliases.from_records(recs)
        items = [{"drug": "Atogepant", "url": "https://www.ema.europa.eu/en/medicines/human/EPAR/aquipta", "classification": "approval", "date": "2023-08-11"}]
        status = dr.region_status(a.of("Atogepant"), a, recs[2], items)
        self.assertIn("FDA (US): approved for the use shown", status)
        self.assertIn("EMA (EU): approval (2023-08-11)", status)
        self.assertIn("CDSCO (India): not verified", status)
        self.assertIn("EMA (EU): not verified", dr.region_status(a.of("Vyepti"), a, recs[0], items))      # one regulator's page says nothing about another drug


class UsedRows(unittest.TestCase):
    def offered(self, sentence, kind="web"):
        return {"S1": {"title": "t", "text": sentence, "label": "[t](http://x)", "url": "http://x", "kind": kind, "sentences": [f"1. {sentence}"], "study_type": ["Randomized Controlled Trial"] if kind == "paper" else None}}

    def item(self, drug, **kw):
        return dict({"drug": drug, "drug_kind": "generic", "source_id": "S1", "sentence": 1, "relevance": "treats_condition", "statement_type": "used_in_practice", "category": "Off-label",
                     "indication": "prevent migraine", "population": "adults", "dose_sentence": 0}, **kw)

    def test_a_brand_name_of_an_fda_row_is_a_duplicate_not_an_off_label_row(self):
        state = base_state([label("Galcanezumab-gnlm", ["Emgality"], "Emgality is indicated for the preventive treatment of migraine.")])
        rows, rejected = landscape.verify_drug_decisions([self.item("Emgality")], self.offered("Emgality is used to prevent migraine in adults."), state)
        self.assertEqual(rows, [])
        self.assertIn("duplicate", rejected[0][1])

    def test_a_source_that_only_describes_use_is_evidence_supported_not_off_label(self):
        state = base_state([])
        rows, _ = landscape.verify_drug_decisions([self.item("Somedrug")], self.offered("Somedrug is used to prevent migraine in adults."), state)
        self.assertTrue(rows[0]["category"].startswith("Evidence-supported"))
        self.assertEqual(rows[0]["use_kind"], "preventive")

    def test_a_study_that_reports_a_benefit_is_evidence_and_one_that_reports_none_is_only_studied(self):
        state = base_state([])
        good, _ = landscape.verify_drug_decisions([self.item("Somedrug", statement_type="study_result")], self.offered("Somedrug reduced monthly migraine days versus placebo in adults.", "paper"), state)
        none, _ = landscape.verify_drug_decisions([self.item("Otherdrug", statement_type="study_result")], self.offered("Otherdrug showed no significant difference from placebo for migraine.", "paper"), state)
        self.assertTrue(good[0]["category"].startswith("Evidence-supported"))
        self.assertEqual(good[0]["evidence_kind"], "randomized controlled trial")
        self.assertTrue(none[0]["category"].startswith("Studied"))

    def test_reconcile_merges_brand_and_generic_and_says_off_label_only_from_a_label(self):
        state = base_state([label("Galcanezumab-gnlm", ["Emgality"], "Emgality is indicated for the preventive treatment of migraine."),
                            label("Gabapentin", ["Neurontin"], "Gabapentin is indicated for postherpetic neuralgia.", relevant=False, klass="Anti-epileptic")])
        ev = "Evidence-supported (described as used in practice; FDA label status not verified)"
        rows = [{"drug": "Gabapentin", "category": ev, "label": "[a](u)"}, {"drug": "Emgality (galcanezumab-gnlm)", "category": ev, "label": "[b](u)"}, {"drug": "Newdrug", "category": ev, "label": "[c](u)"},
                {"drug": "Newdrug", "category": ev, "label": "[d](u)"}]
        kept, labels = landscape.reconcile_used_rows(state, rows, [None, None, None, None])
        self.assertEqual([r["drug"] for r in kept], ["Gabapentin", "Newdrug"])
        self.assertEqual(len(labels), 2)
        self.assertTrue(kept[0]["category"].startswith("Off-label"))      # a label exists, and not for migraine
        self.assertEqual(kept[0]["drug_class"], "Anti-epileptic")
        self.assertTrue(kept[1]["category"].startswith("Evidence-supported"))      # no label retrieved: not called off-label
        self.assertEqual(kept[1]["also_cited"], ["[d](u)"])


class Checks(unittest.TestCase):
    def test_validate_reports_each_kind_of_problem(self):
        a = dr.Aliases.from_records([label("Galcanezumab-gnlm", ["Emgality"])])
        rows = [{"drug": "Galcanezumab-gnlm", "category": "FDA-approved", "kind": "preventive", "url": "u"}, {"drug": "Emgality", "category": "Off-label", "kind": "preventive", "source": "s"},
                {"drug": "Zolmitriptan", "category": "FDA-approved", "kind": "acute", "url": "u"}, {"drug": "Nosource", "category": "Evidence-supported", "kind": "preventive"}]
        text = " ".join(dr.validate(rows, a, "preventive", "atogepant reduced migraine days", ["Atogepant"]))
        self.assertIn("listed more than once", text)
        self.assertIn("acute treatment but is listed among the preventive", text)
        self.assertIn("no supporting source", text)
        self.assertIn("Atogepant is discussed in the report but is not in the drug table", text)

    def test_a_row_with_both_names_covers_a_source_that_uses_only_the_brand(self):
        state = base_state([])
        state["used_rows_final"] = [{"drug": "erenumab (Aimovig)", "category": "Evidence-supported", "use_kind": "preventive", "source": "s", "relevance": "treats_condition"}]
        state["used_drugs_trace"] = {"model_items": [{"drug": "Aimovig", "drug_kind": "brand", "relevance": "treats_condition", "statement_type": "used_in_practice"}]}
        self.assertEqual(landscape.drug_checks(state, "Aimovig reduced migraine days."), [])

    def test_a_clean_state_passes(self):
        state = base_state([label("Galcanezumab-gnlm", ["Emgality"], "Emgality is indicated for the preventive treatment of migraine.")])
        self.assertEqual(landscape.drug_checks(state, "Galcanezumab (Emgality) reduced migraine days."), [])


class LabelVerification(unittest.TestCase):
    EV = landscape.USED_CATEGORIES["off-label"]

    def row(self, drug, category=None, label="[a](u)"):
        return {"drug": drug, "category": category or self.EV, "label": label, "excerpt": "a sentence", "purpose": "prevent migraine", "population": "not stated in this sentence", "relevance": "treats_condition",
                "source": "S1", "evidence_kind": "randomized controlled trial", "dose": None}

    def test_the_drugs_own_label_makes_it_approved_and_gives_its_use_type(self):
        state = base_state([])
        found = {"generic": "atogepant", "generic_names": ["ATOGEPANT"], "brands": ["QULIPTA"], "label_date": "2025-04-01", "drug_class": "CGRP receptor antagonist",
                 "indications": "QULIPTA is indicated for the preventive treatment of migraine in adults. Limitations of use: none."}
        kept, _ = landscape.reconcile_used_rows(state, [self.row("atogepant")], [found])
        self.assertEqual(kept[0]["category"], landscape.FDA_APPROVED)
        self.assertEqual(kept[0]["use_kind"], "preventive")
        self.assertEqual(kept[0]["drug_class"], "CGRP receptor antagonist")
        state["used_rows_final"] = kept
        md = landscape.landscape_markdown(state, kept, [])
        self.assertIn("FDA (US): approved for the use shown (label 2025-04-01)", md)
        self.assertIn("EMA (EU): not verified", md)
        self.assertIn("CGRP receptor antagonist", md)

    def test_a_label_that_does_not_name_the_condition_is_off_label_and_a_guideline_stays_a_guideline(self):
        state = base_state([])
        found = {"generic": "x", "generic_names": ["GABAPENTIN"], "brands": [], "label_date": "2024-01-01", "indications": "indicated for postherpetic neuralgia and partial seizures."}
        kept, _ = landscape.reconcile_used_rows(state, [self.row("gabapentin"), self.row("metoprolol", "Guideline-recommended")], [found, found])
        self.assertTrue(kept[0]["category"].startswith("Off-label"))
        self.assertEqual(kept[1]["category"], "Guideline-recommended")
        self.assertIn("indications do not name this condition", kept[1]["fda_note"])

    def test_a_drug_with_no_label_found_is_unverified_not_off_label(self):
        kept, _ = landscape.reconcile_used_rows(base_state([]), [self.row("somedrug")], [None])
        self.assertTrue(kept[0]["category"].startswith("Evidence-supported"))
        self.assertIn("not verified", kept[0]["category"])

    def test_a_label_that_says_not_indicated_does_not_count(self):
        self.assertIsNone(dr.label_names_condition("Drugx is not indicated for the treatment of migraine.", ["migrain"]))
        self.assertIsNotNone(dr.label_names_condition("Drugx is indicated for the prophylaxis of migraine headache.", ["migrain"]))

    def test_the_brand_and_generic_rows_become_one_row_with_both_names(self):
        state = base_state([])
        found_g = {"generic": "erenumab", "generic_names": [], "brands": [], "indications": None}
        found_b = {"generic": "Aimovig", "generic_name": "erenumab-aooe", "brand": "AIMOVIG", "generic_names": ["ERENUMAB-AOOE"], "brands": ["AIMOVIG"], "indications": None}
        kept, labels = landscape.reconcile_used_rows(state, [self.row("erenumab"), self.row("Aimovig (erenumab-aooe)", label="[b](u)")], [found_g, found_b])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["drug"], "erenumab (Aimovig)")
        self.assertEqual(kept[0]["also_cited"], ["[b](u)"])


class LabelLookupByName(unittest.TestCase):
    def test_a_name_with_marks_and_brackets_is_cleaned_and_a_brand_name_is_found(self):
        import asyncio
        import async_tools as at
        asked = []

        async def fake_get(params):
            asked.append(params["search"])
            if params["search"].startswith("openfda.brand_name:"):
                return {"results": [{"set_id": "s1", "effective_time": "20240101", "openfda": {"generic_name": ["FREMANEZUMAB-VFRM"], "brand_name": ["AJOVY"]},
                                     "indications_and_usage": ["AJOVY is indicated for the preventive treatment of migraine in adults."]}]}
            return None
        with unittest.mock.patch.object(at, "_fda_get", fake_get):
            records = asyncio.run(at._fda_records("AJOVY® (fremanezumab-vfrm)", ["migrain"], 1))
        self.assertTrue(all("®" not in q and "(" not in q for q in asked), asked)
        self.assertEqual(records[0]["generic"], "Fremanezumab-Vfrm")      # the label's own generic name
        self.assertEqual(records[0]["use_kind"], "preventive")
        self.assertEqual(dr.search_names("AJOVY® (fremanezumab-vfrm)"), ["fremanezumab-vfrm", "AJOVY"])


if __name__ == "__main__":
    unittest.main()
