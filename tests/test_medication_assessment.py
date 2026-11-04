import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import clinical_tools
import landscape
import medication_assessment as ma
import qa_stream
import verification

CHILD = "Three year old child has global developmental delay, regression of milestones and autistic tendencies. What could be the work-up and management?"
CORE = "Medications do not treat the core features of the condition."
COMORBID = "Risperidone is used for irritability in children with the condition."
LABEL = {"generic": "Drugx", "model_relevant": True, "model_relation": "treats_associated_symptom_or_comorbidity", "url": "https://example.org/label", "label_date": "2026-01-01",
         "indication_excerpt": "Drugx is indicated for irritability associated with condition Z in children ages 5 to 17 years.", "route": "Oral",
         "dose_statement": "Start 0.25 mg once daily. Do not exceed 3 mg per day.", "label_extras": {"max_dose": "Do not exceed 3 mg per day.", "adverse_effects": "Somnolence was common."}}
WEIGHT_LABEL = {**LABEL, "generic": "Drugw", "dose_statement": "Give 0.5 mg/kg once daily in two divided doses.", "label_extras": {}}


def paper(pmid="111111", abstract=f"{CORE} {COMORBID}"):
    return {"pmid": pmid, "title": "Review", "abstract": abstract, "cite": "Smith 2024 [x]", "year": 2024, "journal": "J", "date": "2024", "authors": ["Smith A"], "doi": None}


def state(**extra):
    s = qa_stream.new_state(CHILD)
    s.update(papers={"111111": paper()}, regulatory=[], web=[], guidelines=[], source_status={})
    s.update(extra)
    return s


def pick(role="core_no_drug", n=1, source="111111"):
    return {"id": source, "n": n, "role": role}


class Picks(unittest.TestCase):
    def test_the_sentence_shown_is_the_sources_own_never_the_models(self):
        pool = ma.source_pool(state())
        picks = ma.verify_picks([{**pick(), "sentence": "Medicines cure everything."}], pool)
        self.assertEqual([p["sentence"] for p in picks], [CORE])
        self.assertEqual(picks[0]["source"], "PMID 111111 (Smith 2024)")

    def test_a_pick_with_an_unknown_source_number_or_role_is_dropped(self):
        pool = ma.source_pool(state())
        bad = [pick(source="999999"), pick(n=9), pick(n=0), pick(role="invented"), {"id": "111111", "n": "x", "role": "core_no_drug"}, "text", None]
        self.assertEqual(ma.verify_picks(bad, pool), [])

    def test_a_no_medicine_pick_must_really_say_so(self):
        s = state(papers={"111111": paper(abstract="Early therapy improves language outcomes in many children.")})
        self.assertEqual(ma.verify_picks([pick()], ma.source_pool(s)), [])

    def test_a_model_failure_is_reported_and_picks_nothing(self):
        def boom(*a):
            raise RuntimeError("limit")
        picks, error = ma.pick_statements(CHILD, state(), boom)
        self.assertEqual(picks, [])
        self.assertIn("RuntimeError", error)

    def test_retracted_papers_are_never_offered(self):
        s = state(papers={"111111": {**paper(), "retracted": True}})
        self.assertEqual(ma.source_pool(s), {})


class Outcomes(unittest.TestCase):
    def verified(self, s, picks=()):
        return ma.verify_picks(list(picks), ma.source_pool(s))

    def test_1_a_work_up_question_still_gets_the_section_with_an_explicit_outcome(self):
        a = ma.assess(state(), CHILD, [], [], [])
        self.assertIn(a["outcome"], ma.OUTCOMES)
        md = ma.markdown(a)
        self.assertTrue(ma.has_section(md))
        self.assertIn("**Outcome:", md)

    def test_2_non_drug_management_ends_in_no_specific_medication_with_the_quoted_source(self):
        s = state()
        a = ma.assess(s, CHILD, [], self.verified(s, [pick()]), [])
        self.assertEqual(a["outcome"], "none_indicated")
        md = ma.markdown(a)
        self.assertIn(CORE, md)
        self.assertIn("PMID 111111", md)
        self.assertNotIn("Drugx", md)

    def test_3_an_established_medicine_is_listed_with_fields_copied_from_the_label(self):
        s = state(regulatory=[LABEL])
        a = ma.assess(s, "A 12-year-old child, weight 40 kg, has condition Z. Which medicine?", [], [], [])
        self.assertEqual(a["outcome"], "identified")
        entry = a["entries"][0]
        self.assertEqual(entry["max_dose"], "Do not exceed 3 mg per day.")
        self.assertEqual(entry["route"], "Oral")
        self.assertEqual(entry["frequency"], "Start 0.25 mg once daily.")
        self.assertEqual(entry["contraindications"], "")      # the label gave none: never filled in
        md = ma.markdown(a, primary=True)
        self.assertIn("| Maximum dose | Do not exceed 3 mg per day. |", md)
        self.assertIn("Not stated in the retrieved sources:", md)
        self.assertIn("contraindications", md.split("Not stated in the retrieved sources:")[1])
        self.assertIn("FDA-labelled", md)

    def test_4_an_associated_symptom_medicine_is_shown_as_conditional_and_not_as_core_treatment(self):
        s = state()
        row = {"drug": "Risperidone", "purpose": "irritability", "population": "children", "dose": "", "label": "PMID 111111 (Smith 2024)", "category": "Guideline", "relevance": "treats_associated_symptom_or_comorbidity"}
        a = ma.assess(s, CHILD, [row], self.verified(s, [pick(), pick("comorbidity_drug", 2)]), [])
        md = ma.markdown(a)
        self.assertEqual(a["outcome"], "identified")
        self.assertIn("Associated problems (if present in this patient)", md)
        self.assertIn("No FDA-labelled indication for this condition", md)
        self.assertIn("Medicines and the core problem", md)

    def test_5_a_child_with_weight_based_dosing_and_no_weight_needs_more_information(self):
        s = state(regulatory=[WEIGHT_LABEL])
        a = ma.assess(s, CHILD, [], [], [])
        self.assertEqual(a["outcome"], "need_info")
        self.assertTrue(any("weight" in n for n in a["needs"]))
        self.assertIn("No dose is recommended for this patient here", ma.markdown(a))
        known = ma.assess(s, CHILD + " Weight 14 kg.", [], [], [])
        self.assertEqual(known["outcome"], "identified")
        self.assertFalse(any("weight" in n for n in known["needs"]))

    def test_6_failed_medicine_sources_never_become_no_medicine_exists(self):
        s = state(source_status={"fda_label": {"retrieval_status": "failed", "evidence_status": "unknown", "error": "timeout"}})
        a = ma.assess(s, CHILD, [], self.verified(s, [pick()]), ["FDA labels: the interpretation did not complete (x)"])
        self.assertEqual(a["outcome"], "insufficient")
        md = ma.markdown(a)
        self.assertIn("not evidence that no medicine exists", md)
        self.assertIn("What could not be verified", md)
        self.assertIn(CORE, md)      # what a source did say is still shown

    def test_7_nothing_retrieved_is_insufficient_not_none_indicated(self):
        a = ma.assess(state(papers={}), CHILD, [], [], [])
        self.assertEqual(a["outcome"], "insufficient")
        self.assertNotEqual(a["outcome"], "none_indicated")

    def test_8_a_drug_question_gets_the_detail_fields_and_a_dose_only_from_a_source(self):
        s = state(regulatory=[LABEL])
        row = {"drug": "Othermed", "purpose": "condition Z", "population": "not stated", "dose": "", "label": "PMID 111111 (Smith 2024)", "category": "Review", "relevance": "treats_condition"}
        a = ma.assess(s, "Which medicines treat condition Z in a 10 year old weighing 30 kg, with doses?", [row], [], [])
        other = [e for e in a["entries"] if e["drug"] == "Othermed"][0]
        self.assertEqual(other["dose"], "")      # no source gave a dose: none is written
        md = ma.markdown(a, primary=True)
        self.assertIn("#### Othermed", md)
        self.assertNotIn("| Dose |", md.split("#### Othermed")[1])      # the row is left out and the field is listed as not stated
        self.assertIn("dose", md.split("#### Othermed")[1].split("Not stated in the retrieved sources:")[1])

    def test_a_crashed_assessment_still_prints_a_section_that_proves_nothing(self):
        md = ma.markdown(ma.fallback("the medication assessment did not complete (KeyError)"))
        self.assertTrue(ma.has_section(md))
        self.assertIn("not evidence that no medicine exists", md)

    def test_the_wording_does_not_trip_the_no_drugs_contradiction_check(self):
        for outcome in ("none_indicated", "insufficient"):
            a = {"outcome": outcome, "label": ma.OUTCOMES[outcome], "entries": [], "picks": [], "facts": {"child": True}, "needs": [], "incomplete": outcome == "insufficient", "failures": [], "sources_read": 0}
            st = state(regulatory=[LABEL])
            kinds = {i["kind"] for i in verification.find_issues(ma.markdown(a), st)}
            self.assertNotIn("contradiction_no_drugs", kinds)


class LabelExtras(unittest.TestCase):
    LABEL = {"dosage_and_administration": ["Start 0.25 mg daily. Do not exceed 3 mg per day. Reassess the need for treatment every 6 months. Monitor weight every 4 weeks."],
             "contraindications": ["Known hypersensitivity to the drug."], "adverse_reactions": ["The most common adverse reactions were somnolence and weight gain."],
             "drug_interactions": ["Strong enzyme inhibitors increase exposure to the drug."], "warnings_and_cautions": ["Monitor for metabolic changes during treatment."]}

    def test_each_field_is_a_verbatim_label_sentence(self):
        extras = clinical_tools.label_extras(self.LABEL)
        self.assertEqual(extras["max_dose"], "Do not exceed 3 mg per day.")
        self.assertEqual(extras["duration"], "Reassess the need for treatment every 6 months.")
        self.assertEqual(extras["contraindications"], "Known hypersensitivity to the drug.")
        self.assertIn("somnolence", extras["adverse_effects"])
        self.assertIn("enzyme inhibitors", extras["interactions"])
        self.assertIn("Monitor", extras["monitoring"])

    def test_a_section_the_label_does_not_have_stays_empty(self):
        extras = clinical_tools.label_extras({"dosage_and_administration": ["Take one tablet."]})
        self.assertIsNone(extras["contraindications"])
        self.assertIsNone(extras["max_dose"])
        self.assertIsNone(extras["duration"])


class Mandatory(unittest.TestCase):
    def test_the_drug_jobs_stay_for_a_question_that_does_not_ask_about_medicines(self):
        plan = {"condition": "x", "trial_condition": "x", "trial_intervention": "", "drugs": [], "intent": "diagnosis", "skills": ["latest-evidence"],
                "understanding": {"info_needed": ["diagnosis"], "medicines_needed": False}}
        kept, _ = qa_stream.legacy_jobs_for_intent(plan, qa_stream.deep_jobs(dict(plan), "What is the diagnostic approach for x?"))
        self.assertTrue(qa_stream.LEGACY_BY_NEED["drugs"] & set(kept))

    def test_the_drug_skill_is_always_asked_to_write_searches(self):
        calls = []

        def ask(question, task, payload, max_tokens, skill="drug-intelligence"):
            calls.append(skill)
            return {"queries": []}
        qa_stream.route_deep("What is the diagnostic approach for x in children?", {"skills": ["latest-evidence"], "understanding": {"info_needed": ["diagnosis"], "medicines_needed": False},
                                                                                 "entities": {}, "condition": "x", "intent": "diagnosis"}, ask=ask)
        self.assertIn("drug-intelligence", calls)

    def test_the_section_survives_verification_and_numbering(self):
        import presentation
        s = state(regulatory=[LABEL])
        a = ma.assess(s, CHILD, [], ma.verify_picks([pick()], ma.source_pool(s)), [])
        text = "## Clinical summary\n\n**Bottom line** A work-up is needed (PMID 111111).\n\n" + ma.markdown(a) + "\n\n"
        answer = verification.verify_and_repair(text, s)["answer"]
        self.assertTrue(ma.has_section(answer))
        self.assertIn("Maximum dose", answer)
        numbered, _ = presentation.numberize(answer, s["papers"])
        self.assertTrue(ma.has_section(presentation.hide_unneeded_sections(numbered)))
        self.assertIn(CORE, numbered)


class AlwaysOnSections(unittest.TestCase):
    """Drugs and clinical trials follow the same rule: searched and shown for every question, limited to what matches the condition, with a plain line when nothing was found."""

    def test_the_trial_skill_is_always_asked_to_write_searches(self):
        calls = []

        def ask(question, task, payload, max_tokens, skill="x"):
            calls.append(skill)
            return {"queries": []}
        qa_stream.route_deep("What is the diagnostic approach for x in children?", {"skills": ["latest-evidence"], "understanding": {"info_needed": ["diagnosis"], "medicines_needed": False},
                                                                                 "entities": {}, "condition": "x", "intent": "diagnosis"}, ask=ask)
        self.assertIn("clinical-trials", calls)
        self.assertIn("drug-intelligence", calls)

    def test_the_trial_jobs_stay_for_a_work_up_question(self):
        plan = {"condition": "x", "trial_condition": "x", "trial_intervention": "", "drugs": [], "intent": "diagnosis", "skills": ["latest-evidence"],
                "understanding": {"info_needed": ["diagnosis"], "medicines_needed": False}}
        kept, _ = qa_stream.legacy_jobs_for_intent(plan, qa_stream.deep_jobs(dict(plan), "What is the diagnostic approach for x?"))
        self.assertTrue(qa_stream.LEGACY_BY_NEED["trials"] & set(kept))

    def test_no_trial_found_is_stated_plainly_never_hidden_as_not_applicable(self):
        s = state(section_states={"trials": {"state": "not_applicable"}}, understanding={"info_needed": ["diagnosis"]})
        s["source_status"]["trials"] = {"retrieval_status": "success", "evidence_status": "not_found_after_search", "error": None}
        md = landscape.trials_markdown(s)
        self.assertIn("returned no trial records", md)
        self.assertNotIn("Not applicable", md)
        import presentation
        self.assertIn("## Clinical trials", presentation.hide_unneeded_sections(md))

    def test_the_drug_table_is_not_replaced_by_a_not_listed_line(self):
        s = state(regulatory=[LABEL], medicines_needed=False)
        self.assertIn("Drugx", landscape.landscape_markdown(s, [], []))
        self.assertNotIn("not listed", landscape.top_drugs_markdown(s).lower())


if __name__ == "__main__":
    unittest.main()
