"""stream_answer is a set of named stages (stages.py) run in the order the orchestrator agent chose. Run end to end with every search and model call faked."""
import json
import os
import unittest
from unittest import mock

os.environ.setdefault("ORCHESTRATOR_AGENT", "off")
import qa_stream
import stages

PUB = json.dumps({"results": [{"pmid": str(30000000 + i), "title": f"Migraine prevention trial {i}", "year": 2023, "journal": "J Neurol", "authors": ["Smith A"], "study_type": ["Randomized Controlled Trial"],
                               "abstract_excerpt": "Erenumab reduced migraine days.", "publication_date": "2023-01-01", "url": "http://x"} for i in range(5)]})
TRI = json.dumps({"results": [{"nct_id": "NCT00000001", "title": "t", "status": "RECRUITING", "phase": "PHASE3", "conditions": ["Migraine"], "interventions": ["erenumab"]}]})
EMPTY = json.dumps({"results": []})


def fake_jobs(found=True):
    def run_jobs(jobs, timeout, essential=None, extra_until=0.6):
        if not found:
            return {}, {}
        return {k: (PUB if k.startswith("pubmed") else TRI if k.startswith("trials") else EMPTY) for k in jobs}, {}
    return run_jobs


def run(order=None, found=True, skip=None, focus=False):
    plan = {"intent": "drug_treatment", "condition": "migraine"}
    if focus:
        plan.update(focus=focus if isinstance(focus, list) else ["drugs"], drugs=[] if focus == ["drug_discovery"] else ["erenumab"], skills=["drug-intelligence"])
        if isinstance(focus, list) and "drug_facts" in focus:
            plan["attributes"] = ["contraindications"]
    if order is not None:
        plan["stages"] = order
    if skip:
        plan["skip_stages"] = skip

    def fake_llm(*a, **k):
        yield "**Bottom line** Erenumab helps (PMID 30000000)."
    patches = [mock.patch.object(qa_stream, "run_jobs", fake_jobs(found)), mock.patch.object(qa_stream, "stream_llm", fake_llm), mock.patch.object(qa_stream, "run_llm", lambda *a, **k: "{}"),
               mock.patch.object(qa_stream, "fast_drug_decisions", lambda s, q: None), mock.patch.object(qa_stream, "triage_evidence", lambda s, q, k: None),
               mock.patch.object(qa_stream, "run_escalation_phase", lambda *a, **k: {}), mock.patch.object(qa_stream, "hybrid_rerank", lambda *a, **k: None),
               mock.patch.object(qa_stream, "cross_rerank", lambda *a, **k: None), mock.patch.object(qa_stream, "routing_enabled", lambda: False),
               mock.patch.object(qa_stream, "legacy_deep_enabled", lambda: True)]
    for p in patches:
        p.start()
    try:
        return list(qa_stream.stream_answer("What is the treatment of migraine in adults?", plan=plan))
    finally:
        for p in patches:
            p.stop()


class FocusedRun(unittest.TestCase):
    def test_a_focused_follow_up_writes_no_report_sections(self):
        events = run(focus=True)
        text = "".join(e.get("text", "") for e in events if e["type"] == "token")
        final = events[-1]
        self.assertEqual(final["type"], "final")
        for heading in ("## Clinical summary", "Latest findings", "## Clinical trials", "Key studies"):
            self.assertNotIn(heading, text)
            self.assertNotIn(heading, final["markdown"])
        self.assertNotIn("None retrieved", text)      # an empty table is never printed
        self.assertIn("No information on this was found", text)
        self.assertNotIn("What you asked", final["markdown"])      # no lead text: the answer is the table

    def test_a_normal_run_still_writes_the_summary(self):
        text = "".join(e.get("text", "") for e in run() if e["type"] == "token")
        self.assertIn("Clinical summary", text)


class FocusedDrugFacts(unittest.TestCase):
    def test_a_question_about_contraindications_gets_a_contraindications_table_only(self):
        events = run(focus=["drug_facts"])
        md = events[-1]["markdown"]
        self.assertEqual(events[-1]["type"], "final")
        self.assertIn("| Drug | Contraindications | Source |", md)
        for gone in ("Side effects of the drugs", "Dose / route", "## Clinical summary", "Treatment Drug Landscape", "Common side effects"):
            self.assertNotIn(gone, md)


class FocusedKinds(unittest.TestCase):
    def test_a_trials_follow_up_writes_only_the_trials_table(self):
        events = run(focus=["trials"])
        md = events[-1]["markdown"]
        self.assertEqual(events[-1]["type"], "final")
        self.assertIn("Clinical trials", md)
        for gone in ("## Clinical summary", "Treatment Drug Landscape", "Latest findings"):
            self.assertNotIn(gone, md)

    def test_an_evidence_follow_up_writes_the_evidence_but_no_summary_drug_or_trial_section(self):
        events = run(focus=["evidence"])
        md = events[-1]["markdown"]
        self.assertEqual(events[-1]["type"], "final")
        self.assertNotIn("## Clinical summary", md)
        self.assertNotIn("Treatment Drug Landscape", md)
        self.assertNotIn("## Clinical trials", md)


class FocusedSafety(unittest.TestCase):
    def test_a_side_effects_follow_up_writes_a_side_effects_table_and_not_the_drug_landscape(self):
        events = run(focus=["safety"])
        md = events[-1]["markdown"]
        self.assertEqual(events[-1]["type"], "final")
        self.assertIn("## Side effects of the drugs from the earlier answers", md)
        self.assertIn("| Drug | Common side effects |", md)
        for gone in ("## Clinical summary", "Treatment Drug Landscape", "Safety warnings (FDA boxed warnings)", "## Clinical trials"):
            self.assertNotIn(gone, md)


class FocusSections(unittest.TestCase):
    def test_what_is_asked_decides_the_parts(self):
        import orchestrator as o
        plan = lambda *a: {"info_needed": list(a)}
        self.assertEqual(o.focus_sections(plan("dosing"), {}, True, False), ["drugs"])
        self.assertEqual(o.focus_sections(plan("safety"), {}, True, False), ["safety"])
        self.assertEqual(o.focus_sections(plan("safety", "drug_interaction"), {}, True, False), ["safety"])
        self.assertEqual(o.focus_sections(plan("safety"), {}, False, False), [])      # no stored drugs to look up
        self.assertEqual(o.focus_sections(plan("trial_status"), {}, True, False), ["trials"])
        self.assertEqual(o.focus_sections(plan("efficacy"), {}, False, False), ["evidence"])
        self.assertEqual(o.focus_sections(plan("dosing", "trial_status"), {}, True, False), ["drugs", "trials"])

    def test_the_whole_report_when_the_topic_changes_or_the_asks_are_broad_or_the_agent_says_so(self):
        import orchestrator as o
        self.assertEqual(o.focus_sections({"info_needed": ["dosing"]}, {}, True, True), [])      # another condition or population
        self.assertEqual(o.focus_sections({"info_needed": ["treatment_options", "efficacy", "safety"]}, {}, False, False), ["evidence"])      # alternatives for the same case: one researched answer, not the whole report
        self.assertEqual(o.focus_sections({"info_needed": ["dosing"]}, {"scope": "full"}, True, False), [])
        self.assertEqual(o.focus_sections({"info_needed": ["dosing"]}, {}, False, False), [])      # no stored drugs to look up


class FocusedTables(unittest.TestCase):
    MD = ("## Treatment Drug Landscape\n\nintro\n\n### Drugs used for treatment of this condition\n\n| Drug |\n|---|\n| a |\n\n### Other drugs: used for symptoms (not established)\n\n| Drug |\n|---|\n| None retrieved |\n\n"
          "### Recently approved or updated (FDA)\n\nx\n\n### Safety warnings (FDA boxed warnings)\n\n| Drug |\n|---|\n| a |\n")

    def test_a_dosage_question_keeps_only_the_drug_tables_that_have_rows(self):
        out = qa_stream.focused_landscape(self.MD, ["dosing"])
        self.assertIn("Drugs used for treatment", out)
        for gone in ("Safety warnings", "Recently approved", "Other drugs: used for symptoms"):
            self.assertNotIn(gone, out)

    def test_a_safety_question_has_its_own_table_so_the_boxed_warning_table_is_not_reprinted(self):
        self.assertNotIn("Safety warnings", qa_stream.focused_landscape(self.MD, ["safety"]))


class StageMachine(unittest.TestCase):
    def test_the_default_order_is_the_one_the_skill_describes(self):
        events = run()
        record = events[-1]["state"]["stage_order"]
        self.assertEqual(events[-1]["type"], "final")
        self.assertEqual((record["chosen_by"], record["ran"]), ("default", list(stages.DEFAULT_ORDER)))

    def test_the_agents_order_is_followed(self):
        wanted = ["fast_search", "fast_answer", "deep_search", "rank", "drugs", "read_evidence", "escalate", "write_start", "write_stream", "figures", "assemble", "limitations", "verify", "finalize"]
        events = run(wanted)
        record = events[-1]["state"]["stage_order"]
        self.assertEqual(events[-1]["type"], "final")
        self.assertEqual((record["chosen_by"], record["ran"]), ("agent", wanted))

    def test_an_impossible_order_is_repaired_not_obeyed(self):
        events = run(list(stages.DEFAULT_ORDER)[::-1])
        ran = events[-1]["state"]["stage_order"]["ran"]
        self.assertEqual(events[-1]["type"], "final")
        for name in ran:
            for needed in stages.STAGES[name][1]:
                self.assertLess(ran.index(needed), ran.index(name))

    def test_a_missing_required_stage_is_added_figures_included(self):
        events = run(["fast_search", "fast_answer", "deep_search", "rank", "read_evidence", "drugs", "write_stream"])
        ran = events[-1]["state"]["stage_order"]["ran"]
        self.assertEqual(events[-1]["type"], "final")
        for required in stages.DEFAULT_ORDER:
            self.assertIn(required, ran)      # the agent cannot drop a stage, figures included

    def test_nothing_found_ends_the_run_after_the_fast_search(self):
        events = run(found=False)
        self.assertEqual(events[-1]["type"], "final")
        self.assertTrue(events[-1]["markdown"].startswith("I answer clinical research questions"))
        self.assertNotIn("stage_order", events[-1]["state"])      # the run stopped before the agent's order was read

    def test_the_report_is_the_same_whatever_valid_order_is_chosen(self):
        a, b = run()[-1]["markdown"], run(["fast_search", "fast_answer", "deep_search", "rank", "drugs", "read_evidence", "escalate", "write_start", "write_stream", "assemble", "limitations", "verify", "finalize"])[-1]["markdown"]
        self.assertEqual(a, b)


class OrderRules(unittest.TestCase):
    def test_every_stage_reads_only_earlier_stages_in_the_default_order(self):
        for i, name in enumerate(stages.DEFAULT_ORDER):
            for needed in stages.STAGES[name][1]:
                self.assertLess(stages.DEFAULT_ORDER.index(needed), i)

    def test_unknown_and_repeated_names_are_dropped(self):
        order = stages.clean_order(["verify", "bogus", "verify", "fast_search"])
        self.assertNotIn("bogus", order)
        self.assertEqual(len(order), len(set(order)))

    def test_no_choice_means_the_default(self):
        self.assertEqual(stages.clean_order(None), list(stages.DEFAULT_ORDER))
        self.assertEqual(stages.clean_order([]), list(stages.DEFAULT_ORDER))


if __name__ == "__main__":
    unittest.main()


class FocusedDrugDiscovery(unittest.TestCase):
    def test_a_follow_up_that_asks_for_the_drugs_of_the_condition_is_focused_without_earlier_drugs(self):
        events = run(focus=["drug_discovery"])
        md = events[-1]["markdown"]
        self.assertEqual(events[-1]["type"], "final")
        for gone in ("## Clinical summary", "Latest findings", "## Clinical trials", "Key studies"):
            self.assertNotIn(gone, md)
        self.assertTrue(qa_stream.stream_answer)

    def test_the_orchestrator_adds_drug_discovery_when_medicines_are_asked_but_not_the_earlier_drugs(self):
        import orchestrator as o
        plan = {"info_needed": ["treatment_options"], "medicines_needed": True}
        self.assertIn("drug_discovery", o.focus_sections(plan, {"scope": "focused"}, False, False))
        self.assertNotIn("drug_discovery", o.focus_sections({"info_needed": ["prognosis"], "medicines_needed": False}, {"scope": "focused"}, False, False))
        self.assertNotIn("drug_discovery", o.focus_sections({"info_needed": ["dosing"], "medicines_needed": True}, {}, True, False))      # about the earlier drugs: their own parts

    def test_a_name_with_a_trailing_letter_or_numeral_is_a_drug_name(self):
        import landscape
        for name in ("penicillin V", "factor VIII", "amoxicillin"):
            self.assertTrue(landscape.DRUG_NAME.fullmatch(name), name)
