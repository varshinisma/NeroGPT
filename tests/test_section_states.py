"""Section states in the report: not applicable / not retrieved / no evidence found / retrieval failed / timed out, inside the existing headings and tables; a claim of absence is never made for a section
whose search was skipped, failed or did not finish (false_absence_claim); the missing_drug_landscape rule is NOT relaxed."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import landscape  # noqa: E402
import qa_stream  # noqa: E402
import retrieval_routing as rr  # noqa: E402
import verification  # noqa: E402
from tests.test_regression import BASE, make_state  # noqa: E402


def empty_state(**sections):
    state = qa_stream.new_state("How does condition X work?")
    state["source_status"] = {"fda_fast": {"retrieval_status": "success", "evidence_status": "not_found_after_search", "error": None}}
    state["section_states"] = {name: {"state": value, "needs": []} for name, value in sections.items()}
    return state


class Renderers(unittest.TestCase):
    def test_every_state_reads_differently_and_the_failure_states_never_claim_an_absence(self):
        texts = {landscape.STATE_LINES[k] for k in ("not_applicable", "not_retrieved", "no_relevant_evidence", "failed", "timed_out")}
        self.assertEqual(len(texts), 5)
        for state in ("failed", "timed_out", "not_retrieved", "not_generated"):
            self.assertIn("does not show that", landscape.STATE_LINES[state])
        self.assertNotIn("does not show", landscape.STATE_LINES["no_relevant_evidence"])      # only a completed search may say that nothing relevant was found
        self.assertIs(landscape.STATE_LINES["failed"], rr.STATE_TEXT["failed"])      # one wording, shared with the routing module

    def test_the_drug_landscape_keeps_its_headings_and_tables_in_every_state(self):
        for sec in ("not_applicable", "not_retrieved", "no_relevant_evidence", "failed", "timed_out"):
            md = landscape.landscape_markdown(empty_state(drugs=sec), [], [])
            self.assertTrue(md.startswith("## Treatment Drug Landscape"))
            self.assertIn("### Drugs used for treatment of this condition", md)
            self.assertIn("### Other drugs: used for symptoms", md)
            self.assertEqual(md.count("| Drug | Indication | Dose / route (if sourced) | Key evidence |"), 2)      # both tables keep their columns
            self.assertIn(landscape.STATE_LINES[sec], md)
            self.assertNotIn("No FDA drug label with an indication matching", md)      # no absence is claimed about a search that did not complete

    def test_a_not_applicable_drug_section_does_not_say_none_was_found(self):
        md = landscape.landscape_markdown(empty_state(drugs="not_applicable"), [], [])
        self.assertIn("Not applicable", md)
        self.assertNotIn("None retrieved", md)

    def test_without_a_routed_state_the_earlier_wording_is_unchanged(self):
        state = qa_stream.new_state("How to treat condition X?")
        state["source_status"] = {"fda_fast": {"retrieval_status": "success", "evidence_status": "not_found_after_search", "error": None}}
        md = landscape.landscape_markdown(state, [], [])
        self.assertIn("None retrieved", md)
        self.assertIn("No FDA drug label with an indication matching", md)

    def test_rows_are_shown_whatever_the_state(self):
        state = make_state()
        state["section_states"] = {"drugs": {"state": "not_applicable", "needs": []}}
        md = landscape.landscape_markdown(state, [], [])
        self.assertIn("Risperidone", md)      # a verified FDA record is shown: the state only explains an EMPTY section

    def test_the_trials_section_states(self):
        for sec in ("not_applicable", "not_retrieved", "failed", "timed_out", "no_relevant_evidence"):
            md = landscape.trials_markdown(empty_state(trials=sec))
            self.assertTrue(md.startswith("## Clinical trials"))
            self.assertIn(landscape.STATE_LINES[sec], md)
        self.assertIn("trial status is unknown", landscape.trials_markdown(qa_stream.new_state("q")).lower())      # no completed registry result is not a negative finding


class ComputingTheStates(unittest.TestCase):
    def record(self, ops, needs):
        return {"all_ops": ops, "needs": {k: {"state": v} for k, v in needs.items()}}

    def op(self, target, need_id, skill="latest-evidence", origin="generated"):
        return rr.Operation(id=f"{target}:{need_id}", need_id=need_id, need=need_id, target=target, query="alpha beta gamma delta", skill=skill, origin=origin)

    def test_a_section_nobody_searched_is_not_applicable(self):
        states = qa_stream.compute_section_states(self.record([self.op("pubmed", "a")], {"a": "covered"}))
        self.assertEqual({k: v["state"] for k, v in states.items()}, {"drugs": "not_applicable", "trials": "not_applicable", "literature": "covered"})

    def test_failures_and_empty_results_are_kept_apart(self):
        ops = [self.op("trials", "t", "clinical-trials"), self.op("fda_label", "d", "drug-intelligence"), self.op("pubmed", "l")]
        states = qa_stream.compute_section_states(self.record(ops, {"t": "failed", "d": "no_relevant_evidence", "l": "timed_out"}))
        self.assertEqual({k: v["state"] for k, v in states.items()}, {"trials": "failed", "drugs": "no_relevant_evidence", "literature": "timed_out"})

    def test_a_skill_that_wrote_a_baseline_after_a_generation_failure_is_not_called_not_applicable(self):
        ops = [self.op("trials", "baseline-clinical-trials", "clinical-trials", "baseline")]
        states = qa_stream.compute_section_states(self.record(ops, {"baseline-clinical-trials": "no_relevant_evidence"}))
        self.assertEqual(states["trials"]["state"], "no_relevant_evidence")

    def test_one_covering_search_covers_the_section(self):
        ops = [self.op("fda_label", "d1", "drug-intelligence"), self.op("regulator", "d2", "drug-intelligence")]
        states = qa_stream.compute_section_states(self.record(ops, {"d1": "failed", "d2": "covered"}))
        self.assertEqual(states["drugs"]["state"], "covered")


class FalseAbsenceClaims(unittest.TestCase):
    def issues(self, text, **states):
        state = make_state()
        state["section_states"] = {name: {"state": value, "needs": []} for name, value in states.items()}
        return [i["kind"] for i in verification.find_issues(text, state)]

    def test_no_trials_found_is_flagged_when_the_trial_search_failed_was_skipped_or_unfinished(self):
        text = BASE + "No trials were found for this condition.\n"
        for state in ("failed", "timed_out", "not_retrieved", "not_applicable", "not_generated"):
            self.assertIn("false_absence_claim", self.issues(text, trials=state), state)

    def test_the_same_sentence_is_allowed_when_a_search_completed(self):
        text = BASE + "No trials were found for this condition.\n"
        self.assertNotIn("false_absence_claim", self.issues(text, trials="no_relevant_evidence"))
        self.assertNotIn("false_absence_claim", self.issues(text, trials="covered"))
        self.assertNotIn("false_absence_claim", self.issues(text))      # no routed state: the earlier rules only

    def test_a_scoped_sentence_is_not_a_false_claim(self):
        text = BASE + "No trials were found, but the registry could not be retrieved, so this does not show that none exist.\n"
        self.assertNotIn("false_absence_claim", self.issues(text, trials="failed"))

    def test_the_generated_state_lines_are_not_flagged(self):
        for sec in ("not_applicable", "failed"):
            text = BASE + landscape.STATE_LINES[sec] + "\n"
            self.assertNotIn("false_absence_claim", self.issues(text, trials=sec, drugs=sec, literature=sec))

    def test_drug_and_literature_absence_claims_are_checked_too(self):
        self.assertIn("false_absence_claim", self.issues(BASE + "No drug treatment was identified.\n", drugs="failed"))
        self.assertIn("false_absence_claim", self.issues(BASE + "No studies were found on this question.\n", literature="timed_out"))

    def test_the_repair_removes_the_false_sentence(self):
        state = make_state()
        state["section_states"] = {"trials": {"state": "failed", "needs": []}}
        result = verification.verify_and_repair(BASE + "No trials were found for this condition.\n\nExercise helps.\n", state)
        self.assertNotIn("No trials were found", result["answer"])

    def test_missing_drug_landscape_is_not_relaxed(self):
        state = make_state()
        state["section_states"] = {"drugs": {"state": "not_applicable", "needs": []}}
        kinds = [i["kind"] for i in verification.find_issues("**Evidence searched on 2026-10-06**\n\nText without the section.\n", state)]
        self.assertIn("missing_drug_landscape", kinds)      # the heading is still required: a not-applicable section keeps it and says so


if __name__ == "__main__":
    unittest.main()
