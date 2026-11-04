import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import plan_audit  # noqa: E402
import retrieval_routing as rr  # noqa: E402

QUESTION = "Child aged three has aaaa bbbb and cccc regression. What is the work-up and management?"


def op(target="pubmed", query="aaaa bbbb child work-up", need="diagnostic work-up", need_id="n1", skill="latest-evidence", serves="diagnosis", origin="generated"):
    return rr.Operation(id=need_id + target + query[:6], need_id=need_id, need=need, target=target, query=query, skill=skill, serves=serves, origin=origin)


def plan(**extra):
    understanding = {"info_needed": ["diagnosis"], "medicines_needed": False, **extra.pop("understanding", {})}
    return {"condition": "aaaa bbbb cccc regression", "intent": "case_evaluation", "drugs": [], "understanding": understanding,
            "pubmed_query": "(aaaa bbbb OR cccc regression) AND (child OR pediatric) AND (workup)", **extra}


def checks(findings):
    return {f["check"] for f in findings}


class AuditPlan(unittest.TestCase):
    def test_a_consistent_plan_has_no_finding(self):
        self.assertEqual(plan_audit.audit_plan(QUESTION, plan(), [op()], ["latest-evidence"]), [])

    def test_medicine_and_trial_searches_without_a_reason_are_reported(self):
        ops = [op(), op("fda_label", "drug label", need="labels", need_id="n2", skill="drug-intelligence", serves=""),
               op("trials", "", need="trials", need_id="n3", skill="clinical-trials", serves="")]
        found = checks(plan_audit.audit_plan(QUESTION, plan(), ops, ["latest-evidence", "drug-intelligence", "clinical-trials"]))
        self.assertTrue({"unneeded_medicine_search", "unneeded_trial_search"} <= found)

    def test_a_medicine_search_is_justified_by_the_plan_or_the_question(self):
        ops = [op(), op("fda_label", "drug label", need_id="n2", skill="drug-intelligence", serves="dosing")]
        for extra in ({"understanding": {"medicines_needed": True, "info_needed": ["diagnosis", "dosing"]}}, {"drugs": ["somedrug"]}):
            self.assertNotIn("unneeded_medicine_search", checks(plan_audit.audit_plan(QUESTION, plan(**extra), ops, ["latest-evidence", "drug-intelligence"])))

    def test_an_information_need_without_a_search_and_a_search_without_a_need(self):
        found = checks(plan_audit.audit_plan(QUESTION, plan(understanding={"info_needed": ["diagnosis", "prognosis"]}), [op(), op(need="", need_id="")], ["latest-evidence"]))
        self.assertTrue({"need_without_search", "search_without_need"} <= found)

    def test_duplicate_searches_and_a_lost_patient_group(self):
        ops = [op(query="aaaa bbbb child work-up"), op(query="aaaa bbbb child work-up", need_id="n2"), op(query="cccc regression causes", need_id="n3")]
        found = checks(plan_audit.audit_plan(QUESTION, plan(), ops, ["latest-evidence"]))
        self.assertIn("duplicate_search", found)
        self.assertIn("patient_context_lost", found)

    def test_a_condition_the_question_does_not_contain_is_reported(self):
        self.assertIn("condition_not_in_question", checks(plan_audit.audit_plan(QUESTION, plan(condition="zzzz yyyy xxxx"), [op()], ["latest-evidence"])))

    def test_a_failed_plan_is_reported(self):
        self.assertIn("plan_failed", checks(plan_audit.audit_plan(QUESTION, plan(plan_status="fallback"), [op()], [])))


class AuditNeedsAndSections(unittest.TestCase):
    def test_only_uncovered_essential_needs_are_reported(self):
        needs = {"n1": {"need": "a", "priority": "essential", "state": "covered"}, "n2": {"need": "b", "priority": "essential", "state": "no_relevant_evidence", "reason": "empty"},
                 "n3": {"need": "c", "priority": "supporting", "state": "failed"}}
        found = plan_audit.audit_needs(needs)
        self.assertEqual([f["check"] for f in found], ["essential_need_uncovered"])
        self.assertIn("'b'", found[0]["detail"])

    REPORT = "## Treatment Drug Landscape\n\n{landscape}\n\n## Clinical trials\n\n{trials}\n\n## Key studies\n\nx"

    def decisions(self, plan_, landscape, trials):
        return plan_audit.section_decisions(plan_, self.REPORT.format(landscape=landscape, trials=trials))

    def test_sections_that_do_not_apply_say_so(self):
        out = self.decisions(plan(), "*Not listed: this question does not ask about medicines.*", "Not applicable to this question.")
        self.assertTrue(out["landscape"]["ok"] and out["trials"]["ok"])
        self.assertEqual((out["landscape"]["state"], out["trials"]["state"]), ("not_applicable", "not_applicable"))

    def test_a_filled_section_the_plan_did_not_call_for_is_reported(self):
        out = self.decisions(plan(), "| Drug | Category |\n| x | y |", "- NCT0000")
        self.assertFalse(out["landscape"]["ok"])
        self.assertFalse(out["trials"]["ok"])

    def test_a_section_the_plan_called_for_may_not_claim_it_does_not_apply(self):
        out = self.decisions(plan(understanding={"medicines_needed": True}), "Not applicable.", "- NCT0000")
        self.assertFalse(out["landscape"]["ok"])

    def test_a_missing_section_is_reported(self):
        self.assertEqual(plan_audit.section_decisions(plan(), "## Key studies\n\nx")["landscape"]["state"], "missing")


if __name__ == "__main__":
    unittest.main()
