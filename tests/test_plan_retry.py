import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import qa_stream

GOOD = json.dumps({"is_clinical": True, "intent": "diagnosis", "condition": "alpha beta", "drugs": [], "info_needed": ["diagnosis"], "asked": "alpha", "purpose": "beta", "answer_format": "stepwise_plan", "case_facts": []})


class PlanRetry(unittest.TestCase):
    def run_plan(self, replies):
        original, calls = qa_stream.run_llm, []

        def fake(*args, **kwargs):
            calls.append(1)
            reply = replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply
        qa_stream.run_llm = fake
        try:
            return qa_stream.plan_with_skill("Child with alpha beta: how to evaluate?"), len(calls)
        finally:
            qa_stream.run_llm = original

    def test_a_cut_off_reply_is_asked_for_once_more(self):
        plan, calls = self.run_plan(['{"is_clinical": true, "intent": "diag', GOOD])
        self.assertEqual((calls, plan["plan_status"], plan["intent"]), (2, "retried", "diagnosis"))

    def test_a_good_first_reply_needs_no_second_call(self):
        plan, calls = self.run_plan([GOOD])
        self.assertEqual((calls, plan["plan_status"]), (1, "ok"))

    def test_two_failures_fall_back_to_the_question_and_say_so(self):
        plan, calls = self.run_plan([RuntimeError("timeout"), "not json"])
        self.assertEqual((calls, plan["plan_status"]), (2, "fallback"))
        self.assertTrue(plan["condition"])


if __name__ == "__main__":
    unittest.main()
