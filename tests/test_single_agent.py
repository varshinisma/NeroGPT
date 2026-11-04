import inspect
import unittest
from unittest import mock

import orchestrator
import qa_stream


class PlanFromTheAgent(unittest.TestCase):
    Q = "Treatment of migraine in adults"

    def test_the_agents_plan_is_finished_by_code_without_a_planner_call(self):
        with mock.patch.object(qa_stream, "plan_with_skill", side_effect=AssertionError("old planner called")):
            plan = qa_stream.plan_for(self.Q, {"intent": "drug_treatment", "condition": "migraine", "medicines_needed": True})
        self.assertEqual((plan["plan_status"], plan["intent"], plan["condition"]), ("ok", "drug_treatment", "migraine"))
        for key in ("understanding", "conditions", "skills", "pubmed_query", "trial_condition"):
            self.assertIn(key, plan)

    def test_the_agent_plans_when_no_plan_was_given(self):
        with mock.patch.object(orchestrator, "plan_only", return_value={"intent": "safety", "condition": "migraine"}), mock.patch.object(qa_stream, "plan_with_skill", side_effect=AssertionError):
            self.assertEqual(qa_stream.plan_for(self.Q)["intent"], "safety")

    def test_the_old_planner_is_the_fallback(self):
        for failing in (mock.Mock(return_value=None), mock.Mock(side_effect=RuntimeError("down"))):
            with mock.patch.object(orchestrator, "plan_only", failing), mock.patch.object(qa_stream, "plan_with_skill", return_value={"intent": "old"}):
                self.assertEqual(qa_stream.plan_for(self.Q), {"intent": "old"})

    def test_switching_the_agent_planner_off_uses_the_old_planner(self):
        with mock.patch.object(orchestrator, "PLAN_BY_AGENT", False), mock.patch.object(orchestrator, "plan_only", side_effect=AssertionError), mock.patch.object(qa_stream, "plan_with_skill", return_value={"intent": "old"}):
            self.assertEqual(qa_stream.plan_for(self.Q), {"intent": "old"})

    def test_stream_answer_accepts_the_plan(self):
        self.assertIn("plan", inspect.signature(qa_stream.stream_answer).parameters)


class AgentChoosesTheOrder(unittest.TestCase):
    def test_the_agents_instructions_list_the_stages_and_the_skill_steps(self):
        text = orchestrator.instructions()
        for word in ("fast_search", "write_stream", "verify", "Step 7", "Fast path (about 5 seconds)"):
            self.assertIn(word, text)

    def test_the_order_goes_through_the_plan(self):
        def run(system, prompt, tools):
            by = {t.__name__: getattr(t, "entrypoint", t) for t in tools}
            by["research"](kind="new", intent="drug_treatment", condition="migraine", stages=["fast_search", "fast_answer", "deep_search"])
        plan = orchestrator.plan_only("Treatment of migraine", run)
        self.assertEqual(plan["stages"], ["fast_search", "fast_answer", "deep_search"])
        self.assertEqual(qa_stream.finish_plan(plan, "ok", "Treatment of migraine")["stages"], ["fast_search", "fast_answer", "deep_search"])


if __name__ == "__main__":
    unittest.main()
