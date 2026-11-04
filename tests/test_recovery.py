import json
import unittest

import orchestrator
import qa_stream
import recovery


def fresh(**over):
    state = {"papers": {}, "trials_by_id": {}, "web": [], "regulatory": [], "retrieval_errors": {}, "source_status": {}, "_search_raw": {}, "medicines_needed": True}
    state.update(over)
    return state


def agent(*actions):
    """A fake agent: calls retry_search for each (source, query), then proceed(note)."""
    def run(system, prompt, tools):
        by = {t.__name__: getattr(t, "entrypoint", t) for t in tools}
        for source, query in actions[:-1]:
            by["retry_search"](source, query)
        by["proceed"](actions[-1])
    return run


class Problems(unittest.TestCase):
    def test_a_group_with_content_has_no_problem(self):
        papers = {str(i): {} for i in range(3)}
        self.assertEqual(recovery.problems(fresh(papers=papers, trials_by_id={"N": {}}, regulatory=[{}]), "evidence"), [])
        self.assertEqual(recovery.problems(fresh(trials_by_id={"N": {}}), "trials"), [])
        self.assertEqual(recovery.problems(fresh(regulatory=[{}]), "drugs"), [])

    def test_empty_groups_are_problems(self):
        self.assertEqual(len(recovery.problems(fresh(), "evidence")), 1)
        self.assertEqual(len(recovery.problems(fresh(), "trials")), 1)
        self.assertEqual(len(recovery.problems(fresh(), "drugs")), 1)

    def test_no_drug_label_is_no_problem_when_medicines_are_not_needed(self):
        self.assertEqual(recovery.problems(fresh(medicines_needed=False), "drugs"), [])

    def test_retries_are_checked(self):
        plan = {"condition": "migraine", "drugs": []}
        raw = [{"source": "trials", "query": "migraine prevention"}, {"source": "trials", "query": "migraine prevention"}, {"source": "pubmed", "query": "migraine"},
               {"source": "trials", "query": "unrelated words entirely"}, {"source": "trials", "query": "migraine adults"}, {"source": "trials", "query": "migraine children"}]
        kept = recovery.clean_retries(raw, "trials", "Treatment of migraine in adults", plan)
        self.assertEqual(kept, [{"source": "trials", "query": "migraine prevention"}, {"source": "trials", "query": "migraine adults"}])

    def test_skip_stages_are_limited_to_the_optional_ones(self):
        self.assertEqual(recovery.clean_skip(["Figures", "full text", "verification", 3]), ["full_text"])
        self.assertEqual(recovery.clean_skip(None), [])


class AgentDecides(unittest.TestCase):
    def test_the_agent_retries_then_proceeds(self):
        out = orchestrator.recover("report", "trials", "Treatment of migraine in adults", {"condition": "migraine"},
                                   agent(("trials", "migraine"), "no registered trial"))
        self.assertEqual(out, {"retries": [{"source": "trials", "query": "migraine"}], "note": "no registered trial"})

    def test_agent_failure_means_no_retries(self):
        def boom(*a):
            raise RuntimeError("down")
        self.assertEqual(orchestrator.recover("r", "trials", "q migraine", {}, boom), {"retries": [], "note": ""})


class RecoverGroup(unittest.TestCase):
    def run_group(self, state, group, recover, searches):
        skipped = []
        qa_stream.recover_group(state, group, "Treatment of migraine in adults", {"condition": "migraine"}, skipped, recover=recover, searches=searches)
        return skipped

    def test_nothing_empty_means_no_agent_call(self):
        calls = []
        self.run_group(fresh(trials_by_id={"N": {}}), "trials", lambda *a: calls.append(1), {})
        self.assertEqual(calls, [])

    def test_a_retry_fills_the_group(self):
        trial = {"results": [{"nct_id": "NCT00000001", "title": "t", "status": "RECRUITING"}]}
        state = fresh()
        skipped = self.run_group(state, "trials", lambda *a: {"retries": [{"source": "trials", "query": "migraine"}], "note": ""}, {"trials": lambda q: json.dumps(trial)})
        self.assertIn("NCT00000001", state["trials_by_id"])
        self.assertEqual(skipped, [])
        self.assertEqual(state["recovery"][0]["still_empty"], [])

    def test_an_empty_retry_is_reported_in_the_limitations(self):
        state = fresh()
        skipped = self.run_group(state, "trials", lambda *a: {"retries": [{"source": "trials", "query": "migraine"}], "note": "no registered trial"}, {"trials": lambda q: json.dumps({"results": []})})
        self.assertEqual(skipped, ["trials: no registered trial"])

    def test_a_crash_never_stops_the_report(self):
        state = fresh()
        self.run_group(state, "trials", lambda *a: 1 / 0, {})
        self.assertIn("error", state["recovery"][0])


class PlanSkipsStages(unittest.TestCase):
    def test_the_plan_keeps_only_known_optional_stages(self):
        plan = qa_stream.finish_plan({"intent": "drug_treatment", "condition": "migraine", "skip_stages": ["full_text", "writing"]}, "ok", "Treatment of migraine")
        self.assertEqual(plan["skip_stages"], ["full_text"])
        self.assertEqual(qa_stream.finish_plan({"condition": "migraine"}, "ok", "Treatment of migraine")["skip_stages"], [])


if __name__ == "__main__":
    unittest.main()
