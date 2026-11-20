"""The agent-driven run (async_agent_run.py): the stage tools, the guardrails and the completion guard. The pipeline is a fake (stages that only record themselves) and the agent is a script that calls the tools."""
import asyncio
import unittest
from unittest import mock

import async_agent_run as agent_run
import stages as stages_mod


class FakeRun:
    """The parts of AsyncPipeline the agent run uses."""
    question = "How to treat migraine in adults?"

    def __init__(self, fail=None):
        self.ran, self.halted, self.focused = ["fast_search", "fast_answer"], False, False
        self.papers, self.trials, self.web, self.skipped = {"1": {}, "2": {}}, {}, [], []
        self.state = {"retrieval_errors": {}, "regulatory": [], "hybrid_trace": {}}
        self.plan_box = {"plan": {"condition": "migraine", "drugs": [], "intent": "drug_treatment"}}
        self.fail = fail or {}
        self.rank_trace = {"cross_encoder": "reranked the first 2 papers"}

    async def run_stage(self, name):
        self.ran.append(name)
        if name in self.fail:
            raise RuntimeError(self.fail[name])
        if name == "rank":
            self.state["hybrid_trace"] = dict(self.rank_trace)
        await asyncio.sleep(0)
        yield {"type": "stage_event", "stage_name": name}
        if name == "finalize":
            yield {"type": "final", "markdown": "report"}
        else:
            yield {"type": "progress", "stage_name": name}


class ScriptedAgent:
    """arun(prompt) calls the tools of the script, in order, like the model would."""
    def __init__(self, tools, script, results):
        self.tools, self.script, self.results = tools, script, results

    async def arun(self, prompt):
        for name, kwargs in self.script:
            self.results.append((name, await self.tools[name](**kwargs)))


def maker(script, results):
    return lambda ctx, tools: ScriptedAgent(tools, script, results)


NORMAL = [(n, {}) for n in ("collect_deep_evidence", "rank_evidence", "read_evidence", "analyze_drugs", "write_report", "list_limitations", "verify_report", "finalize_report")]


async def collect(run, make_agent):
    return [e async for e in agent_run.drive(run, make_agent)]


def ran_after_fast(run):
    return [n for n in run.ran if n not in stages_mod.FIRST]


class AgentRunTests(unittest.TestCase):
    def test_a_normal_agent_run_runs_every_stage_once_in_a_valid_order_and_ends_with_final(self):
        run, results = FakeRun(), []
        events = asyncio.run(collect(run, maker(NORMAL, results)))
        self.assertEqual(events[-1]["type"], "final")
        for name in stages_mod.DEFAULT_ORDER:
            self.assertEqual(run.ran.count(name), 1, name)
        for index, name in enumerate(run.ran):      # nothing ran before the stages it reads
            self.assertTrue(all(r in run.ran[:index] for r in stages_mod.STAGES[name][1]), name)
        self.assertEqual(run.state["agent"]["guard_ran"], [])
        self.assertEqual(run.state["stage_order"]["chosen_by"], "agent_loop")

    def test_the_events_are_the_stage_events_unchanged_and_in_order(self):
        run, results = FakeRun(), []
        events = asyncio.run(collect(run, maker(NORMAL, results)))
        stage_names = [e["stage_name"] for e in events if e["type"] == "stage_event"]
        self.assertEqual(stage_names, ran_after_fast(run))

    def test_a_tool_refuses_before_its_prerequisites_and_runs_nothing(self):
        run, results = FakeRun(), []
        asyncio.run(collect(run, maker([("verify_report", {}), ("write_report", {})], results)))
        self.assertTrue(results[0][1].startswith("NOT RUN"))
        self.assertIn("list_limitations", results[0][1])
        self.assertTrue(results[1][1].startswith("NOT RUN"))
        self.assertNotIn("verify", run.ran[:5])      # refused: the guard ran it later, in order, not the agent

    def test_the_papers_cannot_be_read_before_they_are_ranked(self):
        run, results = FakeRun(), []
        asyncio.run(collect(run, maker([("collect_deep_evidence", {}), ("read_evidence", {})], results)))
        self.assertIn("needs rank (call rank_evidence)", results[1][1])
        self.assertLess(run.ran.index("rank"), run.ran.index("read_evidence"))

    def test_finalize_needs_verify(self):
        run, results = FakeRun(), []
        script = NORMAL[:6] + [("finalize_report", {})]
        asyncio.run(collect(run, maker(script, results)))
        self.assertIn("needs verify", results[-1][1])
        self.assertEqual(run.ran.count("finalize"), 1)      # the guard ran verify, then finalize
        self.assertLess(run.ran.index("verify"), run.ran.index("finalize"))

    def test_search_more_adds_papers_marks_the_ranking_stale_and_ranking_runs_again(self):
        run, results = FakeRun(), []

        async def fake_search(self, source, query):
            run.papers["3"] = {}
            run.papers["4"] = {}
        script = [("collect_deep_evidence", {}), ("rank_evidence", {}), ("search_more", {"source": "pubmed", "query": "migraine prevention adults"}), ("read_evidence", {}), ("rank_evidence", {}), ("read_evidence", {})]
        with mock.patch.object(agent_run.AgentContext, "_do_search", fake_search):
            asyncio.run(collect(run, maker(script, results)))
        self.assertIn("added 2 papers", results[2][1])
        self.assertIn("STALE", results[2][1])
        self.assertIn("rank_evidence first", results[3][1])      # read_evidence refused while stale
        self.assertEqual(run.ran.count("rank"), 2)
        self.assertTrue(results[5][1].startswith("ok"))
        self.assertLess(len(run.ran), 30)
        self.assertEqual(run.state["agent"]["searches"][0]["added"]["papers"], 2)

    def test_search_more_is_refused_after_reading_and_for_unrelated_queries_and_beyond_the_limit(self):
        run, results = FakeRun(), []
        called = []

        async def fake_search(self, source, query):
            called.append((source, query))
        script = [("search_more", {"source": "pubmed", "query": "migraine adults"}),      # before collect_deep_evidence
                  ("collect_deep_evidence", {}), ("search_more", {"source": "pubmed", "query": "quantum chromodynamics"}), ("search_more", {"source": "nowhere", "query": "migraine adults"}),
                  ("search_more", {"source": "web", "query": "migraine adults A"}), ("search_more", {"source": "web", "query": "migraine adults B"}), ("search_more", {"source": "web", "query": "migraine adults C"}),
                  ("search_more", {"source": "web", "query": "migraine adults D"}),
                  ("rank_evidence", {}), ("read_evidence", {}), ("search_more", {"source": "trials", "query": "migraine adults"})]
        with mock.patch.object(agent_run.AgentContext, "_do_search", fake_search):
            asyncio.run(collect(run, maker(script, results)))
        verdicts = [r[1].splitlines()[0] for r in results]
        self.assertIn("call collect_deep_evidence first", verdicts[0])
        self.assertIn("must use words of the question", verdicts[2])
        self.assertIn("source must be one of", verdicts[3])
        self.assertEqual(len(called), 3)      # AGENT_MAX_SEARCHES = 3
        self.assertIn("limit of 3", verdicts[7])
        self.assertIn("already being read", verdicts[10])

    def test_a_skipped_cross_encoder_reaches_the_limitations_and_the_agent(self):
        run, results = FakeRun(), []
        run.rank_trace = {"cross_encoder": "skipped: the model is not available (still loading)"}
        asyncio.run(collect(run, maker(NORMAL[:2], results)))
        self.assertIn("skipped", results[1][1])
        self.assertTrue(any(s.startswith("ranking: skipped") for s in run.skipped))

    def test_the_guard_finishes_the_run_when_the_agent_stops_early(self):
        run, results = FakeRun(), []
        events = asyncio.run(collect(run, maker(NORMAL[:2], results)))
        self.assertEqual(events[-1]["type"], "final")
        self.assertEqual(run.state["agent"]["guard_ran"], [n for n in stages_mod.DEFAULT_ORDER if n not in stages_mod.FIRST and n not in ("deep_search", "rank")])
        self.assertEqual(run.ran.count("verify"), 1)

    def test_a_failing_agent_gives_the_fixed_pipeline(self):
        run = FakeRun()

        def broken(ctx, tools):
            raise RuntimeError("model down")
        events = asyncio.run(collect(run, broken))
        self.assertEqual(ran_after_fast(run), [n for n in stages_mod.DEFAULT_ORDER if n not in stages_mod.FIRST])
        self.assertEqual(events[-1]["type"], "final")
        self.assertIn("model down", run.state["agent"]["error"])

    def test_a_stage_error_is_told_to_the_agent_and_the_guard_retries_it_once_like_the_fixed_run(self):
        run, results = FakeRun(fail={"read_evidence": "boom"}), []
        script = NORMAL[:3]
        with self.assertRaises(RuntimeError):      # the guard retries it once, exactly as the fixed run would have failed
            asyncio.run(collect(run, maker(script, results)))
        self.assertTrue(results[2][1].startswith("ERROR in read_evidence"))

    def test_an_agent_that_is_too_slow_is_cut_off_and_the_run_is_completed(self):
        run = FakeRun()

        class Slow:
            async def arun(self, prompt):
                await asyncio.sleep(30)
        with mock.patch.dict("os.environ", {"AGENT_BUDGET_S": "0.01"}), mock.patch.object(agent_run, "budget_s", return_value=-59.9):
            events = asyncio.run(collect(run, lambda ctx, tools: Slow()))
        self.assertEqual(events[-1]["type"], "final")
        self.assertEqual(ran_after_fast(run), [n for n in stages_mod.DEFAULT_ORDER if n not in stages_mod.FIRST])

    def test_a_halted_run_stops_the_agent_and_the_guard(self):
        run, results = FakeRun(), []
        run.halted = True
        events = asyncio.run(collect(run, maker(NORMAL, results)))
        self.assertEqual(events, [])
        self.assertTrue(all(r[1].startswith(("STOPPED", "NOT RUN")) for r in results))
        self.assertEqual(run.ran, ["fast_search", "fast_answer"])

    def test_the_agent_is_offered_every_tool_with_a_description(self):
        tools = agent_run.build_tools(agent_run.AgentContext(FakeRun()))
        self.assertEqual(set(tools), set(agent_run.GROUPS) | {"search_more"})
        for name, fn in tools.items():
            self.assertTrue((fn.__doc__ or "").strip(), name)
        self.assertEqual(set(stages_mod.DEFAULT_ORDER) - set(stages_mod.FIRST), {s for names in agent_run.GROUPS.values() for s in names})      # every stage is reachable

    def test_the_agent_cannot_leave_the_figures_out_of_the_report(self):
        import inspect
        tools = agent_run.build_tools(agent_run.AgentContext(FakeRun()))
        self.assertEqual(list(inspect.signature(tools["write_report"]).parameters), [])      # no include_figures switch
        self.assertIn("figures", agent_run.GROUPS["write_report"])

    def test_the_drug_stage_is_started_before_the_reading_is_finished(self):
        run, results = FakeRun(), []
        asyncio.run(collect(run, maker([(n, {}) for n in ("collect_deep_evidence", "rank_evidence", "read_evidence", "write_report")], results)))
        self.assertIn("NOT RUN", dict(results)["write_report"])
        self.assertIn("drugs", dict(results)["write_report"])
        self.assertNotIn("escalate", run.ran[:run.ran.index("drugs")] if "drugs" in run.ran else [])      # the guard then starts the drugs first, then finishes the reading

    def test_in_the_normal_order_the_drugs_start_before_escalate_and_write_start(self):
        run = FakeRun()
        asyncio.run(collect(run, maker(NORMAL, [])))
        order = run.ran
        self.assertLess(order.index("read_evidence"), order.index("drugs"))
        self.assertLess(order.index("drugs"), order.index("escalate"))
        self.assertLess(order.index("escalate"), order.index("write_start"))

    def test_a_stage_that_finished_beside_the_fast_answer_is_complete_and_one_that_did_not_is_not(self):
        run = FakeRun()
        run.ran = ["fast_search", "fast_answer", "deep_search", "rank"]
        run.prefetched = {"deep_search"}      # rank was started but did not finish
        ctx = agent_run.AgentContext(run)
        self.assertIn("deep_search", ctx.completed)
        self.assertNotIn("rank", ctx.completed)

    def test_deep_search_and_rank_run_beside_the_fast_answer_and_their_events_come_after_it(self):
        import time
        import async_runner

        class Fake:
            focused, halted = False, False
            def __init__(self):
                self.ran, self.prefetched, self.state = [], set(), {}
            def event(self, kind, **kw):
                return {"type": kind, **kw}
            async def run_stage(self, name):
                self.ran.append(name)
                await asyncio.sleep(0.3 if name == "fast_answer" else 0.1)
                yield {"type": "stage_event", "stage_name": name}
            _prefetch = async_runner.AsyncPipeline._prefetch

        async def go():
            fake = Fake()
            async def no_agent(run, make_agent=None):
                return
                yield
            with mock.patch.object(async_runner.async_agent_run, "drive", no_agent):
                began = time.time()
                events = [e async for e in async_runner.AsyncPipeline.run(fake)]
                return fake, events, time.time() - began
        fake, events, took = asyncio.run(go())
        names = [e.get("stage_name") for e in events if e["type"] == "stage_event"]
        self.assertEqual(names, ["fast_search", "fast_answer", "deep_search", "rank"])      # the page sees the old order
        self.assertLess(took, 0.45)      # 0.1 + 0.3 + 0.1 + 0.1 = 0.6 one after another
        self.assertEqual(fake.prefetched, {"deep_search", "rank"})

    def test_a_failed_overlapped_stage_is_left_undone_for_the_agent(self):
        import async_runner

        class Fake:
            focused, halted = False, False
            def __init__(self):
                self.ran, self.prefetched, self.state = [], set(), {}
            def event(self, kind, **kw):
                return {"type": kind, **kw}
            async def run_stage(self, name):
                self.ran.append(name)
                if name == "deep_search":
                    raise RuntimeError("boom")
                yield {"type": "stage_event", "stage_name": name}
            _prefetch = async_runner.AsyncPipeline._prefetch

        async def go():
            fake = Fake()
            async def no_agent(run, make_agent=None):
                return
                yield
            with mock.patch.object(async_runner.async_agent_run, "drive", no_agent):
                [e async for e in async_runner.AsyncPipeline.run(fake)]
            return fake
        fake = asyncio.run(go())
        self.assertNotIn("deep_search", fake.ran)
        self.assertNotIn("rank", fake.ran)      # rank needs deep_search: not started
        self.assertEqual(fake.prefetched, set())
        self.assertIn("deep_search", fake.state["overlap_errors"])

    def test_there_is_no_off_switch_and_the_runner_always_hands_over_to_the_agent(self):
        self.assertFalse(hasattr(agent_run, "enabled"))
        source = open("async_runner.py", encoding="utf-8").read()
        self.assertIn("async_agent_run.drive(self)", source)
        self.assertNotIn("clean_order(asked_order)", source)      # the fixed stage loop is gone from the main run

    def test_a_follow_up_is_told_to_the_agent(self):
        run = FakeRun()
        run.focus_sections = ["drugs"]
        self.assertIn("FOLLOW-UP", agent_run.brief(agent_run.AgentContext(run)))


if __name__ == "__main__":
    unittest.main()
