import tempfile
import unittest
from pathlib import Path
from unittest import mock

import latency_report
import observability as obs
from tests import test_stage_machine as stage_tests


class Numbers(unittest.TestCase):
    def test_percentiles(self):
        self.assertEqual(obs.percentile([1, 2, 3, 4, 100], 50), 3)
        self.assertEqual(obs.percentile([1, 2, 3, 4, 100], 95), 80.8)
        self.assertIsNone(obs.percentile([], 50))

    def test_model_tool_and_wait_time_do_not_double_count_parallel_calls(self):
        spans = [{"kind": "stage", "name": "deep", "start": 0.0, "end": 10.0},
                 {"kind": "model", "name": "writer", "start": 1.0, "end": 5.0, "ttft": 0.5, "gen_s": 3.5, "out_chars": 400, "in_chars": 4000},
                 {"kind": "model", "name": "writer", "start": 2.0, "end": 6.0, "ttft": 0.7, "gen_s": 3.3, "out_chars": 400, "in_chars": 8000},      # runs beside the first: 1 to 6 is 5 s of wall time
                 {"kind": "tool", "name": "pubmed", "start": 4.0, "end": 8.0}]      # 4 to 6 overlaps the models: only 6 to 8 counts as tool time
        s = obs.summarize(spans, 10.0, {"model_retries": 1})
        self.assertEqual((s["model_wall_s"], s["tool_wall_s"], s["orchestration_wait_s"]), (5.0, 2.0, 3.0))
        self.assertEqual(s["stages"]["deep"], {"seconds": 10.0, "model_s": 5.0, "tool_s": 2.0, "orchestration_wait_s": 3.0})
        self.assertEqual((s["model_calls"], s["tool_calls"], s["retries"], s["input_tokens_max"]), (2, 1, 1, 2000))
        self.assertEqual(s["model_by_purpose"]["writer"]["calls"], 2)


class Recording(unittest.TestCase):
    def test_a_run_is_recorded_with_ttft_and_written_to_the_log(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(obs, "OBS_DIR", Path(d)), mock.patch.object(obs, "RUNS_FILE", Path(d) / "runs.jsonl"):
            obs._run.clear()
            obs.begin_run("a question")
            obs.begin_run("again")      # an open run is kept (the API opens it before the routing decision)
            self.assertEqual(obs._run["question"], "a question")

            @obs.model_call
            def fake_model(instructions, prompt, max_tokens, purpose="writer"):
                yield "ab"
                yield "cd"
            self.assertEqual("".join(fake_model("sys", "prompt", 10, purpose="triage")), "abcd")
            with obs.tool("pubmed"):
                pass
            with self.assertRaises(ValueError), obs.tool("web"):
                raise ValueError("boom")
            obs.record("stage", "rank", obs.now() - 0.1, obs.now())
            summary = obs.end_run({"first_token_s": 1.2})
            self.assertEqual((summary["model_calls"], summary["tool_calls"], summary["errors"]), (1, 2, 1))
            self.assertEqual(summary["first_token_s"], 1.2)
            runs = obs.load_runs(Path(d) / "runs.jsonl")
            self.assertEqual(len(runs), 1)
            model = [s for s in runs[0]["spans"] if s["kind"] == "model"][0]
            self.assertEqual((model["name"], model["out_chars"], model["chunks"]), ("triage", 4, 2))
            self.assertIsNotNone(model["ttft"])
            page = latency_report.html_report(runs)
            self.assertIn("Waterfall", page)
            self.assertIn("p95", latency_report.text_report(runs))

    def test_a_stubbed_run_records_its_stages_and_puts_the_summary_in_the_state(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(obs, "OBS_DIR", Path(d)), mock.patch.object(obs, "RUNS_FILE", Path(d) / "runs.jsonl"):
            obs._run.clear()
            events = stage_tests.run()
            summary = events[-1]["state"]["observability"]
            self.assertIn("stages", summary)
            self.assertTrue({"fast_search", "fast_answer", "deep_search", "write_stream"} <= set(summary["stages"]))      # (the finalize stage ends after the summary is taken; the model is stubbed here)
            self.assertIn("orchestration_wait_s", summary["stages"]["deep_search"])
            self.assertEqual(len(obs.load_runs(Path(d) / "runs.jsonl")), 1)


if __name__ == "__main__":
    unittest.main()
