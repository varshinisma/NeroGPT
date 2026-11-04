import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import tracing


class TracingFallbackTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.patches = [mock.patch.object(tracing, "TRACE_DIR", Path(self.dir.name)), mock.patch.dict(tracing._state, {"quota_hit": False}), mock.patch.dict(os.environ, {"LOCAL_TRACING": ""})]
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in self.patches:
            patch.stop()
        self.dir.cleanup()

    def lines(self):
        out = []
        for path in Path(self.dir.name).glob("*.jsonl"):
            out += [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        return out

    def test_nothing_is_wrapped_or_written_without_any_tracing_setup(self):
        with mock.patch.object(tracing, "enabled", return_value=False):
            def f(x):
                return x + 1
            self.assertIs(tracing.traced("f")(f), f)
        self.assertEqual(self.lines(), [])

    def test_local_tracing_writes_one_line_per_call_and_returns_the_result(self):
        with mock.patch.dict(os.environ, {"LOCAL_TRACING": "true"}):
            f = tracing.traced("add")(lambda x: x + 1)
            self.assertEqual(f(1), 2)
        rows = self.lines()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["name"], "add")
        self.assertEqual(rows[0]["output"], "2")
        self.assertIsNone(rows[0]["error"])

    def test_an_error_is_recorded_and_still_raised(self):
        with mock.patch.dict(os.environ, {"LOCAL_TRACING": "true"}):
            def boom():
                raise ValueError("bad")
            with self.assertRaises(ValueError):
                tracing.traced("boom")(boom)()
        self.assertIn("ValueError", self.lines()[0]["error"])

    def test_a_generator_is_traced_without_changing_what_it_yields(self):
        with mock.patch.dict(os.environ, {"LOCAL_TRACING": "true"}):
            def gen():
                yield 1
                yield 2
            self.assertEqual(list(tracing.traced("gen")(gen)()), [1, 2])
        self.assertEqual(self.lines()[0]["output"], "<stream of 2 items>")

    def test_a_quota_message_switches_langsmith_off_and_the_local_trace_takes_over(self):
        calls = {"langsmith": 0, "plain": 0}

        def plain(x):
            calls["plain"] += 1
            return x

        def via_langsmith(x):
            calls["langsmith"] += 1
            return x
        wrapped = tracing.make_wrapper(plain, "step", "chain", langsmith_fn=via_langsmith)
        wrapped(1)
        self.assertEqual(calls, {"langsmith": 1, "plain": 0})
        self.assertEqual(self.lines(), [])      # LangSmith works and no local trace was asked for
        self.assertTrue(tracing.note_quota_message("Rate limit exceeded ... Monthly unique traces usage limit exceeded"))
        wrapped(2)
        self.assertEqual(calls, {"langsmith": 1, "plain": 1})
        self.assertEqual(len(self.lines()), 1)

    def test_an_unrelated_log_line_does_not_count_as_quota(self):
        self.assertFalse(tracing.note_quota_message("connection reset by peer"))
        self.assertFalse(tracing.quota_hit())


if __name__ == "__main__":
    unittest.main()
