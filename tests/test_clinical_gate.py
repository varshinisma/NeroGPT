import time
import unittest

import social_gate


def events(n=3, delay=0.0):
    for i in range(n):
        time.sleep(delay)
        yield {"type": "stage", "i": i}


class Closing:
    def __init__(self, n=50):
        self.n, self.closed = n, False

    def __iter__(self):
        for i in range(self.n):
            time.sleep(0.01)
            yield {"i": i}

    def close(self):
        self.closed = True


class Converse(unittest.TestCase):
    def test_the_model_writes_the_reply_to_a_message_that_is_not_clinical(self):
        self.assertEqual(social_gate.converse("how are you", lambda s, p: "I'm doing well, thanks. What would you like to look into?"), "I'm doing well, thanks. What would you like to look into?")
        self.assertIn("MESSAGE: how are you", (lambda: (lambda box: (social_gate.converse("how are you", lambda s, p: box.append(p) or "Hi!"), box)[1][0])([]))())

    def test_clinical_and_every_failure_mean_research(self):
        self.assertIsNone(social_gate.converse("migraine", lambda s, p: "CLINICAL"))
        self.assertIsNone(social_gate.converse("migraine", lambda s, p: " clinical. "))
        self.assertIsNone(social_gate.converse("migraine", lambda s, p: ""))

        def boom(system, prompt):
            raise RuntimeError("no model")
        self.assertIsNone(social_gate.converse("hi", boom))      # a failed check never blocks research

    def test_no_greeting_or_reply_is_written_in_code(self):
        source = open("social_gate.py", encoding="utf-8").read()
        for fixed in ('"hello"', '"thanks"', "You're welcome", "Hello. Which"):
            self.assertNotIn(fixed, source)

    def test_the_early_and_the_parallel_limits(self):
        self.assertTrue(social_gate.needs_early("how are you"))
        self.assertFalse(social_gate.needs_early("what is migraine prophylaxis"))
        self.assertTrue(social_gate.needs_gate("what is migraine prophylaxis"))
        self.assertFalse(social_gate.needs_gate("What are the first-line medications for generalised anxiety disorder in older adults with kidney disease?"))
        self.assertFalse(social_gate.needs_gate(""))


class Gated(unittest.TestCase):
    def test_a_clinical_message_gets_every_event_in_order(self):
        gated = social_gate.Gated(events(4), lambda: None)
        self.assertEqual([e["i"] for e in gated], [0, 1, 2, 3])
        self.assertFalse(gated.rejected)

    def test_a_reply_means_no_events_and_the_run_is_closed(self):
        run = Closing()
        gated = social_gate.Gated(run, lambda: "Hello there")
        self.assertEqual(list(gated), [])
        self.assertTrue(gated.rejected)
        self.assertEqual(gated.reply, "Hello there")
        time.sleep(0.3)
        self.assertTrue(run.closed)

    def test_a_gate_that_takes_too_long_or_fails_lets_the_run_through(self):
        slow = social_gate.Gated(events(2), lambda: (time.sleep(1.0), "late reply")[1], wait_s=0.1)
        self.assertEqual(len(list(slow)), 2)
        self.assertFalse(slow.rejected)

        def boom():
            raise RuntimeError("x")
        self.assertEqual(len(list(social_gate.Gated(events(2), boom))), 2)

    def test_the_run_goes_on_while_the_gate_decides(self):
        began = time.time()
        list(social_gate.Gated(events(3, delay=0.2), lambda: (time.sleep(0.4), None)[1]))
        self.assertLess(time.time() - began, 0.75)

    def test_a_run_error_reaches_the_caller(self):
        def bad():
            yield {"i": 0}
            raise ValueError("stage failed")
        with self.assertRaises(ValueError):
            list(social_gate.Gated(bad(), lambda: None))

    def test_a_run_that_is_never_started(self):
        rejected = social_gate.Rejected("Hi, I'm here.")
        self.assertTrue(rejected.rejected)
        self.assertEqual((list(rejected), rejected.reply), ([], "Hi, I'm here."))
        self.assertTrue(social_gate.Rejected().reply)


class ApiWiring(unittest.TestCase):
    def setUp(self):
        with open("api.py", encoding="utf-8") as handle:
            self.source = handle.read()

    def test_a_very_short_first_message_is_read_before_any_routing_plan_or_search(self):
        gate = self.source.index("social_gate.needs_early(clean)")
        self.assertLess(gate, self.source.index('"stage": "UNDERSTANDING"'))
        self.assertLess(gate, self.source.index("orchestrator.decide("))
        self.assertLess(gate, self.source.index("events = runner(asked"))
        self.assertIn("return", self.source[gate:self.source.index('"stage": "UNDERSTANDING"')])

    def test_a_plan_that_says_not_clinical_starts_no_run(self):
        check = self.source.index('plan.get("is_clinical") is False')
        self.assertLess(check, self.source.index("events = runner(asked"))
        self.assertIn("social_gate.Rejected(social_gate.converse(asked))", self.source)

    def test_nothing_in_the_api_is_a_fixed_reply(self):
        self.assertNotIn("social_reply", self.source)
        self.assertNotIn("social_gate.REPLY", self.source)


if __name__ == "__main__":
    unittest.main()
