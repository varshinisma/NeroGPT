import time
import unittest
from unittest import mock

import qa_stream
from tests import test_stage_machine as t


def run_with_slow_escalation(early: bool, seconds: float = 0.6):
    marks = {}

    def slow_escalation(*a, **k):
        time.sleep(seconds)
        marks["escalation_ended"] = time.time()
        return {}

    def fake_llm(*a, **k):
        yield "**Latest** Erenumab helps (PMID 30000000)."
    patches = [mock.patch.object(qa_stream, "run_jobs", t.fake_jobs(True)), mock.patch.object(qa_stream, "stream_llm", fake_llm), mock.patch.object(qa_stream, "run_llm", lambda *a, **k: "{}"),
               mock.patch.object(qa_stream, "fast_drug_decisions", lambda s, q: None), mock.patch.object(qa_stream, "triage_evidence", lambda s, q, k: None),
               mock.patch.object(qa_stream, "run_escalation_phase", slow_escalation), mock.patch.object(qa_stream, "hybrid_rerank", lambda *a, **k: None),
               mock.patch.object(qa_stream, "cross_rerank", lambda *a, **k: None), mock.patch.object(qa_stream, "routing_enabled", lambda: False),
               mock.patch.object(qa_stream, "legacy_deep_enabled", lambda: True), mock.patch.object(qa_stream, "EARLY_LATEST", early)]
    for p in patches:
        p.start()
    seen, events = [], []
    try:
        for event in qa_stream.stream_answer("What is the treatment of migraine in adults?", plan={"intent": "drug_treatment", "condition": "migraine"}):
            events.append(event)
            if event["type"] == "token" and "Erenumab helps" in event.get("text", "") and event.get("stage") == "EVIDENCE_ENRICHING":
                seen.append(time.time())
    finally:
        for p in patches:
            p.stop()
    return events, seen, marks


class EarlyLatest(unittest.TestCase):
    def test_the_latest_findings_are_shown_while_the_escalation_runs(self):
        events, seen, marks = run_with_slow_escalation(True)
        self.assertTrue(seen)
        self.assertLess(seen[0], marks["escalation_ended"])      # shown before the escalation finished
        self.assertIn("early_latest_started_s", events[-1]["state"])

    def test_without_the_switch_they_come_after_the_escalation(self):
        events, seen, marks = run_with_slow_escalation(False)
        self.assertTrue(seen)
        self.assertGreaterEqual(seen[0], marks["escalation_ended"])
        self.assertNotIn("early_latest_started_s", events[-1]["state"])

    def test_the_report_text_is_the_same_either_way(self):
        early = run_with_slow_escalation(True, 0.1)[0][-1]["markdown"]
        late = run_with_slow_escalation(False, 0.1)[0][-1]["markdown"]
        strip = lambda m: [l for l in m.splitlines() if "Evidence searched on" not in l]
        self.assertEqual(strip(early), strip(late))


if __name__ == "__main__":
    unittest.main()


class OnlyJudgedPapers(unittest.TestCase):
    def test_the_early_latest_findings_wait_for_the_relevance_reading_and_read_the_judged_papers(self):
        seen = {}

        def slow_triage(state, question, skipped):
            time.sleep(0.4)
            for p in state["papers"].values():
                p["triage"] = {"relevance": "direct"}      # judged relevant

        real = qa_stream.build_evidence_context

        def spy(state, question):
            context = real(state, question)
            seen.setdefault("papers_in_context", len(context["papers"]))
            seen.setdefault("at", time.time())
            return context

        def slow_escalation(*a, **k):
            time.sleep(1.0)
            seen["escalation_ended"] = time.time()
            return {}

        def fake_llm(*a, **k):
            yield "x"
        patches = [mock.patch.object(qa_stream, "run_jobs", t.fake_jobs(True)), mock.patch.object(qa_stream, "stream_llm", fake_llm), mock.patch.object(qa_stream, "run_llm", lambda *a, **k: "{}"),
                   mock.patch.object(qa_stream, "fast_drug_decisions", lambda s, q: None), mock.patch.object(qa_stream, "triage_evidence", slow_triage),
                   mock.patch.object(qa_stream, "run_escalation_phase", slow_escalation), mock.patch.object(qa_stream, "hybrid_rerank", lambda *a, **k: None),
                   mock.patch.object(qa_stream, "cross_rerank", lambda *a, **k: None), mock.patch.object(qa_stream, "routing_enabled", lambda: False),
                   mock.patch.object(qa_stream, "legacy_deep_enabled", lambda: True), mock.patch.object(qa_stream, "build_evidence_context", spy)]
        for p in patches:
            p.start()
        try:
            events = list(qa_stream.stream_answer("What is the treatment of migraine in adults?", plan={"intent": "drug_treatment", "condition": "migraine"}))
        finally:
            for p in patches:
                p.stop()
        self.assertGreater(seen["papers_in_context"], 0)      # the writer saw the judged papers, not an empty list
        self.assertLess(seen["at"], seen["escalation_ended"])      # and it started before the escalation ended
        self.assertIn("early_latest_started_s", events[-1]["state"])
