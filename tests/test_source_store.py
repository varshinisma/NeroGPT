import tempfile
import unittest
from pathlib import Path
from unittest import mock

import qa_stream
import source_store
from tests import test_stage_machine as t


def paper(pmid, text):
    return {"pmid": pmid, "title": text, "abstract": text, "year": "2023", "journal": "J", "authors": ["Smith A"], "cite": f"Smith et al., J, 2023 [PMID {pmid}]"}


class Store(unittest.TestCase):
    def test_save_load_seed_and_cover(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(source_store, "STORE_DIR", Path(d)):
            state = {"papers": {"111": paper("111", "erenumab dose in adults"), "222": paper("222", "erenumab dosing schedule")}, "trials_by_id": {"NCT1": {"nct_id": "NCT1"}}, "web": [{"url": "http://a"}]}
            source_store.save("chat-1", state)
            stored = source_store.load("chat-1")
            fresh = {"papers": {"333": paper("333", "other")}, "trials_by_id": {}, "web": []}
            counts = source_store.seed(fresh, stored)
            self.assertEqual(counts, {"papers": 2, "trials_by_id": 1, "web": 1})
            self.assertEqual(set(fresh["papers"]), {"111", "222", "333"})
            self.assertTrue(source_store.covered(fresh, "erenumab", qa_stream.COVER_WORDS["drugs"]))
            self.assertFalse(source_store.covered(fresh, "erenumab", qa_stream.COVER_WORDS["safety"]))      # the stored papers say nothing of side effects
            self.assertFalse(source_store.covered(fresh, "other drug", qa_stream.COVER_WORDS["drugs"]))
            self.assertEqual(source_store.load("nobody"), {})

    def test_a_covered_drug_is_not_searched_again(self):
        plan = {"focus": ["drugs"], "drugs": ["erenumab", "fremanezumab"], "condition": "migraine", "stored_cover": {"drugs": ["erenumab"]}}
        ops = qa_stream.dosing_paper_operations("dosage of erenumab and fremanezumab", plan, [])
        queries = " ".join(o.query for o in ops)
        self.assertNotIn("erenumab", queries)
        self.assertIn("fremanezumab", queries)

    def test_a_run_started_from_the_store_reuses_it(self):
        prior = {"papers": {"444": paper("444", "erenumab dose")}, "trials_by_id": {}, "web": []}
        plan = {"intent": "drug_treatment", "condition": "migraine", "focus": ["trials"], "drugs": ["erenumab"], "prior_sources": prior}
        # the same harness as the other stage tests, with the plan carrying the store
        import json
        events = list(self._run(plan))
        self.assertEqual(events[-1]["state"]["reused_sources"]["papers"], 1)
        self.assertIn("444", events[-1]["state"]["papers"])
        self.assertNotIn("prior_sources", json.dumps(events[-1]["state"]))

    def _run(self, plan):
        def fake_llm(*a, **k):
            yield "x"
        patches = [mock.patch.object(qa_stream, "run_jobs", t.fake_jobs(True)), mock.patch.object(qa_stream, "stream_llm", fake_llm), mock.patch.object(qa_stream, "run_llm", lambda *a, **k: "{}"),
                   mock.patch.object(qa_stream, "fast_drug_decisions", lambda s, q: None), mock.patch.object(qa_stream, "triage_evidence", lambda s, q, k: None),
                   mock.patch.object(qa_stream, "run_escalation_phase", lambda *a, **k: {}), mock.patch.object(qa_stream, "hybrid_rerank", lambda *a, **k: None),
                   mock.patch.object(qa_stream, "cross_rerank", lambda *a, **k: None), mock.patch.object(qa_stream, "routing_enabled", lambda: False), mock.patch.object(qa_stream, "legacy_deep_enabled", lambda: True)]
        for p in patches:
            p.start()
        try:
            return list(qa_stream.stream_answer("Are there trials for these?", plan=plan))
        finally:
            for p in patches:
                p.stop()


class MultiAsk(unittest.TestCase):
    def test_several_asks_in_one_message_give_several_parts(self):
        import orchestrator as o
        plan = {"info_needed": ["dosing", "safety", "trial_status"]}
        self.assertEqual(o.focus_sections(plan, {}, True, False), ["drugs", "safety", "trials"])
        events = t.run(focus=["drugs", "safety", "trials"])
        md = events[-1]["markdown"]
        self.assertIn("Clinical trials", md)
        self.assertIn("Side effects", md)
        self.assertNotIn("Clinical summary", md)


if __name__ == "__main__":
    unittest.main()
