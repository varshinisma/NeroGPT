import json
import unittest

import qa_stream as q


class MergeStage(unittest.TestCase):
    """merge_group was moved out of stream_answer unchanged: it can now be run by itself."""

    def box(self):
        plan = {"intent": "drug_treatment", "condition": "cond", "understanding": {"medicines_needed": True}, "plan_status": "ok", "skills": []}
        raw = {"pubmed_recent": json.dumps({"results": [{"pmid": "1", "title": "T", "abstract": "A b c.", "year": 2025, "date": "2025-01-01", "journal": "J", "study_type": ["Review"], "url": "u", "doi": "d"}]})}
        return {"plan": plan, "raw_evidence": raw}

    def test_a_finished_group_enters_the_state(self):
        state = q.new_state("What treats it?")
        skipped = []
        q.merge_group(state, "What treats it?", "evidence", self.box(), skipped)
        self.assertEqual(state["intent"], "drug_treatment")
        self.assertIn("1", state["papers"])
        self.assertEqual(state["medicines_needed"], True)
        self.assertTrue(state["core_stems"])

    def test_a_group_without_a_plan_adds_nothing_but_the_evidence_group_still_merges(self):
        state = q.new_state("What treats it?")
        q.merge_group(state, "What treats it?", "trials", {"raw_trials": {}}, [])
        self.assertEqual(state["intent"], None)
        q.merge_group(state, "What treats it?", "evidence", {"raw_evidence": self.box()["raw_evidence"]}, [])
        self.assertIn("1", state["papers"])

    def test_a_failed_group_is_recorded(self):
        state = q.new_state("What treats it?")
        q.merge_group(state, "What treats it?", "drugs", {"plan": self.box()["plan"], "error_drugs": "drugs retrieval crashed"}, [])
        self.assertEqual(state["source_status"]["deep_drugs"]["retrieval_status"], "failed")


if __name__ == "__main__":
    unittest.main()
