import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import qa_stream


class CutOffParts(unittest.TestCase):
    def test_a_complete_part_is_left_alone(self):
        text = "## Latest findings\n\n**A study:**\n2026 (RCT)\n*Smith et al., J, 2026 [1]*\nIt found a result (PMID 123456)."
        self.assertEqual(qa_stream.trim_part(text), (text, False))

    def test_a_part_cut_inside_an_entry_loses_that_whole_entry(self):
        text = ("## Latest findings\n\n**A study:**\n2026 (RCT)\n*Smith et al., J, 2026 [1]*\nIt found a result (PMID 123456).\n\n"
                "**Another study:**\n2026 (Review)\n*Jones et al., The South African journal of")
        kept, cut = qa_stream.trim_part(text)
        self.assertTrue(cut)
        self.assertIn("A study", kept)
        self.assertNotIn("Another study", kept)
        self.assertNotIn("South African", kept)

    def test_one_cut_line_is_dropped_when_there_is_only_one_block(self):
        kept, cut = qa_stream.trim_part("Line one is complete.\nLine two stops in the mid")
        self.assertTrue(cut)
        self.assertEqual(kept, "Line one is complete.")


class NoIndirectLabels(unittest.TestCase):
    def test_no_writing_prompt_asks_for_an_indirect_label(self):
        for name in ("GROUND", "LATEST_TASK", "CURRENT_TASK"):
            text = getattr(qa_stream, name)
            self.assertNotIn("must be labelled 'indirect evidence'", text, name)
            self.assertNotIn("labelled 'indirect evidence' with", text, name)
        self.assertNotIn("Indirect evidence (different or related population)", qa_stream.CURRENT_TASK)
        self.assertIn("no label", qa_stream.LATEST_TASK)


if __name__ == "__main__":
    unittest.main()
