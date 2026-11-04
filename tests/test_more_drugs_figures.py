import unittest
import source_figures as sf
import qa_stream as qs


class Widened(unittest.TestCase):
    def test_required_keys_skip_adds_beyond_report(self):
        self.assertNotIn("adds_beyond_report", sf.REQUIRED_KEYS)
        self.assertEqual(len(sf.REQUIRED_KEYS), 4)

    def test_limits_raised(self):
        self.assertEqual(sf.MAX_FIGURES, 8)
        self.assertGreaterEqual(qs.MAX_USED_DRUGS, 24)
        self.assertGreaterEqual(qs.DRUG_PAPERS, 36)


if __name__ == "__main__":
    unittest.main()
