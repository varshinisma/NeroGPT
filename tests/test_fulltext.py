import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import fulltext
import verification

JATS = """<article><front><article-meta><abstract><p>Background text.</p></abstract></article-meta></front><body>
<sec><title>Introduction</title><p>Why this matters.</p></sec>
<sec><title>Methods</title><p>Children were randomised to drug or placebo for 12 weeks.</p><sec><title>Outcome</title><p>The primary outcome was a symptom score.</p></sec></sec>
<sec><title>Results</title><p>The score fell by 4.2 points more with the drug.</p><table-wrap><table><tr><td>cell</td></tr></table></table-wrap></sec>
<sec><title>Discussion</title><p>Long talk.</p></sec>
<sec><title>Conclusions</title><p>The drug helped in this trial.</p></sec></body></article>"""


class Sections(unittest.TestCase):
    def test_methods_results_and_conclusion_are_found_by_their_titles_and_the_rest_is_ignored(self):
        got = fulltext.sections_from_xml(JATS)
        self.assertEqual(sorted(got), ["conclusion", "methods", "results"])
        self.assertIn("randomised", got["methods"])
        self.assertIn("primary outcome", got["methods"])      # a sub-section belongs to its section
        self.assertIn("4.2 points", got["results"])
        self.assertNotIn("cell", got["results"])
        self.assertNotIn("Why this matters", " ".join(got.values()))

    def test_unreadable_xml_gives_nothing(self):
        self.assertEqual(fulltext.sections_from_xml("<not xml"), {})

    def test_long_sections_are_cut_to_their_limit(self):
        long = {"results": "Result sentence one. " * 200, "conclusion": "First. " + "Closing sentence here. " * 80}
        cut = fulltext.shorten(long)
        self.assertLessEqual(len(cut["results"]), fulltext.LIMITS["results"])
        self.assertLessEqual(len(cut["conclusion"]), fulltext.LIMITS["conclusion"])
        self.assertTrue(cut["results"].endswith("."))


class Attach(unittest.TestCase):
    def test_only_open_access_papers_are_read_and_a_failure_leaves_the_abstract_alone(self):
        papers = [{"pmid": "1", "pmcid": "PMC1"}, {"pmid": "2", "pmcid": None}, {"pmid": "3", "pmcid": "PMC3"}, {"pmid": "4", "pmcid": "PMC4", "retracted": True}]

        def fetch(pmcid):
            if pmcid == "PMC3":
                raise RuntimeError("down")
            return JATS, []
        record = fulltext.attach(papers, fetch, cap_s=3)
        self.assertEqual((record["asked"], record["read"]), (2, 1))
        self.assertIn("body", papers[0])
        self.assertNotIn("body", papers[1])
        self.assertNotIn("body", papers[2])
        self.assertNotIn("body", papers[3])

    def test_a_paper_not_in_the_open_access_set_is_skipped(self):
        papers = [{"pmid": "1", "pmcid": "PMC1"}]
        record = fulltext.attach(papers, lambda pmcid: None, cap_s=3)
        self.assertEqual(record["read"], 0)
        self.assertNotIn("body", papers[0])

    def test_switch_off(self):
        os.environ["FULL_TEXT"] = "off"
        try:
            self.assertFalse(fulltext.enabled())
        finally:
            del os.environ["FULL_TEXT"]
        self.assertTrue(fulltext.enabled())


class Verification(unittest.TestCase):
    def test_numbers_from_the_read_sections_count_as_retrieved_text(self):
        papers = {"1": {"title": "t", "abstract": "short", "body_text": "mean change of 4.2 points", "year": 2024, "date": "2024"}}
        self.assertIn("4.2 points", verification.evidence_blob({"papers": papers}))


if __name__ == "__main__":
    unittest.main()
