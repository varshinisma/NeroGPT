import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import chunking
import qa_stream
import verification

ABSTRACT = ("Background: Sleep problems are common in children. Methods: Children were randomised to drug or placebo for 12 weeks. "
            "Results: Sleep onset fell by 18 minutes with the drug. Adverse events were mild. Conclusions: The drug shortened sleep onset in this trial.")


def paper(pmid="1", abstract=ABSTRACT, body=None):
    return {"pmid": pmid, "title": "Drug for sleep", "abstract": abstract, "body": body or {}, "year": 2024, "date": "2024", "type": ["Randomized Controlled Trial"], "cite": f"A et al., J, 2024 [PMID {pmid}]"}


def by_keyword(question, texts):
    return [5.0 if "Sleep onset fell" in t or "shortened" in t else -5.0 for t in texts]


class Passages(unittest.TestCase):
    def test_a_text_is_cut_into_short_windows_of_sentences(self):
        parts = chunking.passages(ABSTRACT)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(len(x) <= 420 for x in parts))
        self.assertEqual(" ".join(parts).replace("  ", " ").split()[:3], ABSTRACT.split()[:3])

    def test_a_short_text_is_one_passage_and_nothing_gives_none(self):
        self.assertEqual(len(chunking.passages("One sentence only.")), 1)
        self.assertEqual(chunking.passages(""), [])

    def test_full_text_sections_are_chunked_with_their_section_name(self):
        chunks = chunking.paper_chunks(paper(body={"results": "The score fell. It stayed lower at follow-up. No harm was seen."}))
        self.assertEqual({c["section"] for c in chunks}, {"abstract", "results"})


class Selection(unittest.TestCase):
    def test_the_best_passages_are_kept_in_reading_order(self):
        p = paper()
        trace = chunking.select("sleep onset in children", [p], by_keyword, keep=2)
        self.assertEqual(trace["papers"], 1)
        self.assertEqual(len(p["passages"]), 2)
        self.assertTrue(all(c["score"] == 5.0 for c in p["passages"]))
        positions = [ABSTRACT.index(c["text"][:20]) for c in p["passages"]]
        self.assertEqual(positions, sorted(positions))

    def test_passages_are_pieces_of_the_papers_own_text(self):
        p = paper()
        chunking.select("q", [p], by_keyword)
        blob = verification.evidence_blob({"papers": {"1": p}})
        for c in p["passages"]:
            self.assertIn(c["text"][:30], blob)

    def test_a_scorer_failure_leaves_the_paper_without_passages(self):
        def broken(question, texts):
            raise RuntimeError("model gone")
        p = paper()
        trace = chunking.select("q", [p], broken)
        self.assertNotIn("passages", p)
        self.assertIn("skipped", trace)

    def test_a_wrong_number_of_scores_is_a_failure_too(self):
        p = paper()
        trace = chunking.select("q", [p], lambda q, texts: [1.0])
        self.assertNotIn("passages", p)
        self.assertIn("skipped", trace)

    def test_switch_off(self):
        os.environ["CHUNKING"] = "off"
        try:
            self.assertFalse(chunking.enabled())
        finally:
            del os.environ["CHUNKING"]
        self.assertTrue(chunking.enabled())


class WriterView(unittest.TestCase):
    def test_the_writer_gets_the_passages_and_only_the_start_of_the_abstract(self):
        p = paper()
        chunking.select("q", [p], by_keyword, keep=2)
        view = qa_stream.paper_view(p, 800, set())
        self.assertIn("relevant_passages", view)
        self.assertLessEqual(len(view["abstract"]), 270)
        self.assertEqual(len(view["relevant_passages"]), 2)

    def test_without_passages_the_view_is_the_abstract_as_before(self):
        view = qa_stream.paper_view(paper(), 800, set())
        self.assertNotIn("relevant_passages", view)
        self.assertIn("Sleep onset fell", view["abstract"])


if __name__ == "__main__":
    unittest.main()
