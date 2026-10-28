"""Finding papers by meaning and in the full text: the planner's alternative wordings, Europe PMC full-text search, the embedding call in batches with one retry, and the note when the ranking by meaning was unavailable. Offline."""
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import clinical_tools  # noqa: E402
import hybrid_search  # noqa: E402
import landscape  # noqa: E402
import qa_stream  # noqa: E402
import retrieval_routing as rr  # noqa: E402

PLAN = {"plan_status": "ok", "condition": "condition x", "pubmed_query": '("aaa bbb" OR "ccc") AND ("child" OR "pediatric") AND ("workup" OR "diagnosis")',
        "semantic_variants": ["early onset ddd", "delayed milestones"], "understanding": {"info_needed": ["diagnosis"]}}


class Variants(unittest.TestCase):
    def test_variants_are_plain_phrases(self):
        self.assertEqual(qa_stream.clean_variants(['early (delay)', '"x" OR y', "ab", "early (delay)", 5, "ORAL motor delay"]), ["early delay", "x y", "ORAL motor delay"])
        self.assertEqual(qa_stream.clean_variants("not a list"), [])
        self.assertEqual(len(qa_stream.clean_variants([f"phrase number {i}" for i in range(20)])), 5)

    def test_the_plan_jobs_search_the_variants_in_pubmed_and_the_full_text(self):
        jobs = qa_stream.plan_pubmed_jobs(PLAN)
        self.assertTrue({"pubmed_plan_variants", "pubmed_plan_epmc_fulltext", "pubmed_plan_epmc_variants", "pubmed_plan_main"} <= set(jobs))

    def test_the_full_text_search_can_be_switched_off_and_needs_no_variants(self):
        with mock.patch.dict(os.environ, {"ENABLE_FULLTEXT_SEARCH": "false"}):
            names = set(qa_stream.plan_pubmed_jobs(PLAN))
        self.assertNotIn("pubmed_plan_epmc_fulltext", names)
        self.assertNotIn("pubmed_plan_epmc_variants", names)
        self.assertIn("pubmed_plan_variants", names)
        self.assertNotIn("pubmed_plan_variants", qa_stream.plan_pubmed_jobs({**PLAN, "semantic_variants": []}))

    def test_the_variant_search_uses_every_variant_in_one_query(self):
        seen = []
        with mock.patch.object(qa_stream, "search_pubmed", lambda q, **k: seen.append(q) or "{}"), mock.patch.object(qa_stream, "cached", lambda ns, key, ttl, fn: fn()):
            qa_stream.plan_pubmed_jobs(PLAN)["pubmed_plan_variants"]()
        self.assertEqual(seen, ['"early onset ddd" OR "delayed milestones"'])


class FullText(unittest.TestCase):
    def test_europe_pmc_searches_the_body_of_open_access_articles(self):
        sent = []
        with mock.patch.object(clinical_tools, "_get", lambda url, params: sent.append(params["query"]) or json.dumps({"resultList": {"result": []}}).encode()):
            clinical_tools.search_europepmc_quick('"aaa" AND "bbb"', fulltext=True)
            clinical_tools.search_europepmc_quick('"aaa" AND "bbb"')
        self.assertTrue(sent[0].startswith('BODY:("aaa" AND "bbb")') and "OPEN_ACCESS:y" in sent[0])
        self.assertNotIn("BODY:", sent[1])

    def test_a_failed_full_text_search_is_an_error_not_a_pubmed_search(self):
        import urllib.error

        def failing(url, params):
            raise urllib.error.URLError("down")
        with mock.patch.object(clinical_tools, "_get", failing):
            self.assertIn("error", json.loads(clinical_tools.search_europepmc_quick("x y z", fulltext=True)))

    def test_a_routed_search_may_ask_for_the_full_text(self):
        self.assertTrue(rr.validate_args("europepmc", {"fulltext": True})["fulltext"])
        self.assertNotIn("fulltext", rr.validate_args("europepmc", {"fulltext": "yes"}))
        self.assertNotIn("fulltext", rr.validate_args("pubmed", {"fulltext": True}))


class Embedding(unittest.TestCase):
    def fake(self, statuses):
        calls = []

        def post(url, headers=None, json=None, timeout=None):
            calls.append(len(json["input"]))
            code = statuses.pop(0) if statuses else 200

            class R:
                status_code = code

                def raise_for_status(self):
                    if code >= 400:
                        raise RuntimeError(code)

                def json(self):
                    return {"data": [{"index": i, "embedding": [float(i), 1.0]} for i in range(len(json["input"]))]}
            return R()
        return post, calls

    def test_a_large_pool_is_embedded_in_batches(self):
        post, calls = self.fake([])
        with mock.patch.dict(os.environ, {"MISTRAL_API_KEY": "k"}), mock.patch.object(hybrid_search.httpx, "post", post), mock.patch.object(hybrid_search, "_cache", {}):
            vectors = hybrid_search.embed([f"text {i}" for i in range(60)])
        self.assertEqual(len(vectors), 60)
        self.assertEqual(sorted(calls), [12, 24, 24])

    def test_one_rate_limit_is_retried(self):
        post, calls = self.fake([429])
        with mock.patch.dict(os.environ, {"MISTRAL_API_KEY": "k"}), mock.patch.object(hybrid_search.httpx, "post", post), mock.patch.object(hybrid_search, "_cache", {}), mock.patch.object(hybrid_search.time, "sleep", lambda s: None):
            self.assertEqual(len(hybrid_search.embed(["a", "b"])), 2)
        self.assertEqual(len(calls), 2)

    def test_two_failures_give_none(self):
        post, _ = self.fake([429, 429])
        with mock.patch.dict(os.environ, {"MISTRAL_API_KEY": "k"}), mock.patch.object(hybrid_search.httpx, "post", post), mock.patch.object(hybrid_search, "_cache", {}), mock.patch.object(hybrid_search.time, "sleep", lambda s: None):
            self.assertIsNone(hybrid_search.embed(["a", "b"]))


class Limitation(unittest.TestCase):
    def test_the_limitations_say_when_the_ranking_by_meaning_was_unavailable(self):
        state = {"source_status": {}, "full_text": {"read": 0}, "hybrid_trace": {"hybrid": "lexical + source order (embedding call unavailable)"}}
        self.assertIn("meaning-based (semantic) ranking", landscape.limitations_markdown(state, []))
        state["hybrid_trace"] = {"hybrid": "semantic + lexical + source order"}
        self.assertNotIn("meaning-based", landscape.limitations_markdown(state, []))


class RankingByIntentQueries(unittest.TestCase):
    def papers(self):
        mk = lambda pmid, title, abstract: {"pmid": pmid, "title": title, "abstract": abstract}
        return {"1": mk("1", "Pneumonia in children", "antibiotics for pneumonia"), "2": mk("2", "Weather and mood", "rain and mood in adults"),
                "3": mk("3", "Chromosomal microarray yield", "genetic testing yield in global delay"), "4": mk("4", "Sleep in infants", "sleep habits"), "5": mk("5", "Diet study", "diet and weight")}

    def test_a_paper_that_matches_an_intent_query_rises_even_if_the_question_misses_it(self):
        question = "What could be done for this child?"
        with mock.patch.object(hybrid_search, "embed", lambda texts, timeout=None: None):
            plain = self.papers()
            hybrid_search.rerank(question, plain, {})
            planned = self.papers()
            trace = {}
            hybrid_search.rerank(question, planned, trace, queries=["genetic testing yield global delay"])
        self.assertEqual(list(planned)[0], "3")
        self.assertEqual(planned["3"]["hybrid"]["best_query"], "genetic testing yield global delay")
        self.assertEqual(trace["queries_used"], 2)
        self.assertNotEqual(list(plain)[0], "3")

    def test_the_semantic_part_uses_each_query_too(self):
        def fake(texts, timeout=None):
            return [[1.0, 0.0] if ("genetic" in t or "yield" in t) else [0.0, 1.0] for t in texts]
        question = "What could be done for this child?"
        papers = self.papers()
        with mock.patch.object(hybrid_search, "embed", fake):
            hybrid_search.rerank(question, papers, {}, queries=["genetic testing yield"])
        self.assertEqual(papers["3"]["hybrid"]["best_semantic_rank"], 1)

    def test_the_cross_encoder_also_reads_a_paper_with_the_query_it_matched(self):
        papers = self.papers()
        for pmid, p in papers.items():
            p["hybrid"] = {"best_query": "genetic testing" if pmid == "3" else "What could be done?"}
        seen = []

        def scorer(query, texts):
            seen.append(query)
            return [5.0 if query == "genetic testing" else -5.0 for _ in texts]
        hybrid_search.cross_rerank("What could be done?", papers, {}, top=5, scorer=scorer)
        self.assertEqual(seen, ["What could be done?", "genetic testing"])
        self.assertEqual(list(papers)[0], "3")

    def test_the_intent_queries_come_from_the_plan_the_routing_and_the_variants(self):
        state = {"routing": {"operations": [{"target": "pubmed", "query": "genetic testing yield in global developmental delay", "status": "completed_found"}, {"target": "trials", "query": "x", "status": "completed_found"}]}}
        out = qa_stream.intent_queries(state, {"condition": "condition x", "understanding": {"info_needed": ["diagnosis", "treatment_options"]}, "semantic_variants": ["early onset ddd"]})
        self.assertIn("condition x diagnosis", out)
        self.assertIn("genetic testing yield in global developmental delay", out)
        self.assertIn("condition x early onset ddd", out)
        self.assertNotIn("x", out)


if __name__ == "__main__":
    unittest.main()
