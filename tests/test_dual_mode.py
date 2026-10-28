"""Legacy and routed deep searches together: one merged set without duplicates, a routing failure that never breaks the report, validation of model-written searches, per-source rate limits,
the comparison log, and the configuration flags. Offline."""
import json
import os
import sys
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import landscape  # noqa: E402
import merge  # noqa: E402
import qa_stream  # noqa: E402
import ratelimit  # noqa: E402
import retrieval_routing as rr  # noqa: E402


def paper(pmid, title, doi=None):
    return {"pmid": pmid, "title": title, "doi": doi, "authors": ["Smith J"], "journal": "J", "year": "2026", "publication_date": "2026-01-01", "study_type": [], "abstract_excerpt": "text", "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"}


def raw(*papers):
    return json.dumps({"results": list(papers)})


class MergeWithoutDuplicates(unittest.TestCase):
    def ingest(self, data):
        papers, trials, web = {}, {}, []
        qa_stream.ingest(papers, trials, web, data)
        return papers, trials, web

    def test_the_same_pmid_from_both_modes_is_one_record_with_both_searches(self):
        p = paper("111", "Melatonin for sleep problems in children with autism spectrum disorder")
        papers, _, _ = self.ingest({"pubmed_recent": raw(p), "pubmed_op1": raw(p)})
        self.assertEqual(len(papers), 1)
        self.assertEqual(set(papers["111"]["found_by"]), {"pubmed_recent", "pubmed_op1"})

    def test_the_same_doi_under_another_pmid_is_one_record(self):
        a, b = paper("111", "A first title that is long enough to identify", doi="10.1/ABC"), paper("222", "A different wording of the title entirely", doi="https://doi.org/10.1/abc")
        papers, _, _ = self.ingest({"pubmed_recent": raw(a), "pubmed_epmc_op2": raw(b)})
        self.assertEqual(list(papers), ["111"])
        self.assertIn("pubmed_epmc_op2", papers["111"]["found_by"])

    def test_the_same_normalized_title_is_one_record(self):
        a, b = paper("111", "Genetic Testing in Neurodevelopmental Disorders."), paper("222", "genetic testing in  neurodevelopmental disorders")
        papers, _, _ = self.ingest({"pubmed_a": raw(a), "pubmed_b": raw(b)})
        self.assertEqual(len(papers), 1)

    def test_a_short_title_is_not_used_to_merge(self):
        a, b = paper("111", "Case report"), paper("222", "Case report")
        self.assertEqual(len(self.ingest({"pubmed_a": raw(a), "pubmed_b": raw(b)})[0]), 2)

    def test_a_trial_from_both_modes_is_one_record_and_a_page_from_both_is_one_page(self):
        trial = {"nct_id": "NCT00000001", "title": "t", "recruitment_status": "RECRUITING", "url": "u"}
        page = [{"title": "T", "url": "https://www.example.org/page/", "snippet": "s"}]
        again = [{"title": "T", "url": "https://example.org/page", "snippet": "s"}]
        _, trials, web = self.ingest({"trials_fast": json.dumps({"results": [trial]}), "trials_op1": json.dumps({"results": [trial]}), "web_standard": json.dumps(page), "web_op3": json.dumps(again)})
        self.assertEqual(len(trials), 1)
        self.assertEqual(set(trials["NCT00000001"]["found_by"]), {"trials_fast", "trials_op1"})
        self.assertEqual(len(web), 1)
        self.assertEqual(set(web[0]["found_by"]), {"web_standard", "web_op3"})

    def test_title_normalisation(self):
        self.assertEqual(merge.normalize_title("<i>Autism</i>: A Review!"), "autism a review")
        self.assertEqual(merge.normalize_doi("DOI: 10.1/X"), "10.1/x")


class RoutingFailureKeepsTheReport(unittest.TestCase):
    def test_an_error_a_skipped_stage_and_an_unfinished_stage_are_recorded(self):
        self.assertEqual(qa_stream.routing_outcome({"trace": {"error": "ValueError: bad"}})["retrieval_status"], "failed")
        self.assertEqual(qa_stream.routing_outcome({"trace": {"skipped": "the plan was not ready in time"}})["retrieval_status"], "failed")
        self.assertEqual(qa_stream.routing_outcome(None)["retrieval_status"], "failed")
        partial = qa_stream.routing_outcome({"unfinished": True, "trace": {}})
        self.assertEqual(partial["retrieval_status"], "partial")
        self.assertIn("timed out", partial["error"])

    def test_a_working_stage_and_a_switched_off_stage_are_not_failures(self):
        self.assertIsNone(qa_stream.routing_outcome({"trace": {"skills": {"latest-evidence": {"outcome": "ok"}}}, "ops": []}))
        self.assertIsNone(qa_stream.routing_outcome({"trace": {"skipped": "deep routing is off"}}))

    def test_every_skill_failing_to_write_a_search_is_a_failure(self):
        trace = {"skills": {"latest-evidence": {"outcome": "malformed"}, "drug-intelligence": {"outcome": "all_rejected"}}}
        self.assertEqual(qa_stream.routing_outcome({"trace": trace, "ops": []})["retrieval_status"], "failed")

    def test_the_limitations_name_the_failure_first_and_the_report_is_still_written(self):
        state = {"source_status": {"deep_routing": {"retrieval_status": "failed", "evidence_status": "unavailable", "error": "boom"}, "pubmed_fast": {"retrieval_status": "success", "evidence_status": "found", "error": None}},
                 "full_text": {"read": 0}}
        text = landscape.limitations_markdown(state, [])
        self.assertEqual(text.splitlines()[2], "- Deep routing unavailable; results use the standard search only.")
        self.assertNotIn("deep-retrieval stage", text)
        self.assertIn("Findings are drawn from abstracts", text)

    def test_a_routing_stage_whose_model_always_fails_still_has_baseline_searches(self):
        def broken(*args, **kwargs):
            raise RuntimeError("model unavailable")
        plan = {"intent": "x", "condition": "condition x", "skills": ["latest-evidence"], "drugs": [], "plan_status": "ok", "understanding": {"info_needed": ["diagnosis"]}}
        routed = qa_stream.route_deep("What helps children with condition x?", plan, ask=broken)
        self.assertTrue(routed["ops"])      # the question itself as a literature search: added by code, with no model call
        self.assertTrue(all(o.origin == "baseline" for o in routed["ops"]))


class ModelWrittenSearchesAreChecked(unittest.TestCase):
    QUESTION = "What medication is used for tics in children?"
    ANCHORS = rr.anchor_words(QUESTION, "tics")

    def test_an_unknown_source_is_rejected(self):
        ops, rejected = rr.parse_operations("latest-evidence", {"queries": [{"need": "n", "target": "twitter", "query": "tics children medication"}]}, self.QUESTION, self.ANCHORS)
        self.assertEqual(ops, [])
        self.assertTrue(rejected)

    def test_an_invalid_or_off_topic_query_is_rejected(self):
        for query in ("", "football league results this season champions", 12345):
            ops, rejected = rr.parse_operations("latest-evidence", {"queries": [{"need": "n", "target": "pubmed", "query": query}]}, self.QUESTION, self.ANCHORS)
            self.assertEqual(ops, [], query)
            self.assertTrue(rejected, query)

    def test_a_reply_that_is_not_a_list_of_queries_gives_no_search_and_one_rejection(self):
        ops, rejected = rr.parse_operations("latest-evidence", {"queries": "pubmed: tics"}, self.QUESTION, self.ANCHORS)
        self.assertEqual((ops, len(rejected)), ([], 1))

    def test_the_number_of_routed_searches_is_capped(self):
        items = [{"need": f"n{i}", "target": "pubmed", "query": f"tics children medication trial variant{i} outcomes"} for i in range(30)]
        ops, _ = rr.parse_operations("latest-evidence", {"queries": items}, self.QUESTION, self.ANCHORS, max_ops=5)
        self.assertLessEqual(len(ops), 5)
        self.assertEqual(rr.MAX_OPERATIONS, int(os.getenv("ROUTING_MAX_QUERIES", "14")))


class RateLimit(unittest.TestCase):
    def test_calls_to_one_source_are_spaced_by_its_rate_with_a_fake_clock(self):
        now = [0.0]
        waits = []

        def sleep(seconds):
            waits.append(seconds)
            now[0] += seconds
        limiter = ratelimit.SourceLimiter(clock=lambda: now[0], sleep=sleep)
        with mock.patch.dict(os.environ, {"RATE_PUBMED_PER_S": "3"}):
            starts = []
            for _ in range(7):
                limiter.acquire("pubmed")
                starts.append(now[0])
        gaps = [b - a for a, b in zip(starts, starts[1:])]
        self.assertTrue(all(g >= 1 / 3 - 1e-9 for g in gaps))
        self.assertLessEqual(sum(1 for s in starts if s < 1.0), 3)      # at most 3 requests start within the first second

    def test_sources_are_limited_independently(self):
        now = [0.0]
        limiter = ratelimit.SourceLimiter(clock=lambda: now[0], sleep=lambda s: now.__setitem__(0, now[0] + s))
        limiter.acquire("pubmed")
        self.assertEqual(limiter.acquire("fda"), 0.0)

    def test_threads_together_never_exceed_the_rate(self):
        limiter = ratelimit.SourceLimiter()
        stamps, lock = [], threading.Lock()

        def call():
            limiter.acquire("fda")
            with lock:
                stamps.append(time.monotonic())
        with mock.patch.dict(os.environ, {"RATE_FDA_PER_S": "20"}):
            threads = [threading.Thread(target=call) for _ in range(12)]
            [t.start() for t in threads]
            [t.join() for t in threads]
        stamps.sort()
        for i in range(len(stamps)):
            inside = [s for s in stamps if stamps[i] <= s < stamps[i] + 1.0]
            self.assertLessEqual(len(inside), 21)      # 20 per second, plus the call that starts the window

    def test_pubmed_stays_under_three_per_second_without_a_key_and_ten_with_one(self):
        env = {k: v for k, v in os.environ.items() if k not in ("RATE_PUBMED_PER_S", "NCBI_API_KEY")}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(ratelimit.per_second("pubmed"), 2.5)
            os.environ["NCBI_API_KEY"] = "x"
            self.assertEqual(ratelimit.per_second("pubmed"), 8.0)

    def test_every_data_source_goes_through_the_limiter(self):
        self.assertEqual(ratelimit.source_of_url("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"), "pubmed")
        self.assertEqual(ratelimit.source_of_url("https://www.ebi.ac.uk/europepmc/webservices/rest/search"), "europepmc")
        self.assertEqual(ratelimit.source_of_url("https://clinicaltrials.gov/api/v2/studies"), "clinicaltrials")
        self.assertEqual(ratelimit.source_of_url("https://api.fda.gov/drug/label.json"), "fda")


class ComparisonLogAndFlags(unittest.TestCase):
    def test_the_log_counts_what_each_mode_found_and_what_the_other_did_not(self):
        data = {"pubmed_recent": raw(paper("1", "a" * 30), paper("2", "b" * 30)), "pubmed_op1": raw(paper("2", "b" * 30), paper("3", "c" * 30), paper("4", "d" * 30)),
                "trials_op2": json.dumps({"error": "timed out"})}
        log = merge.search_log(data, {"pubmed_op1", "trials_op2"})
        rows = {r["search"]: r for r in log["searches"]}
        self.assertEqual((rows["pubmed_recent"]["mode"], rows["pubmed_recent"]["returned"], rows["pubmed_recent"]["new_vs_other_mode"]), ("legacy", 2, 1))
        self.assertEqual((rows["pubmed_op1"]["mode"], rows["pubmed_op1"]["returned"], rows["pubmed_op1"]["new_vs_other_mode"]), ("routing", 3, 2))
        self.assertTrue(rows["trials_op2"]["error"])
        self.assertEqual(log["summary"], {"legacy": 2, "legacy_only": 1, "routing": 3, "routing_only": 2, "overlap": 1})
        self.assertIn("overlap: 1", log["summary_line"])

    def test_the_new_flags_switch_each_mode_and_the_old_names_still_work(self):
        with mock.patch.dict(os.environ, {"ENABLE_LEGACY_SEARCH": "false", "ENABLE_DEEP_ROUTING": "true", "DEEP_ROUTING": "off", "LEGACY_DEEP": "on"}):
            self.assertFalse(qa_stream.legacy_deep_enabled())
            self.assertTrue(qa_stream.routing_enabled())      # the new name wins over the old one
        env = {k: v for k, v in os.environ.items() if k not in ("ENABLE_LEGACY_SEARCH", "ENABLE_DEEP_ROUTING")}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.dict(os.environ, {"DEEP_ROUTING": "off", "LEGACY_DEEP": "off"}):
            self.assertFalse(qa_stream.routing_enabled())
            self.assertFalse(qa_stream.legacy_deep_enabled())

    def test_the_routed_calls_do_not_use_the_planner_model(self):
        import inspect
        self.assertEqual(inspect.signature(qa_stream.ask_json).parameters["purpose"].default, "triage")      # the planner setting applies to purpose="planning" only
        with mock.patch.dict(os.environ, {"PLANNER_PROVIDER": "groq", "PLANNER_MODEL": "some/model"}):
            self.assertEqual(qa_stream.planner_choice(), ("groq", "some/model"))
            self.assertNotEqual(qa_stream.model_id_for("triage"), "some/model")


if __name__ == "__main__":
    unittest.main()
