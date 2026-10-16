"""The guideline retrieval moved from fixed config topics to model-written `guideline_sites` searches. The authoritative-domain validation (config/care_topics.json) is preserved: only pages of the
listed organisations can become guideline rows, a failed search is never an empty one, and the old rows are merged, not overwritten."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import guidelines  # noqa: E402
import qa_stream  # noqa: E402
import retrieval_routing as rr  # noqa: E402

CONFIG = guidelines.load_config()
DOMAIN = next(iter(CONFIG["authoritative_domains"]))
ORG = CONFIG["authoritative_domains"][DOMAIN]
GOOD = {"title": "Guideline for condition X in children", "url": f"https://www.{DOMAIN}/guidance/x", "snippet": "The committee recommends early behavioural support for children with condition X, with review at six months."}
OTHER = {"title": "Blog about condition X", "url": "https://someblog.example.com/post", "snippet": "A personal blog about condition X in children and what worked for us at home."}


class Filter(unittest.TestCase):
    def test_only_authoritative_organisations_are_kept_with_their_name(self):
        kept = json.loads(guidelines.filter_authoritative(json.dumps([GOOD, OTHER])))
        self.assertEqual([k["url"] for k in kept], [GOOD["url"]])
        self.assertEqual(kept[0]["organisation"], ORG)

    def test_a_search_that_completed_with_no_authoritative_page_is_a_valid_empty_result(self):
        self.assertEqual(rr.classify_raw(guidelines.filter_authoritative(json.dumps([OTHER])))[0], "completed_empty")

    def test_a_failed_search_is_never_turned_into_an_empty_one(self):
        error = json.dumps({"error": "429 too many requests"})
        self.assertEqual(guidelines.filter_authoritative(error), error)
        self.assertEqual(rr.classify_raw(guidelines.filter_authoritative(error))[0], "failed")
        self.assertEqual(rr.classify_raw(guidelines.filter_authoritative("Web search failed: boom"))[0], "failed")

    def test_excluded_hosts_are_still_excluded(self):
        for host in CONFIG.get("excluded_hosts", [])[:2]:
            kept = json.loads(guidelines.filter_authoritative(json.dumps([{**GOOD, "url": f"https://{host}/x"}])))
            self.assertEqual(kept, [])


class Rows(unittest.TestCase):
    def test_a_row_needs_the_organisation_a_title_an_excerpt_and_the_condition(self):
        rows = guidelines.rows_from_results([GOOD, OTHER], CONFIG, ["condi"])
        self.assertEqual([r["url"] for r in rows], [GOOD["url"]])
        self.assertEqual((rows[0]["organisation"], rows[0]["topic"]), (ORG, "guideline"))
        self.assertTrue(rows[0]["excerpt"])
        self.assertEqual(guidelines.rows_from_results([GOOD], CONFIG, ["zzzzz"]), [])      # the page does not mention the condition
        self.assertEqual(guidelines.rows_from_results([{**GOOD, "snippet": "short"}], CONFIG, ["condi"]), [])      # no usable excerpt

    def test_duplicates_and_junk_are_ignored(self):
        self.assertEqual(len(guidelines.rows_from_results([GOOD, GOOD, None, "x", {}], CONFIG, ["condi"])), 1)


class Ingest(unittest.TestCase):
    def state(self):
        s = qa_stream.new_state("How to treat condition X in children?")
        s["core_stems"] = ["condi"]
        return s

    def test_routed_guideline_pages_become_guideline_rows(self):
        s = self.state()
        qa_stream.ingest_guideline_ops(s, {"web_guideline_op_3": json.dumps([{**GOOD, "organisation": ORG}])})
        self.assertEqual([g["url"] for g in s["guidelines"]], [GOOD["url"]])
        self.assertTrue(s["guideline_search_ok"])
        qa_stream.ingest_guideline_ops(s, {"web_guideline_op_3": json.dumps([GOOD])})      # the same page again: not duplicated
        self.assertEqual(len(s["guidelines"]), 1)

    def test_a_failed_routed_search_adds_nothing_and_is_recorded_as_failed(self):
        s = self.state()
        raw = {"web_guideline_op_3": json.dumps({"error": "boom"})}
        qa_stream.ingest_guideline_ops(s, raw)
        self.assertEqual(s.get("guidelines", []), [])
        qa_stream.record_source_status(s, raw)
        self.assertEqual(s["source_status"]["web_guideline_op_3"]["retrieval_status"], "failed")

    def test_the_old_topic_rows_are_merged_with_the_routed_ones_not_overwritten(self):
        s = self.state()
        qa_stream.ingest_guideline_ops(s, {"web_guideline_op_3": json.dumps([GOOD])})
        legacy = {"title": "Older guideline", "url": f"https://www.{DOMAIN}/old", "excerpt": "An excerpt that is long enough to be usable here.", "organisation": ORG, "topic": "t1", "label": "T1"}
        qa_stream.ingest_guidelines(s, {"guidelines_deep": json.dumps({"results": [legacy], "failed": 0, "topics": 1})})
        self.assertEqual(sorted(g["url"] for g in s["guidelines"]), sorted([GOOD["url"], legacy["url"]]))

    def test_the_guideline_adapter_applies_the_filter_and_the_pages_reach_the_verifier(self):
        adapters = rr.make_adapters(search_pubmed=lambda *a, **k: "{}", search_europepmc_quick=lambda *a, **k: "{}", search_clinical_trials=lambda *a, **k: "{}", search_fda_labels=lambda *a, **k: "{}",
                                    guideline_search=lambda q: json.dumps([GOOD, OTHER]), regulator_search=lambda q: "[]", web_search=lambda q: "[]", anchors={"condition"}, guideline_filter=guidelines.filter_authoritative)
        op = rr.Operation(id="g", need_id="n", need="n", target="guideline_sites", query="condition X guideline recommendations children")
        raw = adapters["guideline_sites"](op)
        self.assertEqual([i["url"] for i in json.loads(raw)], [GOOD["url"]])
        s = self.state()
        qa_stream.ingest_guideline_ops(s, {"web_guideline_op_0": raw})
        import verification
        self.assertIn(verification._norm_url(GOOD["url"]), {verification._norm_url(g["url"]) for g in s["guidelines"]})      # a guideline URL is a known source for the citation check


if __name__ == "__main__":
    unittest.main()
