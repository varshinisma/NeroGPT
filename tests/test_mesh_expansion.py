"""MeSH expansion and PubMed 'similar articles': a lookup of the vocabulary's other names for the condition words, added to the deep searches (legacy and routed) and never to the fast search. No network: the NCBI calls are replaced."""
import inspect
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import clinical_tools
import mesh_expand
import qa_stream
import retrieval_routing as rr

HEADING = ["Heading Alpha", "Alpha Heading, Old", "Alpha name", "Alpha other name", "A very long name " * 5]


def fake_get(url, params):
    """NCBI's MeSH search and summary for one made-up term ('alpha'); every other term has no heading."""
    if url.endswith("esearch.fcgi"):
        if "alpha" in params["term"]:
            return json.dumps({"esearchresult": {"querytranslation": '"heading alpha"[MeSH Terms] OR alpha[Text Word]', "idlist": ["900", "901"]}}).encode()
        return json.dumps({"esearchresult": {"querytranslation": "beta[All Fields]", "idlist": ["902"]}}).encode()
    return json.dumps({"result": {"900": {"ds_meshterms": ["Other Heading", "Other name"]}, "901": {"ds_meshterms": HEADING}}}).encode()


class IsolatedCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        patches = [mock.patch.object(mesh_expand, "CACHE_FILE", Path(self.tmp.name) / "cache" / "mesh.json"), mock.patch.object(mesh_expand, "_memory", {}),
                   mock.patch.object(mesh_expand, "_disk_loaded", False)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.tmp.cleanup)


class Lookup(IsolatedCache):
    def test_the_heading_comes_from_the_sources_own_mapping_not_the_id_order(self):
        with mock.patch.object(mesh_expand, "_get", fake_get):
            record = mesh_expand.lookup("alpha")
        self.assertEqual(record["heading"], "Heading Alpha")
        self.assertEqual(record["entry_terms"], ["Alpha name", "Alpha other name"])      # the inverted form (comma) and the very long name are dropped

    def test_a_term_without_a_heading_adds_nothing(self):
        with mock.patch.object(mesh_expand, "_get", fake_get):
            self.assertEqual(mesh_expand.lookup("beta"), {"heading": None, "entry_terms": []})
            self.assertEqual(mesh_expand.clauses("beta", mesh_expand.lookup("beta")), [])

    def test_a_failed_lookup_is_none_and_is_not_remembered(self):
        with mock.patch.object(mesh_expand, "_get", side_effect=OSError("down")):
            self.assertIsNone(mesh_expand.lookup("alpha"))
        with mock.patch.object(mesh_expand, "_get", fake_get):
            self.assertEqual(mesh_expand.lookup("alpha")["heading"], "Heading Alpha")

    def test_a_term_is_looked_up_once_and_kept_on_disk(self):
        calls = []

        def counting(url, params):
            calls.append(url)
            return fake_get(url, params)
        with mock.patch.object(mesh_expand, "_get", counting):
            mesh_expand.lookup("alpha")
            mesh_expand.lookup("  ALPHA ")
        self.assertEqual(len(calls), 2)      # a search and a summary, once
        self.assertIn("heading alpha", json.loads(mesh_expand.CACHE_FILE.read_text(encoding="utf-8")).get("alpha", {}).get("heading", "").lower())

    def test_each_source_gets_its_own_syntax(self):
        record = {"heading": "Heading Alpha", "entry_terms": ["Alpha name"]}
        self.assertEqual(mesh_expand.clauses("alpha", record, "pubmed"), ['"Heading Alpha"[MeSH Terms]', '"Heading Alpha"', '"Alpha name"'])
        self.assertEqual(mesh_expand.clauses("alpha", record, "europepmc"), ['MESH:"Heading Alpha"', '"Heading Alpha"', '"Alpha name"'])

    def test_a_late_or_failed_lookup_leaves_the_term_out(self):
        with mock.patch.object(mesh_expand, "lookup", side_effect=lambda t: None if t == "gamma" else {"heading": "Heading Alpha", "entry_terms": []}):
            found = mesh_expand.expand(["alpha", "gamma"], "pubmed")
        self.assertEqual(list(found), ["alpha"])


class RoutedWait(IsolatedCache):
    def test_a_routed_search_waits_only_briefly_and_the_late_lookup_is_kept_for_the_next_search(self):
        import threading
        release = threading.Event()

        def slow_fetch(term):
            release.wait(5)
            return {"heading": "Heading Alpha", "entry_terms": []}
        with mock.patch.object(mesh_expand, "_fetch", slow_fetch), mock.patch.object(mesh_expand, "ROUTING_WAIT_S", 0.2):
            started = __import__("time").time()
            first = mesh_expand.clauses_for_terms(["alpha", "delta"], ["alpha"], "pubmed")
            waited = __import__("time").time() - started
            release.set()
            for _ in range(50):      # the lookup that was late finishes in the background
                if "alpha" in mesh_expand._memory:
                    break
                __import__("time").sleep(0.05)
            second = mesh_expand.clauses_for_terms(["alpha", "delta"], ["alpha"], "pubmed")
        self.assertEqual(first, {})
        self.assertLess(waited, 2.0)
        self.assertIn("alpha", second)


class Routing(unittest.TestCase):
    ANCHORS = {"alpha", "delta"}

    def test_a_routed_search_without_mesh_is_written_as_before(self):
        self.assertEqual(rr.boolean_form("alpha delta treatment options", self.ANCHORS), rr.boolean_form("alpha delta treatment options", self.ANCHORS, expand=None))

    def test_a_condition_made_of_several_required_words_is_widened_as_one_term(self):
        form = rr.boolean_form("alpha beta gamma delta treatment", {"alpha", "beta", "gamma", "delta"}, expand=lambda words: {"alpha beta": ['"Heading AB"[MeSH Terms]']})
        self.assertIn('((alpha AND beta) OR "Heading AB"[MeSH Terms])', form)
        self.assertEqual(form.count("beta"), 1)      # its words are written once, inside the term
        self.assertIn(" AND gamma", form)

    def test_only_the_plans_condition_terms_that_a_routed_search_requires_are_looked_up(self):
        asked = []
        with mock.patch.object(mesh_expand, "expand", side_effect=lambda terms, target, timeout=None: asked.append(terms) or {}):
            mesh_expand.clauses_for_terms(["alpha", "beta", "year"], ["alpha beta", "gamma delta", "alpha"], "pubmed")
        self.assertEqual(asked, [["alpha beta", "alpha"]])      # "year" and "gamma delta" are never looked up

    def test_the_required_words_get_their_alternatives(self):
        form = rr.boolean_form("alpha delta treatment options", self.ANCHORS, expand=lambda words: {"alpha": ['"Heading Alpha"[MeSH Terms]', '"Alpha name"']})
        self.assertIn('(alpha OR "Heading Alpha"[MeSH Terms] OR "Alpha name")', form)
        self.assertIn(" AND delta", form)

    def test_both_routed_text_sources_use_it_and_the_flag_off_does_not(self):
        seen = {}

        def fake_pubmed(query, **kw):
            seen["pubmed"] = query
            return "{}"

        def fake_epmc(query, **kw):
            seen["epmc"] = query
            return "{}"
        calls = []

        def mesh(words, target):
            calls.append(target)
            return {words[0]: [f"EXTRA-{target}"]}
        op = mock.Mock(query="alpha delta treatment", args={})
        none = lambda *a, **k: "{}"
        adapters = rr.make_adapters(search_pubmed=fake_pubmed, search_europepmc_quick=fake_epmc, search_clinical_trials=none, search_fda_labels=none, guideline_search=none, regulator_search=none,
                                    web_search=none, anchors={"alpha", "delta"}, mesh=mesh)
        adapters["pubmed"](op)
        adapters["europepmc"](op)
        self.assertIn("EXTRA-pubmed", seen["pubmed"])
        self.assertIn("EXTRA-europepmc", seen["epmc"])
        self.assertEqual(calls, ["pubmed", "europepmc"])
        plain = rr.make_adapters(search_pubmed=fake_pubmed, search_europepmc_quick=fake_epmc, search_clinical_trials=none, search_fda_labels=none, guideline_search=none, regulator_search=none,
                                 web_search=none, anchors={"alpha", "delta"})
        plain["pubmed"](op)
        self.assertNotIn("EXTRA", seen["pubmed"])

    def test_the_real_adapters_follow_the_flag(self):
        with mock.patch.dict(os.environ, {"ENABLE_MESH_EXPANSION": "false"}), mock.patch.object(qa_stream.rr, "make_adapters") as made:
            qa_stream.real_adapters({"alpha"})
        self.assertIsNone(made.call_args.kwargs["mesh"])
        with mock.patch.dict(os.environ, {"ENABLE_MESH_EXPANSION": "true"}), mock.patch.object(qa_stream.rr, "make_adapters") as made:
            qa_stream.real_adapters({"alpha"})
        self.assertIsNotNone(made.call_args.kwargs["mesh"])

    def test_the_anchors_carry_the_plans_condition_terms(self):
        plan = {"condition": "condition x", "plan_status": "ok", "pubmed_query": '("aaa bbb" OR "ccc") AND ("child") AND ("workup" OR "diagnosis")', "understanding": {}}
        anchors = qa_stream.routing_anchors("What work-up does a child with condition x need?", plan)
        self.assertEqual(anchors.mesh_terms, ["condition x", "aaa bbb", "ccc"])
        self.assertIsInstance(anchors, set)


class Legacy(unittest.TestCase):
    def setUp(self):
        patch = mock.patch.object(mesh_expand, "prefetch")
        self.prefetch = patch.start()
        self.addCleanup(patch.stop)

    PLAN = {"condition": "condition x", "plan_status": "ok", "pubmed_query": '("aaa bbb" OR "ccc") AND ("workup" OR "diagnosis")', "understanding": {"info_needed": []}}

    def test_the_terms_are_the_condition_and_the_subject_phrases(self):
        self.assertEqual(qa_stream.mesh_terms(self.PLAN, '("aaa bbb"[tiab] OR "ccc") AND ("child" OR "pediatric")'), ["condition x", "aaa bbb", "ccc"])      # the population group is not looked up

    def test_the_intent_groups_are_kept_and_the_condition_is_widened(self):
        seen = []
        with mock.patch.object(mesh_expand, "records", return_value={"condition x": {"heading": "Heading Alpha", "entry_terms": []}}), \
                mock.patch.object(qa_stream, "search_pubmed", side_effect=lambda q, **k: seen.append(q) or "{}"), mock.patch.object(qa_stream, "cached", side_effect=lambda kind, key, ttl, fn: fn()):
            jobs = qa_stream.mesh_jobs(self.PLAN, *qa_stream.planned_pubmed_query(self.PLAN))
            jobs["pubmed_plan_mesh"]()
        self.assertEqual(seen, ['("aaa bbb" OR "ccc" OR "Heading Alpha"[MeSH Terms] OR "Heading Alpha") AND ("workup" OR "diagnosis")'])

    def test_the_population_group_is_kept_as_written_and_only_the_condition_group_is_widened(self):
        plan = {**self.PLAN, "pubmed_query": '("aaa bbb" OR "ccc") AND ("child" OR "pediatric") AND ("workup" OR "diagnosis")'}
        seen = []
        with mock.patch.object(mesh_expand, "records", return_value={"aaa bbb": {"heading": "Heading Alpha", "entry_terms": []}}) as looked_up,                 mock.patch.object(qa_stream, "search_pubmed", side_effect=lambda q, **k: seen.append(q) or "{}"), mock.patch.object(qa_stream, "cached", side_effect=lambda kind, key, ttl, fn: fn()):
            qa_stream.mesh_jobs(plan, *qa_stream.planned_pubmed_query(plan))["pubmed_plan_mesh"]()
        self.assertEqual(looked_up.call_args.args[0], ["condition x", "aaa bbb", "ccc"])
        self.assertEqual(seen, ['("aaa bbb" OR "ccc" OR "Heading Alpha"[MeSH Terms] OR "Heading Alpha") AND ("child" OR "pediatric") AND ("workup" OR "diagnosis")'])

    def test_the_pubmed_and_europe_pmc_searches_share_one_set_of_lookups(self):
        seen = []
        with mock.patch.object(mesh_expand, "records", return_value={"condition x": {"heading": "Heading Alpha", "entry_terms": ["Alpha name"]}}) as looked_up, \
                mock.patch.object(qa_stream, "search_pubmed", return_value="{}"), mock.patch.object(qa_stream, "search_europepmc_quick", side_effect=lambda q, **k: seen.append(q) or "{}"), \
                mock.patch.object(qa_stream, "cached", side_effect=lambda kind, key, ttl, fn: fn()):
            jobs = qa_stream.mesh_jobs(self.PLAN, *qa_stream.planned_pubmed_query(self.PLAN))
            jobs["pubmed_plan_mesh"]()
            jobs["pubmed_plan_epmc_mesh"]()
        self.assertEqual(looked_up.call_count, 1)
        self.assertIn('MESH:"Heading Alpha"', seen[0])

    def test_a_condition_without_a_heading_runs_no_search(self):
        with mock.patch.object(mesh_expand, "records", return_value={}), mock.patch.object(qa_stream, "search_pubmed") as search:
            jobs = qa_stream.mesh_jobs(self.PLAN, *qa_stream.planned_pubmed_query(self.PLAN))
            out = json.loads(jobs["pubmed_plan_mesh"]())
        search.assert_not_called()
        self.assertEqual(out["results"], [])

    def test_the_flag_turns_the_jobs_off(self):
        with mock.patch.dict(os.environ, {"ENABLE_MESH_EXPANSION": "false"}):
            self.assertEqual(qa_stream.mesh_jobs(self.PLAN, *qa_stream.planned_pubmed_query(self.PLAN)), {})
            self.assertFalse({"pubmed_plan_mesh", "pubmed_plan_epmc_mesh"} & set(qa_stream.plan_pubmed_jobs(self.PLAN)))


class Similar(unittest.TestCase):
    @staticmethod
    def raw(*pmids, retracted=()):
        return json.dumps({"results": [{"pmid": p, "study_type": ["Retracted Publication"] if p in retracted else []} for p in pmids]})

    def test_the_seeds_are_the_first_papers_of_the_main_search(self):
        raw = {"pubmed_recent": self.raw("9"), "pubmed_plan_main": self.raw("1", "2", "3", "4", "5", "6"), "trials_x": "{}"}
        self.assertEqual(qa_stream.seed_pmids(raw), ["1", "2", "3", "4", "5"])

    def test_a_paper_found_by_several_searches_comes_before_one_that_is_only_first(self):
        raw = {"pubmed_plan_main": self.raw("1", "2", "3"), "pubmed_bestmatch": self.raw("8", "3"), "pubmed_plan_diagnosis": self.raw("9", "3", "2")}
        self.assertEqual(qa_stream.seed_pmids(raw, limit=3), ["3", "2", "1"])      # 3 by three searches, 2 by two, then the best-placed of the rest

    def test_a_paper_only_a_newest_first_search_returned_comes_last(self):
        raw = {"pubmed_plan_main": self.raw("1"), "pubmed_recent": self.raw("7")}
        self.assertEqual(qa_stream.seed_pmids(raw), ["1", "7"])

    def test_a_term_asked_by_two_searches_at_once_is_fetched_once(self):
        import threading
        calls = []

        def slow(term):
            calls.append(term)
            import time
            time.sleep(0.2)
            return {"heading": None, "entry_terms": []}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(mesh_expand, "CACHE_FILE", Path(tmp) / "m.json"), mock.patch.object(mesh_expand, "_memory", {}), \
                mock.patch.object(mesh_expand, "_disk_loaded", True), mock.patch.object(mesh_expand, "_fetch", slow):
            threads = [threading.Thread(target=mesh_expand.lookup, args=("alpha",)) for _ in range(4)]
            [t.start() for t in threads]
            [t.join() for t in threads]
        self.assertEqual(calls, ["alpha"])

    def test_a_retracted_paper_is_not_a_seed_and_the_other_searches_fill_up(self):
        raw = {"pubmed_plan_main": self.raw("1", "2", retracted=("1",)), "pubmed_recent": self.raw("2", "3")}
        self.assertEqual(qa_stream.seed_pmids(raw), ["2", "3"])

    def test_the_neighbours_exclude_the_seeds_and_the_best_scores_come_first(self):
        linked = {"linksets": [{"linksetdbs": [{"linkname": "pubmed_pubmed", "links": [{"id": "1", "score": 99}, {"id": "20", "score": 50}, {"id": "21", "score": 40}, {"id": "22", "score": 30}]}]},
                               {"linksetdbs": [{"linkname": "pubmed_pubmed", "links": [{"id": "2", "score": 99}, {"id": "21", "score": 70}, {"id": "23", "score": 10}]}]}]}
        captured = {}
        with mock.patch.object(clinical_tools, "_get", return_value=json.dumps(linked).encode()) as get, \
                mock.patch.object(clinical_tools, "search_pubmed_quick", side_effect=lambda q, **k: captured.update(q=q, **k) or "{}"):
            clinical_tools.similar_articles(["1", "2"], per_seed=2, max_results=10)
        self.assertEqual(get.call_args.args[1]["id"], ["1", "2"])
        self.assertEqual(captured["q"], "21[uid] OR 20[uid] OR 23[uid]")      # per seed the best two, once each, best score first; the seeds are not returned

    def test_a_failure_is_reported_not_raised(self):
        with mock.patch.object(clinical_tools, "_get", side_effect=urllib.error.URLError("down")):
            self.assertIn("error", json.loads(clinical_tools.similar_articles(["1"])))
        self.assertEqual(json.loads(clinical_tools.similar_articles([]))["results"], [])


class FastSearchUntouched(unittest.TestCase):
    def test_the_fast_search_knows_nothing_about_the_expansion(self):
        for fn in (qa_stream.fast_jobs, qa_stream._fast_jobs, clinical_tools.search_pubmed_quick, clinical_tools.search_europepmc_quick, clinical_tools.search_pubmed):
            source = inspect.getsource(fn).lower()
            self.assertNotIn("mesh", source)
            self.assertNotIn("similar", source)


if __name__ == "__main__":
    unittest.main()
