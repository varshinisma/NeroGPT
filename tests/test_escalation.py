"""Offline tests of the escalation phase and of triage_evidence(only=...): fake model replies, fake search tools, no network.

The phase runs BEFORE the evidence writer: wave 1 (needs that found nothing or failed) overlaps the relevance reading; wave 2 (needs whose records were all judged irrelevant) follows it; the new records are read
without marking the old ones uncertain; each need escalates at most once; supporting needs never; one hard deadline bounds everything and unfinished work is reported.
"""
import json
import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import qa_stream  # noqa: E402
import retrieval_routing as rr  # noqa: E402
from tests.test_routing import EMPTY, ERROR, TRANSIENT, FakeAdapters  # noqa: E402
from tests.test_routing_stage import DRUGS, FakeModel, LATEST, QUESTION, plan  # noqa: E402


def papers_json(*pmids):
    return json.dumps({"results": [{"pmid": p, "title": f"Title {p}", "year": 2024, "authors": ["A B"], "journal": "J", "abstract_excerpt": f"Abstract {p}. It reports a result.",
                                    "study_type": ["Randomized Controlled Trial"], "publication_date": "2024-01-01"} for p in pmids]})


def web_json(*urls):
    return json.dumps([{"title": f"Page {u}", "url": f"https://{u}.org/x", "snippet": f"Snippet about {u}."} for u in urls])


def finished_thread():
    t = threading.Thread(target=lambda: None)
    t.start()
    t.join()
    return t


def make_state():
    state = qa_stream.new_state(QUESTION)
    return state


class TriageOfNewRecordsOnly(unittest.TestCase):
    def setUp(self):
        self.original = qa_stream.ask_json
        self.offered = []

        def fake(question, task, payload, max_tokens, skill="x"):
            ids = [p["id"] for p in payload.get("papers", [])]
            self.offered += ids
            return {"papers": [f"{i}|direct|emerging|1" for i in ids]}
        qa_stream.ask_json = fake

    def tearDown(self):
        qa_stream.ask_json = self.original

    def state_with(self, old, new):
        state = make_state()
        qa_stream.ingest(state["papers"], state["trials_by_id"], state["web"], {"pubmed_fast": papers_json(*old, *new)})
        for pmid in old:
            state["papers"][pmid]["triage"] = {"relevance": "indirect", "role": None, "population": "", "evidence": "old"}
        return state

    def test_only_the_new_records_are_read_and_the_old_decisions_stay(self):
        state = self.state_with(["1", "2"], ["3", "4"])
        qa_stream.triage_evidence(state, QUESTION, [], only={"3", "4"})
        self.assertEqual(sorted(self.offered), ["3", "4"])
        self.assertEqual(state["papers"]["1"]["triage"]["evidence"], "old")
        self.assertEqual(state["papers"]["3"]["triage"]["relevance"], "direct")

    def test_records_outside_the_set_are_never_marked_uncertain(self):
        state = self.state_with(["1"], ["3"])
        state["papers"]["9"] = {**state["papers"]["1"], "pmid": "9"}      # a record nobody has read yet and that is not in the set
        del state["papers"]["9"]["triage"]
        qa_stream.triage_evidence(state, QUESTION, [], only={"3"})
        self.assertNotIn("triage", state["papers"]["9"])

    def test_a_failed_reading_marks_only_the_new_records_uncertain(self):
        state = self.state_with(["1"], ["3"])

        def broken(*args, **kwargs):
            raise RuntimeError("model down")
        qa_stream.ask_json = broken
        skipped = []
        qa_stream.triage_evidence(state, QUESTION, skipped, only={"3"})
        self.assertEqual(state["papers"]["3"]["triage"]["relevance"], "uncertain")
        self.assertEqual(state["papers"]["1"]["triage"]["evidence"], "old")
        self.assertTrue(skipped)

    def test_the_first_reading_still_reads_the_first_records_and_marks_the_rest(self):
        state = make_state()
        qa_stream.ingest(state["papers"], state["trials_by_id"], state["web"], {"pubmed_fast": papers_json(*[str(i) for i in range(1, 9)])})
        qa_stream.triage_evidence(state, QUESTION, [])
        self.assertTrue(all("triage" in p for p in state["papers"].values()))


class Phase(unittest.TestCase):
    def setUp(self):
        self.original_ask, self.original_triage = qa_stream.ask_json, qa_stream.triage_evidence
        self.triage_calls = []
        self.relevance = {}      # pmid -> relevance given by the fake reading (default direct)

        def fake_triage(state, question, skipped, only=None):
            self.triage_calls.append(None if only is None else set(only))
            for pmid, p in state["papers"].items():
                if (only is None or pmid in only) and not p.get("triage"):
                    p["triage"] = {"relevance": self.relevance.get(pmid, "direct"), "role": None, "population": "", "evidence": "x"}
            for nct, t in state["trials_by_id"].items():
                if (only is None or nct in only) and not t.get("triage"):
                    t["triage"] = {"relevance": "direct", "kind": "treatment_trial", "evidence": "x"}
        qa_stream.triage_evidence = fake_triage

    def tearDown(self):
        qa_stream.ask_json, qa_stream.triage_evidence = self.original_ask, self.original_triage

    def routed(self, script, latest=LATEST, drugs=DRUGS, p=None, budget=5):
        """The first round, run for real through the routing stage with fake sources; returns (state, stage, adapters) with the first-round results already ingested."""
        adapters = FakeAdapters(script)
        stage = qa_stream.run_routing_stage(QUESTION, p or plan(), make_adapters=lambda anchors: adapters, budget_s=budget, ask=FakeModel({"latest-evidence": [latest], "drug-intelligence": [drugs]}))
        state = make_state()
        qa_stream.ingest_routed(state, stage["raw"])
        qa_stream.triage_evidence(state, QUESTION, [])      # the relevance reading of the first round (the real one runs in the triage thread)
        self.triage_calls.clear()
        return state, stage, adapters

    def phase(self, state, stage, adapters, ask=None, scorer=None, deadline=10, triage_thread=None):
        qa_stream.ask_json = ask or FakeModel({"latest-evidence": [{"queries": []}]})
        skipped = []
        record = qa_stream.run_escalation_phase(state, QUESTION, plan(), stage, triage_thread or finished_thread(), skipped, make_adapters=lambda anchors: adapters, scorer=scorer or (lambda q, texts: [5.0] * len(texts)),
                                                ask=qa_stream.ask_json, deadline_s=deadline)
        return record, skipped

    def alternates(self, **per_need):
        return FakeModel({"latest-evidence": [{"queries": [{"need_id": n, "target": t, "query": q, "priority": "essential"} for n, (t, q) in per_need.items()]}]})

    def test_an_essential_need_that_found_nothing_is_tried_once_in_an_alternate_source(self):
        state, stage, adapters = self.routed({"pubmed": [EMPTY], "guideline_sites": [web_json("g")], "fda_label": [EMPTY], "regulator": [web_json("r")]})
        record, skipped = self.phase(state, stage, adapters, ask=self.alternates(head=("europepmc", "sertraline versus fluoxetine adolescents depression efficacy comparison trial"),
                                                                                 l1=("regulator", "sertraline fluoxetine adolescent depression label update")))
        searches = [s for w in record["waves"] for s in w["searches"]]
        self.assertTrue(searches)
        self.assertTrue(all(s["from"]["round"] == 1 for s in searches))
        self.assertEqual({s["need_id"] for s in searches}, {s["need_id"] for s in searches})
        self.assertEqual(len({s["need_id"] for s in searches}), len(searches))      # at most one escalation per need
        self.assertEqual(adapters.targets_called().count("europepmc"), 1)

    def test_a_failed_search_escalates_with_the_reason_failed_and_is_never_called_no_evidence(self):
        state, stage, adapters = self.routed({"pubmed": [ERROR], "guideline_sites": [web_json("g")], "fda_label": [found_labels()], "regulator": [web_json("r")]}, drugs={"queries": []})
        record, _ = self.phase(state, stage, adapters, ask=self.alternates(head=("europepmc", "sertraline versus fluoxetine adolescents depression efficacy comparison trial")))
        first = next(w for w in record["waves"] if w["wave"] == 1)["searches"][0]
        self.assertEqual(first["from"]["reason"], "failed")
        self.assertEqual(first["from"]["from_target"], "pubmed")

    def test_covered_needs_add_no_search_and_no_model_call(self):
        state, stage, adapters = self.routed({"pubmed": [papers_json("11", "12")], "guideline_sites": [web_json("g")], "fda_label": [found_labels()], "regulator": [web_json("r")]})
        ask = self.alternates()
        before = list(adapters.calls)
        record, _ = self.phase(state, stage, adapters, ask=ask)
        self.assertEqual(adapters.calls, before)
        self.assertEqual(record["waves"], [])
        self.assertEqual(ask.calls, [])      # no alternates were requested

    def test_supporting_needs_never_escalate(self):
        latest = {"queries": [{"need_id": "s1", "need": "background", "target": "pubmed", "query": "sertraline fluoxetine adolescent depression background reviews", "priority": "supporting"}]}
        state, stage, adapters = self.routed({"pubmed": [EMPTY]}, latest=latest, drugs={"queries": []})
        record, _ = self.phase(state, stage, adapters, ask=self.alternates(s1=("web", "sertraline fluoxetine adolescent depression background")))
        self.assertEqual(record["waves"], [])
        self.assertNotIn("web", adapters.targets_called())

    def test_wave_two_escalates_a_need_whose_records_were_all_judged_irrelevant(self):
        self.relevance = {"21": "not_relevant", "22": "not_relevant"}      # PubMed answers, but nothing it returns bears on the question
        state, stage, adapters = self.routed({"pubmed": [papers_json("21", "22")], "guideline_sites": [web_json("g")], "fda_label": [found_labels()], "regulator": [web_json("r")], "europepmc": [papers_json("31")]},
                                             drugs={"queries": []})
        record, _ = self.phase(state, stage, adapters, ask=self.alternates(head=("europepmc", "sertraline versus fluoxetine adolescents depression efficacy comparison trial")))
        waves = {w["wave"]: w for w in record["waves"]}
        self.assertNotIn(1, waves)      # nothing was empty: wave 1 had nothing to do
        self.assertEqual([s["target"] for s in waves[2]["searches"]], ["europepmc"])
        self.assertIn({"31"}, self.triage_calls)      # ONLY the new record is read
        self.assertEqual(state["papers"]["21"]["triage"]["relevance"], "not_relevant")      # the old decisions are untouched

    def test_web_results_are_screened_by_the_local_scorer_and_borderline_ones_count(self):
        latest = {"queries": [{"need_id": "w1", "need": "background pages", "target": "web", "query": "sertraline fluoxetine adolescent depression background information", "priority": "essential"}]}
        script = {"web": [web_json("a", "b")], "pubmed": [papers_json("41")]}
        state, stage, adapters = self.routed(script, latest=latest, drugs={"queries": []})
        far_off = lambda q, texts: [-12.0] * len(texts)      # the scorer says every page is off topic (below the calibrated -10)
        record, _ = self.phase(state, stage, adapters, ask=self.alternates(w1=("europepmc", "sertraline fluoxetine adolescent depression background information review")), scorer=far_off)
        self.assertEqual([s["target"] for w in record["waves"] for s in w["searches"] if s["need_id"] == "w1"], ["europepmc"])      # clearly off topic: one alternate-source search
        state, stage, adapters = self.routed(script, latest=latest, drugs={"queries": []})
        borderline = lambda q, texts: [-7.0] * len(texts)      # inside the borderline band: left to the model-based assessment, counted as relevant for coverage
        record, _ = self.phase(state, stage, adapters, scorer=borderline)
        self.assertEqual(record["needs"]["w1"]["state"], "covered")
        self.assertEqual(record["waves"], [])

    def test_guideline_and_regulator_pages_are_never_rejected_by_a_topical_score(self):
        state, stage, adapters = self.routed({"pubmed": [papers_json("41")], "guideline_sites": [web_json("g1", "g2")], "fda_label": [found_labels()],
                                              "regulator": [json.dumps([{"title": "FDA Newsroom", "url": "https://www.fda.gov/news-events", "snippet": "News."}])]}, drugs=DRUGS)
        far_off = lambda q, texts: [-12.0] * len(texts)
        record, _ = self.phase(state, stage, adapters, scorer=far_off)
        states = {v["need"]: v["state"] for v in record["needs"].values()}
        self.assertEqual(states.get("safety communications"), "covered")      # an official regulator page: the score says nothing about it
        self.assertEqual(states.get("guideline recommendations"), "covered")      # an authoritative guideline page

    def test_each_need_escalates_at_most_once_and_the_cap_holds(self):
        latest = {"queries": [{"need_id": f"e{i}", "need": f"need {i}", "target": "pubmed", "query": f"sertraline fluoxetine adolescent depression topic {i} evidence", "priority": "essential"} for i in range(6)]}
        state, stage, adapters = self.routed({"pubmed": [EMPTY]}, latest=latest, drugs={"queries": []})
        ask = self.alternates(**{f"e{i}": ("europepmc", f"sertraline fluoxetine adolescent depression topic {i} other index") for i in range(6)})
        record, skipped = self.phase(state, stage, adapters, ask=ask)
        searches = [s for w in record["waves"] for s in w["searches"]]
        self.assertLessEqual(len(searches), rr.MAX_ESCALATIONS)
        self.assertEqual(len({s["need_id"] for s in searches}), len(searches))
        self.assertEqual(len([c for c in adapters.calls if c[0] == "europepmc"]), len(searches))

    def test_the_hard_deadline_is_kept_unfinished_work_is_reported_and_late_results_are_recorded(self):
        qa_stream.ESCALATION_MIN_S = 0.05
        self.addCleanup(setattr, qa_stream, "ESCALATION_MIN_S", 2.0)

        def slow(op):
            time.sleep(0.9)
            return papers_json("51")
        state, stage, adapters = self.routed({"pubmed": [EMPTY], "europepmc": [slow], "guideline_sites": [web_json("g")], "fda_label": [found_labels()], "regulator": [web_json("r")]}, drugs={"queries": []})
        began = time.monotonic()
        record, skipped = self.phase(state, stage, adapters, ask=self.alternates(head=("europepmc", "sertraline versus fluoxetine adolescents depression efficacy comparison trial")), deadline=0.4)
        self.assertLess(time.monotonic() - began, 0.75)      # it did not wait for the slow source
        self.assertTrue(record["unfinished"])
        self.assertTrue(any("did not finish" in s for s in skipped))      # explicit in the report's limitations
        self.assertNotIn("51", state["papers"])      # the late result was NOT silently merged

    def test_a_relevance_reading_that_is_still_running_at_the_deadline_skips_wave_two_and_says_so(self):
        gate = threading.Event()
        running = threading.Thread(target=lambda: gate.wait(5), daemon=True)
        running.start()
        state, stage, adapters = self.routed({"pubmed": [papers_json("61")], "guideline_sites": [web_json("g")], "fda_label": [found_labels()], "regulator": [web_json("r")]}, drugs={"queries": []})
        record, skipped = self.phase(state, stage, adapters, deadline=0.3, triage_thread=running)
        gate.set()
        self.assertTrue(any("relevance reading of the retrieved records did not finish" in s for s in skipped))
        self.assertNotIn(2, [w["wave"] for w in record["waves"]])

    def test_without_a_routed_stage_the_old_short_wait_is_used(self):
        state = make_state()
        record = qa_stream.run_escalation_phase(state, QUESTION, plan(), {}, finished_thread(), [], make_adapters=lambda a: {}, deadline_s=5)
        self.assertEqual(record["waves"], [])

    def test_a_transient_failure_is_retried_on_the_same_source_before_any_escalation(self):
        state, stage, adapters = self.routed({"pubmed": [TRANSIENT, papers_json("71")], "guideline_sites": [web_json("g")], "fda_label": [found_labels()], "regulator": [web_json("r")]}, drugs={"queries": []})
        record, _ = self.phase(state, stage, adapters)
        self.assertEqual(record["waves"], [])      # no escalation: the retry on the same source succeeded
        self.assertIn(2, [r.attempts for r in stage["execution"].results.values()])      # exactly one retry


def found_labels():
    return json.dumps({"results": [{"generic": "sertraline", "brands": [], "indication_excerpt": "x", "url": "u", "label_date": "2024"}], "searched_terms": ["depression"]})


if __name__ == "__main__":
    unittest.main()
