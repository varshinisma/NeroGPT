"""Regression contracts for dynamic query shaping, evidence gating, and provenance reconciliation."""
import json
import os
import sys
import unittest
from unittest import mock
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import clinical_tools  # noqa: E402
import landscape  # noqa: E402
import qa_stream  # noqa: E402
import report  # noqa: E402
import presentation  # noqa: E402
import verification  # noqa: E402
import source_figures  # noqa: E402
import presentation  # noqa: E402


class DynamicQueryTests(unittest.TestCase):
    def test_verified_understanding_retains_valid_secondary_intents_only(self):
        plan = {"intent": "diagnosis", "secondary_intents": ["drug_treatment", "diagnosis", "not_an_intent"],
                "info_needed": ["diagnosis", "treatment_options", "made_up_need"], "case_facts": []}
        understood = qa_stream.verified_understanding(plan, "What diagnostic work-up and management are appropriate?")
        self.assertEqual(understood["secondary_intents"], ["drug_treatment"])
        self.assertEqual(understood["info_needed"], ["diagnosis", "treatment_options"])

    def test_query_cleanup_generalizes_to_a_different_domain(self):
        words, _, _ = qa_stream.keywords("A 42-year-old adult asks about current migraine prevention")
        self.assertIn("migraine", words)
        self.assertNotIn("year", words)

    def test_planner_fallback_preserves_workup_and_management_information_needs(self):
        understanding = qa_stream.verified_understanding({}, "What is the diagnostic work-up, management and follow-up for this presentation?")
        self.assertIn("diagnosis", understanding["info_needed"])
        self.assertIn("treatment_options", understanding["info_needed"])
        self.assertEqual(understanding["answer_format"], "stepwise_plan")


class FutureDateNoteTests(unittest.TestCase):
    def test_only_papers_used_as_evidence_get_a_future_date_note(self):
        papers = {"1": {"pmid": "1", "year": "2999", "triage": {"relevance": "direct"}},
                  "2": {"pmid": "2", "year": "2999", "triage": {"relevance": "uncertain"}},
                  "3": {"pmid": "3", "year": "2999"},
                  "4": {"pmid": "4", "year": "2001", "triage": {"relevance": "direct"}}}
        notes = qa_stream.future_date_notes(papers)
        self.assertEqual(len(notes), 1)
        self.assertIn("PMID 1 ", notes[0])      # a screened-out paper named in a note would become a numbered reference


class RecentSearchTests(unittest.TestCase):
    def test_relevance_picks_the_papers_and_the_date_only_orders_them(self):
        found = json.dumps({"results": [{"pmid": "1", "year": "2024"}, {"pmid": "2", "year": "2026"}, {"pmid": "3", "year": "2026"}, {"pmid": "4", "year": ""}]})
        with mock.patch.object(qa_stream, "search_pubmed", return_value=found) as search:
            out = json.loads(qa_stream.best_of_recent("q", 8))
        self.assertEqual(search.call_args.kwargs["sort"], "relevance")      # never the date sort: it would fill the slots with the newest papers whatever their fit
        self.assertEqual([p["pmid"] for p in out["results"]], ["2", "3", "1", "4"])      # newest first; one year keeps the relevance order

    def test_an_error_or_empty_answer_is_passed_on_unchanged(self):
        for raw in ('{"error": "down"}', '{"results": []}', "not json"):
            with mock.patch.object(qa_stream, "search_pubmed", return_value=raw):
                self.assertEqual(qa_stream.best_of_recent("q", 8), raw)

    def test_the_recent_jobs_use_it(self):
        plan = {"condition": "condition x", "plan_status": "ok", "pubmed_query": '("aaa bbb") AND ("child") AND ("workup")', "understanding": {}}
        with mock.patch.object(qa_stream, "best_of_recent", return_value="{}") as recent, mock.patch.object(qa_stream.mesh_expand, "prefetch"):
            qa_stream.plan_pubmed_jobs(plan)["pubmed_plan_recent"]()
        self.assertEqual(recent.call_args.args[1], 6)


class QueryFocusTests(unittest.TestCase):
    QUESTION = "Child has aaaa bbbb with cccc regression and dddd features. What is the work-up and management?"
    QUERY = '(aaaa bbbb OR cccc regression OR dddd spectrum disorder) AND (child OR pediatric) AND (workup OR management)'

    def plan(self, **extra):
        return {"pubmed_query": self.QUERY, "main_problem": "aaaa bbbb with cccc regression", "associated_findings": ["dddd features"], **extra}

    def test_a_finding_that_only_accompanies_the_main_problem_leaves_the_condition_group(self):
        out = qa_stream.focus_condition_group(self.plan(), self.QUESTION)
        self.assertEqual(out["pubmed_query"], '(aaaa bbbb OR cccc regression) AND (child OR pediatric) AND (workup OR management)')
        self.assertEqual(out["query_focus"]["dropped_from_condition_group"], ["dddd spectrum disorder"])

    def test_nothing_changes_when_the_names_are_not_in_the_question(self):
        for extra in ({"main_problem": "zzzz yyyy"}, {"associated_findings": ["qqqq"]}, {"main_problem": "zzzz yyyy aaaa bbbb"}, {"main_problem": ""}, {"associated_findings": []}):
            out = qa_stream.focus_condition_group(self.plan(**extra), self.QUESTION)
            self.assertEqual(out["pubmed_query"], self.QUERY)
            self.assertNotIn("query_focus", out)

    def test_the_group_is_never_emptied(self):
        plan = self.plan(pubmed_query="(dddd spectrum disorder OR dddd features) AND (child)")
        self.assertEqual(qa_stream.focus_condition_group(plan, self.QUESTION)["pubmed_query"], "(dddd spectrum disorder OR dddd features) AND (child)")

    def test_a_query_of_another_shape_is_left_alone(self):
        plan = self.plan(pubmed_query="aaaa bbbb AND dddd")
        self.assertEqual(qa_stream.focus_condition_group(plan, self.QUESTION)["pubmed_query"], "aaaa bbbb AND dddd")


class SeparateFindingsTests(unittest.TestCase):
    QUERY = '(("aaaa bbbb" OR "cccc regression") AND ("dddd features" OR "dddd spectrum")) AND ("work-up" OR "management")'
    FACTS = ["Child aged three", "aaaa bbbb", "cccc regression", "dddd features"]

    def plan(self, **extra):
        return {"pubmed_query": self.QUERY, "drugs": [], "understanding": {"case_facts": self.FACTS}, **extra}

    def test_the_groups_of_a_wrapped_query_are_found(self):
        self.assertEqual(qa_stream.top_level_groups(self.QUERY), ['"aaaa bbbb" OR "cccc regression"', '"dddd features" OR "dddd spectrum"', '"work-up" OR "management"'])
        self.assertEqual(qa_stream.top_level_groups("(a OR b"), [])

    def test_a_group_tied_to_one_other_finding_is_taken_out_of_the_query(self):
        out = qa_stream.separate_findings(self.plan())
        self.assertEqual(out["pubmed_query"], '("aaaa bbbb" OR "cccc regression") AND ("work-up" OR "management")')
        self.assertEqual(out["query_focus"]["dropped_groups"], ['"dddd features" OR "dddd spectrum"'])

    def test_a_group_with_a_named_medicine_or_a_short_case_is_never_taken_out(self):
        self.assertEqual(qa_stream.separate_findings(self.plan(drugs=["dddd"]))["pubmed_query"], self.QUERY)
        short = self.plan(understanding={"case_facts": self.FACTS[:2]})
        self.assertEqual(qa_stream.separate_findings(short)["pubmed_query"], self.QUERY)

    def test_a_word_shared_with_the_main_group_does_not_hide_the_other_finding(self):
        query = '("aaaa bbbb" OR "cccc regression") AND ("dddd features" OR "regressive dddd") AND ("work-up")'
        self.assertEqual(qa_stream.separate_findings(self.plan(pubmed_query=query))["pubmed_query"], '("aaaa bbbb" OR "cccc regression") AND ("work-up")')

    def test_a_trials_section_nobody_asked_for_lists_only_directly_relevant_trials(self):
        state = {"section_states": {"trials": {"state": "not_applicable"}}, "understanding": {"info_needed": ["diagnosis"]},
                 "trials_by_id": {"NCT1": {"nct_id": "NCT1", "triage": {"relevance": "direct", "kind": "treatment_trial"}},
                                  "NCT2": {"nct_id": "NCT2", "triage": {"relevance": "indirect", "kind": "treatment_trial"}}}}
        shown = landscape.trials_markdown(state)
        self.assertIn("NCT1", shown)      # the trial search is mandatory: a trial that matches the condition is shown
        self.assertIn("| NCT2", shown)      # the trial section is mandatory for every question: a related (indirect) treatment trial is shown too
        state["understanding"]["info_needed"].append("trial_status")
        self.assertIn("NCT2", landscape.trials_markdown(state))

    def test_the_patient_group_stays(self):
        query = '("aaaa bbbb" OR "cccc regression") AND ("dddd features") AND ("child" OR "pediatric") AND ("work-up")'
        self.assertEqual(qa_stream.separate_findings(self.plan(pubmed_query=query))["pubmed_query"], '("aaaa bbbb" OR "cccc regression") AND ("child" OR "pediatric") AND ("work-up")')


class PruneSkillsTests(unittest.TestCase):
    def test_medicine_and_trial_skills_need_a_reason_in_the_plan(self):
        plan = {"skills": ["latest-evidence", "drug-intelligence", "clinical-trials"], "intent": "case_evaluation", "drugs": [], "understanding": {"info_needed": ["diagnosis"], "medicines_needed": False}}
        out = qa_stream.prune_unjustified_skills(plan)
        self.assertEqual(out["skills"], ["latest-evidence"])
        self.assertEqual(out["skills_pruned"], ["drug-intelligence", "clinical-trials"])

    def test_the_skills_stay_when_the_plan_calls_for_them(self):
        plan = {"skills": ["latest-evidence", "drug-intelligence", "clinical-trials"], "intent": "drug_treatment", "drugs": [], "understanding": {"info_needed": ["treatment_options", "trial_status"], "medicines_needed": True}}
        self.assertEqual(qa_stream.prune_unjustified_skills(plan)["skills"], ["latest-evidence", "drug-intelligence", "clinical-trials"])

    def test_a_failed_plan_keeps_every_skill(self):
        plan = {"plan_status": "fallback", "skills": ["drug-intelligence"], "understanding": {}}
        self.assertEqual(qa_stream.prune_unjustified_skills(plan)["skills"], ["drug-intelligence"])


class RoutedPubmedTests(unittest.TestCase):
    def test_a_newest_first_request_returns_the_best_matches_not_the_newest_papers(self):
        with mock.patch.object(qa_stream, "search_pubmed", return_value='{"results": [{"pmid": "1", "year": "2023"}, {"pmid": "2", "year": "2025"}]}') as search:
            out = json.loads(qa_stream.routed_pubmed("q", max_results=8, years_back=2, sort="date"))
        self.assertEqual(search.call_args.kwargs["sort"], "relevance")
        self.assertEqual([p["pmid"] for p in out["results"]], ["2", "1"])

    def test_other_sorts_are_passed_through(self):
        with mock.patch.object(qa_stream, "search_pubmed", return_value="{}") as search:
            qa_stream.routed_pubmed("q", max_results=6, years_back=5, sort="relevance")
        self.assertEqual(search.call_args.kwargs, {"max_results": 6, "years_back": 5, "sort": "relevance"})


class RelatedTopicTests(unittest.TestCase):
    PLAN = {"condition": "condition x", "plan_status": "ok", "pubmed_query": '("aaa bbb") AND ("child") AND ("workup")', "understanding": {},
            "related_topics": ["kinds of causes to consider", "first line laboratory tests"]}

    def jobs(self, plan=None):
        with mock.patch.object(qa_stream.mesh_expand, "prefetch"):
            return qa_stream.plan_pubmed_jobs(plan or self.PLAN)

    def test_one_search_per_topic_within_the_patient_group(self):
        jobs = self.jobs()
        self.assertTrue({"pubmed_plan_topic_1", "pubmed_plan_topic_2"} <= set(jobs))
        with mock.patch.object(qa_stream, "search_pubmed", return_value="{}") as search:
            jobs["pubmed_plan_topic_1"]()
        self.assertEqual(search.call_args.args[0], '(kinds of causes to consider) AND ("child")')
        self.assertEqual(search.call_args.kwargs["sort"], "relevance")

    def test_no_topics_no_topic_searches(self):
        self.assertFalse([n for n in self.jobs({**self.PLAN, "related_topics": []}) if "_topic_" in n])

    def test_the_topics_reach_the_ranking_queries(self):
        queries = qa_stream.intent_queries({}, {**self.PLAN, "understanding": {"info_needed": ["diagnosis"]}})
        self.assertTrue(any("first line laboratory tests" in q for q in queries))


class ModelStageTests(unittest.TestCase):
    def test_mistral_reasoning_and_writer_models_can_be_configured_separately(self):
        settings = {"MODEL_PROVIDER": "mistral", "MISTRAL_MODEL": "mistral-medium-latest",
                    "MISTRAL_REASONING_MODEL": "mistral-medium-latest", "MISTRAL_WRITER_MODEL": "mistral-small-latest"}
        settings = {**settings, "PLANNER_PROVIDER": "", "PLANNER_MODEL": "", "TRIAGE_PROVIDER": "", "TRIAGE_MODEL": ""}      # the plan call's own provider is tested separately
        with mock.patch.dict(os.environ, settings, clear=False):
            self.assertEqual(qa_stream.model_id_for("planning"), "mistral-medium-latest")
            self.assertEqual(qa_stream.model_id_for("triage"), "mistral-medium-latest")
            self.assertEqual(qa_stream.model_id_for("writer"), "mistral-small-latest")

    def test_a_rate_limit_message_tells_how_long_to_wait(self):
        self.assertAlmostEqual(qa_stream.retry_after_seconds("Rate limit reached. Please try again in 7.2s.", 3.0), 7.7)
        self.assertEqual(qa_stream.retry_after_seconds("Please try again in 460ms", 3.0), 1.0)      # at least one second
        self.assertEqual(qa_stream.retry_after_seconds("try again in 12m30s", 3.0), qa_stream.RATE_LIMIT_MAX_WAIT_S)      # never longer than the cap
        self.assertEqual(qa_stream.retry_after_seconds("429 too many requests", 6.0), 6.0)      # no hint: the default

    def test_the_triage_calls_can_use_another_provider_and_model(self):
        settings = {"MODEL_PROVIDER": "mistral", "MISTRAL_MODEL": "open-mistral-nemo", "MISTRAL_REASONING_MODEL": "mistral-medium-latest",
                    "TRIAGE_PROVIDER": "groq", "TRIAGE_MODEL": "some/model"}
        with mock.patch.dict(os.environ, settings, clear=False):
            self.assertEqual(qa_stream.triage_choice(), ("groq", "some/model"))
            self.assertEqual(qa_stream.model_id_for("triage"), "some/model")      # never the Mistral model id
        with mock.patch.dict(os.environ, {**settings, "TRIAGE_PROVIDER": "", "TRIAGE_MODEL": ""}, clear=False):
            self.assertEqual(qa_stream.triage_choice(), (None, None))
            self.assertEqual(qa_stream.model_id_for("triage"), "mistral-medium-latest")

    def test_the_search_writing_calls_can_use_their_own_provider_and_model(self):
        settings = {"MODEL_PROVIDER": "mistral", "MISTRAL_MODEL": "open-mistral-nemo", "MISTRAL_REASONING_MODEL": "", "TRIAGE_PROVIDER": "", "TRIAGE_MODEL": "",
                    "QUERY_PROVIDER": "groq", "QUERY_MODEL": "some/model"}
        with mock.patch.dict(os.environ, settings, clear=False):
            self.assertEqual(qa_stream.query_choice(), ("groq", "some/model"))
            self.assertEqual(qa_stream.model_id_for("queries"), "some/model")
            self.assertEqual(qa_stream.model_id_for("triage"), "open-mistral-nemo")      # the relevance reading keeps the default model
        with mock.patch.dict(os.environ, {**settings, "QUERY_PROVIDER": "", "QUERY_MODEL": ""}, clear=False):
            self.assertEqual(qa_stream.query_choice(), (None, None))
            self.assertEqual(qa_stream.model_id_for("queries"), "open-mistral-nemo")

    def test_mistral_models_are_cached_by_stage_and_passed_to_the_provider(self):
        settings = {"MODEL_PROVIDER": "mistral", "MISTRAL_MODEL": "mistral-medium-latest",
                    "MISTRAL_REASONING_MODEL": "mistral-medium-latest", "MISTRAL_WRITER_MODEL": "mistral-small-latest"}
        settings = {**settings, "PLANNER_PROVIDER": "", "PLANNER_MODEL": "", "TRIAGE_PROVIDER": "", "TRIAGE_MODEL": ""}
        with mock.patch.dict(os.environ, settings, clear=False), mock.patch.object(qa_stream.server, "build_model", side_effect=lambda **kw: kw) as build:
            old = qa_stream._models
            qa_stream._models = {}
            try:
                planning = qa_stream.model_for(400, 0.0, "planning")
                writer = qa_stream.model_for(400, None, "writer")
            finally:
                qa_stream._models = old
        self.assertEqual(planning["model_id"], "mistral-medium-latest")
        self.assertEqual(writer["model_id"], "mistral-small-latest")
        self.assertEqual(build.call_count, 2)


class EvidenceSelectionTests(unittest.TestCase):
    # Small manually reviewed fixtures: labels stand for an independent human screen,
    # while `ordered_papers` is the deterministic final-context gate under evaluation.
    EVAL_CASES = [
        {"question": "Three-year-old with developmental delay, loss of acquired skills, and autistic features: work-up and management?",
         "labels": [("810001", "direct"), ("810002", "indirect"), ("810003", "not_relevant"), ("810004", "not_relevant"), ("810005", "not_relevant")],
         "retrieval_order": ["810003", "810001", "810004", "810002", "810005"]},
        {"question": "For an adult with migraine, what preventive medication evidence is available?",
         "labels": [("820001", "direct"), ("820002", "not_relevant"), ("820003", "indirect")],
         "retrieval_order": ["820001", "820002", "820003"]},
        {"question": "Which clinical trials assess an intervention for an adult neurodegenerative condition?",
         "labels": [("830001", "direct"), ("830002", "not_relevant")], "retrieval_order": ["830002", "830001"]},
    ]

    def test_manually_reviewed_eval_set_reports_retrieval_and_context_precision(self):
        retrieval_precisions, context_precisions = [], []
        for case in self.EVAL_CASES:
            by_id = {pid: {"pmid": pid, "year": "2025", "triage": {"relevance": label}}
                     for pid, label in case["labels"]}
            retrieved = [by_id[pid] for pid in case["retrieval_order"]]
            included = qa_stream.ordered_papers(by_id)
            relevant = {pid for pid, label in case["labels"] if label in ("direct", "indirect")}
            retrieval_precisions.append(sum(p["pmid"] in relevant for p in retrieved[:10]) / min(10, len(retrieved)))
            context_precisions.append(sum(p["pmid"] in relevant for p in included[:qa_stream.MAX_CONTEXT_PAPERS]) / max(1, len(included[:qa_stream.MAX_CONTEXT_PAPERS])))
        self.assertAlmostEqual(sum(retrieval_precisions) / len(retrieval_precisions), 47 / 90)
        self.assertAlmostEqual(sum(context_precisions) / len(context_precisions), 1.0)

    def test_missing_or_failed_relevance_decision_fails_closed(self):
        self.assertFalse(qa_stream.usable({"pmid": "1"}))
        self.assertFalse(qa_stream.usable({"pmid": "2", "triage": {"relevance": "uncertain"}}))
        self.assertTrue(qa_stream.usable({"pmid": "3", "triage": {"relevance": "direct"}}))
        self.assertTrue(qa_stream.usable({"pmid": "4", "triage": {"relevance": "indirect"}}))

    def test_title_only_oncology_match_cannot_enter_a_developmental_workup_report(self):
        offered = {"cancer": {"sentences": ["Ewing sarcoma case in a young adult", "The report describes tumour progression and chemotherapy."]}}
        decision = landscape.verify_paper_triage([{"id": "cancer", "relevance": "indirect", "role": "experimental", "need": "diagnosis", "sentence": 1,
                                                   "rationale": "Mentions a young person and a possible differential diagnosis."}], offered, information_needs=["diagnosis", "treatment_options"])
        self.assertEqual(decision["cancer"]["relevance"], "uncertain")
        self.assertFalse(qa_stream.usable({"pmid": "cancer", "triage": decision["cancer"]}))

    def test_relevant_synonym_can_be_retained_when_an_abstract_sentence_supports_a_real_need(self):
        offered = {"synonym": {"sentences": ["Study title", "Loss of previously acquired developmental skills was evaluated in preschool children."]}}
        decision = landscape.verify_paper_triage([{"id": "synonym", "relevance": "direct", "role": "emerging", "need": "diagnosis", "sentence": 2,
                                                   "rationale": "Assesses regression as part of the diagnostic evaluation."}], offered, information_needs=["diagnosis"])
        self.assertEqual(decision["synonym"]["relevance"], "direct")

    def test_case_specific_report_cannot_be_promoted_as_general_workup_evidence(self):
        offered = {"case": {"study_type": ["Case Reports"], "sentences": ["Rare syndrome case", "One child with a rare disorder also had developmental delay."]}}
        decision = landscape.verify_paper_triage([{"id": "case", "relevance": "direct", "role": "experimental", "need": "diagnosis", "scope": "case_specific", "sentence": 2,
                                                   "rationale": "Describes a rare disorder that can co-occur with developmental delay."}], offered, information_needs=["diagnosis"])
        self.assertEqual(decision["case"]["relevance"], "background")
        paper = {"pmid": "100000", "title": "Rare syndrome case", "year": "2026", "authors": ["Lee"], "type": ["Case Reports"], "abstract": "Results: One child had developmental delay.", "triage": decision["case"]}
        self.assertEqual(report.key_study_entries([paper]), [])

    def test_generalizable_case_report_requires_an_explicit_scope_decision(self):
        offered = {"case": {"study_type": ["Case Reports"], "sentences": ["Assessment method", "The diagnostic method was evaluated across the stated clinical population."]}}
        decision = landscape.verify_paper_triage([{"id": "case", "relevance": "direct", "role": "emerging", "need": "diagnosis", "scope": "generalizable", "sentence": 2,
                                                   "rationale": "Evaluates the requested diagnostic approach in the stated population."}], offered, information_needs=["diagnosis"])
        self.assertEqual(decision["case"]["relevance"], "direct")

    def test_case_report_is_not_direct_evidence_for_a_broad_question(self):
        offered = {"case": {"study_type": ["Case Reports"], "sentences": ["Case title", "A child received a syndrome-specific procedure."]}}
        decision = landscape.verify_paper_triage([{"id": "case", "relevance": "direct", "role": "experimental", "need": "diagnosis", "scope": "generalizable", "sentence": 2,
                                                   "rationale": "Addresses one diagnostic component."}], offered, information_needs=["diagnosis", "treatment_options"], broad_question=True)
        self.assertEqual(decision["case"]["relevance"], "indirect")

    def test_non_trial_question_shows_indirect_treatment_trials(self):
        state = qa_stream.new_state("condition work-up")
        state["understanding"] = {"info_needed": ["diagnosis"]}
        state["trials_by_id"] = {"NCT1": {"nct_id": "NCT1", "title": "Related condition trial", "interventions": ["Intervention"], "phase": [], "recruitment_status": "RECRUITING", "last_update": "2026-01-01", "triage": {"relevance": "indirect", "kind": "treatment_trial", "evidence": "x"}}}
        self.assertIn("NCT1", landscape.trials_markdown(state))

    def test_relevance_requires_a_need_from_the_actual_question(self):
        offered = {"record": {"sentences": ["Study title", "The study reports an assessment outcome in the target population."]}}
        decision = landscape.verify_paper_triage([{"id": "record", "relevance": "direct", "role": "emerging", "need": "made_up_need", "sentence": 2,
                                                   "rationale": "Claims a contribution that was not requested."}], offered, information_needs=["diagnosis"])
        self.assertEqual(decision["record"]["relevance"], "uncertain")

    def test_irrelevant_retrieved_papers_do_not_reach_synthesis_or_key_studies(self):
        unrelated = {"pmid": "800001", "year": "2026", "triage": {"relevance": "not_relevant"}}
        direct = {"pmid": "800002", "year": "2024", "triage": {"relevance": "direct"}}
        ordered = qa_stream.ordered_papers({"800001": unrelated, "800002": direct})
        self.assertEqual([p["pmid"] for p in ordered], ["800002"])

    def test_hybrid_order_is_preserved_within_direct_and_indirect_relevance_tiers(self):
        # Dict order is the existing hybrid/cross-encoder rank; recency must not overwrite it.
        lower_rank = {"pmid": "800020", "year": "2026", "triage": {"relevance": "direct"}}
        higher_rank = {"pmid": "800021", "year": "2018", "triage": {"relevance": "direct"}}
        indirect = {"pmid": "800022", "year": "2025", "triage": {"relevance": "indirect"}}
        ranked = qa_stream.ordered_papers({p["pmid"]: p for p in (higher_rank, lower_rank, indirect)})
        self.assertEqual([p["pmid"] for p in ranked], ["800021", "800020", "800022"])

    def test_direct_candidates_precede_indirect_even_when_the_indirect_study_is_newer(self):
        direct = {"pmid": "800024", "year": "2020", "triage": {"relevance": "direct"}}
        indirect = {"pmid": "800025", "year": "2026", "triage": {"relevance": "indirect"}}
        self.assertEqual([p["pmid"] for p in qa_stream.ordered_papers({"a": indirect, "b": direct})], ["800024", "800025"])

    def test_high_quality_design_is_a_tiebreaker_inside_direct_tier_not_a_relevance_label(self):
        direct_case = {"pmid": "800026", "type": ["Case Reports"], "study_type": ["Case Reports"], "triage": {"relevance": "direct"}}
        direct_trial = {"pmid": "800027", "type": ["Randomized Controlled Trial"], "study_type": ["Randomized Controlled Trial"], "triage": {"relevance": "direct"}}
        indirect_review = {"pmid": "800028", "type": ["Systematic Review"], "study_type": ["Systematic Review"], "triage": {"relevance": "indirect"}}
        ordered = qa_stream.ordered_papers({p["pmid"]: p for p in (direct_case, indirect_review, direct_trial)})
        self.assertEqual([p["pmid"] for p in ordered], ["800027", "800026", "800028"])
        self.assertEqual(ordered[-1]["triage"]["relevance"], "indirect")

    def test_key_studies_are_grouped_by_triage_and_numbered_once_after_sorting(self):
        direct = {"pmid": "800029", "year": "2024", "authors": ["Alpha A"], "journal": "J", "type": ["Randomized Controlled Trial"],
                  "study_type": ["Randomized Controlled Trial"], "abstract": "Results: A randomized intervention improved the target clinical outcome across the assessed population.",
                  "triage": {"relevance": "direct", "role": "emerging", "evidence": "Results: A randomized intervention improved the target clinical outcome across the assessed population.", "rationale": "Addresses the requested treatment outcome."}}
        indirect = {"pmid": "800030", "year": "2025", "authors": ["Beta B"], "journal": "J", "type": ["Systematic Review"],
                    "study_type": ["Systematic Review"], "abstract": "Results: A related-population intervention was associated with improvement in the reported clinical outcome for that subgroup.",
                    "triage": {"relevance": "indirect", "role": "emerging", "evidence": "Results: A related-population intervention was associated with improvement in the reported clinical outcome for that subgroup.", "rationale": "Informs a related subgroup only."}}
        model_text = ("## Key studies\n\n1. **Beta et al. (2025)**, systematic review: A related-population intervention was associated with improvement in the reported clinical outcome for that subgroup (PMID 800030).\n"
                      "2. **Alpha et al. (2024)**, randomized controlled trial: A randomized intervention improved the target clinical outcome across the assessed population (PMID 800029).")
        out = report.clean_model_key_studies(model_text, [indirect, direct])
        self.assertLess(out.index("1. **Alpha"), out.index("2. **Beta"))
        self.assertNotIn("Direct evidence", out)
        self.assertNotIn("Indirect evidence", out)
        self.assertEqual(verification.lint_report(out), [])

    def test_reference_numbers_follow_reordered_studies_and_keep_their_pmids(self):
        papers = {"800031": {"pmid": "800031", "authors": ["Alpha A"], "year": "2024"},
                  "800032": {"pmid": "800032", "authors": ["Beta B"], "year": "2025"}}
        text = "## Key studies\n\n1. Direct finding (PMID 800031).\n\n2. Indirect finding (PMID 800032)."
        numbered, order = presentation.numberize(text, papers)
        self.assertEqual(order, ["800031", "800032"])
        self.assertIn("Direct finding [1]", numbered)
        self.assertIn("Indirect finding [2]", numbered)

    def test_screened_relevance_is_shared_by_key_studies_current_evidence_and_figures(self):
        direct = {"pmid": "800033", "year": "2024", "authors": ["Alpha A"], "journal": "J", "type": ["Clinical Trial"], "study_type": ["Clinical Trial"],
                  "abstract": "Results: The intervention improved the clinical outcome in the studied population.",
                  "triage": {"relevance": "direct", "role": "emerging", "evidence": "The intervention improved the clinical outcome in the studied population."}}
        indirect = {"pmid": "800034", "cite": "Beta et al., J, 2025 [PMID 800034]", "year": "2025", "date": "2025", "type": ["Clinical Trial"],
                    "title": "Related subgroup intervention", "abstract": "A related intervention was studied.",
                    "triage": {"relevance": "indirect", "population": "related subgroup", "evidence": "related evidence"}}
        papers = {"800033": direct, "800034": indirect}
        self.assertEqual(qa_stream.paper_view(indirect, 500)["relation"], "indirect (different or related population)")
        table = qa_stream.rows_to_table("## Current evidence\nApproach A || model label || Background || Result || (PMID 800033)\nApproach B || model label || Emerging || Related result || (PMID 800034)", papers)
        self.assertLess(table.index("Approach A"), table.index("Approach B"))
        self.assertIn("| Approach | Evidence level | What the evidence shows | Reference |", table)
        self.assertIn("| Approach A | Background |", table)
        figure = {"pmid": "800034", "path": "figure.png", "relation": "indirect", "label": "related"}
        record = source_figures.display_records([figure], {"1": "800034"})[0]
        self.assertEqual(record["relation"], "indirect")
        self.assertIn("indirect evidence", record["why"])

    def test_question_specific_triage_schema_keeps_relevance_distinct_from_design_strength(self):
        self.assertIn("strong study design does not make an indirect study direct", qa_stream.PAPER_TRIAGE_TASK)
        self.assertIn("specific question need", qa_stream.PAPER_TRIAGE_TASK)
        self.assertIn("rationale", qa_stream.PAPER_TRIAGE_TASK)

    def test_recent_but_unrelated_pediatric_paper_is_excluded_by_screen_result(self):
        papers = {
            "800030": {"pmid": "800030", "year": "2026", "triage": {"relevance": "not_relevant"}},
            "800031": {"pmid": "800031", "year": "2021", "triage": {"relevance": "direct"}},
        }
        self.assertEqual([p["pmid"] for p in qa_stream.ordered_papers(papers)], ["800031"])

    def test_key_studies_require_a_justified_screened_candidate(self):
        candidate = {"pmid": "800040", "year": "2024", "title": "A diagnostic study", "authors": ["Example A"],
                     "study_type": ["Systematic Review"], "abstract": "Results: Diagnostic work-up improved appropriate detection in the studied population.",
                     "triage": {"relevance": "direct", "role": "emerging", "evidence": "The review reports a diagnostic result."}}
        text = ("## Key studies\n\n"
                "1. **Example et al. (2024)**, systematic review: This review reports a diagnostic work-up result for the question's population (PMID 800040).\n"
                "2. **Unrelated et al. (2025)**, randomized controlled trial: This trial concerns another condition and does not inform the question (PMID 800041).")
        out = report.clean_model_key_studies(text, [candidate])
        self.assertIn("PMID 800040", out)
        self.assertNotIn("PMID 800041", out)

    def test_key_study_cannot_be_promoted_without_role_and_evidence_rationale(self):
        background = {"pmid": "800042", "year": "2025", "title": "Context review", "authors": ["Context A"],
                      "study_type": ["Review"], "abstract": "Results: Context only.",
                      "triage": {"relevance": "direct", "role": "background", "evidence": "title sentence"}}
        out = report.clean_model_key_studies("## Key studies\n\n1. **Context et al. (2025)**, review: This study is listed as a key result for this question (PMID 800042).", [background])
        self.assertNotIn("PMID 800042", out)
        self.assertIn("No retrieved record had", out)

    def test_useful_indirect_evidence_remains_available(self):
        record = {"pmid": "800032", "triage": {"relevance": "indirect", "evidence": "relevant to a stated need"}}
        self.assertEqual([p["pmid"] for p in qa_stream.ordered_papers({"800032": record})], ["800032"])

    def test_indirect_triage_requires_a_question_specific_contribution(self):
        offered = {"800035": {"sentences": ["Study title", "A related subgroup's diagnostic result was reported."]}}
        without_reason = landscape.verify_paper_triage([{"id": "800035", "relevance": "indirect", "role": "emerging", "sentence": 2}], offered)
        with_reason = landscape.verify_paper_triage([{"id": "800035", "relevance": "indirect", "role": "emerging", "sentence": 2,
                                                     "rationale": "Informs the diagnostic assessment need for a related subgroup."}], offered)
        self.assertEqual(without_reason["800035"]["relevance"], "uncertain")
        self.assertEqual(with_reason["800035"]["relevance"], "indirect")
        self.assertEqual(with_reason["800035"]["rationale"], "Informs the diagnostic assessment need for a related subgroup.")

    def test_background_context_is_opt_in_and_cannot_outrank_direct_or_indirect(self):
        direct = {"pmid": "800036", "triage": {"relevance": "direct", "evidence": "direct source sentence"}}
        indirect = {"pmid": "800037", "triage": {"relevance": "indirect", "evidence": "related source sentence", "rationale": "Informs a related need."}}
        background = {"pmid": "800038", "triage": {"relevance": "background", "evidence": "context source sentence", "rationale": "Provides general context."}}
        self.assertEqual([p["pmid"] for p in qa_stream.ordered_papers({"b": background, "i": indirect, "d": direct})], ["800036", "800037"])
        self.assertEqual([p["pmid"] for p in qa_stream.ordered_papers({"b": background, "i": indirect, "d": direct}, include_background=True)], ["800036", "800037", "800038"])

    def test_relevance_task_requires_condition_population_intent_and_intervention_reasoning(self):
        task = qa_stream.PAPER_TRIAGE_TASK.lower()
        self.assertIn("exact question", task)
        self.assertIn("population", task)
        self.assertIn("information needs", task)
        self.assertIn("intervention", task)
        self.assertIn("unrelated disease", task)

    def test_indirect_evidence_must_be_restricted_by_the_triage_contract(self):
        self.assertIn("still answers one of the question's stated information needs", qa_stream.PAPER_TRIAGE_TASK)
        self.assertIn("unrelated disease", qa_stream.PAPER_TRIAGE_TASK)
        self.assertIn("diagnostic_or_assessment", qa_stream.TRIAL_TRIAGE_TASK)

class RetrievalAndProvenanceTests(unittest.TestCase):
    def test_clinicaltrials_accepts_planner_concept_lists_and_builds_valid_v2_params(self):
        calls = []
        response = {"studies": [{"protocolSection": {
            "identificationModule": {"nctId": "NCT00000001", "briefTitle": "A study"},
            "statusModule": {"overallStatus": "RECRUITING"}, "designModule": {},
            "armsInterventionsModule": {"interventions": []}, "conditionsModule": {"conditions": ["condition alpha"]},
            "eligibilityModule": {}, "outcomesModule": {"primaryOutcomes": []}, "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Sponsor"}},
        }}]}
        def fake_get(url, params):
            calls.append((url, params))
            return json.dumps(response).encode("utf-8")
        with mock.patch.object(clinical_tools, "_get", side_effect=fake_get):
            result = json.loads(clinical_tools.search_clinical_trials(["autism spectrum disorder", "global developmental delay"], status="RECRUITING"))
        self.assertEqual(len(result["results"]), 1)
        url, params = calls[0]
        self.assertEqual(url, clinical_tools.CTGOV)
        self.assertEqual(params["query.cond"], "autism spectrum disorder global developmental delay")
        self.assertNotIn("[", params["query.cond"])
        self.assertEqual(params["filter.overallStatus"], "RECRUITING")
        self.assertEqual(result["query_trace"][0]["status"], "found")

    def test_clinicaltrials_http_400_is_logged_as_failure_not_empty_results(self):
        error = urllib.error.HTTPError(clinical_tools.CTGOV, 400, "Bad Request", None, None)
        with mock.patch.object(clinical_tools, "_get", side_effect=error):
            result = json.loads(clinical_tools.search_clinical_trials(["condition alpha", "symptom beta"]))
        self.assertIn("HTTP Error 400", result["error"])
        self.assertEqual(result["query_trace"][0]["status"], "failed")
        self.assertEqual(result["query_trace"][0]["params"]["query.cond"], "condition alpha symptom beta")

    def test_invalid_status_filter_is_rejected_before_request(self):
        with mock.patch.object(clinical_tools, "_get") as get:
            result = json.loads(clinical_tools.search_clinical_trials("condition alpha", status="CURRENTLY_ACTIVE"))
        self.assertIn("invalid overall-status", result["error"])
        get.assert_not_called()

    def test_partial_family_status_is_not_hidden_by_one_success(self):
        state = {"source_status": {
            "trials_op_1": {"retrieval_status": "success", "evidence_status": "not_found_after_search"},
            "trials_op_2": {"retrieval_status": "partial", "evidence_status": "unavailable", "error": "timeout"},
        }}
        self.assertEqual(landscape.family_status(state, "trials")["retrieval_status"], "partial")

    def test_timeout_does_not_become_a_negative_trial_result(self):
        state = {"trials_by_id": {}, "source_status": {"trials_fast": {"retrieval_status": "partial", "evidence_status": "unavailable", "error": "timeout"}},
                 "routing": {"unfinished": True}, "understanding": {"info_needed": ["trial_status"]}}
        text = landscape.trials_markdown(state)
        self.assertIn("search was incomplete", text)
        self.assertIn("Trial status is unknown", text)
        self.assertNotIn("No trials were found", text)

    def test_zero_relevant_records_are_distinct_from_a_failed_source(self):
        empty = {"source_status": {"pubmed_op_1": {"retrieval_status": "success", "evidence_status": "not_found_after_search"}}}
        failed = {"source_status": {"pubmed_op_1": {"retrieval_status": "failed", "evidence_status": "unavailable", "error": "timeout"}}}
        self.assertEqual(landscape.family_status(empty, "literature")["evidence_status"], "not_found_after_search")
        self.assertEqual(landscape.family_status(failed, "literature")["evidence_status"], "unavailable")

    def test_fda_candidate_records_are_collected_before_the_executor_shuts_down(self):
        with mock.patch.object(clinical_tools, "_fda_get", return_value={"results": [{"term": "exampledrug"}]}), \
             mock.patch.object(clinical_tools, "_fda_records", return_value=[{"generic": "exampledrug"}]):
            response = json.loads(clinical_tools.search_fda_labels(["condition example"], []))
        self.assertNotIn("error", response)
        self.assertEqual(response["results"], [{"generic": "exampledrug"}])

    def test_unreviewed_registry_records_are_not_assumed_to_be_treatment_trials(self):
        t = {"nct_id": "NCT00000001", "design": "INTERVENTIONAL", "interventions": ["an intervention"]}
        self.assertFalse(landscape.is_treatment_trial(t))

    def test_guidelines_and_trials_mentioned_in_the_report_are_listed_as_sources(self):
        guideline = {"organisation": "A clinical organization", "title": "Clinical practice guideline", "url": "https://guidelines.example.org/g"}
        trial = {"nct_id": "NCT00000001", "title": "Study record", "url": "https://clinicaltrials.gov/study/NCT00000001"}
        state = {"papers": {}, "trials_by_id": {trial["nct_id"]: trial}, "web": [], "regulatory": [], "guidelines": [guideline]}
        answer = f"Guidance [{guideline['title']}]({guideline['url']}). Trial [NCT00000001]({trial['url']})."
        rebuilt = verification.rebuild_sources(answer, state)
        self.assertIn(guideline["url"], rebuilt)
        self.assertIn(trial["url"], rebuilt)

    def test_unlinked_named_policy_statement_is_a_verification_issue(self):
        state = qa_stream.new_state("clinical workup question")
        answer = "**Evidence searched on 2026-10-19**\n\nAmerican Academy of Pediatrics (2019 [AAP Policy Statement]) recommended an approach."
        issues = verification.find_issues(answer, state)
        self.assertIn("unlinked_guideline_citation", {i["kind"] for i in issues})

    def test_final_verification_needs_repair_is_not_relabelled_as_verified(self):
        state = qa_stream.new_state("adult medication question")
        result = verification.verify_and_repair("## Clinical summary\n\nAn unsupported clinical claim.", state, max_repairs=0)
        self.assertEqual(result["status"], "NEEDS_REPAIR")
        self.assertEqual(presentation.verification_label(result["status"]), "Verification needs repair")
        self.assertNotIn("verified", presentation.verification_label(result["status"]).lower())

    def test_material_summary_claims_require_a_retrieved_citation_or_explicit_context_label(self):
        state = qa_stream.new_state("clinical workup question")
        answer = ("## Clinical summary\n\n**Bottom line**\n\n"
                  "Early therapy is effective for this presentation.\n"
                  "General clinical context; not directly supported by the retrieved sources: assess the child in person.\n")
        issues = verification.find_issues(answer, state)
        self.assertIn("uncited_summary_claim", {i["kind"] for i in issues})

    def test_uncited_summary_context_is_qualified_instead_of_leaving_a_fragment(self):
        state = qa_stream.new_state("clinical workup question")
        answer = "## Clinical summary\n\n**Bottom line** A multidisciplinary assessment is appropriate.\n"
        repaired = verification.verify_and_repair(answer, state)
        self.assertIn("General clinical context; not directly supported by the retrieved sources", repaired["answer"])
        self.assertIn("a multidisciplinary assessment is appropriate.", repaired["answer"].lower())
        remaining = verification.verify_and_repair(repaired["answer"], state, repair=False)["issues_after"]
        self.assertNotIn("uncited_summary_claim", {i["kind"] for i in remaining})

    def test_summary_context_label_is_emitted_once_per_summary_line(self):
        state = qa_stream.new_state("clinical workup question")
        answer = "## Clinical summary\n\n**Immediate context** Assess the child in person. Coordinate multidisciplinary care.\n"
        repaired = verification.verify_and_repair(answer, state)
        self.assertEqual(repaired["answer"].count("General clinical context; not directly supported by the retrieved sources:"), 1)
        remaining = verification.verify_and_repair(repaired["answer"], state, repair=False)["issues_after"]
        self.assertNotIn("uncited_summary_claim", {i["kind"] for i in remaining})

    def test_summary_claim_with_a_real_citation_still_needs_the_outcome_in_the_source(self):
        state = qa_stream.new_state("condition X work-up")
        state["papers"] = {"800021": {"pmid": "800021", "title": "Evaluation of condition X", "year": "2025", "authors": ["Smith"],
                                         "journal": "Journal", "abstract": "The study evaluated diagnostic assessment in adults.", "type": ["Clinical Trial"], "study_type": ["Clinical Trial"], "cite": "Smith et al., Journal, 2025 [PMID 800021]"}}
        text = ("## Clinical summary\n\n**Most important recent finding**\n\n"
                "The intervention improved sleep outcomes (Smith et al., Journal, 2025 [PMID 800021]).\n")
        self.assertIn("outcome_not_in_source", {i["kind"] for i in verification.find_issues(text, state)})

    def test_case_report_result_cannot_be_generalized_in_summary(self):
        state = qa_stream.new_state("condition X treatment")
        state["papers"] = {"800022": {"pmid": "800022", "title": "Case report of an intervention", "year": "2025", "authors": ["Smith"],
                                         "journal": "Journal", "abstract": "A patient received the intervention.", "type": ["Case Reports"], "study_type": ["Case Reports"], "cite": "Smith et al., Journal, 2025 [PMID 800022]"}}
        broad = ("## Clinical summary\n\n**Most important recent finding**\n\n"
                 "The intervention was effective (Smith et al., Journal, 2025 [PMID 800022]).\n")
        self.assertIn("case_report_overgeneralization", {i["kind"] for i in verification.find_issues(broad, state)})
        scoped = broad.replace("The intervention was effective", "A single case report described the intervention")
        self.assertNotIn("case_report_overgeneralization", {i["kind"] for i in verification.find_issues(scoped, state)})

    def test_incomplete_literature_search_cannot_support_no_direct_evidence_claim(self):
        state = qa_stream.new_state("condition X work-up")
        state["section_states"] = {"literature": {"state": "timed_out", "needs": ["need-1"]}}
        text = "## Clinical summary\n\nNo direct evidence was found for this question.\n"
        self.assertIn("false_absence_claim", {i["kind"] for i in verification.find_issues(text, state)})

    def test_fda_search_cannot_be_rendered_as_proof_no_approved_treatment_exists(self):
        state = qa_stream.new_state("condition X treatment")
        text = "## Clinical summary\n\nNo FDA-approved treatment exists for this condition.\n"
        self.assertIn("unsupported_fda_absence", {i["kind"] for i in verification.find_issues(text, state)})

    def test_top_drug_summary_distinguishes_completed_negative_search_from_missing_search_status(self):
        completed = qa_stream.new_state("condition X treatment")
        completed["source_status"] = {"fda_fast": {"retrieval_status": "success", "evidence_status": "not_found_after_search"}}
        completed["fda_terms"] = ["condition X"]
        line = landscape.top_drugs_markdown(completed)
        self.assertIn("condition X", line)
        self.assertIn("does not establish", line)
        missing = qa_stream.new_state("condition X treatment")
        self.assertIn("status was not recorded", landscape.top_drugs_markdown(missing))

    def test_irrelevant_recent_paper_is_excluded_from_summary_candidate_context(self):
        state = qa_stream.new_state("condition X work-up")
        state["papers"] = {"800023": {"pmid": "800023", "cite": "Jones et al., Journal, 2026 [PMID 800023]", "year": "2026", "date": "2026", "type": ["Clinical Trial"],
                                         "title": "Recent pediatric unrelated study", "abstract": "A trial of another condition.", "triage": {"relevance": "irrelevant", "role": "not applicable"}}}
        self.assertEqual(qa_stream.build_fast_context(state, "condition X work-up")["papers"], [])

    def test_evidence_lifecycle_separates_unique_records_screening_and_citations(self):
        included = {"source_id": "pmid:800010", "pmid": "800010", "origin": "pubmed_fast", "found_by": ["pubmed_fast", "pubmed_op_0"],
                    "triage": {"relevance": "direct", "evidence": "source sentence"}}
        unreviewed = {"source_id": "pmid:800011", "pmid": "800011", "origin": "pubmed_fast", "found_by": ["pubmed_fast"]}
        state = {"papers": {"800010": included, "800011": unreviewed}}
        life = qa_stream.evidence_lifecycle(state, "Finding (Author et al., Journal, 2025 [PMID 800010]).", "PASS")["papers"]
        self.assertEqual(life["retrieved_occurrences"], 3)
        self.assertEqual(life["deduplicated"], 2)
        self.assertEqual(life["screened"], 1)
        self.assertEqual(life["included"], 1)
        self.assertEqual(life["unreviewed"], 1)
        self.assertEqual(life["citation_checked"], 1)


if __name__ == "__main__":
    unittest.main()
