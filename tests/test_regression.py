"""Regression tests for the NeuroGPT grounding / verification / drug-intelligence / QC fixes.

Run:  .venv\\Scripts\\python.exe -m unittest discover -s tests -v

Everything here uses synthetic data (no network, no model) and none of it is specific to one disease or drug.
"""
import copy
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import landscape  # noqa: E402
import qa_stream  # noqa: E402
import verification  # noqa: E402


def paper(pmid, title, year="2026", journal="Test Journal", author="Smith J", abstract="", protocol=False, retracted=False):
    return {"source_id": f"pmid:{pmid}", "pmid": pmid, "doi": f"10.1000/{pmid}", "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", "year": year, "date": f"{year}-05-01",
            "journal": journal, "authors": [author], "title": title, "type": [], "study_type": [], "abstract": abstract, "cite": f"{author.split()[0]} et al., {journal}, {year} [PMID {pmid}]",
            "retracted": retracted, "protocol": protocol, "origin": "t"}


def make_state():
    s = qa_stream.new_state("How to treat condition X?")
    s["core_stems"] = ["condi"]
    s["papers"]["111111"] = paper("111111", "Exercise improves motor skills in condition X", abstract="Exercise improved motor skills (Hedges' g = 0.95).")
    s["papers"]["222222"] = paper("222222", "Music therapy for condition X: study protocol", journal="Proto Journal", author="Chen L", protocol=True,
                                  abstract="We will assess music therapy for social communication. Protocol only.")
    s["papers"]["333333"] = paper("333333", "Nutraceuticals in youth with and without condition X traits", journal="Mood Journal", author="Woz J",
                                  abstract="Both treatments gave modest improvement in mood.")
    s["trials_by_id"]["NCT01234567"] = {"nct_id": "NCT01234567", "title": "A trial", "interventions": ["Drug A"], "recruitment_status": "RECRUITING", "last_update": "2026-10-01",
                                        "results_posted": False, "design": "INTERVENTIONAL", "phase": ["PHASE3"], "primary_endpoint": "Score", "url": "https://clinicaltrials.gov/study/NCT01234567"}
    s["source_status"] = {k: {"retrieval_status": "success", "evidence_status": "found", "error": None} for k in ("pubmed_fast", "fda_deep", "web_approvals", "trials_deep")}
    s["regulatory"] = [{"generic": "Risperidone", "brands": ["Risperdal"], "jurisdiction": "FDA", "matches_condition": True, "label_date": "2026-06-25", "set_id": "abc",
                        "indication_excerpt": "Treatment of irritability associated with condition X in pediatric patients", "url": "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=abc",
                        "source_id": "fda:abc"}]
    return s


BASE = ("**Evidence searched on 2026-10-06**\n\n### Treatment Drug Landscape\n\nRegulatory records are listed above.\n\n")


class RegressionTests(unittest.TestCase):
    # TEST 1 -------------------------------------------------------------------------------------------------------------
    def test1_removed_source_leaves_no_citation_anywhere(self):
        s = make_state()
        answer = (BASE + "Exercise helped motor skills (Smith et al., Test Journal, 2026 [PMID 111111]).\n"
                  "A claim about something (Fake et al., Fake Journal, 2026 [PMID 999999]).\n\n### Sources\n1. Fake et al. A fake paper. PMID 999999\n")
        r = verification.verify_and_repair(answer, s)
        self.assertNotIn("999999", r["answer"])                       # claim AND its Sources entry are gone
        self.assertIn("111111", r["answer"])
        self.assertEqual(r["repair_count"], 1)

    # TEST 2 -------------------------------------------------------------------------------------------------------------
    def test2_drug_approval_without_regulatory_source_is_not_verified(self):
        s = make_state()
        s["regulatory"] = []
        answer = BASE + "Drugzol is FDA-approved for condition X in children.\n"
        before = verification.verify_and_repair(answer, s, repair=False)
        self.assertEqual(before["status"], "NEEDS_REPAIR")            # QC does not pass an unsourced approval claim
        table = ("#### Other agents named in the retrieved evidence (not in the FDA label records)\n| Drug | Category | Population | Dose / route | Approval status | Evidence |\n"
                 "|---|---|---|---|---|---|\n| Drugzol | Investigational | youth | 5 mg PO | FDA-approved | x |\n")
        fixed = landscape.enforce_other_agents_table(table, s)
        self.assertIn("Not verified in retrieved regulatory sources", fixed)
        self.assertNotIn("FDA-approved", fixed)
        self.assertNotIn("5 mg", fixed)

    # TEST 3 -------------------------------------------------------------------------------------------------------------
    def test3_drug_retrieval_timeout_is_not_reported_as_no_drugs(self):
        s = make_state()
        s["regulatory"] = []
        landscape.record_source_status(s, {"fda_deep": json.dumps({"error": "timed out after 25s (partial)"})})
        self.assertEqual(s["source_status"]["fda_deep"]["retrieval_status"], "partial")
        self.assertEqual(s["source_status"]["fda_deep"]["evidence_status"], "unavailable")
        s["drug_intelligence_status"] = landscape.drug_intelligence_status(s, drug_stage_ok=True)
        self.assertIn(s["drug_intelligence_status"], ("partial", "failed"))
        self.assertIn("does not mean that no approved drug exists", landscape.regulatory_markdown(s))
        r = verification.verify_and_repair(BASE + "There are no approved drug treatments for this condition.\n", s)
        self.assertNotIn("There are no approved drug treatments", r["answer"])
        self.assertIn("incomplete", r["answer"].lower())              # the incompleteness is disclosed

    # TEST 4 -------------------------------------------------------------------------------------------------------------
    def test4_recruiting_trial_is_never_called_effective(self):
        s = make_state()
        self.assertIn("cannot be inferred", landscape.trial_reading(s["trials_by_id"]["NCT01234567"]))
        r = verification.verify_and_repair(BASE + "NCT01234567 shows the intervention is effective and improves symptoms.\n", s)
        self.assertNotIn("improves symptoms", r["answer"])

    # TEST 5 -------------------------------------------------------------------------------------------------------------
    def test5_missing_dose_is_not_retrieved_never_invented_or_half_filled(self):
        s = make_state()
        answer = BASE + ("| Drug | Dose |\n|---|---|\n| Alpha | 0.5-3 mg PO |\n| Beta | 0.5-[value not in retrieved data] PO |\n")
        r = verification.verify_and_repair(answer, s)
        self.assertNotIn("0.5", r["answer"])
        self.assertNotIn("[value not in retrieved data]", r["answer"])
        self.assertIn("Not retrieved", r["answer"])

    # TEST 6 -------------------------------------------------------------------------------------------------------------
    def test6_protocol_only_paper_is_not_efficacy_evidence(self):
        raw = {"pubmed_x": json.dumps({"results": [{"pmid": "555555", "title": "Effect of X: study protocol for a trial", "authors": ["Chen L"], "journal": "J", "year": "2026",
                                                    "publication_date": "2026-01-01", "study_type": ["Journal Article"], "abstract_excerpt": "planned", "doi": None, "url": "u"}]})}
        papers = {}
        qa_stream.ingest(papers, {}, [], raw)
        self.assertTrue(papers["555555"]["protocol"])
        s = make_state()
        r = verification.verify_and_repair(BASE + "Music therapy is effective for social skills (Chen et al., Proto Journal, 2026 [PMID 222222]).\n", s)
        self.assertNotIn("is effective", r["answer"])

    # TEST 7 -------------------------------------------------------------------------------------------------------------
    def test7_indirect_population_is_labelled_indirect(self):
        s = make_state()
        self.assertTrue(verification.is_indirect(s["papers"]["333333"], {"condi"}))
        r = verification.verify_and_repair(BASE + "Nutraceuticals gave modest mood improvement (Woz et al., Mood Journal, 2026 [PMID 333333]).\n", s)
        self.assertIn("indirect evidence", r["answer"])

    # TEST 8 -------------------------------------------------------------------------------------------------------------
    def test8_approved_indication_is_not_off_label_and_not_an_approval_for_the_core_disease(self):
        s = make_state()
        md = landscape.regulatory_markdown(s)
        self.assertIn("irritability associated with condition X", md)             # the label's own wording, quoted
        self.assertIn("not to other symptoms of the condition", md)
        self.assertNotIn("off-label", md.lower())
        r = verification.verify_and_repair(BASE + "Risperidone is used off-label for irritability in condition X.\n", s)
        self.assertNotIn("off-label", r["answer"].lower())

    # TEST 9 -------------------------------------------------------------------------------------------------------------
    def test9_citation_no_longer_in_researchstate_is_needs_repair(self):
        s = make_state()
        answer = BASE + "A finding (Smith et al., Test Journal, 2026 [PMID 111111]).\n"
        self.assertEqual(verification.verify_and_repair(answer, s, repair=False)["status"], "PASS")
        del s["papers"]["111111"]                                                  # source removed from the state
        self.assertEqual(verification.verify_and_repair(answer, s, repair=False)["status"], "NEEDS_REPAIR")

    # TEST 10 ------------------------------------------------------------------------------------------------------------
    def test10_verification_reruns_on_the_repaired_final_string(self):
        s = make_state()
        seen = []
        real = verification.find_issues

        def spy(text, state):
            seen.append(text)
            return real(text, state)

        answer = BASE + "A claim (Fake et al., Fake Journal, 2026 [PMID 999999]).\n"
        with mock.patch.object(verification, "find_issues", side_effect=spy):
            r = verification.verify_and_repair(answer, s)
        self.assertGreaterEqual(len(seen), 2)                                      # before AND after the repair
        self.assertEqual(seen[-1], r["answer"])                                    # the last check ran on the exact text that is returned
        self.assertNotIn("999999", seen[-1])
        self.assertEqual(r["status"], "PASS")

    # ---- extra: retrieval-status semantics, trial relevance, context isolation ----------------------------------------
    def test11_retrieval_status_semantics(self):
        s = make_state()
        s["source_status"] = {}
        landscape.record_source_status(s, {"a": json.dumps({"results": []}), "b": json.dumps({"results": [1]}), "c": json.dumps({"error": "HTTP 500"}),
                                          "d": json.dumps({"error": "timed out after 3s (partial)"}), "e": "Web search failed: No results found."})
        st = s["source_status"]
        self.assertEqual((st["a"]["retrieval_status"], st["a"]["evidence_status"]), ("success", "not_found_after_search"))
        self.assertEqual((st["b"]["retrieval_status"], st["b"]["evidence_status"]), ("success", "found"))
        self.assertEqual((st["c"]["retrieval_status"], st["c"]["evidence_status"]), ("failed", "unavailable"))
        self.assertEqual(st["d"]["retrieval_status"], "partial")
        self.assertEqual(st["e"]["evidence_status"], "unavailable")

    def test12_peripheral_and_syndrome_trials_are_separated(self):
        s = make_state()
        s["trials_by_id"]["NCT09999991"] = {"nct_id": "NCT09999991", "title": "FOX1 Syndrome: characterizing genetic subtype", "interventions": [], "recruitment_status": "RECRUITING",
                                            "last_update": "2026-10-01", "design": "OBSERVATIONAL", "phase": [], "results_posted": False}
        s["trials_by_id"]["NCT09999992"] = {"nct_id": "NCT09999992", "title": "Habituation to dental examination", "interventions": ["Habituation"], "recruitment_status": "RECRUITING",
                                            "last_update": "2026-10-01", "design": "INTERVENTIONAL", "phase": [], "results_posted": False}
        md = landscape.trials_markdown(s)
        self.assertIn("NCT01234567", md)
        self.assertNotIn("NCT09999991", md)      # a syndrome / characterisation record is not a treatment trial
        self.assertNotIn("NCT09999992", md)      # nor is a peripheral intervention
        self.assertIn("2 further registry record(s)", md)      # they are counted in one line, not printed

    def test13_context_views_are_isolated_and_read_only(self):
        s = make_state()
        s["synthesis"] = "PREVIOUS WRITER OUTPUT SHOULD NEVER REACH THE DRUG SKILL"
        before = copy.deepcopy(s)
        drug_view = json.dumps(qa_stream.build_drug_context(s, "q"))
        qa_stream.build_fast_context(s, "q"); qa_stream.build_evidence_context(s, "q"); qa_stream.build_trials_context(s); qa_stream.build_qc_context(s)
        self.assertEqual(s, before)                                                # views never modify the ResearchState
        self.assertNotIn("PREVIOUS WRITER OUTPUT", drug_view)                      # no previous skill output
        self.assertNotIn("SKILL", drug_view)                                       # no SKILL.md text
        self.assertIn("Risperidone", drug_view)                                    # but it knows which drugs the FDA records already cover

    def test14_orphan_references_cannot_remain(self):
        s = make_state()
        answer = BASE + "A finding (Smith et al., Test Journal, 2026 [PMID 111111]).\n\n### Sources\n1. Smith J et al. Exercise. PMID 111111\n2. Old paper. PMID 222222\n"
        r = verification.verify_and_repair(answer, s)
        self.assertIn("PMID 111111", r["answer"].split("### Sources")[-1])
        self.assertNotIn("PMID 222222", r["answer"].split("### Sources")[-1])      # Sources are rebuilt from what the final text cites


    # ---- bugs found by the real autism run -----------------------------------------------------------------------------
    def test15_planner_output_types_are_normalised(self):
        plan = qa_stream.normalize_plan({"condition": ["autism spectrum disorder"], "drugs": "risperidone", "pubmed_query": ["a", "b"], "trial_condition": None, "intent": ["x"]})
        self.assertIsInstance(plan["condition"], str)
        self.assertEqual(plan["drugs"], ["risperidone"])
        jobs = qa_stream.deep_jobs({**plan, "trial_intervention": ""}, "How to treat X?")   # used to raise TypeError and silently kill the deep retrieval
        self.assertIn("fda_deep", jobs)

    def test16_italic_markers_do_not_hide_fabricated_numbers(self):
        s = make_state()
        bad = verification.unsupported_numbers("Trial (*n* = 24) showed g = 0.5.", s)
        self.assertTrue(any("24" in t for t in bad))
        self.assertFalse(verification.unsupported_numbers("Exercise improved motor skills (Hedges' *g* = 0.95).", s))   # supported by the abstract: not flagged

    def test17_claim_checks_ignore_the_sources_list(self):
        s = make_state()
        answer = BASE + "Mood improved (Woz et al., Mood Journal, 2026 [PMID 333333]) *(indirect evidence: related population)*.\n\n### Sources\n1. Woz J et al. PMID 333333\n"
        r = verification.verify_and_repair(answer, s)
        self.assertEqual(r["status"], "PASS")
        self.assertFalse([i for i in r["issues_after"] if i["kind"] == "indirect_unlabeled"])

    def test18_other_agents_table_is_enforced_for_any_column_count(self):
        s = make_state()
        table = ("#### Other agents named in the retrieved evidence (not in the FDA label records)\n| Drug | Category | Population | Dose | Approval | Evidence |\n|---|---|---|---|---|---|\n"
                 "| **Exercise intervention** | Investigational | kids | g = 0.9 | Under investigation | x |\n"
                 "| **Probiotic Z** | Investigational | kids | 10 mg/day | FDA-approved | (Smith et al., 2026 [PMID 111111]) |\n"
                 "| **Risperidone** | Off-label | kids | 1 mg | - | dup |\n")
        out = landscape.enforce_other_agents_table(table, s)
        self.assertNotIn("Exercise", out)                                          # a non-substance is not a drug
        self.assertNotIn("Risperidone", out)                                       # already covered by the FDA record
        self.assertIn("Probiotic Z", out)
        self.assertNotIn("10 mg/day", out)
        self.assertNotIn("FDA-approved", out)
        self.assertIn("Not retrieved", out)

    def test19_results_part_of_an_abstract_is_what_the_model_sees(self):
        text = "Background " + "x" * 700 + " Results After two months the score fell by 5 points. Conclusion modest."
        self.assertTrue(qa_stream.snippet(text, 120).startswith("Results"))


    # ---- bugs found by the second real autism run ---------------------------------------------------------------------
    def test20_planner_dictionary_never_becomes_search_terms(self):
        plan = qa_stream.normalize_plan({"condition": {"primary": "autism spectrum disorder", "synonyms": ["autism", "ASD"], "normalized": "autism spectrum disorder"}})
        self.assertEqual(plan["condition"], "autism spectrum disorder")
        import clinical_tools
        self.assertEqual(clinical_tools.condition_stems(["autism", "children"]), ["autis"])      # question words -> clean stems
        self.assertEqual(clinical_tools.condition_stems(["primary"]), ["prima"])                 # (this is what the old bug fed in: now never used)

    def test21_no_search_terms_is_a_failure_not_none_found(self):
        import clinical_tools
        result = json.loads(clinical_tools.search_fda_labels([], []))
        self.assertIn("error", result)                                                           # never "searched and found nothing"
        self.assertNotIn("results", result)

    def test22_established_needs_a_guideline(self):
        s = make_state()
        answer = BASE + "**Established**\n- Exercise improves motor skills (Smith et al., Test Journal, 2026 [PMID 111111]).\n"
        r = verification.verify_and_repair(answer, s)
        self.assertNotIn("**Established**", r["answer"])
        self.assertIn("Evidence-supported but limited", r["answer"])
        s["papers"]["111111"]["type"] = ["Practice Guideline"]
        r2 = verification.verify_and_repair(answer, s)
        self.assertIn("**Established**", r2["answer"])                                           # allowed when a guideline is cited

    def test23_one_study_cannot_confirm_or_prove(self):
        s = make_state()
        r = verification.verify_and_repair(BASE + "A dose-response relationship was confirmed (Smith et al., Test Journal, 2026 [PMID 111111]).\n", s)
        self.assertNotIn("confirmed", r["answer"])
        self.assertIn("reported", r["answer"])

    def test24_registry_results_statements_must_match_the_registry(self):
        s = make_state()
        s["trials_by_id"]["NCT01234567"]["results_posted"] = True
        r = verification.verify_and_repair(BASE + "NCT01234567 reported no results posted.\nNCT01234567 remains unpublished.\n", s)
        self.assertNotIn("no results posted", r["answer"])
        self.assertNotIn("unpublished", r["answer"])

    def test25_sample_size_is_supported_by_the_source_wording(self):
        s = make_state()
        s["papers"]["111111"]["abstract"] += " We randomised 62 children."
        self.assertFalse(verification.unsupported_numbers("A trial (*n* = 62).", s))
        self.assertTrue(verification.unsupported_numbers("A trial (*n* = 894).", s))


    def test26_empty_other_agents_table_becomes_none_retrieved(self):
        s = make_state()
        table = ("#### Other agents named in the retrieved evidence (not in the FDA label records)\n| Drug | Category | Population | Dose | Approval | Evidence |\n|---|---|---|---|---|---|\n"
                 "| **None retrieved.** |  |  |  |  |  |\n\n**Safety alerts:** None retrieved.")
        out = landscape.enforce_other_agents_table(table, s)
        self.assertIn("None retrieved.", out)
        self.assertNotIn("| Drug |", out)                                                          # no empty table with a fake row

    def test27_overclaim_words_are_softened_anywhere_in_the_prose(self):
        s = make_state()
        r = verification.verify_and_repair(BASE + "1. **Wang et al.**: meta-analysis confirmed exercise helps; a review validated the programme.\n", s)
        self.assertNotIn("confirmed", r["answer"])
        self.assertNotIn("validated", r["answer"])

    def test28_phase1_drug_block_uses_the_deep_regulatory_result(self):
        s = make_state()
        s["source_status"]["fda_fast"] = {"retrieval_status": "partial", "evidence_status": "unavailable", "error": "timed out after 3s (partial)"}
        s["source_status"]["fda_deep"] = {"retrieval_status": "success", "evidence_status": "found", "error": None}
        block = landscape.top_drugs_markdown(s)
        self.assertIn("Risperidone", block)
        self.assertNotIn("unavailable", block)


    def test29_table_is_enforced_even_after_a_stray_line(self):
        s = make_state()
        text = ("#### Other agents named in the retrieved evidence (not in the FDA label records)\nNone retrieved.\n\n| Drug | Category | Population | Dose | Approval | Evidence |\n"
                "|---|---|---|---|---|---|\n| **Memantine** | Off-label | youth | Not retrieved | FDA-endorsed | slides |\n\n---\n**Safety alerts:**\nNone")
        out = landscape.enforce_other_agents_table(text, s)
        self.assertIn("Not verified in retrieved regulatory sources", out)           # the enforcement reached the real table
        self.assertNotIn("FDA-endorsed", out)
        self.assertEqual(out.count("None retrieved."), 0)                            # the stray line is dropped because a row survives
        self.assertIn("**Safety alerts:**", out)

    def test30_malformed_pubmed_ids_are_removed(self):
        s = make_state()
        r = verification.verify_and_repair(BASE + "Recommended for inattention (APA, 2025 [PMID 2025-77457-001]).\nExercise helped (Smith et al., Test Journal, 2026 [PMID 111111]).\n", s)
        self.assertNotIn("2025-77457-001", r["answer"])
        self.assertIn("111111", r["answer"])

    def test31_a_year_is_not_a_sample_size(self):
        s = make_state()
        self.assertTrue(verification.unsupported_numbers("Exercise meta-analysis (*n*=2026).", s))     # 2026 only appears as a year
        s["papers"]["111111"]["abstract"] += " Sixty-two children were randomised: 62 children in total."
        self.assertFalse(verification.unsupported_numbers("An RCT (n = 62).", s))

    def test32_a_doubled_citation_collapses(self):
        s = make_state()
        text, _ = verification.normalize_citations("Finding (*Smith et al., Test Journal, 2026 [PMID 111111]* 2026 [PMID 111111]).", s)
        self.assertEqual(text.count("[PMID 111111]"), 1)


    def test33_several_issues_on_one_line_are_all_repaired_in_one_pass(self):
        s = make_state()
        line = "Dose-dependent effects were confirmed for motor skills (Smith et al., Test Journal, 2026 [PMID 111111])."
        r = verification.verify_and_repair(BASE + line + "\n", s)
        self.assertEqual(r["status"], "PASS")                                       # no leftover after the single repair pass
        self.assertNotIn("confirmed", r["answer"])
        self.assertNotIn("ose-dependent", r["answer"])
        self.assertEqual(r["repair_count"], 1)


    def test34_a_section_cut_by_the_output_limit_is_trimmed_not_shown_half_written(self):
        text, cut = qa_stream.trim_incomplete("- Music therapy (Chen et al., 2026 [PMID 222222]).\n- **ABA ethics and efficacy**: Aut")
        self.assertTrue(cut)
        self.assertNotIn("Aut", text)
        self.assertTrue(text.endswith(")."))
        self.assertEqual(qa_stream.trim_incomplete("Complete sentence.")[1], False)


    def test35_a_table_emptied_by_a_repair_reads_none_retrieved(self):
        s = make_state()
        answer = (BASE + "#### Other agents\n| Drug | Category |\n|---|---|\n| Fakeol | A claim (Fake et al., Fake Journal, 2026 [PMID 999999]) |\n\nAfter.\n")
        r = verification.verify_and_repair(answer, s)
        self.assertNotIn("| Drug | Category |", r["answer"])
        self.assertIn("None retrieved.", r["answer"])
        self.assertEqual(r["status"], "PASS")


    def test36_guideline_recommended_needs_a_cited_guideline(self):
        s = make_state()
        s["web"] = [{"title": "Leucovorin approval press release", "url": "https://example.org/press", "snippet": "x"},
                    {"title": "Autism clinical practice guideline", "url": "https://example.org/guideline", "snippet": "y"}]
        header = "#### Other agents named in the retrieved evidence (not in the FDA label records)\n| D | C | P | Dose | A | E |\n|---|---|---|---|---|---|\n"
        wrong = landscape.enforce_other_agents_table(header + "| Drug1 | Guideline-recommended | kids | x | y | FDA approval (https://example.org/press) |\n", s)
        right = landscape.enforce_other_agents_table(header + "| Drug2 | Guideline-recommended | kids | x | y | (https://example.org/guideline) |\n", s)
        self.assertNotIn("| Guideline-recommended |", wrong)                         # an approval press release is not a guideline
        self.assertIn("Category not verified", wrong)
        self.assertIn("| Guideline-recommended |", right)                            # a cited guideline is accepted


    def test37_confirms_is_softened_and_no_stranded_annotation_is_left(self):
        s = make_state()
        answer = (BASE + "**Insufficient evidence**\n- A claim about 'social communication' improved (Smith et al., Test Journal, 2026 [PMID 111111]). *(Direct, terminated 2026)*\n"
                  "- The review confirms consistent improvement (Smith et al., Test Journal, 2026 [PMID 111111]).\n")
        r = verification.verify_and_repair(answer, s)
        self.assertNotIn("confirms", r["answer"])
        self.assertNotIn("*(Direct, terminated 2026)*", r["answer"])


    def test38_an_indirect_paper_cited_in_a_table_row_is_relabelled(self):
        s = make_state()
        answer = (BASE + "#### Other agents\n| Drug | Category | Population | Dose | Approval | Evidence |\n|---|---|---|---|---|---|\n"
                  "| **Nutra X** | Investigational | Children with condition X | Not retrieved | Not verified | RCT (Woz et al., Mood Journal, 2026 [PMID 333333]) |\n")
        r = verification.verify_and_repair(answer, s)
        self.assertEqual(r["status"], "PASS")
        self.assertIn("Indirect evidence (different or related population)", r["answer"])
        self.assertNotIn("Children with condition X", r["answer"])                   # the misstated population is replaced


    def test39_posted_results_statements_match_the_registry_in_both_directions(self):
        s = make_state()
        s["trials_by_id"]["NCT07654321"] = {"nct_id": "NCT07654321", "title": "T", "interventions": ["x"], "recruitment_status": "TERMINATED", "last_update": "2026-01-01",
                                            "results_posted": True, "design": "INTERVENTIONAL", "url": "https://clinicaltrials.gov/study/NCT07654321"}
        r = verification.verify_and_repair(BASE + "A terminated trial (NCT07654321) found no posted results on social responsiveness.\n"
                                                  "Results were posted for NCT01234567 on the primary endpoint.\n", s)
        self.assertNotIn("found no posted results", r["answer"])                    # the registry says results ARE posted
        self.assertNotIn("Results were posted for NCT01234567", r["answer"])        # the registry says they are NOT

    def test40_drugs_named_only_as_exposure_risks_are_not_listed_as_agents(self):
        s = make_state()
        table = ("#### Other agents named in the retrieved evidence (not in the FDA label records)\n| D | C | P | Dose | A | E |\n|---|---|---|---|---|---|\n"
                 "| **Topiramate** | Indirect | Prenatal exposure cohort | x | y | Prenatal exposure linked to 1.9% ASD risk; no treatment efficacy |\n"
                 "| **Probiotic Z** | Investigational | kids | x | y | RCT (Smith et al., Test Journal, 2026 [PMID 111111]) |\n")
        out = landscape.enforce_other_agents_table(table, s)
        self.assertNotIn("Topiramate", out)
        self.assertIn("Probiotic Z", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
