"""Tests for the four required fixes (list repair + lint, scope/bottom line, source banner + retry, established care) and the advice flag.

Synthetic data only: no network and no model.  Run:  .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import guidelines  # noqa: E402
import landscape  # noqa: E402
import qa_stream  # noqa: E402
import report  # noqa: E402
import verification  # noqa: E402
from tests.test_regression import BASE, make_state, paper  # noqa: E402

RESULT = "Results: Exercise significantly improved motor skills compared with controls (p = 0.01). Conclusions: Larger trials are needed."


def study(pmid, author="Smith J", year="2025", types=("Randomized Controlled Trial",), abstract=RESULT, **kw):
    p = paper(pmid, f"A study {pmid} of condition X", year=year, author=author, abstract=abstract, **kw)
    p["type"] = p["study_type"] = list(types)
    return p


# ---------------------------------------------------------------------------------------------------- FIX 1
class ListRepair(unittest.TestCase):
    def test_sentence_split_keeps_et_al_and_list_number_together(self):
        parts = verification._sentences("3. **Restoy et al. (2024)**: A study of children (n=55). Next sentence here.")
        self.assertEqual(parts[0], "3. **Restoy et al. (2024)**: A study of children (n=55).")
        self.assertEqual(len(parts), 2)
        self.assertEqual(len(verification._sentences("Treatment works, e.g. in trial A vs. trial B. Another sentence.")), 2)

    def test_removing_a_claim_removes_the_whole_list_item_and_renumbers(self):
        s = make_state()
        s["papers"]["444444"] = study("444444", author="Restoy A")
        text = (BASE + "### Key studies\n\n1. **Smith et al. (2025)**, randomized controlled trial: “Exercise improved skills.” (PMID 111111)\n"
                "2. **Fake et al. (2024)**, meta-analysis: “Something.” (PMID 999999)\n"
                "3. **Restoy et al. (2024)**, meta-analysis: “Exercise significantly improved motor skills compared with controls (p = 0.01).” (PMID 444444)\n")
        r = verification.verify_and_repair(text, s)
        body = r["answer"]
        self.assertNotIn("999999", body)
        self.assertRegex(body, r"(?m)^1\. \*\*Smith")
        self.assertRegex(body, r"(?m)^2\. \*\*Restoy")                  # renumbered from 1, no gap, no stub
        self.assertEqual(verification.lint_report(body), [])

    def test_lint_flags_the_reported_defects(self):
        bad = ("### Key studies\n\n2. **Mazurek et al. (2026)**: Systematic review (n=3).\n3. **Restoy et al.\n4. **Sandbank et al. (2023)**: Meta-analysis (n=13).\n")
        kinds = {i["kind"] for i in verification.lint_report(bad)}
        self.assertTrue({"lint_numbering", "lint_stray_bold"} <= kinds)
        fixed, _ = verification.lint_fix(bad)
        self.assertNotIn("Restoy", fixed)                                # a truncated item is removed, never completed
        self.assertRegex(fixed, r"(?m)^1\. \*\*Mazurek")
        self.assertRegex(fixed, r"(?m)^2\. \*\*Sandbank")
        self.assertEqual(verification.lint_report(fixed), [])

    def test_lint_blocks_pass(self):
        s = make_state()
        issues = verification.lint_report("### Key studies\n\n1. **Smith et al.\n")
        qc = verification.quality_control("x", s, issues)
        self.assertEqual(qc["status"], "NEEDS_REPAIR")
        self.assertEqual(qc["checks"]["formatting"], "fail")

    def test_clean_text_has_no_lint(self):
        self.assertEqual(verification.lint_report("## H\n\nA full sentence here that ends properly and is long enough to count.\n\n1. one item here.\n2. two item here.\n\n| a | b |\n|---|---|\n"), [])


class KeyStudies(unittest.TestCase):
    def test_every_entry_has_authors_year_design_finding_pmid(self):
        entries = report.key_study_entries([study("100001"), study("100002", author="Wang T", year="2024", types=("Meta-Analysis",))])
        self.assertEqual(len(entries), 2)
        md = report.key_studies_markdown([study("100001")])
        for part in ("Smith et al." if False else "Smith", "(2025)", "randomized controlled trial", "PMID 100001", "“Exercise significantly improved"):
            self.assertIn(part, md)

    def test_incomplete_entries_are_dropped(self):
        no_design = study("100003", types=("research-article",))
        no_finding = study("100004", abstract="Background: Condition X is a common disorder. Methods: We searched databases.")
        no_pmid = study("100005")
        no_pmid["pmid"] = ""
        protocol = study("100006", protocol=True)
        self.assertEqual(report.key_study_entries([no_design, no_finding, no_pmid, protocol]), [])
        self.assertIn("none is listed", report.key_studies_markdown([no_design, no_finding]))

    def test_finding_is_copied_verbatim_from_the_abstract(self):
        p = study("100007")
        self.assertIn(report.finding_sentence(p), RESULT)

    def test_model_written_key_studies_is_removed(self):
        text = "### Latest findings\n\nA finding.\n\n### Key studies\n\n1. **Smith et al.\n\n### Conflicting or negative evidence\n\nNone."
        out = report.remove_section(text, "Key studies")
        self.assertNotIn("Key studies", out)
        self.assertIn("Conflicting or negative evidence", out)


# ---------------------------------------------------------------------------------------------------- FIX 2
class BottomLine(unittest.TestCase):
    def test_bottom_line_is_counts_not_a_headline(self):
        s = make_state()
        papers = [study("1"), study("2", types=("Meta-Analysis",)), study("3", types=("Meta-Analysis",))]
        md = report.bottom_line_markdown("How to treat X?", s, papers, None, True)
        self.assertIn("2 systematic reviews or meta-analyses", md)
        self.assertIn("1 randomized controlled trial", md)
        for p in papers:
            self.assertNotIn(p["title"], md)                              # never headlines one paper
        self.assertNotRegex(md, r"(?i)\b(effective|improves|recommended)\b")

    def test_bottom_line_with_no_records_says_so(self):
        md = report.bottom_line_markdown("q", {"regulatory": []}, [], None, False)
        self.assertIn("No usable PubMed/Europe PMC record", md)
        self.assertIn("No FDA label", md)


# ---------------------------------------------------------------------------------------------------- FIX 3
OK = {"retrieval_status": "success", "evidence_status": "found", "error": None}
PARTIAL = {"retrieval_status": "partial", "evidence_status": "unavailable", "error": "timed out after 3s (partial)"}
FAILED = {"retrieval_status": "failed", "evidence_status": "unavailable", "error": "HTTP 500"}


class SourceBanner(unittest.TestCase):
    def test_no_banner_when_everything_succeeded(self):
        self.assertEqual(report.source_banner({"source_status": {"pubmed_fast": OK, "fda_fast": OK}}), "")

    def test_banner_names_a_source_that_still_failed_and_says_absence_cannot_be_concluded(self):
        banner = report.source_banner({"source_status": {"pubmed_fast": OK, "web_fast": FAILED, "web_approvals": FAILED}})
        self.assertIn("Web search", banner)
        self.assertNotIn("PubMed", banner)
        self.assertIn("absence of evidence cannot be concluded", banner)
        self.assertTrue(banner.startswith("> "))

    def test_banner_lists_only_sources_still_failed_after_retry(self):
        # fda_fast timed out, but the deep stage redid the search (fda_deep) and it succeeded: not listed
        banner = report.source_banner({"source_status": {"fda_fast": PARTIAL, "fda_deep": OK, "trials_fast": FAILED, "trials_deep": FAILED}})
        self.assertNotIn("FDA", banner)
        self.assertIn("ClinicalTrials.gov", banner)

    def test_partial_when_only_some_jobs_of_a_source_failed(self):
        problems = report.source_problems({"source_status": {"pubmed_recent": OK, "pubmed_negative": PARTIAL}})
        self.assertEqual(problems[0][1], "partial")

    def test_banner_is_not_flagged_by_the_verifier(self):
        s = make_state()
        text = report.source_banner({"source_status": {"web_fast": FAILED}}) + "\n" + BASE + "Exercise helped (Smith et al., Test Journal, 2026 [PMID 111111]).\n"
        r = verification.verify_and_repair(text, s)
        self.assertEqual(verification.lint_report(r["answer"]), [])
        self.assertIn("Source warning", r["answer"])


class Retry(unittest.TestCase):
    def test_one_retry_with_backoff_then_success(self):
        calls = []

        def flaky():
            calls.append(1)
            return json.dumps({"error": "HTTP 503"}) if len(calls) == 1 else json.dumps({"results": [1]})
        out = qa_stream.with_retry(flaky, retries=1, pause=0.0)()
        self.assertEqual(len(calls), 2)
        self.assertIn("results", out)

    def test_retry_is_bounded(self):
        calls = []
        qa_stream.with_retry(lambda: calls.append(1) or json.dumps({"error": "x"}), retries=1, pause=0.0)()
        self.assertEqual(len(calls), 2)                                  # never more than one retry

    def test_every_fast_source_is_wrapped_in_a_retry(self):
        calls = {"n": 0}
        original = qa_stream._fast_jobs

        def fake(question):
            def flaky():
                calls["n"] += 1
                return json.dumps({"error": "boom"}) if calls["n"] == 1 else json.dumps({"results": []})
            return {"pubmed_fast": flaky}
        qa_stream._fast_jobs = fake
        try:
            out = qa_stream.fast_jobs("q")["pubmed_fast"]()
        finally:
            qa_stream._fast_jobs = original
        self.assertEqual(calls["n"], 2)
        self.assertNotIn("error", out)


# ---------------------------------------------------------------------------------------------------- FIX 4
def fake_search(results_by_word):
    def search(query):
        for word, items in results_by_word.items():
            if word in query:
                return json.dumps(items)
        return json.dumps([])
    return search


GOOD = {"title": "Autism spectrum disorder in under 19s: support and management (2021)", "url": "https://www.nice.org.uk/guidance/cg170",
        "snippet": "This guideline covers diagnosing and managing autism spectrum disorder in children and young people, including psychosocial interventions."}
OFFSITE = {"title": "Autism treatment blog", "url": "https://random-blog.example.com/autism", "snippet": "Ten autism tips that work for every child in our experience."}


class EstablishedCare(unittest.TestCase):
    def test_every_question_gets_the_same_generic_topics_from_the_config(self):
        first = [t["id"] for t in guidelines.topics_for("How to treat autism in children?")]
        second = [t["id"] for t in guidelines.topics_for("What is first-line therapy for hypertension?")]
        self.assertEqual(first, second)
        for needed in ("assessment", "medications", "first_line", "monitoring"):
            self.assertIn(needed, first)

    def test_failed_search_is_reported_unavailable_not_none(self):
        res = guidelines.find_guidelines("How to treat autism in children?", ["autism"], lambda q: "Web search failed: timeout")
        self.assertIn("error", res)
        md = guidelines.established_care_markdown([], False)
        self.assertIn("search unavailable", md)
        self.assertNotIn("No guideline retrieved", md)

    def test_off_topic_guideline_is_not_used(self):
        wrong = {"title": "Hypertension guideline", "url": "https://www.nice.org.uk/guidance/ng136", "snippet": "This guideline covers diagnosing and managing hypertension in adults."}
        res = guidelines.find_guidelines("How to treat autism in children?", ["autism"], fake_search({"medications": [wrong]}))
        self.assertTrue(all(not r.get("title") for r in res["results"]))

    def test_only_allow_listed_organisations_are_used_with_verbatim_excerpt(self):
        res = guidelines.find_guidelines("How to treat autism in children?", ["autism"], fake_search({"medications": [OFFSITE, GOOD]}))
        rows = {r["topic"]: r for r in res["results"]}
        self.assertEqual(rows["medications"]["organisation"], "NICE (UK)")
        self.assertEqual(rows["medications"]["year"], "2021")
        self.assertIn(rows["medications"]["excerpt"], GOOD["snippet"])       # verbatim (here: the whole snippet)
        self.assertTrue(rows["first_line"].get("none"))                       # nothing retrieved for this topic
        md = guidelines.established_care_markdown(res["results"], True)
        self.assertIn("No guideline retrieved", md)
        self.assertNotIn("random-blog", md)


    def test_verifier_rejects_an_established_care_row_without_a_retrieved_guideline(self):
        s = make_state()
        s["guidelines"] = [{"url": GOOD["url"], "title": GOOD["title"], "organisation": "NICE (UK)", "excerpt": GOOD["snippet"], "year": "2021"}]
        table = ("### Established care\n\n| Topic | Guideline | Organisation | Year | Excerpt |\n|---|---|---|---|\n"
                 f"| Speech | [NICE]({GOOD['url']}) | NICE (UK) | 2021 | “Guideline text.” |\n"
                 "| Sleep | Melatonin is first-line for every child | | | |\n"
                 "| OT | No guideline retrieved | | | |\n")
        kinds = [i["kind"] for i in verification.find_issues(BASE + table, s)]
        self.assertEqual(kinds.count("established_care_unsourced"), 1)   # only the model-style claim row
        repaired = verification.verify_and_repair(BASE + table, s)["answer"]
        self.assertNotIn("Melatonin", repaired)
        self.assertIn("No guideline retrieved", repaired)


# ---------------------------------------------------------------------------------------------------- advice flag
class AdviceFlag(unittest.TestCase):
    def test_flagged_not_reworded_and_note_added_before_sources(self):
        s = make_state()
        text = BASE + "You should start risperidone for your child (Smith et al., Test Journal, 2026 [PMID 111111]).\n"
        r = verification.verify_and_repair(text, s)
        self.assertIn("You should start risperidone for your child", r["answer"])        # the sentence is untouched
        self.assertIn(verification.GENERAL_NOTE, r["answer"])
        self.assertTrue(any("flagged" in x for x in r["repairs"]))
        self.assertLess(r["answer"].index(verification.GENERAL_NOTE), r["answer"].index("### Sources"))
        self.assertEqual(r["status"], "PASS")                                             # low severity: never blocks

    def test_ordinary_evidence_wording_is_not_flagged(self):
        s = make_state()
        text = BASE + "Exercise improved motor skills in a meta-analysis (Smith et al., Test Journal, 2026 [PMID 111111]).\n"
        self.assertEqual([i for i in verification.find_issues(text, s) if i["kind"] == "personal_advice"], [])


if __name__ == "__main__":
    unittest.main()


class GuidelineHosts(unittest.TestCase):
    def test_a_rate_limited_topic_is_retried_once(self):
        calls = []

        def search(q):
            calls.append(q)
            return "Web search failed: rate limit" if len(calls) == 1 else json.dumps([GOOD])
        res = guidelines.find_guidelines("autism", ["autism"], search, guidelines.load_config() | {"default": {"topics": [{"id": "t", "label": "T", "query": "q"}]}, "overrides": {}})
        self.assertEqual(len(calls), 2)
        self.assertEqual(res["results"][0]["organisation"], "NICE (UK)")

    def test_pubmed_and_pmc_articles_are_not_guidelines(self):
        cfg = guidelines.load_config()
        self.assertIsNone(guidelines.organisation_for("https://pmc.ncbi.nlm.nih.gov/articles/PMC10422951/", cfg))
        self.assertIsNone(guidelines.organisation_for("https://pubmed.ncbi.nlm.nih.gov/123456/", cfg))
        self.assertEqual(guidelines.organisation_for("https://www.nichd.nih.gov/health/topics/autism", cfg), "US National Institute of Child Health and Human Development")



class OpeningAndGuidelineBanner(unittest.TestCase):
    def test_guideline_search_failure_is_not_a_top_banner(self):
        self.assertEqual(report.source_banner({"source_status": {"guidelines_deep": FAILED}}), "")
        self.assertIn("Web search", report.source_banner({"source_status": {"guidelines_deep": FAILED, "web_fast": FAILED}}))


class DoiLookalikeInUrl(unittest.TestCase):
    def test_a_url_containing_a_doi_shaped_path_is_not_an_unretrieved_doi(self):
        s = make_state()
        url = "https://publications.aap.org/toolkits/resources/10.1542/peo_document591/82079/nutrition-toolkit"
        s["guidelines"] = [{"url": url, "title": "Toolkit", "organisation": "American Academy of Pediatrics", "excerpt": "Guidance text for families.", "year": "2025"}]
        text = BASE + f"| Topic | Guideline |\n|---|---|\n| Feeding | [Toolkit]({url}) |\n"
        self.assertEqual([i for i in verification.find_issues(text, s) if i["kind"] == "unretrieved_doi"], [])


class RepairsOnTheSameLine(unittest.TestCase):
    def test_wording_edit_survives_an_earlier_sentence_removal_on_the_same_line(self):
        s = make_state()
        text = (BASE + "There are no pharmacologic treatments for condition X. Exercise improved motor skills, with effects confirmed in a meta-analysis "
                "(Smith et al., Test Journal, 2026 [PMID 111111]).\n")
        r = verification.verify_and_repair(text, s)
        self.assertNotIn("confirmed", r["answer"])
        self.assertEqual(r["status"], "PASS")


class ModelWrittenKeyStudies(unittest.TestCase):
    def test_incomplete_model_entries_are_dropped_and_list_renumbered(self):
        t = ("### Latest\n\nx\n\n### Key studies\n\n1. **Wang et al. (2026)**, meta-analysis: exercise improved motor skills in children with ASD (PMID 42758334).\n"
             "2. **Restoy et al.\n3. Bad entry (2024) with no pmid at all but plenty of words in the sentence here.\n"
             "4. **Mazurek et al. (2026)**, systematic review: behavioral sleep interventions improved some sleep outcomes in autistic children (PMID 42685447).\n\n### Conflicting\n\nNone.")
        out = report.clean_model_key_studies(t)
        self.assertNotIn("Restoy", out)
        self.assertNotIn("Bad entry", out)
        self.assertRegex(out, r"(?m)^1\. \*\*Wang")
        self.assertRegex(out, r"(?m)^2\. \*\*Mazurek")
        self.assertIn("### Conflicting", out)
        self.assertEqual(verification.lint_report(out), [])

    def test_falls_back_to_code_built_list_when_the_model_gave_nothing_complete(self):
        candidate = study("100001")
        candidate["triage"] = {"relevance": "direct", "role": "emerging", "evidence": "A study of condition X reported an outcome."}
        out = report.clean_model_key_studies("### Key studies\n\n1. **Restoy et al.\n", [candidate], None)
        self.assertIn("PMID 100001", out)


class SourcesEntryIsNotReadAsACitation(unittest.TestCase):
    def test_title_starting_with_a_year_does_not_trigger_a_citation_mismatch(self):
        s = make_state()
        p = study("555555", author="Ringel MD", year="2025")
        p["title"] = "2025 American Thyroid Association Management Guidelines for Adult Patients"
        s["papers"]["555555"] = p
        text = BASE + "Guideline context (Ringel et al., Test Journal, 2025 [PMID 555555]).\n"
        r = verification.verify_and_repair(text, s)
        self.assertEqual(r["status"], "PASS")
        self.assertIn("PMID 555555", r["answer"])


class EmptyLabel(unittest.TestCase):
    def test_a_label_with_nothing_under_it_is_removed(self):
        text = "### What remains\n- **Efficacy**: music therapy dosage (protocol pending).\n\n**Effectiveness:**\n\n### Established care\n\nText for the next section here."
        fixed, actions = verification.lint_fix(text)
        self.assertNotIn("Effectiveness", fixed)
        self.assertIn("music therapy dosage", fixed)
        self.assertIn("removed an empty label", actions)

    def test_a_label_followed_by_items_is_kept(self):
        text = "**Efficacy:**\n- music therapy dosage effects on social communication.\n"
        self.assertIn("Efficacy", verification.lint_fix(text)[0])




class TopDrugsTable(unittest.TestCase):
    ARI = ("1 INDICATIONS AND USAGE Aripiprazole is indicated for the treatment of: \u2022 Schizophrenia \u2022 Irritability Associated with Autistic Disorder "
           "\u2022 Treatment of Tourette\u2019s Disorder Aripiprazole is an atypical antipsychotic. The oral formulations are indicated for: \u2022 Schizophrenia (14.1)")
    RIS = ("with lithium or valproate, for the treatment of acute manic or mixed episodes associated with Bipolar I Disorder ( 1.2 ) "
           "Treatment of irritability associated with autistic disorder ( 1.3 ) 1.1 Schizophrenia Risperidone tablets are indicated for the treatment of schizophrenia.")

    def test_each_row_shows_the_complete_label_clause_for_the_condition(self):
        self.assertEqual(landscape.condition_clause(self.ARI, ["autis"]), "Irritability Associated with Autistic Disorder")
        self.assertEqual(landscape.condition_clause(self.RIS, ["autis"]), "Treatment of irritability associated with autistic disorder")

    def test_the_clause_is_copied_from_the_label_text(self):
        for text in (self.ARI, self.RIS):
            self.assertIn(landscape.condition_clause(text, ["autis"]).lower(), text.lower())

    def test_table_has_a_label_link_and_no_cut_off_marker(self):
        s = make_state()
        s["core_stems"] = ["autis"]
        s["regulatory"] = [{"generic": "Risperidone", "matches_condition": True, "model_relevant": True, "model_evidence": "Risperidone is indicated for the treatment of irritability associated with condition X.", "indication_excerpt": self.RIS, "url": "https://dailymed.nlm.nih.gov/x"}]
        s["source_status"] = {"fda_fast": {"retrieval_status": "success", "evidence_status": "found", "error": None}}
        md = landscape.top_drugs_markdown(s)
        self.assertIn("[DailyMed](https://dailymed.nlm.nih.gov/x)", md)
        self.assertNotIn("...", md)

    def test_a_long_unmatched_excerpt_is_cut_at_a_word_with_a_marker(self):
        out = landscape.condition_clause("word " * 100, ["autis"])
        self.assertTrue(out.endswith(" ..."))
        self.assertLessEqual(len(out), 205)




class CompleteText(unittest.TestCase):
    TEXT = ("1 INDICATIONS AND USAGE Risperidone tablets are an atypical antipsychotic indicated for: Treatment of schizophrenia ( 1.1 ) Treatment of irritability associated with "
            "autistic disorder ( 1.3 ) 1.1 Schizophrenia Risperidone tablets are indicated for the treatment of schizophrenia. 1.3 Irritability Associated with Autistic Disorder "
            "Risperidone tablets are indicated for the treatment of irritability associated with autistic disorder, including symptoms of aggression towards others, "
            "temper tantrums, and quickly changing moods. Efficacy was established in 3 short-term trials in children and adolescents (ages 5 to 17 years).")

    def test_fda_statement_is_the_complete_formal_sentence_with_the_population(self):
        import clinical_tools
        out = clinical_tools.indication_statement(self.TEXT, ["autis"], "Risperidone")
        self.assertTrue(out.startswith("Risperidone tablets are indicated for the treatment of irritability"))
        self.assertTrue(out.rstrip().endswith("(ages 5 to 17 years)."))
        self.assertNotIn("1.3", out)
        self.assertIn(out.split(" Efficacy")[0], self.TEXT)

    def test_fda_table_shows_the_full_statement_without_a_cut_marker(self):
        s = make_state()
        statement = "Risperidone tablets are indicated for the treatment of irritability associated with autistic disorder, including symptoms of aggression towards others."
        s["regulatory"][0]["indication_statement"] = statement
        md = landscape.landscape_markdown(s, [], [])
        self.assertIn(statement, md)
        self.assertNotIn("...", md)

    def test_used_in_practice_excerpt_is_the_whole_sentence(self):
        sentence = "Beside the evidence for the labelled drugs, quetiapine and ziprasidone are still commonly used in clinical practice in the treatment of irritability and aggressive behavior in children and adolescents."
        self.assertEqual(landscape.complete_sentence(sentence), sentence)

    def test_a_snippet_that_stops_mid_sentence_ends_at_a_clause_and_says_so(self):
        out = landscape.complete_sentence("Quetiapine is commonly used in clinical practice for irritability, aggression and")
        self.assertTrue(out.endswith("(source text ends here)"))
        self.assertNotIn("...", out)
        self.assertNotIn("aggression and", out)




class ClickableLinks(unittest.TestCase):
    def test_pdf_has_real_link_annotations_for_markdown_links(self):
        import re
        import tempfile
        import pdf_export
        md = ("# t\n\n| Drug | Source | FDA label |\n|---|---|---|\n| Melatonin | [A review of drugs](https://example.org/review) | "
              "[DailyMed label: Melatonin](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=abc-123) |\n")
        with tempfile.TemporaryDirectory() as d:
            pdf_export.markdown_to_pdf(md, d + "/t.pdf", "t")
            raw = open(d + "/t.pdf", "rb").read().decode("latin-1")
        uris = re.findall(r"/URI \(([^)]*)\)", raw)      # a real link annotation, not just coloured text
        self.assertIn("https://example.org/review", uris)
        self.assertIn("https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=abc-123", uris)

    def test_the_used_in_practice_table_has_no_per_drug_label_column_but_safety_alerts_keep_the_label_link(self):
        rows = [{"drug": "Quetiapine", "excerpt": "Quetiapine is commonly used.", "label": "[A review](https://x.org/r)", "fda_label": "https://dailymed.nlm.nih.gov/q",
                 "purpose": "irritability", "population": "children"}]
        s = make_state()
        s["drug_labels"] = [{"drug": "Quetiapine", "url": "https://dailymed.nlm.nih.gov/q", "boxed_warning": "Increased mortality in elderly patients."}]
        md = landscape.landscape_markdown(s, rows, [{"drug": "NTI164", "nct": "NCT07257939", "phase": "Phase 3", "status": "Recruiting", "updated": "2026-09-30",
                                                      "url": "https://clinicaltrials.gov/study/NCT07257939", "fda_label": None, "conditions": "Autism", "primary": "ABC score", "ages": "from 6 Years"}])
        self.assertIn("| Drug | Indication | Dose / route (if sourced) | Key evidence |", md)      # the specification's six columns
        self.assertNotIn("FDA label of this drug", md)
        used_part = md.split("### Safety warnings")[0]
        self.assertIn("[Quetiapine](https://dailymed.nlm.nih.gov/q)", used_part)      # the drug name opens that drug's own DailyMed page
        safety = landscape.safety_alerts_markdown([{"drug": "Quetiapine", "url": "https://dailymed.nlm.nih.gov/q", "boxed_warning": "Increased mortality in elderly patients."}])
        self.assertIn("https://dailymed.nlm.nih.gov/q", safety)      # where the boxed warning came from


class AboutTheCondition(unittest.TestCase):

    def test_trial_search_records_the_registered_conditions(self):
        import clinical_tools, inspect
        self.assertIn("conditionsModule", inspect.getsource(clinical_tools.search_clinical_trials))


class DrugLabelLinksSurviveVerification(unittest.TestCase):
    def test_a_retrieved_drug_label_link_is_kept_and_an_invented_one_is_removed(self):
        s = make_state()
        good = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=03dc86d3-575f-4c65-b06d-2613efbcd943"
        s["drug_labels"] = [{"drug": "melatonin", "url": good}]
        table = (BASE + "| Drug | FDA label of this drug |\n|---|---|\n"
                 f"| melatonin | [DailyMed label: melatonin]({good}) |\n"
                 "| invented | [DailyMed label: invented](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=ffffffff-0000-0000-0000-000000000000) |\n")
        out = verification.verify_and_repair(table, s)["answer"]
        self.assertIn(good, out)
        self.assertNotIn("ffffffff-0000", out)




class SafetyAndRegulatorUpdates(unittest.TestCase):
    def test_boxed_warning_is_shown_in_the_labels_words_with_a_link(self):
        labels = [{"drug": "Sertraline", "url": "https://dailymed.nlm.nih.gov/s", "boxed_warning": "BOXED WARNING Suicidality and Antidepressant Drugs Antidepressants increased the risk of suicidal thinking in children."},
                  {"drug": "Melatonin", "url": "https://dailymed.nlm.nih.gov/m", "boxed_warning": None}]
        md = landscape.safety_alerts_markdown(labels)
        self.assertIn("Suicidality and Antidepressant Drugs", md)
        self.assertNotIn("BOXED WARNING Suicidality", md)
        self.assertIn("[Sertraline](https://dailymed.nlm.nih.gov/s)", md)
        self.assertNotIn("[Melatonin]", md)      # no boxed warning: no row (it is named in the note under the table instead)
        self.assertIn("No boxed warning on the retrieved label of: Melatonin.", md)

    def test_no_boxed_warning_says_so_without_claiming_safety(self):
        md = landscape.safety_alerts_markdown([{"drug": "Melatonin", "url": "https://x.org/m", "boxed_warning": None}])
        self.assertIn("No boxed warning was found", md)
        self.assertIn("not a complete safety review", md)


    def test_no_announcement_is_stated_honestly(self):
        s = make_state()
        s["core_stems"] = ["autis"]
        self.assertIn("This does not mean none exists", landscape.regulatory_updates_markdown(s))

    def test_the_verifier_keeps_label_links_and_does_not_rejudge_code_built_tables(self):
        s = make_state()
        s["drug_labels"] = [{"drug": "sertraline", "url": "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=00179766-980b-44b0-99d3-1fee2bb27e37", "boxed_warning": "Suicidality in children."}]
        text = (BASE + "#### Safety alerts (FDA boxed warnings of the drugs listed above)\n\n| Drug | Boxed warning | Label |\n|---|---|---|\n"
                "| sertraline | \"Suicidality in children.\" | [DailyMed label: sertraline](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=00179766-980b-44b0-99d3-1fee2bb27e37) |\n"
                "\n#### Recently approved or updated (regulator announcements retrieved)\n\n| Date | Type | Announcement | Page says | Link |\n|---|---|---|---|\n"
                "| date not stated | Approval or regulatory action | T | The FDA approved a label update for autism symptoms. | [fda.gov](https://www.fda.gov/x) |\n")
        s["web"] = [{"source": "web_regulatory", "title": "T", "url": "https://www.fda.gov/x", "snippet": "The FDA approved a label update."}]
        issues = [i for i in verification.find_issues(text, s) if i["severity"] in ("high", "medium")]
        self.assertEqual([i["kind"] for i in issues if i["kind"] in ("unsupported_approval_claim", "unretrieved_url")], [])



class ProfessionalPresentation(unittest.TestCase):
    def papers(self):
        st = make_state()
        st["papers"]["111111"].update(authors=["Smith J", "Lee K", "Wu H"], title="Exercise for condition X", journal="Test Journal", date="2026-05-01", doi="10.1000/111111")
        st["papers"]["333333"].update(authors=["Woz J"], title="Nutraceuticals in youth", journal="Mood Journal", date="2026-05-01", doi="10.1000/333333")
        return st

    def test_citations_become_numbers_in_order_of_first_use_and_the_references_list_them(self):
        import presentation
        st = self.papers()
        text = ("Exercise helped (Smith et al., Test Journal, 2026 [PMID 111111]). Nutraceuticals trended (Woz et al., Mood Journal, 2026 [PMID 333333]); "
                "exercise again (Smith et al., Test Journal, 2026 [PMID 111111]).\n\n| Approach | Reference |\n|---|---|\n| Exercise | (PMID 111111) |\n| Source | PMID 333333 (Woz et al., Mood Journal, 2026) |\n"
                "\n### Sources\n1. Smith J et al. Exercise. PMID 111111\n2. Woz J et al. PMID 333333\n3. FDA drug label: Risperidone. https://dailymed.nlm.nih.gov/x\n")
        out, order = presentation.numberize(text, st["papers"])
        self.assertEqual(order, ["111111", "333333"])
        self.assertIn("Exercise helped [1]. Nutraceuticals trended [2]; exercise again [1].", out)
        self.assertIn("| Exercise | [1] |", out)
        self.assertIn("| Source | [2] |", out)
        self.assertNotIn("(Smith et al.", out.split("## Sources")[0])
        refs = out.split("## Sources")[1]
        self.assertIn("1. Smith J, Lee K, Wu H, et al. Exercise for condition X *Test Journal*", refs)
        self.assertIn("[PMID 111111](https://pubmed.ncbi.nlm.nih.gov/111111/)", refs)
        self.assertNotIn("et al..", refs)
        self.assertIn("[FDA drug label: Risperidone](https://dailymed.nlm.nih.gov/x)", out)      # non-paper sources keep their link, unnumbered

    def test_a_retracted_paper_is_never_numbered_as_a_reference(self):
        import presentation
        st = self.papers()
        st["papers"]["111111"]["retracted"] = True
        out, order = presentation.numberize("Excluded retracted work (PMID 111111).", st["papers"])
        self.assertEqual(order, [])
        self.assertIn("PMID 111111", out)

    def test_the_closing_note_is_short_and_hides_the_repair_log(self):
        import presentation
        ok = presentation.closing_note("PASS")
        self.assertIn(presentation.DISCLAIMER, ok)
        self.assertIn("checked against the retrieved sources", ok)
        for internal in ("Repaired", "[remaining]", "Repair passes", "landscape status"):
            self.assertNotIn(internal, ok)
        self.assertIn("could not be fully verified", presentation.closing_note("NEEDS_REPAIR"))

    def test_stray_letter_labels_and_loose_separators_are_removed_from_model_text(self):
        text = "**(A)**\n| Approach | Level |\n|---|---|\n| X | Emerging |\n\n---\n\n**(B)**\n\n### Key studies\n\n1. one\n\n---\n(D)\n### What remains\n- a\n"
        out = qa_stream.tidy_model_sections(text)
        for gone in ("(A)", "(B)", "(D)", "---\n\n"):
            self.assertNotIn(gone, out.replace("|---|---|", ""))
        self.assertIn("### Key studies", out)
        self.assertIn("| X | Emerging |", out)

    def test_established_in_a_table_cell_needs_a_guideline(self):
        s = make_state()
        row = "| Exercise | Established | Improved motor skills (PMID 111111) | (PMID 111111) |"
        text = BASE + "| Approach | Evidence level | Shows | Reference |\n|---|---|---|---|\n" + row + "\n"
        kinds = [i["kind"] for i in verification.find_issues(text, s)]
        self.assertIn("established_without_guideline", kinds)
        fixed = verification.verify_and_repair(text, s)["answer"]
        self.assertNotIn("| Established |", fixed)
        self.assertIn("Evidence-supported but limited", fixed)

    def test_limitations_use_plain_names_not_job_names(self):
        s = make_state()
        s["source_status"] = {"pubmed_drugs": {"retrieval_status": "partial", "evidence_status": "unavailable", "error": "timed out after 25s (partial)"},
                              "guidelines_deep": {"retrieval_status": "failed", "evidence_status": "unavailable", "error": "x"}}
        md = landscape.limitations_markdown(s, [], [])
        self.assertIn("## Evidence limitations", md)
        self.assertIn("The PubMed / Europe PMC search timed out", md)
        self.assertIn("guideline-organisation search", md)
        for internal in ("pubmed_drugs", "guidelines_deep", "retrieval partial", "`"):
            self.assertNotIn(internal, md)

    def test_trial_phase_and_status_are_readable(self):
        self.assertEqual(landscape.nice_phase(["PHASE1", "PHASE2"]), "Phase 1/2")
        self.assertEqual(landscape.nice_phase(["PHASE4"]), "Phase 4")
        self.assertEqual(landscape.nice_phase(["NA"]), "Phase not applicable")
        self.assertEqual(landscape.nice_phase(None), "Phase not applicable")
        self.assertEqual(landscape.nice_status("NOT_YET_RECRUITING"), "Not yet recruiting")

    def test_a_termination_reason_is_cut_at_a_clause_not_with_dots(self):
        t = {"recruitment_status": "TERMINATED", "why_stopped": "During COVID-19, study execution and participant compliance were severely affected by the lockdown and many families withdrew", "results_posted": False}
        reading = landscape.trial_reading(t)
        self.assertNotIn("...", reading)
        self.assertIn("During COVID-19", reading)


class GuidelineSnapshot(unittest.TestCase):
    def state(self):
        s = make_state()
        s["guidelines"] = [
            {"organisation": "NICE (UK)", "title": "Autism spectrum disorder in under 19s", "url": "https://www.nice.org.uk/guidance/cg170", "year": "2021",
             "text": "Skip to content. Autism spectrum disorder in under 19s: support and management. Offer a personalised plan of support to every child and young person with autism."},
            {"organisation": "SIGN", "title": "A report", "url": "https://www.sign.ac.uk/x.pdf", "year": "2015",
             "text": "244 Clinical Practice Guideline: Autism assessment. of the evidence for occupational therapy to assist people."},
            {"organisation": "AAP", "title": "Autism toolkit", "url": "https://www.aap.org/autism", "year": None,
             "text": "Sleep problems are common in children with autism and behavioral strategies are the first approach recommended by pediatricians."}]
        return s

    def test_one_real_recommendation_sentence_per_page_and_titles_and_fragments_are_skipped(self):
        import guidelines
        md = guidelines.guideline_snapshot_markdown(self.state())
        self.assertIn("Offer a personalised plan of support to every child", md)
        self.assertIn("behavioral strategies are the first approach recommended", md)
        self.assertNotIn("Skip to content", md)
        self.assertNotIn("244 Clinical Practice Guideline", md)
        self.assertNotIn("of the evidence for occupational therapy", md)
        self.assertNotIn("sign.ac.uk", md)      # that page had no usable sentence
        self.assertIn("[Autism spectrum disorder in under 19s](https://www.nice.org.uk/guidance/cg170)", md)

    def test_nothing_is_printed_when_no_guideline_was_retrieved(self):
        import guidelines
        self.assertEqual(guidelines.guideline_snapshot_markdown(make_state()), "")


class SafetyGrouping(unittest.TestCase):
    def test_labels_with_the_same_warning_share_one_row_and_each_drug_keeps_its_own_link(self):
        same = "INCREASED MORTALITY IN ELDERLY PATIENTS WITH DEMENTIA-RELATED PSYCHOSIS Elderly patients with dementia-related psychosis treated with antipsychotic drugs are at an increased risk of death. {} is not approved for the treatment of patients with dementia-related psychosis [see Warnings and Precautions (5.1)]."
        labels = [{"drug": "Ziprasidone", "url": "https://dailymed.nlm.nih.gov/z", "boxed_warning": same.format("Ziprasidone")},
                  {"drug": "Paliperidone", "url": "https://dailymed.nlm.nih.gov/p", "boxed_warning": same.format("Paliperidone extended-release tablets")},
                  {"drug": "Sertraline", "url": "https://dailymed.nlm.nih.gov/s", "boxed_warning": "Suicidality and Antidepressant Drugs Antidepressants increased the risk of suicidal thinking in children."}]
        md = landscape.safety_alerts_markdown(labels)
        rows = [l for l in md.splitlines() if l.startswith("| ") and "---" not in l and "Drug(s)" not in l]
        self.assertEqual(len(rows), 2)      # the two antipsychotics together, sertraline alone
        self.assertIn("[Ziprasidone](https://dailymed.nlm.nih.gov/z), [Paliperidone](https://dailymed.nlm.nih.gov/p)", md)
        self.assertEqual(md.count("INCREASED MORTALITY"), 1)



class DoseAndRoute(unittest.TestCase):
    LABEL = ("2 DOSAGE AND ADMINISTRATION Recommended dosage Schizophrenia (2.1) 10 mg/day. Irritability associated with autistic disorder - pediatric patients (2.4) 2 mg/day 5 to 10 mg/day 15 mg/day "
             "2.1 Schizophrenia Adults The recommended starting dose is 10 or 15 mg/day. "
             "2.4 Irritability Associated with Autistic Disorder Pediatric Patients (6 to 17 years) The recommended dosage range for the treatment of pediatric patients with irritability "
             "associated with autistic disorder is 5 to 15 mg/day. Dosing should be initiated at 2 mg/day [see Clinical Studies (14.4)] . Adjust at intervals of no less than one week. "
             "2.5 Tourette's Disorder The recommended starting dose is 2 mg/day.")

    def test_only_the_section_for_this_condition_is_taken_word_for_word(self):
        import clinical_tools
        out = clinical_tools.dosage_statement(self.LABEL, ["autis"])
        self.assertTrue(out.startswith("Pediatric Patients (6 to 17 years) The recommended dosage range"))
        self.assertIn("5 to 15 mg/day", out)
        self.assertIn("initiated at 2 mg/day", out)
        self.assertNotIn("Schizophrenia", out)
        self.assertNotIn("Tourette", out)      # the next section is not included
        self.assertNotIn("[see", out)          # label cross-references are not part of the dosing wording
        self.assertNotIn(" .", out)

    def test_no_dosing_section_for_the_condition_means_no_dose_is_shown(self):
        import clinical_tools
        self.assertIsNone(clinical_tools.dosage_statement("2 DOSAGE AND ADMINISTRATION 2.1 Schizophrenia Adults The recommended starting dose is 10 mg/day.", ["autis"]))

    def test_the_table_has_the_dose_and_route_column_with_label_text_or_an_honest_gap(self):
        s = make_state()
        s["regulatory"][0].update(route="Oral", dose_statement="Dosing should be initiated at 2 mg/day.")
        md = landscape.landscape_markdown(s, [], [])
        self.assertIn("Dose / route (if sourced)", md)
        self.assertIn("Route: Oral. \"Dosing should be initiated at 2 mg/day.\"", md)
        s["regulatory"][0].update(dose_statement=None)
        gap = landscape.landscape_markdown(s, [], [])
        self.assertIn("No dosing section for this condition was found in the retrieved label text", gap)
        self.assertNotIn("Not retrieved", gap)      # no empty placeholder cell

    def test_a_label_dose_passes_the_verifier_but_an_invented_one_does_not(self):
        s = make_state()
        s["regulatory"][0].update(route="Oral", dose_statement="Dosing should be initiated at 2 mg/day.")
        good = verification.find_issues(BASE + landscape.landscape_markdown(s, [], []) + "\n", s)
        self.assertEqual([i for i in good if i["kind"] == "unsupported_number"], [])
        bad = verification.find_issues(BASE + "Risperidone is given at 7.5 mg twice daily.\n", s)
        self.assertTrue(any(i["kind"] == "unsupported_number" for i in bad))



class FollowsTheSpecification(unittest.TestCase):
    def test_the_landscape_is_two_tables_treatment_drugs_and_other_drugs(self):
        st = make_state()
        st["drug_labels"] = []
        used = [{"drug": "Melatonin", "excerpt": "Melatonin is commonly used.", "label": "[R](https://x.org/r)", "purpose": "sleep problems", "population": "children", "category": "Off-label (described as used in practice)"},
                {"drug": "Drugzol", "excerpt": "Drugzol is recommended for irritability.", "label": "[AAP](https://aap.org/x)", "purpose": "irritability", "population": "children", "relevance": "treats_condition", "category": "Guideline-recommended"}]
        trial = [{"drug": "NTI164", "nct": "NCT07", "phase": "Phase 3", "status": "Recruiting", "updated": "2026-09-30", "url": "https://clinicaltrials.gov/study/NCT07", "conditions": "Autism", "primary": "Score", "ages": "6+"}]
        md = landscape.landscape_markdown(st, used, trial)
        header = "| Drug | Indication | Dose / route (if sourced) | Key evidence |"
        self.assertEqual(md.count(header), 2)      # the same four columns in the treatment table and the other-drugs table
        first_heading = "### Drugs used for treatment of this condition"
        second_heading = "### Other drugs: used for symptoms (not established treatments of this condition)"
        self.assertLess(md.index(first_heading), md.index(second_heading))
        treatment, other = md.split(second_heading)[0], md.split(second_heading)[1]
        self.assertIn("FDA-approved for the use shown", treatment)
        self.assertIn("Guideline-recommended", treatment)
        self.assertNotIn("Off-label", treatment.split("*Approved:")[1].split("\n\n", 1)[1])      # no symptom drug in the treatment table
        self.assertNotIn("Emerging: investigational", treatment)
        self.assertIn("Melatonin", other)      # symptom-directed and investigational drugs are in the second table, which has no category column
        self.assertNotIn("NTI164", other)
        self.assertNotIn("FDA-approved for the use shown", other.split("### Recently approved")[0])
        order = [md.index(x) for x in ("FDA-approved for the use shown", "Guideline-recommended; no FDA", "Melatonin")]
        self.assertEqual(order, sorted(order))
        self.assertTrue(md.startswith("## Treatment Drug Landscape"))



    def test_the_fast_answer_adds_the_most_notable_latest_drug_as_investigational(self):
        s = make_state()
        s["core_stems"] = ["condi"]
        s["trials_by_id"] = {
            "NCT3": {"nct_id": "NCT3", "phase": ["PHASE2"], "recruitment_status": "RECRUITING", "last_update": "2026-08-01", "url": "https://clinicaltrials.gov/study/NCT3", "conditions": ["Condition X"], "title": "t", "drug_interventions": ["Seconddrug"]},
            "NCT4": {"nct_id": "NCT4", "phase": ["PHASE3"], "recruitment_status": "RECRUITING", "last_update": "2026-09-01", "url": "https://clinicaltrials.gov/study/NCT4", "conditions": ["Condition X"], "title": "t", "drug_interventions": ["Latestdrug"]}}
        for t in s["trials_by_id"].values():
            t["triage"] = {"relevance": "direct", "kind": "treatment_trial", "evidence": "t"}      # the model's verified decision on the registry record
        md = landscape.top_drugs_markdown(s)
        self.assertIn("Top treatment drugs (established first, then the most notable latest drug)", md)
        self.assertIn("Latestdrug (latest, investigational)", md)
        self.assertNotIn("Seconddrug", md)      # the more recently updated trial wins
        self.assertIn("not an approved treatment", md)
        self.assertLess(md.index("Risperidone"), md.index("Latestdrug"))      # established first

    def test_the_prompt_asks_for_the_documents_order_and_names(self):
        task = qa_stream.EVIDENCE_TASK
        positions = [task.index(h) for h in ("## Latest findings", "## Current evidence", "## Key studies", "## Conflicting evidence", "## What remains under investigation")]
        self.assertEqual(positions, sorted(positions))
        held = qa_stream.HOLD_FROM
        self.assertTrue(held.search("## Key studies\n1. x"))      # Key studies, Conflicting evidence and What remains are shown AFTER the drugs and trials
        self.assertTrue(held.search("## Conflicting evidence\nNone"))
        self.assertFalse(held.search("## Current evidence\n| a | b |"))

    def test_section_names_follow_the_document(self):
        self.assertTrue(landscape.trials_markdown(make_state()).startswith("## Clinical trials"))
        self.assertTrue(landscape.limitations_markdown(make_state(), [], []).startswith("## Evidence limitations"))


class LatePhaseRule(unittest.TestCase):
    def test_phase_one_two_is_not_late_phase(self):
        self.assertFalse(landscape.is_late_phase({"phase": ["PHASE1", "PHASE2"]}))
        self.assertFalse(landscape.is_late_phase({"phase": ["PHASE1"]}))
        self.assertFalse(landscape.is_late_phase({"phase": ["NA"]}))
        self.assertFalse(landscape.is_late_phase({"phase": []}))
        self.assertTrue(landscape.is_late_phase({"phase": ["PHASE2", "PHASE3"]}))
        self.assertTrue(landscape.is_late_phase({"phase": ["PHASE4"]}))


class CitationShapes(unittest.TestCase):
    def test_a_pmid_written_as_a_link_is_numbered_like_any_other_citation(self):
        import presentation
        st = make_state()
        st["papers"]["111111"].update(authors=["Smith J"], title="T", journal="J", date="2026-05-01")
        text = "| Approach | Reference |\n|---|---|\n| Exercise | (PMID [111111](https://pubmed.ncbi.nlm.nih.gov/111111/)) |\n\nAlso [PMID 111111](https://pubmed.ncbi.nlm.nih.gov/111111/).\n"
        out, order = presentation.numberize(text, st["papers"])
        self.assertEqual(order, ["111111"])
        self.assertIn("| Exercise | [1] |", out)
        self.assertIn("Also [1].", out)



class DoseSummary(unittest.TestCase):
    SOURCE = ("Pediatric Patients (6 to 17 years) The recommended dosage range for the treatment of pediatric patients with irritability associated with autistic disorder is 5 to 15 mg/day. "
              "Dosing should be initiated at 2 mg/day. The dose should be increased to 5 mg/day, with subsequent increases to 10 or 15 mg/day if needed. "
              "Dose adjustments of up to 5 mg/day should occur gradually, at intervals of no less than one week.")
    GOOD = ("For children and adolescents aged 6 to 17 years the recommended range is 5 to 15 mg/day. Treatment starts at 2 mg/day and is raised to 5 mg/day, then to 10 or 15 mg/day if needed. "
            "Adjustments of up to 5 mg/day are made gradually, at least one week apart.")

    def test_a_faithful_short_summary_is_accepted(self):
        self.assertTrue(landscape.dose_summary_ok(self.GOOD, self.SOURCE))

    def test_a_summary_with_a_number_that_is_not_in_the_label_is_rejected(self):
        self.assertFalse(landscape.dose_summary_ok(self.GOOD.replace("5 to 15 mg/day", "5 to 20 mg/day"), self.SOURCE))
        self.assertFalse(landscape.dose_summary_ok(self.GOOD + " Doses above 30 mg/day are not advised.", self.SOURCE))

    def test_too_long_or_formatted_summaries_are_rejected(self):
        five = " ".join(f"Sentence number {i} says dose 2 mg/day." for i in range(5))
        self.assertFalse(landscape.dose_summary_ok(five, self.SOURCE))
        self.assertFalse(landscape.dose_summary_ok("**" + self.GOOD + "**", self.SOURCE))
        self.assertFalse(landscape.dose_summary_ok("", self.SOURCE))

    def test_one_point_zero_and_one_are_the_same_number(self):
        self.assertTrue(landscape.dose_summary_ok("The dose may be raised to 1 mg per day for patients over 20 kg, then reviewed after four days of treatment.",
                                                  "increase to 1.0 mg per day for patients greater than 20 kg. After a minimum of four days the dose may be reviewed."))

    def test_the_fallback_is_the_labels_own_first_sentences(self):
        out = landscape.dose_fallback(self.SOURCE, limit=300)
        self.assertTrue(self.SOURCE.startswith(out))
        self.assertLessEqual(len(out), 300)
        self.assertTrue(out.endswith("."))

    def test_the_table_shows_the_summary_not_the_whole_dosing_section(self):
        s = make_state()
        s["regulatory"][0].update(route="Oral", dose_statement=self.SOURCE, dose_summary=self.GOOD)
        md = landscape.landscape_markdown(s, [], [])
        self.assertIn("Route: Oral. " + self.GOOD, md)
        self.assertNotIn("Pediatric Patients (6 to 17 years) The recommended dosage range", md)
        self.assertIn("short summary of the label's dosing section", md)      # the note says it is a summary, checked against the label
        self.assertEqual([i for i in verification.find_issues(BASE + md + "\n", s) if i["kind"] == "unsupported_number"], [])


class DoseSummaryRules(unittest.TestCase):
    def test_number_words_in_the_label_count_as_numbers(self):
        source = "After a minimum of four days, the dose may be increased to 0.5 mg per day. Maintain this dose for a minimum of 14 days."
        self.assertTrue(landscape.dose_summary_ok("After at least 4 days the dose may be raised to 0.5 mg per day, and kept for 14 days before any further change.", source))
        self.assertFalse(landscape.dose_summary_ok("After at least 5 days the dose may be raised to 0.5 mg per day, and kept for 14 days before any further change.", source))

    def test_a_claim_about_what_the_label_does_not_say_is_rejected(self):
        source = "Dosing should be initiated at 2 mg/day. The dose should be increased to 5 mg/day, with subsequent increases to 10 or 15 mg/day if needed."
        ok = "Dosing starts at 2 mg/day and is increased to 5 mg/day, with later steps to 10 or 15 mg/day if needed."
        self.assertTrue(landscape.dose_summary_ok(ok, source))
        self.assertFalse(landscape.dose_summary_ok(ok + " The label does not specify a maximum dose.", source))
        self.assertFalse(landscape.dose_summary_ok(ok + " There is no maximum dose.", source))


class DoseSummaryWordingSlips(unittest.TestCase):
    SOURCE = "Dosing should be initiated at 2 mg/day. The dose should be increased to 5 mg/day, with subsequent increases to 10 or 15 mg/day if needed, at intervals of no less than one week."

    def test_comments_about_the_text_are_rejected(self):
        ok = "Dosing starts at 2 mg/day, goes to 5 mg/day and then to 10 or 15 mg/day if needed, at least one week apart."
        self.assertTrue(landscape.dose_summary_ok(ok, self.SOURCE))
        for slip in (" No weight-specific adjustments are mentioned in the text.", " According to the label this is usual.", " Nothing more is mentioned."):
            self.assertFalse(landscape.dose_summary_ok(ok + slip, self.SOURCE), slip)


class TableRowsAreCheckedToo(unittest.TestCase):
    def test_overclaim_wording_inside_a_table_row_is_softened(self):
        s = make_state()
        text = (BASE + "| Approach | Evidence level | Shows | Reference |\n|---|---|---|---|\n"
                "| Exercise | Evidence-supported but limited | Exercise improved motor skills (dose-response confirmed). | (PMID 111111) |\n")
        r = verification.verify_and_repair(text, s)
        self.assertNotIn("confirmed", r["answer"])
        self.assertEqual(r["status"], "PASS")

    def test_a_citation_written_as_a_link_is_still_checked(self):
        s = make_state()
        row = "| Exercise | Established | Improved motor skills. | (PMID [111111](https://pubmed.ncbi.nlm.nih.gov/111111/)) |"
        text = BASE + "| Approach | Evidence level | Shows | Reference |\n|---|---|---|---|\n" + row + "\n"
        fixed = verification.verify_and_repair(text, s)["answer"]
        self.assertNotIn("| Established |", fixed)      # no guideline among the cited sources: not 'Established'
        invented = BASE + "| Approach | Evidence level | Shows | Reference |\n|---|---|---|---|\n| X | Emerging | Improved. | (PMID [999999](https://pubmed.ncbi.nlm.nih.gov/999999/)) |\n"
        self.assertNotIn("999999", verification.verify_and_repair(invented, make_state())["answer"])      # an invented PMID written as a link is removed too (a fresh run: one repair pass each)




class OnlyDrugTopicsAreSearched(unittest.TestCase):
    def test_only_drug_related_guideline_topics_are_searched(self):
        import guidelines
        asked = []

        def search(query):
            asked.append(query)
            return json.dumps([])
        res = guidelines.find_guidelines("How to treat autism in children?", ["autism"], search)
        topics = {r["topic"] for r in res["results"]}
        self.assertEqual(topics, {"medications", "first_line"})
        self.assertLessEqual(len(asked), 8)      # four topics, each at most one retry: not the eleven searches that took 14 s


class ReferenceColumnRepair(unittest.TestCase):
    def test_a_citation_left_in_the_last_cell_moves_into_the_reference_column(self):
        text = ("## Current evidence\n| Approach | Evidence level | What the evidence shows | Reference |\n|---|---|---|---|\n"
                "| Exercise | Emerging | Improved motor skills [4]. |\n| Diet | Limited | Fewer symptoms [PMID 123, 456]. |\n| Full | Emerging | Ok | [7] |\n")
        fixed = qa_stream.tidy_model_sections(text)
        self.assertIn("| Exercise | Emerging | Improved motor skills. | [4] |", fixed)
        self.assertIn("| Diet | Limited | Fewer symptoms. | [PMID 123, 456] |", fixed)
        self.assertIn("| Full | Emerging | Ok | [7] |", fixed)


# ---------------------------------------------------------------------------------------------------- the model interprets, code verifies
OFFERED = {
    "111111": {"title": "Pharmacotherapy review", "label": "PMID 111111 (Smith et al.)", "url": "", "kind": "paper",
               "text": "Pharmacotherapy of irritability: risperidone is widely used, and melatonin is commonly prescribed for sleep problems in children with condition X. "
                       "Dexmedetomidine is being tested in a phase 2 trial."},
    "https://www.aap.org/x": {"title": "AAP guidance", "label": "[AAP: guidance](https://www.aap.org/x)", "url": "https://www.aap.org/x", "kind": "guideline",
                              "text": "The AAP recommends melatonin for sleep problems in children with condition X."},
    "https://blog.example.com/y": {"title": "A blog", "label": "[A blog](https://blog.example.com/y)", "url": "https://blog.example.com/y", "kind": "web",
                                   "text": "Our experts say melatonin is recommended for sleep problems in children with condition X."},
}


def drug_item(**kw):
    item = {"drug": "Melatonin", "source_id": "111111", "relevance": "treats_associated_symptom_or_comorbidity", "statement_type": "used_in_practice", "drug_kind": "generic", "category": "Off-label", "indication": "sleep problems",
            "population": "children with condition X", "quote": "melatonin is commonly prescribed for sleep problems in children with condition X"}
    item.update(kw)
    return item


class ModelInterpretationIsVerifiedAgainstTheSource(unittest.TestCase):
    def test_quote_matching_ignores_case_spacing_and_quote_marks_but_not_content(self):
        text = "The “FDA”   approved   Drugzol for pediatric patients."
        self.assertIsNotNone(landscape.locate(text, 'the "fda" approved drugzol for pediatric patients'))
        self.assertIsNotNone(landscape.locate(text, "... approved Drugzol for pediatric patients."))
        self.assertIsNone(landscape.locate(text, "The FDA approved Drugzol for adults and children"))
        self.assertIsNone(landscape.locate(text, "too short"))

    def test_a_supported_drug_becomes_a_row_with_the_sources_own_wording(self):
        rows, rejected = landscape.verify_drug_decisions([drug_item()], OFFERED, make_state())
        self.assertEqual(rejected, [])
        self.assertEqual(rows[0]["drug"], "Melatonin")
        self.assertEqual(rows[0]["purpose"], "sleep problems")
        self.assertEqual(rows[0]["excerpt"], "melatonin is commonly prescribed for sleep problems in children with condition X")
        self.assertEqual(rows[0]["category"], "Off-label (described as used in practice)")
        self.assertEqual(rows[0]["population"], "children with condition X")

    def test_every_check_rejects_what_the_source_does_not_support(self):
        s = make_state()
        cases = {
            "unknown source id": drug_item(source_id="999999"),
            "quote not in the source": drug_item(quote="melatonin cures sleep problems in all children with condition X"),
            "drug not in the quote": drug_item(drug="Quetiapine"),
            "model says not relevant": drug_item(relevance="not_relevant"),
            "model cannot tell": drug_item(relevance="undetermined"),
            "category not allowed": drug_item(category="Approved"),
            "not a drug name": drug_item(drug="Exercise programme 12"),
        }
        for name, item in cases.items():
            rows, rejected = landscape.verify_drug_decisions([item], OFFERED, s)
            self.assertEqual(rows, [], name)
            self.assertEqual(len(rejected), 1, name)

    def test_a_trial_only_drug_the_model_marks_irrelevant_is_not_listed(self):
        item = drug_item(drug="Dexmedetomidine", relevance="not_relevant", quote="Dexmedetomidine is being tested in a phase 2 trial", indication="a phase 2 trial")
        self.assertEqual(landscape.verify_drug_decisions([item], OFFERED, make_state())[0], [])

    def test_guideline_recommended_needs_a_guideline_organisation_page(self):
        s = make_state()
        guide = drug_item(source_id="https://www.aap.org/x", category="Guideline-recommended", quote="The AAP recommends melatonin for sleep problems in children with condition X")
        blog = drug_item(source_id="https://blog.example.com/y", category="Guideline-recommended", quote="melatonin is recommended for sleep problems in children with condition X")
        self.assertEqual(landscape.verify_drug_decisions([guide], OFFERED, s)[0][0]["category"], "Guideline-recommended")
        self.assertEqual(landscape.verify_drug_decisions([blog], OFFERED, s)[0][0]["category"], "Off-label (described as used in practice)")      # downgraded, not trusted

    def test_a_drug_with_its_own_fda_row_is_not_repeated(self):
        item = drug_item(drug="Risperidone", quote="risperidone is widely used", indication="irritability")
        self.assertEqual(landscape.verify_drug_decisions([item], OFFERED, make_state())[0], [])

    def test_a_relevant_label_decision_needs_a_sentence_of_the_label_and_an_irrelevant_one_needs_none(self):
        rec = {"generic": "Drugzol", "matches_condition": True}
        offered = {"fda:Drugzol": {"indication_text": "Drugzol is indicated for the treatment of irritability associated with condition X.", "record": rec}}
        out = landscape.verify_fda_decisions([{"id": "fda:Drugzol", "relevant": True, "sentence": 7}], offered)
        self.assertEqual(out["unreviewed"], ["Drugzol"])
        self.assertFalse(landscape.fda_relevant(rec))      # no verified decision: the label is not listed, whatever the retrieval flag says
        out = landscape.verify_fda_decisions([{"id": "fda:Drugzol", "relevant": True, "relation": "treats_associated_symptom_or_comorbidity", "sentence": 1}], offered)
        self.assertEqual(out["confirmed"], ["Drugzol"])
        self.assertTrue(landscape.fda_relevant(rec))
        self.assertEqual(rec["model_evidence"], "Drugzol is indicated for the treatment of irritability associated with condition X.")
        rec2 = {"generic": "Otherol", "matches_condition": True}
        out = landscape.verify_fda_decisions([{"id": "fda:Otherol", "relevant": False}], {"fda:Otherol": {"indication_text": "Otherol is indicated for hypertension in adults.", "record": rec2}})
        self.assertEqual(out["excluded"], ["Otherol"])
        self.assertFalse(landscape.fda_relevant(rec2))

    def test_an_fda_label_the_model_excluded_is_not_in_the_treatment_table(self):
        s = make_state()
        s["regulatory"][0]["model_relevant"] = False
        treatment = landscape.landscape_markdown(s, [], []).split("### Other drugs")[0].split("### Drugs used for treatment")[1]
        self.assertNotIn("Risperidone", treatment)


REG_OFFERED = {
    "reg:1": {"title": "FDA Takes Action on Drugzol for Condition X", "url": "https://www.fda.gov/news-events/press-announcements/a",
              "text": "Skip to main content. The FDA today approved Drugzol tablets for patients with symptom Y, a feature that occurs in condition X. Posted September 22, 2025."},
    "reg:2": {"title": "FDA Moves to Add Warning to Painex", "url": "https://www.fda.gov/news-events/press-announcements/b",
              "text": "The FDA initiated a label change for Painex to reflect evidence that use in pregnancy may be associated with an increased risk of condition X. Release: September 22, 2025."},
    "reg:3": {"title": "Advisory committee calendar", "url": "https://www.fda.gov/advisory-committees/c", "text": "Meeting calendar and registration for committee members of the agency."},
    "reg:4": {"title": "FDA Approves Zetamab for Condition Z", "url": "https://www.fda.gov/news-events/press-announcements/d",
              "text": "The FDA approved Zetamab for adults with condition Z. Condition Z is sometimes mentioned together with condition X in the literature."},
}
REG_ITEMS = [
    {"id": "reg:1", "classification": "approval", "relation": "treats_associated_symptom_or_comorbidity", "concerns": "Drugzol tablets for symptom Y", "drug": "Drugzol",
     "sentences": [2], "date": "September 22, 2025"},
    {"id": "reg:2", "classification": "safety_communication", "relation": "risk_or_safety_related", "concerns": "label change for Painex", "drug": "Painex", "sentences": [1], "date": "September 22, 2025"},
    {"id": "reg:3", "classification": "not_relevant", "relation": "unclear", "sentences": [1]},
    {"id": "reg:4", "classification": "approval", "relation": "other_indication", "drug": "Zetamab", "sentences": [1], "date": ""},
]


class RegulatorPagesAreReadByTheModelAndVerified(unittest.TestCase):
    def test_actions_that_bear_on_the_question_are_kept_with_their_type_and_relation(self):
        kept, rejected = landscape.verify_regulator_decisions(REG_ITEMS, REG_OFFERED)
        self.assertEqual([(x["type"], x["relation"]) for x in kept], [("Approval", "treats_associated_symptom_or_comorbidity"), ("Safety communication", "risk_or_safety_related")])
        self.assertEqual(kept[0]["quote"], "The FDA today approved Drugzol tablets for patients with symptom Y, a feature that occurs in condition X.")      # the page's own sentence
        self.assertEqual(kept[0]["date"], "September 22, 2025")
        self.assertEqual(kept[1]["date"], "September 22, 2025")
        self.assertEqual(len(rejected), 2)      # the meeting page and the approval for another indication

    def test_an_action_about_another_indication_is_not_a_treatment_approval(self):
        kept, rejected = landscape.verify_regulator_decisions([REG_ITEMS[3]], REG_OFFERED)
        self.assertEqual(kept, [])
        self.assertIn("other_indication", rejected[0][1])

    def test_an_invented_date_drug_or_sentence_number_is_not_accepted(self):
        bad = [dict(REG_ITEMS[0], date="March 3, 2026", drug="Ghostol"), dict(REG_ITEMS[1], sentences=[9])]
        kept, rejected = landscape.verify_regulator_decisions(bad, REG_OFFERED)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["date"], "date not stated")
        self.assertEqual(kept[0]["drug"], "")
        self.assertEqual(len(rejected), 1)

    def test_a_safety_communication_is_listed_as_an_announcement_and_never_as_a_boxed_warning(self):
        s = make_state()
        s["regulator_items"] = landscape.verify_regulator_decisions(REG_ITEMS, REG_OFFERED)[0]
        md = landscape.landscape_markdown(s, [], [])
        announcements = md.split("### Recently approved or updated")[1].split("### Safety warnings")[0]
        boxed = md.split("### Safety warnings")[1]
        self.assertIn("Safety communication", announcements)
        self.assertIn("Painex", announcements)
        self.assertIn("Approval (associated symptom or comorbidity)", announcements)
        self.assertNotIn("Painex", boxed)      # the boxed-warning section holds boxed warnings only


class TheReportHierarchyIsUnchanged(unittest.TestCase):
    def test_landscape_headings_and_columns_are_the_same_with_model_interpreted_rows(self):
        s = make_state()
        s["regulator_items"] = landscape.verify_regulator_decisions(REG_ITEMS, REG_OFFERED)[0]
        rows, _ = landscape.verify_drug_decisions([drug_item()], OFFERED, s)
        md = landscape.landscape_markdown(s, rows, [])
        headings = [line for line in md.splitlines() if line.startswith("#")]
        self.assertEqual(headings, ["## Treatment Drug Landscape", "### Drugs used for treatment of this condition",
                                    "### Other drugs: used for symptoms (not established treatments of this condition)",
                                    "### Recently approved or updated (FDA, EMA and CDSCO announcements)", "### Safety warnings (FDA boxed warnings)"])
        self.assertIn("| Drug | Indication | Dose / route (if sourced) | Key evidence |", md)
        self.assertIn("| Date | Type | Announcement | What the regulator's page says |", md)
        self.assertIn("melatonin is commonly prescribed for sleep problems", md.split("### Other drugs")[1])

    def test_a_drug_the_model_read_as_treating_an_associated_symptom_says_so_in_the_existing_indication_cell(self):
        s = make_state()
        rows, _ = landscape.verify_drug_decisions([drug_item()], OFFERED, s)
        self.assertIn("[associated symptom or comorbidity, not the condition itself]", landscape.landscape_markdown(s, rows, []).split("### Other drugs")[1])
        s["regulatory"][0]["model_relation"] = "treats_associated_symptom_or_comorbidity"
        self.assertIn("associated symptom or comorbidity", landscape.landscape_markdown(s, [], []).split("### Other drugs")[0])

    def test_ask_json_reads_the_object_even_when_the_model_wraps_it(self):
        original = qa_stream.stream_llm
        qa_stream.stream_llm = lambda *a, **k: iter(["Here you go:\n```json\n", '{"items": [{"id": "reg:1"}]}', "\n```"])
        try:
            self.assertEqual(qa_stream.ask_json("q", "task", {"a": 1}, 100)["items"], [{"id": "reg:1"}])
        finally:
            qa_stream.stream_llm = original


PAPERS_OFFERED = {"1": {"sentences": ["Melatonin for sleep in children with condition X", "Children with condition X slept longer on melatonin.", "Sleep onset improved."]},
                  "2": {"sentences": ["Hearing loss in adults", "Adults with hearing loss were studied."]}}


class EvidenceIsReadByTheModelNotFilteredByKeywords(unittest.TestCase):
    def test_paper_decisions_need_an_existing_sentence_and_population_words_from_the_paper(self):
        items = [{"id": "1", "relevance": "direct", "role": "emerging", "population": "children with condition X", "sentence": 2},
                 {"id": "2", "relevance": "direct", "role": "established", "population": "astronauts", "sentence": 9},
                 {"id": "3", "relevance": "direct", "sentence": 1}]
        out = landscape.verify_paper_triage(items, PAPERS_OFFERED)
        self.assertEqual(out["1"]["relevance"], "direct")
        self.assertEqual(out["1"]["population"], "children with condition X")
        self.assertEqual(out["1"]["evidence"], "Children with condition X slept longer on melatonin.")
        self.assertEqual(out["2"]["relevance"], "uncertain")      # a decision that points at no sentence of the paper is not accepted
        self.assertEqual(out["2"]["population"], "")
        self.assertNotIn("3", out)      # an id that was not offered

    def test_a_paper_the_model_judged_irrelevant_is_not_shown_to_the_writer_whatever_words_it_contains(self):
        a = study("700001", year="2026", types=("Randomized Controlled Trial",))
        a["triage"] = {"relevance": "not_relevant", "role": None, "population": "", "evidence": ""}
        b = study("700002", year="2020", types=("Case Reports",))
        b["triage"] = {"relevance": "direct", "role": "emerging", "population": "", "evidence": "x"}
        self.assertEqual([p["pmid"] for p in qa_stream.ordered_papers({"700001": a, "700002": b}, set())], ["700002"])
        self.assertFalse(qa_stream.usable(a, set()))

    def test_order_is_relevance_then_evidence_strength_within_relevance_tier(self):
        old_review = study("700011", year="2015", types=("Systematic Review",))
        old_review["triage"] = {"relevance": "indirect", "role": "established", "population": "", "evidence": "x"}
        new_case = study("700012", year="2026", types=("Case Reports",))
        new_case["triage"] = {"relevance": "direct", "role": "emerging", "population": "", "evidence": "x"}
        mid = study("700013", year="2022", types=("Randomized Controlled Trial",))
        mid["triage"] = {"relevance": "direct", "role": "experimental", "population": "", "evidence": "x"}
        order = [p["pmid"] for p in qa_stream.ordered_papers({p["pmid"]: p for p in (old_review, new_case, mid)}, set())]
        self.assertEqual(order, ["700013", "700012", "700011"])

    def test_without_a_triage_decision_no_keyword_rule_removes_a_paper(self):
        p = study("700021", year="2025")
        p["title"] = "A study of something that shares no word with the question"
        self.assertFalse(qa_stream.usable(p, {"zzzzz"}))
        self.assertFalse(verification.is_indirect(p, {"zzzzz"}))

    def test_indirect_means_the_triage_said_indirect(self):
        p = study("700022")
        p["triage"] = {"relevance": "indirect", "role": None, "population": "", "evidence": "x"}
        self.assertTrue(verification.is_indirect(p, set()))

    def test_trial_decisions_decide_which_records_are_treatment_trials(self):
        offered = {"NCT1": {"sentences": ["Drugzol for condition X", "Conditions: condition X", "Interventions: Drugzol", "Primary outcome: score"]}}
        out = landscape.verify_trial_triage([{"id": "NCT1", "relevance": "direct", "kind": "treatment_trial", "sentence": 3}, {"id": "NCT1x", "relevance": "direct", "kind": "other", "sentence": 1}], offered)
        self.assertEqual(list(out), ["NCT1"])
        self.assertTrue(landscape.is_treatment_trial({"triage": out["NCT1"]}))
        self.assertFalse(landscape.is_treatment_trial({"triage": {"relevance": "direct", "kind": "diagnostic_or_assessment", "evidence": "x"}, "design": "INTERVENTIONAL", "interventions": ["x"]}))

    def test_triage_runs_the_evidence_synthesis_skill_over_all_retrieved_records_and_stores_verified_decisions(self):
        s = make_state()
        s["papers"]["111111"]["triage"] = None
        loaded = []
        original_ask = qa_stream.ask_json

        def fake(question, task, payload, tokens, skill="drug-intelligence"):
            loaded.append(skill)
            if "papers" in payload:
                return {"papers": [{"id": p["id"], "relevance": "direct", "role": "emerging", "population": "", "sentence": 2} for p in payload["papers"]]}
            return {"trials": [{"id": t["id"], "relevance": "direct", "kind": "treatment_trial", "sentence": 1} for t in payload["trials"]]}
        qa_stream.ask_json = fake
        try:
            skipped = []
            qa_stream.triage_evidence(s, "q", skipped)
        finally:
            qa_stream.ask_json = original_ask
        self.assertEqual(set(loaded), {"evidence-synthesis"})
        self.assertEqual(s["papers"]["111111"]["triage"]["relevance"], "direct")
        self.assertEqual(s["trials_by_id"]["NCT01234567"]["triage"]["kind"], "treatment_trial")
        self.assertEqual(skipped, [])

    def test_a_failed_triage_call_leaves_the_records_unfiltered_and_says_so(self):
        s = make_state()
        s["papers"]["111111"].pop("triage", None)
        original_ask = qa_stream.ask_json

        def broken(*a, **k):
            raise RuntimeError("model down")
        qa_stream.ask_json = broken
        try:
            skipped = []
            qa_stream.triage_evidence(s, "q", skipped)
        finally:
            qa_stream.ask_json = original_ask
        self.assertNotIn("triage", s["papers"]["111111"])
        self.assertFalse(qa_stream.usable(s["papers"]["111111"], set()))
        self.assertTrue(skipped)


class DatesAreConsistentWithTheSearchDate(unittest.TestCase):
    def test_a_finding_dated_after_the_search_date_is_shown_as_an_advance_publication(self):
        text = "## Latest findings\n- **2026 Dec**: A study found a result [PMID 1].\n- **2026 Sep**: Another study found a result [PMID 2].\n- **2027**: A third [PMID 3].\n"
        fixed, actions = verification.fix_future_dates(text, {"searched_on": "2026-10-08"})
        self.assertIn("**Advance publication (issue dated 2026 Dec)**", fixed)
        self.assertIn("**2026 Sep**", fixed)
        self.assertIn("**Advance publication (issue dated 2027)**", fixed)
        self.assertEqual(len(actions), 2)

    def test_the_check_is_general_and_runs_inside_the_verification_loop(self):
        s = make_state()
        s["searched_on"] = "2026-10-08"
        s["papers"]["111111"]["date"] = "2026 Dec"
        out = verification.verify_and_repair(BASE + "## Latest findings\n- **2026 Dec**: A study of exercise reported motor skills outcomes (PMID 111111).\n", s)
        self.assertIn("Advance publication (issue dated 2026 Dec)", out["answer"])
        self.assertTrue(any("advance-publication" in r for r in out["repairs"]))

    def test_the_writer_is_told_when_a_papers_date_is_later_than_today(self):
        p = study("700031")
        p["date"] = "2099 Dec"
        self.assertTrue(qa_stream.paper_view(p, 300)["after_search_date"])
        p["date"] = "2020 Jan"
        self.assertNotIn("after_search_date", qa_stream.paper_view(p, 300))


class QuoteAndReviewerDetails(unittest.TestCase):
    def test_a_quote_may_skip_text_with_an_ellipsis_but_every_piece_must_be_in_the_source_in_order(self):
        text = "Aripiprazole is indicated for the treatment of: \u2022 Schizophrenia \u2022 Irritability Associated with Autistic Disorder \u2022 Tourette's Disorder"
        self.assertIsNotNone(landscape.locate(text, "Aripiprazole is indicated for the treatment of: ... Irritability Associated with Autistic Disorder"))
        self.assertIsNone(landscape.locate(text, "Irritability Associated with Autistic Disorder ... Aripiprazole is indicated for the treatment of"))
        self.assertIsNone(landscape.locate(text, "Aripiprazole is indicated for the treatment of: ... Hypertension in older adults"))

    def test_a_brand_name_in_brackets_is_dropped_from_the_drug_name(self):
        offered = {"u": {"title": "t", "label": "l", "url": "", "kind": "web", "text": "Stimulant medications such as methylphenidate are commonly prescribed for hyperactivity in children with condition X."}}
        item = drug_item(drug="Methylphenidate (Ritalin)", source_id="u", quote="methylphenidate are commonly prescribed for hyperactivity in children with condition X",
                         indication="hyperactivity", population="children with condition X")
        rows, rejected = landscape.verify_drug_decisions([item], offered, make_state())
        self.assertEqual([r["drug"] for r in rows], ["Methylphenidate"], rejected)



class TheModelPointsAtSentencesAndCodeTakesTheText(unittest.TestCase):
    SRC = {"u": {"title": "t", "label": "l", "url": "", "kind": "web",
                 "text": "Medication for condition X is an individual decision. Melatonin is commonly used to address sleep problems in children. Exercise helps motor skills."}}

    def test_the_quote_is_the_sources_own_sentence_chosen_by_number(self):
        item = {"drug": "Melatonin", "source_id": "u", "sentence": 2, "relevance": "treats_associated_symptom_or_comorbidity", "statement_type": "used_in_practice", "drug_kind": "generic", "category": "Off-label", "indication": "addressing sleep problems"}
        rows, rejected = landscape.verify_drug_decisions([item], self.SRC, make_state())
        self.assertEqual(rejected, [])
        self.assertEqual(rows[0]["excerpt"], "Melatonin is commonly used to address sleep problems in children.")      # inflection in the model's phrase is fine: its words are in the sentence

    def test_a_wrong_or_missing_number_or_a_sentence_without_the_drug_is_rejected(self):
        base = {"drug": "Melatonin", "source_id": "u", "relevance": "treats_condition", "statement_type": "used_in_practice", "drug_kind": "generic", "category": "Off-label", "indication": "sleep problems"}
        for sentence in (9, 0, "x", None, 1, 3):
            rows, _ = landscape.verify_drug_decisions([dict(base, sentence=sentence)], self.SRC, make_state())
            self.assertEqual(rows, [], sentence)

    def test_an_indication_with_words_the_sentence_does_not_contain_is_never_shown(self):
        item = {"drug": "Melatonin", "source_id": "u", "sentence": 2, "relevance": "treats_condition", "statement_type": "used_in_practice", "drug_kind": "generic", "category": "Off-label", "indication": "aggression and self-injury"}
        rows = landscape.verify_drug_decisions([item], self.SRC, make_state())[0]
        self.assertEqual(rows[0]["purpose"], "as stated in the quoted source sentence")      # the row stays, the model's unsupported wording does not
        self.assertNotIn("aggression", landscape.landscape_markdown(make_state(), rows, []))



class NavigationTextIsNotPartOfASentence(unittest.TestCase):
    def test_a_page_navigation_fragment_is_split_off(self):
        sentences = landscape.numbered_sentences("More Press Announcements . The FDA today initiated a label change for Drugzol in children with condition X.")
        self.assertEqual(sentences[-1], "The FDA today initiated a label change for Drugzol in children with condition X.")
        self.assertNotIn("Press Announcements", sentences[-1])


class FdaLabelsAreInterpretedByTheModel(unittest.TestCase):
    def test_labels_are_decided_by_the_model_asked_again_once_and_never_by_a_keyword(self):
        records = [{"generic": "Aaa", "indication_excerpt": "Aaa is indicated for sleep problems in children with condition X.", "matches_condition": True},
                   {"generic": "Bbb", "indication_excerpt": "Bbb is indicated for hypertension in adults.", "matches_condition": False},
                   {"generic": "Ccc", "indication_excerpt": "Ccc is indicated for condition X in children and adults.", "matches_condition": False}]
        calls = []
        original = qa_stream.ask_json

        def fake(question, task, payload, tokens, skill="drug-intelligence"):
            names = [x["id"].split(":")[1] for x in payload["fda_labels"]]
            calls.append(names)
            answers = {"Aaa": {"relevant": True, "relation": "treats_associated_symptom_or_comorbidity", "sentence": 1}, "Bbb": {"relevant": False},
                       "Ccc": {"relevant": True, "relation": "treats_condition", "sentence": 1}}
            return {"fda_labels": [dict(answers[n], id=x["id"]) for n, x in zip(names, payload["fda_labels"]) if len(calls) > 1 or n != "Ccc"]}      # the first answer omits Ccc
        qa_stream.ask_json = fake
        try:
            out = qa_stream.interpret_fda_labels("How to treat condition X?", records)
        finally:
            qa_stream.ask_json = original
        self.assertEqual(calls, [["Aaa", "Bbb", "Ccc"], ["Ccc"]])      # the second call asks only for the label that got no decision
        self.assertEqual([landscape.fda_relevant(r) for r in records], [True, False, True])      # Bbb matched no condition word and is out; Ccc matched none but the model judged it relevant
        self.assertEqual(records[0]["model_relation"], "treats_associated_symptom_or_comorbidity")
        self.assertEqual(out["unreviewed"], [])

    def test_a_label_with_no_decision_is_not_listed_in_the_top_drugs_table(self):
        s = make_state()
        s["regulatory"] = [{"generic": "Zzz", "matches_condition": True, "indication_excerpt": "Zzz is indicated for condition X.", "url": "https://dailymed.nlm.nih.gov/z"}]
        s["trials_by_id"] = {}
        self.assertNotIn("Zzz", landscape.top_drugs_markdown(s))


class IdentifiersAndLabelsAreHandledGenerally(unittest.TestCase):
    def test_a_doi_with_brackets_is_one_identifier(self):
        found = verification.DOI_RE.findall("see doi 10.1016/S2213-8587(24)00123-X and (10.1000/abc.def).")
        self.assertIn("10.1016/S2213-8587(24)00123-X", found)
        self.assertIn("10.1000/abc.def", [f.rstrip(".,;)") for f in found])

    def test_a_drug_named_by_the_question_returns_every_distinct_label(self):
        import clinical_tools
        labels = [{"set_id": "a1", "effective_time": "20260101", "indications_and_usage": ["Drugzol is indicated for glycemic control in type 2 diabetes."], "openfda": {"brand_name": ["Drugzol"]}},
                  {"set_id": "b2", "effective_time": "20260201", "indications_and_usage": ["Drugzol is indicated for chronic weight management in adults with obesity."], "openfda": {"brand_name": ["Drugzol W"]}},
                  {"set_id": "c3", "effective_time": "20260301", "indications_and_usage": ["Drugzol is indicated for glycemic control in type 2 diabetes."], "openfda": {"brand_name": ["Drugzol"]}}]
        original = clinical_tools._fda_get
        clinical_tools._fda_get = lambda params: {"results": labels}
        try:
            records = clinical_tools._fda_records("DRUGZOL", ["weigh"], 4)
        finally:
            clinical_tools._fda_get = original
        self.assertEqual([r["set_id"] for r in records], ["a1", "b2"])      # the duplicate label is dropped, both indications are offered to the model
        self.assertEqual([r["matches_condition"] for r in records], [False, True])


class TheAnnouncementsTableIsRecentDatedAndReadable(unittest.TestCase):
    def state(self, *dates):
        s = make_state()
        s["searched_on"] = "2026-10-08"
        s["regulator_items"] = [{"classification": "approval", "type": "Approval", "relation": "treats_condition", "title": f"Page {i}", "url": f"https://www.fda.gov/p{i}", "quote": f"The FDA approved Drugzol {i}.",
                                 "concerns": "", "drug": "", "date": d} for i, d in enumerate(dates)]
        return s

    def test_only_actions_dated_within_three_years_are_listed_newest_first_and_the_rest_are_counted(self):
        md = landscape.regulatory_updates_markdown(self.state("September 22, 2025", "date not stated", "June 2, 2021", "March 3, 2026"))
        rows = [l for l in md.splitlines() if l.startswith("| ") and "Page" in l]
        self.assertEqual([r.split("|")[1].strip() for r in rows], ["March 3, 2026", "September 22, 2025"])
        self.assertIn("2 further regulator page(s) were read but are not listed", md)

    def test_with_no_dated_action_the_section_says_so_instead_of_listing_undated_pages(self):
        md = landscape.regulatory_updates_markdown(self.state("date not stated"))
        self.assertIn("No FDA, EMA or CDSCO announcement dated within the last 3 years", md)
        self.assertNotIn("| Page 0", md)      # the undated page is not listed as recent

    def test_square_brackets_in_a_title_cannot_break_the_link(self):
        s = self.state("September 22, 2025")
        s["regulator_items"][0]["title"] = "Revised Written Request [POST-FDAAA]"
        md = landscape.regulatory_updates_markdown(s)
        self.assertIn("[Revised Written Request (POST-FDAAA)](https://www.fda.gov/p0)", md)

    def test_a_snippets_cut_off_end_and_mid_sentence_fragments_are_not_quotable(self):
        got = landscape.numbered_sentences("Skip nav. The FDA today initiated the approval of Drugzol tablets for condition Y. Individuals have problems with", complete_only=True)
        self.assertEqual(got, ["The FDA today initiated the approval of Drugzol tablets for condition Y."])

    def test_a_pages_own_date_is_read_from_its_text(self):
        import clinical_tools

        class Reply:
            status_code = 200
            headers = {"content-type": "text/html; charset=UTF-8"}
            content = b"<html><script>var a='January 1, 2001'</script><body><p>FDA News Release</p><p>Release: September 22, 2025</p></body></html>"

        class Client:
            def get(self, url, timeout=6):
                return Reply()
        original = clinical_tools._http
        clinical_tools._http = Client()
        try:
            self.assertEqual(clinical_tools.page_date("https://www.fda.gov/x"), "September 22, 2025")
        finally:
            clinical_tools._http = original


class OnlyUseInPracticeAndOnlyTheConditionItselfGoInTheTreatmentTable(unittest.TestCase):
    SRC = {"u": {"title": "t", "label": "l", "url": "https://www.aap.org/x", "kind": "guideline",
                 "text": "The guideline recommends melatonin for insomnia in autistic children. Atomoxetine (k=3, RB 0.49) improved core symptoms in children."}}

    def item(self, **kw):
        base = {"drug": "Melatonin", "source_id": "u", "sentence": 1, "relevance": "treats_associated_symptom_or_comorbidity", "statement_type": "used_in_practice", "drug_kind": "generic", "category": "Guideline-recommended", "indication": "insomnia in autistic children"}
        base.update(kw)
        return base

    def test_a_sentence_reporting_a_study_result_is_listed_as_studied_never_as_used_in_practice(self):
        study = self.item(drug="Atomoxetine", sentence=2, statement_type="study_result", indication="improved core symptoms in children", category="Off-label")
        s = make_state()
        rows, rejected = landscape.verify_drug_decisions([study, self.item(drug="Zzzzol", statement_type=None)], self.SRC, s)
        self.assertEqual([r["drug"] for r in rows], ["Atomoxetine"])
        self.assertTrue(rows[0]["category"].startswith("Studied"))
        self.assertIn("neither a statement of use", rejected[0][1])
        md = landscape.landscape_markdown(s, rows, [])
        self.assertNotIn("Atomoxetine", md.split("### Other drugs")[0])      # a studied drug is never in the table of drugs used for treatment

    def test_a_guideline_drug_for_an_associated_symptom_is_in_the_other_drugs_table_and_says_so(self):
        s = make_state()
        rows, _ = landscape.verify_drug_decisions([self.item()], self.SRC, s)
        self.assertEqual(rows[0]["category"], "Guideline-recommended")
        md = landscape.landscape_markdown(s, rows, [])
        treatment, other = md.split("### Other drugs")[0], md.split("### Other drugs")[1]
        self.assertNotIn("Melatonin", treatment)
        self.assertIn("Melatonin", other)
        self.assertIn("[guideline-recommended]", other)
        self.assertIn("[associated symptom or comorbidity, not the condition itself]", other)

    def test_a_guideline_drug_for_the_condition_itself_is_in_the_treatment_table(self):
        s = make_state()
        rows, _ = landscape.verify_drug_decisions([self.item(relevance="treats_condition")], self.SRC, s)
        self.assertIn("Melatonin", landscape.landscape_markdown(s, rows, []).split("### Other drugs")[0])

    def test_the_evidence_cell_shows_the_sentence_even_when_it_is_long(self):
        s = make_state()
        long_sentence = "The guideline recommends melatonin for insomnia in autistic children when sleep hygiene has not helped, " * 2 + "with review after three months."
        src = {"u": dict(self.SRC["u"], text=long_sentence)}
        rows, _ = landscape.verify_drug_decisions([self.item()], src, s)
        self.assertIn("The guideline recommends melatonin", landscape.landscape_markdown(s, rows, []).split("### Other drugs")[1])


class OnlySpecificDrugsAreListed(unittest.TestCase):
    SRC = {"u": {"title": "t", "label": "l", "url": "", "kind": "web",
                 "text": "Many medications are prescribed off-label, such as SSRIs and Strattera, for anxiety in children with condition X. Stimulants such as Ritalin treat attention problems."}}

    def item(self, **kw):
        base = {"drug": "Strattera", "drug_kind": "brand", "source_id": "u", "sentence": 1, "relevance": "treats_associated_symptom_or_comorbidity", "statement_type": "used_in_practice",
                "category": "Off-label", "indication": "anxiety in children"}
        base.update(kw)
        return base

    def test_a_drug_class_is_not_a_drug_row_but_a_brand_name_is(self):
        rows, rejected = landscape.verify_drug_decisions([self.item(drug="SSRIs", drug_kind="class"), self.item(), self.item(drug="Ritalin", drug_kind=None)], self.SRC, make_state())
        self.assertEqual([r["drug"] for r in rows], ["Strattera"])
        self.assertEqual(len(rejected), 2)

    def test_a_snippets_heading_marks_and_doubled_full_stops_are_cleaned(self):
        text = landscape._clean_source("### Stimulant Medications Doctors prescribe Ritalin. Strattera may be a better option..")
        self.assertNotIn("#", text)
        self.assertFalse(text.endswith(".."))

    def test_a_brand_is_resolved_to_its_generic_by_the_fda_label(self):
        import clinical_tools
        calls = []

        def fake(params):
            calls.append(params["search"])
            if "brand_name" in params["search"]:
                return {"results": [{"set_id": "s1", "effective_time": "20260101", "openfda": {"brand_name": ["Strattera"], "generic_name": ["ATOMOXETINE HYDROCHLORIDE"]}, "boxed_warning": []}]}
            return {"results": []}
        original = clinical_tools._fda_get
        clinical_tools._fda_get = fake
        try:
            found = clinical_tools.fda_label_link("Strattera")
        finally:
            clinical_tools._fda_get = original
        self.assertEqual(found["generic_name"], "ATOMOXETINE HYDROCHLORIDE")
        self.assertEqual(len(calls), 2)


class EveryListedDrugIsAccountedForInTheSafetySection(unittest.TestCase):
    def test_drugs_without_a_boxed_warning_or_without_a_label_are_named_instead_of_silently_missing(self):
        labels = [{"drug": "Drugzol", "url": "https://dailymed.nlm.nih.gov/a", "boxed_warning": "INCREASED MORTALITY IN ELDERLY PATIENTS."},
                  {"drug": "Calmol", "url": "https://dailymed.nlm.nih.gov/b", "boxed_warning": None}]
        md = landscape.safety_alerts_markdown(labels, ["Drugzol", "Calmol", "Mixol"])
        self.assertIn("[Drugzol](https://dailymed.nlm.nih.gov/a)", md)
        self.assertIn("No boxed warning on the retrieved label of: Calmol.", md)
        self.assertIn("No FDA label could be matched for: Mixol.", md)
        self.assertNotIn("[Calmol]", md)

    def test_with_no_boxed_warning_at_all_the_section_still_says_which_drugs_were_checked(self):
        md = landscape.safety_alerts_markdown([{"drug": "Calmol", "url": "https://dailymed.nlm.nih.gov/b", "boxed_warning": None}], ["Calmol"])
        self.assertIn("No boxed warning was found", md)
        self.assertIn("Calmol", md)


class ACombinationProductIsOneProductNotSeveralDrugs(unittest.TestCase):
    def test_active_ingredients_joined_by_the_registry_are_shown_as_one_product(self):
        r = {"generic": "Avobenzone, Homosalate, Octisalate, Octocrylene", "brands": []}
        self.assertEqual(landscape.product_name(r), "Combination product (one product: avobenzone + homosalate + octisalate + octocrylene)")
        r = {"generic": "Titanium Dioxide, Zinc Oxide", "brands": ["Neutrogena Mineral Sunscreen"]}
        self.assertEqual(landscape.product_name(r), "Neutrogena Mineral Sunscreen (one product: titanium dioxide + zinc oxide)")
        self.assertEqual(landscape.product_name({"generic": "Risperidone", "brands": ["Risperdal"]}), "Risperidone (Risperdal)")

    def test_the_tables_use_the_product_name(self):
        s = make_state()
        s["regulatory"] = [{"generic": "Aaa, Bbb", "brands": [], "matches_condition": True, "model_relevant": True, "model_evidence": "Aaa and Bbb are indicated for condition X.",
                            "indication_excerpt": "Aaa and Bbb are indicated for condition X.", "url": "https://dailymed.nlm.nih.gov/x", "label_date": "2026-01-01"}]
        self.assertIn("one product: aaa + bbb", landscape.top_drugs_markdown(s))
        self.assertIn("one product: aaa + bbb", landscape.landscape_markdown(s, [], []))


class ALabelMustMatchWhatTheQuestionAsksFor(unittest.TestCase):
    def offered(self):
        rec = {"generic": "Sunx"}
        return rec, {"fda:Sunx": {"indication_text": "Sunx helps prevent sunburn and decreases the risk of skin cancer.", "record": rec}}

    def test_a_preventive_label_is_not_relevant_to_a_question_about_treatment(self):
        rec, offered = self.offered()
        out = landscape.verify_fda_decisions([{"id": "fda:Sunx", "purpose": "prevent", "relevant": True, "relation": "treats_associated_symptom_or_comorbidity", "sentence": 1}], offered, "treat_or_manage")
        self.assertEqual(out["excluded"], ["Sunx"])
        self.assertFalse(landscape.fda_relevant(rec))

    def test_a_label_for_treatment_is_never_excluded_by_a_question_about_prevention(self):
        rec, offered = self.offered()
        out = landscape.verify_fda_decisions([{"id": "fda:Sunx", "purpose": "treat_or_manage", "relevant": True, "relation": "treats_condition", "sentence": 1}], offered, "prevent")
        self.assertEqual(out["confirmed"], ["Sunx"])

    def test_the_same_label_is_relevant_when_the_question_asks_about_prevention(self):
        rec, offered = self.offered()
        out = landscape.verify_fda_decisions([{"id": "fda:Sunx", "purpose": "prevent", "relevant": True, "relation": "treats_condition", "sentence": 1}], offered, "prevent")
        self.assertEqual(out["confirmed"], ["Sunx"])

    def test_an_unknown_question_type_changes_nothing(self):
        rec, offered = self.offered()
        out = landscape.verify_fda_decisions([{"id": "fda:Sunx", "purpose": "prevent", "relevant": True, "sentence": 1}], offered, "other")
        self.assertEqual(out["confirmed"], ["Sunx"])


class TheFastAnswerKeepsTheSearchOrderAndSaysWhenLabelsAreStillBeingRead(unittest.TestCase):
    def test_papers_the_triage_has_not_read_keep_the_search_engines_own_order(self):
        a, b, c = study("710001", year="2019"), study("710002", year="2026"), study("710003", year="2022")
        got = [p["pmid"] for p in qa_stream.ordered_papers({"710001": a, "710002": b, "710003": c}, set(), allow_untriaged=True)]
        self.assertEqual(got, ["710001", "710002", "710003"])      # not newest-first, not by study design

    def test_the_top_drugs_table_names_labels_still_being_read(self):
        s = make_state()
        s["regulatory"] = [{"generic": "Zzz", "matches_condition": True, "indication_excerpt": "Zzz is indicated for condition X.", "url": "https://dailymed.nlm.nih.gov/z"}]
        s["trials_by_id"] = {}
        self.assertIn("1 retrieved FDA label(s) were still being read", landscape.top_drugs_markdown(s))
        s["regulatory"][0]["model_relevant"] = False
        self.assertNotIn("still being read", landscape.top_drugs_markdown(s))


class TheFastSearchNeverLeavesTheFirstAnswerWithoutPapers(unittest.TestCase):
    PRIMARY = json.dumps({"results": [{"pmid": "1", "title": "A"}, {"pmid": "2", "title": "B"}]})
    FALLBACK = json.dumps({"results": [{"pmid": "2", "title": "B"}, {"pmid": "3", "title": "C"}]})

    def test_the_primarys_papers_come_first_and_the_fallbacks_new_ones_follow(self):
        got = json.loads(qa_stream.primary_with_fallback(lambda: self.PRIMARY, lambda: self.FALLBACK, grace=2.0))
        self.assertEqual([r["pmid"] for r in got["results"]], ["1", "2", "3"])

    def test_a_slow_primary_leaves_the_fallback_to_answer_alone(self):
        import time

        def slow():
            time.sleep(1.5)
            return self.PRIMARY
        got = json.loads(qa_stream.primary_with_fallback(slow, lambda: self.FALLBACK, grace=0.3))
        self.assertEqual([r["pmid"] for r in got["results"]], ["2", "3"])

    def test_a_failing_primary_is_covered_by_the_fallback(self):
        def broken():
            raise RuntimeError("down")
        got = json.loads(qa_stream.primary_with_fallback(broken, lambda: self.FALLBACK, grace=0.5))
        self.assertEqual([r["pmid"] for r in got["results"]], ["2", "3"])


class ALabelSentenceIsShownWithItsContext(unittest.TestCase):
    def test_a_bullet_gets_its_lead_in_and_a_full_following_sentence_is_added(self):
        text = "Drugzol is indicated for the treatment of: \u2022 Schizophrenia \u2022 Irritability Associated with Condition X \u2022 Tics"
        sentences = landscape.numbered_sentences(text, 900, 12)
        self.assertEqual(landscape.with_context(sentences, 3), "Drugzol is indicated for the treatment of: Irritability Associated with Condition X.")
        full = landscape.numbered_sentences("Otherol is indicated for irritability in condition X. Efficacy was established in 3 trials in children aged 5 to 17 years.", 900, 12)
        self.assertEqual(landscape.with_context(full, 1), "Otherol is indicated for irritability in condition X. Efficacy was established in 3 trials in children aged 5 to 17 years.")

    def test_the_decision_stores_the_sentence_with_its_context(self):
        rec = {"generic": "Drugzol"}
        text = "Drugzol is indicated for the treatment of: \u2022 Schizophrenia \u2022 Irritability Associated with Condition X"
        offered = {"fda:Drugzol": {"indication_text": text, "record": rec, "sentences": landscape.numbered_sentences(text, 900, 12)}}
        landscape.verify_fda_decisions([{"id": "fda:Drugzol", "relevant": True, "relation": "treats_condition", "sentence": 3}], offered)
        self.assertEqual(rec["model_evidence"], "Drugzol is indicated for the treatment of: Irritability Associated with Condition X")


class FutureDatesAreMarkedWhateverTheBulletLayout(unittest.TestCase):
    def test_the_date_and_study_type_inside_one_bold_span_is_handled(self):
        text = "## Latest findings\n- **2026 Dec: Case Reports**, A case study showed a result (PMID 1).\n- **2026 Sep: Meta-Analysis**, Another result (PMID 2).\n"
        fixed, actions = verification.fix_future_dates(text, {"searched_on": "2026-10-09"})
        self.assertIn("**Advance publication (issue dated 2026 Dec): Case Reports**", fixed)
        self.assertIn("**2026 Sep: Meta-Analysis**", fixed)
        self.assertEqual(len(actions), 1)


class FutureDatesWithADayAreMarkedToo(unittest.TestCase):
    def test_a_date_with_a_day_is_compared_to_the_search_date(self):
        from datetime import datetime
        today = datetime(2026, 10, 9)
        later = ["2027 Jan 15", "2026 Dec 01", "2026 Dec", "2026 Oct 20", "2026-10-20", "2026-12-01", "2027"]
        not_later = ["2026 Oct 09", "2026 Oct 5", "2026 Oct", "2026 Sep 21", "2026", "2025 Dec 31"]
        self.assertEqual([verification.date_after(x, today) for x in later], [True] * len(later))
        self.assertEqual([verification.date_after(x, today) for x in not_later], [False] * len(not_later))

    def test_the_bullet_with_a_day_in_its_date_is_rewritten(self):
        text = "## Latest findings\n- **2027 Jan 15**: *Emerging evidence* \u2013 A method (PMID 1).\n- **2026 Sep 21**: *Emerging evidence* \u2013 Another (PMID 2).\n"
        fixed, actions = verification.fix_future_dates(text, {"searched_on": "2026-10-09"})
        self.assertIn("**Advance publication (issue dated 2027 Jan 15)**", fixed)
        self.assertIn("**2026 Sep 21**", fixed)
        self.assertEqual(len(actions), 1)


class AStalledModelCallIsAbandoned(unittest.TestCase):
    def test_a_generator_that_goes_silent_raises_instead_of_hanging(self):
        import time

        def silent():
            yield "a"
            time.sleep(5)
            yield "b"
        got = []
        with self.assertRaises(TimeoutError):
            for chunk in qa_stream._with_deadline(silent(), first=1.0, idle=0.5):
                got.append(chunk)
        self.assertEqual(got, ["a"])

    def test_a_normal_generator_passes_through_and_an_error_is_re_raised(self):
        self.assertEqual(list(qa_stream._with_deadline(iter(["x", "y"]), first=1.0, idle=1.0)), ["x", "y"])

        def broken():
            yield "x"
            raise ValueError("boom")
        with self.assertRaises(ValueError):
            list(qa_stream._with_deadline(broken(), first=1.0, idle=1.0))


class TablePaddingIsCollapsed(unittest.TestCase):
    def test_padding_spaces_and_long_separator_rows_become_compact(self):
        text = "## Current evidence\n| Approach                  | Evidence level      | Shows |\n|--------------------------|---------------------|-------|\n| Exercise                  | Emerging            | Improved [1]. |\n"
        fixed = qa_stream.tidy_model_sections(text)
        self.assertIn("| Approach | Evidence level | Shows |", fixed)
        self.assertIn("|---|---|---|", fixed)


class DoseLookupIsPickedByTheModelAndCheckedByCode(unittest.TestCase):
    PAGES = [{"title": "Drugzol: how to take it", "url": "https://example.org/drugzol", "snippet": "The usual dose is 50 mg at bedtime. Drugzol is a medicine. Tell your doctor about other medicines."},
             {"title": "Unrelated", "url": "https://example.org/x", "snippet": "Drugzol was studied in 120 adults who slept well. Other drugs are given 5 mg daily."}]

    def test_only_sentences_with_an_amount_or_route_from_a_page_about_the_drug_are_offered(self):
        found = landscape.dose_candidates("Drugzol", self.PAGES)
        texts = [c["text"] for c in found]
        self.assertIn("The usual dose is 50 mg at bedtime.", texts)
        self.assertNotIn("Drugzol is a medicine.", texts)
        self.assertFalse(any("Other drugs are given 5 mg daily" in t for t in texts))      # another page, and it names no Drugzol dose

    def test_a_pick_counts_only_for_a_candidate_of_that_drug(self):
        candidates = {"Drugzol": landscape.dose_candidates("Drugzol", self.PAGES)}
        good = landscape.verify_dose_picks([{"drug": "drugzol", "candidate": "c1"}, {"drug": "Drugzol", "candidate": "c99"}, {"drug": "Other", "candidate": "c1"}, "x"], candidates)
        self.assertEqual(list(good), ["Drugzol"])
        self.assertEqual(good["Drugzol"]["url"], "https://example.org/drugzol")
        self.assertEqual(landscape.verify_dose_picks([{"drug": "Drugzol", "candidate": "c99"}], candidates), {})


class OriginalFiguresAreIncludedOnlyWhenTheyQualify(unittest.TestCase):
    CAPTION = "Figure 1. Forest plot of the mean difference in symptom score for drug A versus placebo."

    @staticmethod
    def fig(**over):
        base = {"id": "PMC1:f1", "label": "Figure 1", "caption": OriginalFiguresAreIncludedOnlyWhenTheyQualify.CAPTION, "image": "https://cdn.ncbi.nlm.nih.gov/pmc/blobs/aa/1/bb/x-f001.png",
                "page_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC1/", "pmid": "111", "pmcid": "PMC1", "doi": None, "title": "A trial of drug A", "cite": "Smith et al., J, 2026 [PMID 111]",
                "design": "Randomized Controlled Trial", "licence": {"url": "https://creativecommons.org/licenses/by/4.0/", "label": "CC BY 4.0"}}
        base.update(over)
        return base

    @staticmethod
    def png_bytes():
        import io
        from PIL import Image
        buffer = io.BytesIO()
        Image.new("RGB", (40, 30), (200, 30, 30)).save(buffer, "PNG")
        return buffer.getvalue()

    def run_select(self, found, rubric=None, inspect=None, cited=("111",), direct=("111",)):
        import tempfile
        from pathlib import Path
        import source_figures
        data = self.png_bytes()

        class Reply:
            status_code, headers, content = 200, {"content-type": "image/png"}, data
        good_rubric = lambda available: [{"id": f["id"], "helps_answer": True, "clinically_meaningful": True, "adds_beyond_report": True, "not_decorative": True, "answers_what_is_asked": True,
                                          "reason": "Forest plot of the mean difference in symptom score"} for f in available]
        good_inspect = lambda fig: '{"shows": "A forest plot of drug effects", "matches_caption": true, "relevant_to_question": true, "legible": true}'
        trace = {}
        folder = Path(tempfile.mkdtemp())
        out = source_figures.select_figures(found, set(cited), set(direct), rubric or good_rubric, inspect or good_inspect, folder, trace=trace, get=lambda url: Reply())
        return out, trace, folder

    def test_a_relevant_cited_licensed_verified_figure_is_included_with_its_attribution(self):
        out, trace, _ = self.run_select([self.fig()])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["caption"], self.CAPTION)      # the original caption, whole
        self.assertEqual(out[0]["licence"]["label"], "CC BY 4.0")
        self.assertIsNone(out[0]["doi"])      # a missing detail stays missing
        self.assertEqual(trace["image_check"], {"downloaded": 1, "looked_at": 1, "could_not_check": 0, "kept": 1})

    def test_an_irrelevant_figure_is_rejected_by_the_relevance_test(self):
        def rubric(available):
            return [{"id": f["id"], "helps_answer": False, "clinically_meaningful": True, "adds_beyond_report": True, "not_decorative": True, "answers_what_is_asked": True, "reason": "Forest plot of the mean difference"} for f in available]
        self.assertEqual(self.run_select([self.fig()], rubric=rubric)[0], [])

    def test_a_justification_that_is_mostly_a_verbatim_caption_run_is_accepted_but_a_paraphrase_is_not(self):
        import source_figures
        caption = self.CAPTION
        self.assertTrue(source_figures.quote_in_caption("Forest plot of the mean difference in symptom score, which is big", caption))
        self.assertFalse(source_figures.quote_in_caption("this chart shows how much better drug A works than placebo overall", caption))
        self.assertFalse(source_figures.quote_in_caption("mean difference", caption))      # fewer than four consecutive words

    def test_a_justification_that_is_not_in_the_caption_is_rejected(self):
        def rubric(available):
            return [{"id": f["id"], "helps_answer": True, "clinically_meaningful": True, "adds_beyond_report": True, "not_decorative": True, "answers_what_is_asked": True, "reason": "dramatic survival benefit in all patients"} for f in available]
        self.assertEqual(self.run_select([self.fig()], rubric=rubric)[0], [])

    def test_a_figure_from_an_uncited_but_relevant_paper_is_kept_and_labelled_and_an_irrelevant_paper_is_rejected(self):
        import source_figures
        out, trace, _ = self.run_select([self.fig()], cited=())
        self.assertEqual(len(out), 1)
        self.assertFalse(out[0]["cited"])      # the report does not cite this paper: its figure says so
        self.assertEqual(self.run_select([self.fig()], cited=("111",))[0][0]["cited"], True)
        self.assertEqual(source_figures.eligible([self.fig()], set(), {"111"}, require_cited=True), [])      # the strict mode is still available
        out, trace, _ = self.run_select([self.fig()], direct=())      # neither directly nor indirectly relevant
        self.assertEqual(out, [])
        self.assertEqual(trace["removed_by_gate"]["paper_not_direct"], 1)

    def test_unclear_licence_or_a_reproduced_figure_is_rejected(self):
        self.assertEqual(self.run_select([self.fig(licence=None)])[0], [])
        reproduced = self.fig(caption="Figure 1. Forest plot of the mean difference in symptom score. Reproduced from Jones et al. with permission.")
        out, trace, _ = self.run_select([reproduced])
        self.assertEqual(out, [])
        self.assertEqual(trace["removed_by_gate"]["reproduced_or_permission"], 1)

    def test_a_figure_whose_image_cannot_be_verified_is_skipped(self):
        out, trace, _ = self.run_select([self.fig()], inspect=lambda fig: None)      # rate-limited or unreachable
        self.assertEqual(out, [])
        self.assertEqual(trace["image_check"]["could_not_check"], 1)
        mismatch = lambda fig: '{"shows": "A photograph", "matches_caption": false, "relevant_to_question": true, "legible": true}'
        self.assertEqual(self.run_select([self.fig()], inspect=mismatch)[0], [])
        unreadable = lambda fig: '{"shows": "A blur", "matches_caption": true, "relevant_to_question": true, "legible": false}'
        self.assertEqual(self.run_select([self.fig()], inspect=unreadable)[0], [])
        self.assertEqual(self.run_select([self.fig()], inspect=lambda fig: "I cannot see an image")[0], [])

    def test_with_the_image_check_off_the_figure_is_kept_but_labelled_not_machine_checked(self):
        import tempfile
        from pathlib import Path
        import source_figures
        data = self.png_bytes()

        class Reply:
            status_code, headers, content = 200, {"content-type": "binary/octet-stream"}, data      # the bucket labels every file like this: the type is read from the file
        rubric = lambda available: [{"id": f["id"], "helps_answer": True, "clinically_meaningful": True, "adds_beyond_report": True, "not_decorative": True, "answers_what_is_asked": True,
                                     "reason": "Forest plot of the mean difference in symptom score"} for f in available]
        trace = {}
        out = source_figures.select_figures([self.fig()], {"111"}, {"111"}, rubric, None, Path(tempfile.mkdtemp()), trace=trace, get=lambda url: Reply())
        self.assertEqual(len(out), 1)
        self.assertEqual(trace["image_check"]["mode"], "off")
        record = source_figures.display_records(out, {"1": "111"})[0]
        self.assertFalse(record["image_checked"])
        self.assertIsNone(record["image_check"])
        # the other gates still apply with the check off
        self.assertEqual(source_figures.select_figures([self.fig(licence=None)], {"111"}, {"111"}, rubric, None, Path(tempfile.mkdtemp()), get=lambda url: Reply()), [])
        self.assertEqual(source_figures.select_figures([self.fig()], {"111"}, set(), rubric, None, Path(tempfile.mkdtemp()), get=lambda url: Reply()), [])      # a paper that is not relevant

    QUESTION = "How do you manage migraine in adults?"
    TEXTS = {"111": "Migraine prevention in adults: a randomized trial. We enrolled adults with chronic migraine.", "222": "Nerve blocks in adolescents with refractory migraine: a retrospective study."}

    def population_answer(self, **papers):
        return {"question_population": "in adults", "papers": [{"pmid": pmid, "study_population": "x", **item} for pmid, item in papers.items()]}

    def test_a_paper_whose_own_population_matches_the_question_is_kept(self):
        import source_figures
        answer = self.population_answer(**{"111": {"verdict": "match", "evidence_quote": "We enrolled adults with chronic migraine"}})
        trace = {}
        out = source_figures.apply_population_check([self.fig()], answer, self.TEXTS, self.QUESTION, trace)
        self.assertEqual(len(out), 1)
        self.assertEqual(trace["population"]["question_population"], "in adults")

    def test_a_broader_population_is_kept_with_a_caveat_only_when_its_text_is_quoted(self):
        import source_figures
        quoted = self.population_answer(**{"111": {"verdict": "broader", "study_population": "adults and adolescents with migraine", "evidence_quote": "We enrolled adults with chronic migraine"}})
        out = source_figures.apply_population_check([self.fig()], quoted, self.TEXTS, self.QUESTION)
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0]["population_note"].startswith("broader than the question:"))
        unquoted = self.population_answer(**{"111": {"verdict": "broader", "study_population": "everyone", "evidence_quote": "patients of every kind took part in the study"}})
        self.assertEqual(source_figures.apply_population_check([self.fig()], unquoted, self.TEXTS, self.QUESTION), [])

    def test_a_paper_with_a_clearly_different_population_is_rejected(self):
        import source_figures
        answer = self.population_answer(**{"222": {"verdict": "mismatch", "evidence_quote": "in adolescents with refractory migraine"}})
        trace = {}
        self.assertEqual(source_figures.apply_population_check([self.fig(pmid="222")], answer, self.TEXTS, self.QUESTION, trace), [])
        self.assertEqual(trace["population"]["mismatch"], 1)

    def test_an_unknown_population_is_skipped_never_assumed(self):
        import source_figures
        for answer in (self.population_answer(**{"111": {"verdict": "unclear", "evidence_quote": ""}}),
                       self.population_answer(**{"111": {"verdict": "match", "evidence_quote": "all patients were treated successfully overall"}}),      # a quote that is not in the paper
                       self.population_answer(),      # the paper was not mentioned
                       None, "not json"):
            trace = {}
            self.assertEqual(source_figures.apply_population_check([self.fig()], answer, self.TEXTS, self.QUESTION, trace), [], answer)

    def test_a_question_that_names_no_population_rejects_nothing_on_population(self):
        import source_figures
        figs = [self.fig(), self.fig(id="PMC2:f1", pmid="222")]
        for named in ("not specified", "", "in elderly women"):      # the last is not in the question: it is not trusted
            answer = {"question_population": named, "papers": []}
            trace = {}
            self.assertEqual(len(source_figures.apply_population_check(figs, answer, self.TEXTS, self.QUESTION, trace)), 2)
            self.assertFalse(trace["population"]["question_names_a_population"])

    def test_select_figures_applies_the_population_check_before_the_relevance_test_and_fails_safe(self):
        import tempfile
        from pathlib import Path
        import source_figures
        data = self.png_bytes()

        class Reply:
            status_code, headers, content = 200, {"content-type": "binary/octet-stream"}, data
        asked = []

        def rubric(available):
            asked.append([f["id"] for f in available])
            return [{"id": f["id"], "helps_answer": True, "clinically_meaningful": True, "adds_beyond_report": True, "not_decorative": True, "answers_what_is_asked": True,
                     "reason": "Forest plot of the mean difference in symptom score"} for f in available]
        adolescent = self.fig(id="PMC2:f1", pmid="222", pmcid="PMC2")
        answer = self.population_answer(**{"111": {"verdict": "match", "evidence_quote": "We enrolled adults with chronic migraine"}, "222": {"verdict": "mismatch", "evidence_quote": "in adolescents"}})
        population = lambda available: source_figures.apply_population_check(available, answer, self.TEXTS, self.QUESTION)
        out = source_figures.select_figures([self.fig(), adolescent], {"111", "222"}, {"111", "222"}, rubric, None, Path(tempfile.mkdtemp()), get=lambda url: Reply(), population=population)
        self.assertEqual([f["pmid"] for f in out], ["111"])
        self.assertEqual(asked, [["PMC1:f1"]])      # the adolescent figure never reached the relevance test

        def broken(available):
            raise RuntimeError("model unavailable")
        trace = {}
        self.assertEqual(source_figures.select_figures([self.fig()], {"111"}, {"111"}, rubric, None, Path(tempfile.mkdtemp()), trace=trace, get=lambda url: Reply(), population=broken), [])
        self.assertTrue(trace["population_check_failed"])

    def test_no_candidates_or_a_failing_relevance_call_means_no_figures(self):
        self.assertEqual(self.run_select([])[0], [])

        def broken(available):
            raise RuntimeError("model unavailable")
        self.assertEqual(self.run_select([self.fig()], rubric=broken)[0], [])

    def test_the_original_image_bytes_are_saved_unchanged_with_a_checksum(self):
        import hashlib
        from pathlib import Path
        out, _, folder = self.run_select([self.fig()])
        saved = Path(out[0]["path"]).read_bytes()
        self.assertEqual(saved, self.png_bytes())
        self.assertEqual(out[0]["sha256"], hashlib.sha256(self.png_bytes()).hexdigest())
        self.assertEqual((out[0]["width"], out[0]["height"]), (40, 30))
        self.assertEqual(out[0]["pdf_path"], out[0]["path"])      # a PNG needs no converted copy

    def test_the_page_and_the_pdf_are_given_the_same_records_and_an_empty_list_adds_no_section(self):
        import tempfile
        from pathlib import Path
        import pdf_export
        import source_figures
        out, _, _ = self.run_select([self.fig()])
        records = source_figures.display_records(out, {"1": "111"})
        self.assertEqual(records[0]["reference"], "1")
        self.assertEqual(records[0]["licence_label"], "CC BY 4.0")
        self.assertEqual(records[0]["caption"], self.CAPTION)
        texts = []
        original = pdf_export.Paragraph

        class Spy(original):
            def __init__(self, text, *args, **kwargs):
                texts.append(text)
                super().__init__(text, *args, **kwargs)
        pdf_export.Paragraph = Spy
        try:
            folder = Path(tempfile.mkdtemp())
            pdf_export.markdown_to_pdf("# Q\n\n## Key studies\n\n1. A study [1].\n", str(folder / "with.pdf"), "Q", figures=records)
            with_text = " ".join(texts)
            texts.clear()
            pdf_export.markdown_to_pdf("# Q\n\n## Key studies\n\n1. A study [1].\n", str(folder / "without.pdf"), "Q", figures=[])
            without_text = " ".join(texts)
        finally:
            pdf_export.Paragraph = original
        self.assertIn("Figure credits", with_text)
        self.assertNotIn("Source:", with_text)      # no source paragraph under the figure
        self.assertIn("Forest plot of the mean difference in symptom score", with_text)
        self.assertIn("CC BY 4.0", with_text)
        self.assertIn("Cited in this report as [1]", with_text)
        self.assertNotIn("Source:", without_text)

    def test_the_article_xml_is_read_for_figures_captions_and_an_explicit_licence(self):
        import source_figures
        xml = ('<article><front><permissions><license xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="https://creativecommons.org/licenses/by/4.0/"><license-p>CC BY</license-p></license></permissions></front>'
               '<body><fig id="f1"><label>Figure 1</label><caption><p>Forest plot of the mean difference in symptom score for drug A versus placebo.</p></caption>'
               '<graphic xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="x-f001"/></fig>'
               '<fig id="f2"><label>Figure 2</label><caption><p>A scan of one patient.</p></caption><graphic xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="x-f002"/></fig>'
               '<fig id="f3"><label>Figure 3</label><caption><p>No image file in the bucket.</p></caption><graphic xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="x-f003"/></fig></body></article>')
        keys = ["PMC1.1/x-f001.jpg", "PMC1.1/x-f002.png", "PMC1.1/PMC1.1.xml"]
        figs = source_figures.parse_figures(xml, "PMC1", keys)
        self.assertEqual([f["id"] for f in figs], ["PMC1:f1", "PMC1:f2"])      # a figure whose image file is not in the bucket is not a candidate
        self.assertEqual(figs[0]["caption"], self.CAPTION)
        self.assertEqual(figs[0]["image"], "https://pmc-oa-opendata.s3.amazonaws.com/PMC1.1/x-f001.jpg")
        self.assertIn("A scan of one patient", figs[1]["caption"])
        self.assertEqual(source_figures.open_licence(xml)["label"], "CC BY 4.0")
        self.assertIsNone(source_figures.open_licence("<p>All rights reserved.</p>"))
        self.assertEqual(source_figures.open_licence('<a href="https://creativecommons.org/publicdomain/zero/1.0/">x</a>')["label"], "CC0 1.0")


import re


class NoDiseaseOrDrugIsHardCoded(unittest.TestCase):
    NAMES = re.compile(r"\b(?:autis\w*|adhd|tourette\w*|pandas|irritability|risperidone|aripiprazole|ibuprofen|naproxen|melatonin|insomnia|diabetes|hypertension|epilepsy|schizophrenia|asthma|"
                       r"cancer|parkinson\w*|alzheimer\w*|migraine|arthritis|lupus|covid\w*|sertraline|fluoxetine|methylphenidate|atomoxetine)\b", re.I)

    def test_no_production_file_or_config_names_a_disease_or_a_drug(self):
        import pathlib
        root = pathlib.Path(__file__).resolve().parent.parent
        skip = {"audit.py", "phi.py"}      # a development audit script, and the privacy filter whose name lists stop a disease name from being removed as a person's name
        found = []
        for path in list(root.glob("*.py")) + [root / "config" / "care_topics.json"]:
            if path.name in skip:
                continue
            text = path.read_text(encoding="utf-8")
            if path.suffix == ".json":
                text = text.split('"authoritative_domains"')[0] + text.split('"default"')[-1]      # the publisher allow-list is a source policy, not medical knowledge
            for number, line in enumerate(text.splitlines(), 1):
                if self.NAMES.search(line):
                    found.append(f"{path.name}:{number}: {line.strip()[:90]}")
        self.assertEqual(found, [], "a disease or drug name is hard-coded:\n" + "\n".join(found))


class FiguresOfRelatedConditionsAndPlacement(unittest.TestCase):
    @staticmethod
    def fig(**over):
        return OriginalFiguresAreIncludedOnlyWhenTheyQualify.fig(**over)

    def test_a_closely_related_paper_gives_a_figure_labelled_indirect_but_an_unrelated_one_does_not(self):
        import source_figures
        related = self.fig(pmid="333")
        trace = {}
        kept = source_figures.eligible([self.fig(), related, self.fig(pmid="444")], {"111", "333"}, {"111"}, trace, indirect_pmids={"333"})
        self.assertEqual([(f["pmid"], f["relation"]) for f in kept], [("111", "direct"), ("333", "indirect")])
        self.assertEqual(trace["removed_by_gate"]["paper_not_direct"], 1)      # 444 is neither direct nor indirect
        self.assertEqual(source_figures.eligible([related], set(), {"111"}, indirect_pmids={"333"}, require_cited=True), [])
        self.assertEqual(len(source_figures.eligible([related], set(), {"111"}, indirect_pmids={"333"})), 1)      # an uncited related paper is allowed and labelled

    def test_a_figure_belongs_in_the_section_that_cites_its_paper(self):
        import source_figures
        report = ("## Clinical summary\n\nA short summary [2].\n\n## Latest findings\n\n- A finding [1, 2].\n\n## Current evidence\n\n| A | B |\n|---|---|\n| x | y [2] |\n\n"
                  "## Key studies\n\n1. A study [3].\n\n## Sources\n\n[1] x\n[2] y\n")
        self.assertEqual(source_figures.section_for(report, 2), "Current evidence")      # the evidence sections come first, not the summary
        self.assertEqual(source_figures.section_for(report, 3), "Key studies")
        self.assertEqual(source_figures.section_for(report, 1), "Latest findings")
        self.assertIsNone(source_figures.section_for(report, 9))
        self.assertIsNone(source_figures.section_for(report, None))
        self.assertIsNone(source_figures.section_for("", 1))

    def test_a_figure_of_a_paper_the_report_does_not_cite_goes_to_the_section_about_the_same_things(self):
        import source_figures
        report = ("## Latest findings\n\n- Nothing relevant here.\n\n## Current evidence\n\n| Approach | Level | Shows | Ref |\n|---|---|---|---|\n"
                  "| Topiramate | Limited | Reduced headache frequency in chronic migraine. | [4] |\n| Amitriptyline | Emerging | Reduced headache frequency. | [4] |\n\n"
                  "## Treatment Drug Landscape\n\nTopiramate, amitriptyline.\n\n## Key studies\n\n1. A study of botulinum toxin [9].\n")
        title_and_caption = "Comparative effectiveness of topiramate versus amitriptyline for chronic migraine. Figure 2. Trend in headache frequency over time."
        self.assertEqual(source_figures.best_section(report, title_and_caption), "Current evidence")
        self.assertIsNone(source_figures.best_section(report, "A figure about something else entirely, the weather."))
        self.assertIsNone(source_figures.best_section("", title_and_caption))
        records = source_figures.display_records([{**OriginalFiguresAreIncludedOnlyWhenTheyQualify.fig(pmid="999", title="Topiramate versus amitriptyline for chronic migraine",
                                                                                           caption="Figure 2. Trend in headache frequency over time."), "path": "x.png"}], {"1": "111"}, report)
        self.assertEqual(records[0]["section"], "Current evidence")
        self.assertFalse(records[0]["cited"])

    def test_the_pdf_places_each_figure_inside_its_own_section_not_at_the_end(self):
        import tempfile
        from pathlib import Path
        import pdf_export
        import source_figures
        found = self.fig()
        out = [{**found, "path": str(Path(tempfile.mkdtemp()) / "x.png"), "width": 40, "height": 30, "relation": "direct"}]
        from PIL import Image
        Image.new("RGB", (40, 30), (10, 10, 200)).save(out[0]["path"])
        out[0]["pdf_path"] = out[0]["path"]
        report = "# Q\n\n## Current evidence\n\nEvidence text [1].\n\n## Key studies\n\n1. A study [2].\n\n## Evidence limitations\n\nLimits text.\n"
        records = source_figures.display_records(out, {"1": "111"}, report)
        self.assertEqual(records[0]["section"], "Current evidence")
        texts, original = [], pdf_export.Paragraph

        class Spy(original):
            def __init__(self, text, *args, **kwargs):
                texts.append(text)
                super().__init__(text, *args, **kwargs)
        pdf_export.Paragraph = Spy
        try:
            pdf_export.markdown_to_pdf(report, str(Path(tempfile.mkdtemp()) / "a.pdf"), "Q", figures=records)
        finally:
            pdf_export.Paragraph = original
        order = " | ".join(texts)
        self.assertLess(order.index("Evidence text"), order.index("Forest plot of the mean difference"))
        self.assertLess(order.index("Forest plot of the mean difference"), order.index("A study"))      # before the next section starts, not after Evidence limitations
        self.assertEqual(order.count("Figure credits"), 1)      # the attribution is one list at the end, not under each figure
        self.assertNotIn("Source:", order)


class CurrentEvidenceRowsAndConditionSpecificDoses(unittest.TestCase):
    def test_lines_become_a_compact_table_with_the_citation_in_its_own_column(self):
        text = ("## Current evidence\nApproach A || Emerging || One small study reported benefit. || (PMID 111111)\n"
                "Approach B || Insufficient evidence || No result was reported (Smith et al., J, 2025 [PMID 222222]).\n"
                "not a row")
        table = qa_stream.rows_to_table(text)
        self.assertTrue(table.startswith("## Current evidence"))
        self.assertIn("| Approach | Evidence level | What the evidence shows | Reference |\n|---|---|---|---|", table)
        self.assertIn("| Approach A | Emerging | One small study reported benefit. | (PMID 111111) |", table)
        self.assertIn("| Approach B | Insufficient evidence | No result was reported | (PMID 222222) |", table)
        already = "## Current evidence\n\n| Approach | Evidence level |\n|---|---|\n| A | B |"
        self.assertEqual(qa_stream.rows_to_table(already), already)
        self.assertEqual(qa_stream.rows_to_table("## Current evidence\n\n| Approach | Evidence level | What the evidence shows "), "## Current evidence\n\n| Approach | Evidence level | What the evidence shows ")

    def test_a_dose_sentence_must_be_about_the_question_condition(self):
        pages = [{"title": "Drugzol dosage guide", "url": "https://example.org/a", "snippet": "Drugzol: the initial dose is 0.5 mg/kg/day orally for adults with heart disease."},
                 {"title": "Drugzol for condition X", "url": "https://example.org/b", "snippet": "Most children with condition X start Drugzol at 0.5 mg once daily at bedtime."}]
        with_stems = landscape.dose_candidates("Drugzol", pages, stems=["condition"])
        self.assertEqual([c["url"] for c in with_stems], ["https://example.org/b"])      # the first page is about another use
        self.assertEqual(len(landscape.dose_candidates("Drugzol", pages)), 2)      # no condition words known: nothing is filtered


class FigureExplanationsAreVerified(unittest.TestCase):
    CAPTION = "Figure 3. Forest plot of effects. Values above 0 mean more improvement than control. Each dot is a summary effect and each line is its 95% confidence interval."
    ABSTRACT = "In randomized trials the effect of intervention A on social communication was 0.35 (95% CI 0.23 to 0.47) across 32 studies."
    TEXTS = {"111": "A meta-analysis. " + ABSTRACT}

    def figure(self):
        return {"id": "PMC1:f3", "pmid": "111", "caption": self.CAPTION}

    def answer(self, **over):
        item = {"id": "PMC1:f3", "how_to_read": "Each dot is a summary effect and each line is its 95% confidence interval; values above 0 mean more improvement.",
                "how_to_read_quote": "Values above 0 mean more improvement than control", "shows_for_question": "Intervention A changed social communication by 0.35 (95% CI 0.23 to 0.47) across 32 studies.",
                "shows_quote": "the effect of intervention A on social communication was 0.35",
                "helps_treatment": "It lets a doctor compare intervention A with control on social communication across 32 studies.", "helps_quote": "intervention A on social communication was 0.35"}
        item.update(over)
        return {"figures": [item]}

    def test_verified_lines_are_kept(self):
        import source_figures
        trace = {}
        out = source_figures.explain_figures([self.figure()], self.answer(), self.TEXTS, trace)
        self.assertIn("summary effect", out[0]["how_to_read"])
        self.assertIn("0.35", out[0]["shows_for_question"])
        self.assertEqual(trace["explanations"], {"how_to_read_kept": 1, "shows_kept": 1, "helps_kept": 1, "dropped": 0})

    def test_the_how_to_read_line_may_rest_on_what_the_vision_model_saw_when_the_caption_is_short(self):
        import source_figures
        short = {"id": "PMC1:f3", "pmid": "111", "caption": "Figure 2. Trend in headache frequency over time.",
                 "image_check": "A line graph with the x-axis showing time points from baseline to week 12 and the y-axis showing mean headache frequency per month; lower values indicate fewer episodes."}
        answer = {"figures": [{"id": "PMC1:f3", "how_to_read": "The x-axis shows time from baseline to week 12 and the y-axis shows mean headaches per month; lower is better.",
                               "how_to_read_quote": "x-axis showing time points from baseline to week 12", "shows_for_question": "", "shows_quote": ""}]}
        out = source_figures.explain_figures([short], answer, {"111": "A study."})
        self.assertIn("x-axis shows time", out[0]["how_to_read"])
        without = source_figures.explain_figures([{**short, "image_check": None}], answer, {"111": "A study."})
        self.assertIsNone(without[0]["how_to_read"])      # with no caption words and no picture description there is nothing to rest on

    def test_a_number_that_is_not_in_the_paper_removes_the_line_but_not_the_figure(self):
        import source_figures
        out = source_figures.explain_figures([self.figure()], self.answer(shows_for_question="Intervention A changed social communication by 0.62 across 32 studies."), self.TEXTS)
        self.assertIsNone(out[0]["shows_for_question"])
        self.assertIsNotNone(out[0]["how_to_read"])      # the other line is judged on its own
        self.assertEqual(out[0]["id"], "PMC1:f3")      # the figure stays

    def test_a_quote_that_is_not_in_the_text_or_an_overclaim_removes_the_line(self):
        import source_figures
        invented = self.answer(shows_quote="intervention A cured most children in the study", how_to_read_quote="blue squares show the funding sources")
        out = source_figures.explain_figures([self.figure()], invented, self.TEXTS)
        self.assertIsNone(out[0]["shows_for_question"])
        self.assertIsNone(out[0]["how_to_read"])
        overclaim = self.answer(shows_for_question="Intervention A works: social communication improved by 0.35.")
        self.assertIsNone(source_figures.explain_figures([self.figure()], overclaim, self.TEXTS)[0]["shows_for_question"])

    def test_an_unusable_answer_leaves_the_figure_without_lines(self):
        import source_figures
        for answer in (None, "not json", {}, {"figures": []}):
            out = source_figures.explain_figures([self.figure()], answer, self.TEXTS)
            self.assertEqual(len(out), 1)
            self.assertIsNone(out[0]["how_to_read"])
            self.assertIsNone(out[0]["shows_for_question"])

    def test_the_pdf_shows_the_two_lines_only_when_they_exist(self):
        import tempfile
        from pathlib import Path
        import pdf_export
        from PIL import Image
        path = str(Path(tempfile.mkdtemp()) / "x.png")
        Image.new("RGB", (40, 30), (10, 10, 200)).save(path)
        base = {"pdf_path": path, "width": 40, "height": 30, "label": "Figure 3", "caption": self.CAPTION, "title": "T", "pmid": "111", "section": "Key studies", "reference": "1", "cited": True}
        texts, original = [], pdf_export.Paragraph

        class Spy(original):
            def __init__(self, text, *args, **kwargs):
                texts.append(text)
                super().__init__(text, *args, **kwargs)
        pdf_export.Paragraph = Spy
        try:
            report = "# Q\n\n## Key studies\n\n1. A study [1].\n"
            pdf_export.markdown_to_pdf(report, str(Path(tempfile.mkdtemp()) / "a.pdf"), "Q", figures=[{**base, "how_to_read": "Each dot is a summary effect.", "shows_for_question": "A changed it by 0.35."}])
            with_lines = " ".join(texts)
            texts.clear()
            pdf_export.markdown_to_pdf(report, str(Path(tempfile.mkdtemp()) / "b.pdf"), "Q", figures=[{**base, "how_to_read": None, "shows_for_question": None}])
            without_lines = " ".join(texts)
        finally:
            pdf_export.Paragraph = original
        self.assertIn("What this figure shows:", with_lines)
        self.assertIn("How to read this figure:", with_lines)
        self.assertNotIn("What this figure shows:", without_lines)
        self.assertIn("Figure 3. Forest plot of effects", without_lines)      # the figure is still shown


class FiguresMustBeReadableAndNotRepeated(unittest.TestCase):
    @staticmethod
    def picture(rows, row_height, width=900, gap=10):
        import io
        from PIL import Image, ImageDraw
        height = rows * (row_height + gap) + 20
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        for i in range(rows):
            y = 10 + i * (row_height + gap)
            draw.rectangle([20, y, 20 + 300 + (i * 37) % 400, y + row_height], fill="black")      # a block of ink the height of a line of text
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        return buffer.getvalue()

    def test_text_that_would_be_tiny_on_the_page_is_not_readable_and_larger_text_is(self):
        import source_figures
        big = source_figures.text_height_points(self.picture(rows=8, row_height=14, width=700, gap=30))
        tiny = source_figures.text_height_points(self.picture(rows=60, row_height=5, width=1400))
        self.assertGreaterEqual(big, source_figures.MIN_TEXT_POINTS)
        self.assertLess(tiny, source_figures.MIN_TEXT_POINTS)
        self.assertTrue(source_figures.readable_on_the_page(self.picture(rows=8, row_height=14, width=700, gap=30)))
        self.assertFalse(source_figures.readable_on_the_page(self.picture(rows=60, row_height=5, width=1400)))

    def test_a_picture_that_cannot_be_measured_is_kept(self):
        import source_figures
        self.assertIsNone(source_figures.text_height_points(self.picture(rows=1, row_height=14)))      # fewer than three text rows
        self.assertTrue(source_figures.readable_on_the_page(self.picture(rows=2, row_height=14)))
        self.assertTrue(source_figures.readable_on_the_page(b"not an image"))

    def test_a_dense_figure_is_skipped_and_one_paper_may_give_up_to_three_figures(self):
        import tempfile
        from pathlib import Path
        import source_figures
        readable, dense = self.picture(rows=8, row_height=14, width=700, gap=30), self.picture(rows=60, row_height=5, width=1400)
        pictures = {"https://x/a.png": readable, "https://x/b.png": readable, "https://x/c.png": dense, "https://x/d.png": readable}

        class Reply:
            status_code, headers = 200, {"content-type": "binary/octet-stream"}

            def __init__(self, url):
                self.content = pictures[url]
        base = OriginalFiguresAreIncludedOnlyWhenTheyQualify.fig
        found = [base(id="P1:a", image="https://x/a.png", pmid="111"), base(id="P1:b", image="https://x/b.png", pmid="111", caption="Figure 2. Forest plot of the mean difference in symptom score for sleep and anxiety outcomes of drug B in children aged four."), base(id="P2:c", image="https://x/c.png", pmid="222"),
                 base(id="P3:d", image="https://x/d.png", pmid="333")]
        rubric = lambda available: [{"id": f["id"], "helps_answer": True, "clinically_meaningful": True, "adds_beyond_report": True, "not_decorative": True, "answers_what_is_asked": True,
                                     "reason": "Forest plot of the mean difference in symptom score"} for f in available]
        trace = {}
        out = source_figures.select_figures(found, {"111", "222", "333"}, {"111", "222", "333"}, rubric, None, Path(tempfile.mkdtemp()), trace=trace, get=lambda url: Reply(url))
        self.assertEqual([f["id"] for f in out], ["P1:a", "P1:b", "P3:d"])      # two relevant figures of paper 111 are both kept; P2:c is too dense to read
        self.assertEqual(trace["unreadable_before_relevance_test"], 1)      # the dense figure never reached the model: it chooses only among figures that can be read
        self.assertEqual(trace["too_small_to_read"], 0)
        words = ["alpha omega delta", "kappa sigma theta", "lambda gamma zeta", "epsilon upsilon phi", "omicron rho tau"]
        many = [base(id=f"P4:{k}", image="https://x/a.png", pmid="444", caption=f"Figure {k}. Forest plot about {words[k]} outcomes") for k in range(5)]
        trace = {}
        own_words = lambda available: [{"id": f["id"], "helps_answer": True, "clinically_meaningful": True, "adds_beyond_report": True, "not_decorative": True, "answers_what_is_asked": True,
                                        "reason": f["caption"].split(". ", 1)[1]} for f in available]      # each figure justified with its own caption's words
        out = source_figures.select_figures(many, {"444"}, {"444"}, own_words, None, Path(tempfile.mkdtemp()), trace=trace, get=lambda url: Reply(url))
        self.assertEqual(len(out), 3)      # at most three figures of one paper
        self.assertEqual(trace["extra_figures_of_one_paper_removed"], 2)


class FiguresFromRetrievedPapersAreLabelled(unittest.TestCase):
    def test_the_records_and_the_pdf_say_when_the_report_does_not_cite_the_paper(self):
        import tempfile
        from pathlib import Path
        import pdf_export
        import source_figures
        from PIL import Image
        path = str(Path(tempfile.mkdtemp()) / "x.png")
        Image.new("RGB", (40, 30), (10, 10, 200)).save(path)
        figure = {**OriginalFiguresAreIncludedOnlyWhenTheyQualify.fig(), "path": path, "pdf_path": path, "width": 40, "height": 30, "relation": "direct"}
        report = "# Q\n\n## Key studies\n\n1. A study [1].\n"
        cited = source_figures.display_records([figure], {"1": "111"}, report)[0]
        uncited = source_figures.display_records([figure], {"1": "999"}, report)[0]
        self.assertTrue(cited["cited"])
        self.assertFalse(uncited["cited"])
        self.assertIsNone(uncited["reference"])
        texts, original = [], pdf_export.Paragraph

        class Spy(original):
            def __init__(self, text, *args, **kwargs):
                texts.append(text)
                super().__init__(text, *args, **kwargs)
        pdf_export.Paragraph = Spy
        try:
            pdf_export.markdown_to_pdf(report, str(Path(tempfile.mkdtemp()) / "a.pdf"), "Q", figures=[uncited])
        finally:
            pdf_export.Paragraph = original
        joined = " | ".join(texts)
        self.assertIn("Not discussed in the report text", joined)
        self.assertNotIn("Figure from a cited study", joined)
        self.assertNotIn("Figure from a retrieved study", joined)      # the heading label is gone


class RepeatedFiguresOfOnePaperAreDropped(unittest.TestCase):
    def test_captions_that_are_nearly_the_same_are_repeats_and_different_ones_are_not(self):
        import source_figures
        a = "Fig 3. Forest plot of summary estimates for each outcome type by intervention type when only outcomes from randomized controlled trials are included."
        b = "Fig 5. Forest plot of summary estimates for each outcome type by intervention type when only outcomes from randomized controlled trials and low risk of detection bias are included."
        c = "Fig 1. PRISMA diagram of how studies were identified, screened and included."
        self.assertGreaterEqual(source_figures.caption_similarity(a, b), source_figures.SAME_FIGURE)
        self.assertLess(source_figures.caption_similarity(a, c), 0.3)

    def test_a_repeat_from_the_same_paper_is_removed_but_the_same_caption_in_another_paper_is_kept(self):
        import tempfile
        from pathlib import Path
        import source_figures
        data = FiguresMustBeReadableAndNotRepeated.picture(rows=8, row_height=14, width=700, gap=30)

        class Reply:
            status_code, headers, content = 200, {"content-type": "binary/octet-stream"}, data
        base = OriginalFiguresAreIncludedOnlyWhenTheyQualify.fig
        caption = "Figure 3. Forest plot of summary estimates for each outcome type by intervention type when only outcomes from randomized controlled trials are included."
        found = [base(id="P1:a", pmid="111", caption=caption), base(id="P1:b", pmid="111", caption=caption.replace("Figure 3", "Figure 5") + " and low risk of bias"),
                 base(id="P1:c", pmid="111", caption="Figure 2. Network plot of the number of studies comparing each pair of treatments for the primary safety outcome."),
                 base(id="P2:a", pmid="222", caption=caption)]
        rubric = lambda available: [{"id": f["id"], "helps_answer": True, "clinically_meaningful": True, "adds_beyond_report": True, "not_decorative": True, "answers_what_is_asked": True,
                                     "reason": "Forest plot of summary estimates for each outcome type"} if "Forest" in f["caption"] else
                                    {"id": f["id"], "helps_answer": True, "clinically_meaningful": True, "adds_beyond_report": True, "not_decorative": True, "answers_what_is_asked": True,
                                     "reason": "Network plot of the number of studies comparing each pair"} for f in available]
        trace = {}
        out = source_figures.select_figures(found, {"111", "222"}, {"111", "222"}, rubric, None, Path(tempfile.mkdtemp()), trace=trace, get=lambda url: Reply())
        self.assertEqual([f["id"] for f in out], ["P1:a", "P1:c", "P2:a"])
        self.assertEqual(trace["repeats_of_one_paper_removed"], 1)


class VisionProviderIsChosenFromTheSettings(unittest.TestCase):
    def settings(self, **env):
        import os
        keys = ("VISION_PROVIDER", "VISION_MODEL", "GROQ_API_KEY")
        saved = {k: os.environ.get(k) for k in keys}
        try:
            for k in keys:
                os.environ.pop(k, None)
            os.environ.update(env)
            return qa_stream.vision_settings()
        finally:
            for k, v in saved.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v

    def test_groq_is_used_when_its_key_is_set_otherwise_mistral_and_the_model_can_be_overridden(self):
        self.assertEqual(self.settings(GROQ_API_KEY="x"), ("groq", qa_stream.GROQ_VISION_MODEL))
        self.assertEqual(self.settings(), ("mistral", qa_stream.MISTRAL_VISION_MODEL))
        self.assertEqual(self.settings(GROQ_API_KEY="x", VISION_PROVIDER="mistral"), ("mistral", qa_stream.MISTRAL_VISION_MODEL))
        self.assertEqual(self.settings(GROQ_API_KEY="x", VISION_MODEL="some/vision-model"), ("groq", "some/vision-model"))

    def test_a_missing_key_or_an_unreachable_provider_returns_none_so_the_figure_is_skipped(self):
        import os
        import tempfile
        from pathlib import Path
        from PIL import Image
        path = str(Path(tempfile.mkdtemp()) / "x.png")
        Image.new("RGB", (40, 30), (10, 10, 200)).save(path)
        saved = {k: os.environ.get(k) for k in ("VISION_PROVIDER", "GROQ_API_KEY")}
        try:
            os.environ["VISION_PROVIDER"] = "groq"
            os.environ.pop("GROQ_API_KEY", None)
            self.assertIsNone(qa_stream.vision_inspect({"path": path, "pdf_path": path, "caption": "A caption of a figure."}, "A question?"))
        finally:
            for k, v in saved.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v


class RelevanceJustificationMayParaphraseTheCaptionButNotInventFindings(unittest.TestCase):
    CAPTION = "Figure 3. Trend in HIT-6 score over time for topiramate and amitriptyline groups."

    def test_a_verbatim_run_or_a_paraphrase_made_of_caption_words_is_grounded(self):
        import source_figures
        self.assertTrue(source_figures.reason_grounded("Trend in HIT-6 score over time", self.CAPTION))
        self.assertTrue(source_figures.reason_grounded("HIT-6 score trend over time for topiramate and amitriptyline", self.CAPTION))

    def test_a_justification_with_words_the_caption_does_not_contain_is_not_grounded(self):
        import source_figures
        self.assertFalse(source_figures.reason_grounded("dramatic survival benefit in all patients", self.CAPTION))
        self.assertFalse(source_figures.reason_grounded("topiramate cured chronic migraine in every participant", self.CAPTION))
        self.assertFalse(source_figures.reason_grounded("", self.CAPTION))

    def test_a_population_the_model_invented_is_not_trusted_but_the_questions_own_words_are(self):
        import source_figures
        question = "Migraine prophylaxis 2024 evidence?"
        figs = [{"pmid": "111", "id": "a"}]
        invented = {"question_population": "pediatric migraine", "papers": []}
        trace = {}
        self.assertEqual(source_figures.apply_population_check(figs, invented, {"111": "Adults with migraine."}, question, trace), figs)
        self.assertFalse(trace["population"]["question_names_a_population"])      # "pediatric" is not in the question: nothing is rejected on population
        genuine = {"question_population": "in adults", "papers": [{"pmid": "111", "verdict": "match", "evidence_quote": "Adults with migraine were enrolled"}]}
        self.assertTrue(source_figures.apply_population_check(figs, genuine, {"111": "Adults with migraine were enrolled in the trial."}, "How is migraine managed in adults?")[0:1])

    def test_a_disease_named_as_the_population_is_ignored(self):
        import source_figures
        figs = [{"pmid": "111", "id": "a"}, {"pmid": "222", "id": "b"}]
        answer = {"question_population": "migraine", "papers": [{"pmid": "111", "verdict": "mismatch", "evidence_quote": "x"}]}
        trace = {}
        out = source_figures.apply_population_check(figs, answer, {}, "Migraine prophylaxis 2024 evidence?", trace, condition="migraine prophylaxis")
        self.assertEqual(len(out), 2)      # nothing rejected: 'migraine' is the condition, not a population
        self.assertFalse(trace["population"]["question_names_a_population"])



class TheModelMustAlsoSayTheFigureShowsWhatTheQuestionAsksAndTheReaderSeesWhy(unittest.TestCase):
    CAPTION = "Figure 3. Frequencies of behavior categories by severity in children."

    def item(self, **over):
        base = {"id": "P:1", "helps_answer": True, "clinically_meaningful": True, "adds_beyond_report": True, "not_decorative": True, "answers_what_is_asked": True,
                "reason": "Frequencies of behavior categories by severity"}
        base.update(over)
        return base

    def test_a_figure_that_does_not_show_what_the_question_asks_is_dropped(self):
        import source_figures
        available = [{"id": "P:1", "caption": self.CAPTION, "pmid": "1"}]
        self.assertEqual(len(source_figures.verify_rubric([self.item()], available)), 1)
        self.assertEqual(source_figures.verify_rubric([self.item(answers_what_is_asked=False)], available), [])      # describes how common the problem is, not how to treat it
        missing = self.item()
        del missing["answers_what_is_asked"]
        self.assertEqual(source_figures.verify_rubric([missing], available), [])      # a missing answer is not a yes

    def test_the_reader_is_told_why_a_figure_is_shown_and_only_what_was_really_checked(self):
        import source_figures
        figure = {"id": "P:1", "path": "x.png", "pmid": "1", "reason": "Frequencies of behavior categories by severity", "relation": "direct", "population_note": "children with ASD"}
        text = source_figures.display_records([figure], {}, "")[0]["why"]
        self.assertIn("Frequencies of behavior categories by severity", text)
        self.assertIn("directly relevant", text)
        self.assertIn("open licence", text)
        self.assertNotIn("vision model", text)      # no vision check happened: it is not claimed
        with_vision = source_figures.display_records([{**figure, "image_check": "A bar graph.", "relation": "indirect", "population_note": "related condition or population: x"}], {}, "")[0]["why"]
        self.assertIn("vision model confirmed", with_vision)
        self.assertIn("indirect evidence", with_vision)


class AFigureMustItselfPresentWhatTheQuestionAsks(unittest.TestCase):
    def figures(self):
        return [{"id": "P:1", "pmid": "1"}, {"id": "P:2", "pmid": "1"}, {"id": "P:3", "pmid": "2"}]

    def test_only_figures_the_model_says_present_the_asked_information_are_kept(self):
        import source_figures
        answer = {"asked_kind": "ways to manage it and how well they work", "figures": [{"id": "P:1", "presents_it": True, "what_it_presents": "outcome of a treatment versus control"},
                                                                                         {"id": "P:2", "presents_it": False, "what_it_presents": "how common the problem is"}]}
        trace = {}
        kept = source_figures.apply_strict_check(self.figures(), answer, trace)
        self.assertEqual([f["id"] for f in kept], ["P:1"])      # P:2 only describes the problem; P:3 was not judged
        self.assertEqual(kept[0]["presents"], "outcome of a treatment versus control")
        self.assertEqual(trace["strict_check"], {"asked_kind": "ways to manage it and how well they work", "checked": 3, "kept": 1})

    def test_when_the_model_gives_no_usable_answer_the_check_is_skipped_and_an_unclear_verdict_drops_the_figure(self):
        import source_figures
        for answer in (None, {}, "not json", {"figures": []}):
            trace = {}
            self.assertEqual(len(source_figures.apply_strict_check(self.figures(), answer, trace)), 3)      # a failed call does not throw the figures away
            self.assertTrue(trace["strict_check"]["could_not_run"])
        self.assertEqual(source_figures.apply_strict_check(self.figures(), {"figures": [{"id": "P:1", "presents_it": "yes"}]}), [])      # an answer that is not a clear yes drops the figure


class FiguresThatAreNotReadyWhenTheReportIsFinalAreDeliveredLate(unittest.TestCase):
    def job(self, delay):
        import threading
        import time
        box = {"figures": [{"id": "P:1", "path": "x.png", "pmid": "111", "caption": "Figure 1. Plot.", "relation": "direct"}], "trace": {}}
        thread = threading.Thread(target=lambda: time.sleep(delay), daemon=True)
        thread.start()
        qa_stream.FIGURE_JOBS["job-x"] = (thread, box)
        return {"state": {"figure_job": "job-x", "reference_numbers": {"1": "111"}}, "markdown": "## Key studies\n\n1. A study [1].\n"}

    def test_a_job_that_is_not_ready_is_kept_not_thrown_away_and_its_figures_arrive_later(self):
        final = self.job(0.6)
        try:
            self.assertEqual(qa_stream.prepare_figures(final) if False else qa_stream.collect_figures(final["state"], final["markdown"], timeout=0.05), [])
            self.assertTrue(qa_stream.figures_pending(final))      # the job is still there
            late = qa_stream.late_figures(final, timeout=5)
            self.assertEqual([r["id"] if "id" in r else r["pmid"] for r in late], ["111"])
            self.assertEqual(late[0]["section"], "Key studies")
            self.assertFalse(qa_stream.figures_pending(final))
            self.assertTrue(final["state"]["figure_trace"]["delivered_late"])
        finally:
            qa_stream.FIGURE_JOBS.pop("job-x", None)

    def test_a_job_that_is_ready_is_collected_at_once_and_a_missing_job_gives_nothing(self):
        final = self.job(0.0)
        try:
            self.assertEqual(len(qa_stream.collect_figures(final["state"], final["markdown"], timeout=2)), 1)
            self.assertFalse(qa_stream.figures_pending(final))
            self.assertEqual(qa_stream.late_figures(final, timeout=0.1), [])
        finally:
            qa_stream.FIGURE_JOBS.pop("job-x", None)


class TheModelOnlyChoosesAmongFiguresThatCanBeRead(unittest.TestCase):
    def test_a_dense_candidate_is_not_even_offered_to_the_relevance_test(self):
        import tempfile
        from pathlib import Path
        import source_figures
        readable = FiguresMustBeReadableAndNotRepeated.picture(rows=8, row_height=14, width=700, gap=30)
        dense = FiguresMustBeReadableAndNotRepeated.picture(rows=60, row_height=5, width=1400)
        pictures = {"https://x/ok.png": readable, "https://x/dense.png": dense}

        class Reply:
            status_code, headers = 200, {"content-type": "binary/octet-stream"}

            def __init__(self, url):
                self.content = pictures.get(url, readable)
        base = OriginalFiguresAreIncludedOnlyWhenTheyQualify.fig
        found = [base(id="P:dense", image="https://x/dense.png", pmid="111", caption="Figure 1. Network of treatments."), base(id="P:ok", image="https://x/ok.png", pmid="222", caption="Figure 2. Forest plot of effects.")]
        offered = []

        def rubric(available):
            offered.extend(f["id"] for f in available)
            return []
        trace = {}
        source_figures.select_figures(found, {"111", "222"}, {"111", "222"}, rubric, None, Path(tempfile.mkdtemp()), trace=trace, get=lambda url: Reply(url))
        self.assertEqual(offered, ["P:ok"])
        self.assertEqual(trace["unreadable_before_relevance_test"], 1)
        self.assertEqual(trace["figure_fates"][0]["id"], "P:dense")


class HybridSearchFusesSemanticLexicalAndSourceOrder(unittest.TestCase):
    def papers(self):
        return {"1": {"title": "Alpha beta gamma", "abstract": "unrelated words about weather"},
                "2": {"title": "Reducing symptoms with therapy", "abstract": "therapy lowers symptoms"},
                "3": {"title": "Totally other topic", "abstract": "nothing relevant here"}}

    def test_semantic_score_can_lift_a_paper_with_other_words(self):
        import hybrid_search as hs
        papers = self.papers()
        vectors = {"q": [1.0, 0.0], "Alpha beta gamma. unrelated words about weather": [0.0, 1.0], "Reducing symptoms with therapy. therapy lowers symptoms": [1.0, 0.0],
                   "Totally other topic. nothing relevant here": [0.0, 1.0]}
        original = hs.embed
        hs.embed = lambda texts, timeout=None: [vectors["q"] if i == 0 else vectors[t] for i, t in enumerate(texts)]
        try:
            trace = {}
            hs.rerank("treat symptoms", papers, trace)
        finally:
            hs.embed = original
        self.assertEqual(list(papers)[0], "2")
        self.assertIsNotNone(papers["2"]["hybrid"]["semantic"])
        self.assertIn("semantic", trace["hybrid"])

    def test_without_the_embedding_call_the_lexical_and_source_order_still_rank(self):
        import hybrid_search as hs
        papers = self.papers()
        original = hs.embed
        hs.embed = lambda texts, timeout=None: None
        try:
            trace = {}
            hs.rerank("therapy symptoms", papers, trace)
        finally:
            hs.embed = original
        self.assertLess(list(papers).index("2"), list(papers).index("3"))
        self.assertIsNone(papers["2"]["hybrid"]["semantic"])
        self.assertIn("unavailable", trace["hybrid"])

    def test_switched_off_leaves_the_order_alone(self):
        import hybrid_search as hs
        papers = self.papers()
        os.environ["HYBRID_SEARCH"] = "off"
        try:
            hs.rerank("therapy symptoms", papers, {})
        finally:
            del os.environ["HYBRID_SEARCH"]
        self.assertEqual(list(papers), ["1", "2", "3"])

    def test_bm25_prefers_the_document_with_the_query_words(self):
        import hybrid_search as hs
        scores = hs.bm25_scores("therapy symptoms", ["weather report", "therapy for symptoms", "other"])
        self.assertEqual(scores.index(max(scores)), 1)


class StreamlitFrontEndShowsTheSamePipeline(unittest.TestCase):
    def setUp(self):
        try:
            import streamlit_app
        except ImportError:
            self.skipTest("streamlit is not installed")
        self.app = streamlit_app

    def test_figure_text_has_the_lines_and_the_caption(self):
        fig = {"label": "Fig 3", "shows_for_question": "S", "helps_treatment": "H", "how_to_read": "R", "why": "W", "caption": "Original caption."}
        text = self.app.figure_markdown(fig)
        for part in ("What this figure shows:** S", "How it helps with treatment options:** H", "How to read this figure:** R"):
            self.assertIn(part, text)
        self.assertNotIn("Why this figure was chosen", text)
        self.assertEqual(self.app.caption_text(fig), "Fig 3. Original caption.")

    def test_figures_go_inside_the_section_that_cites_their_paper(self):
        md = chr(10).join(["# T", "", "## Clinical summary", "x", "", "## Key studies", "y", ""])
        figs = [{"id": "a", "section": "Key studies"}, {"id": "b", "section": "No such section"}]
        out = self.app.sections_with_figures(md, figs)
        by_head = {part.splitlines()[0]: [f["id"] for f in mine] for part, mine in out}
        self.assertEqual(by_head["## Key studies"], ["a"])
        self.assertEqual(by_head["## Clinical summary"], [])
        self.assertEqual(by_head["## Figures from the published papers"], ["b"])

    def test_credit_line_names_paper_licence_and_whether_cited(self):
        line = self.app.credit_markdown({"label": "Fig 3", "title": "A title.", "pmid": "1", "licence_label": "CC BY 4.0", "licence_url": "https://creativecommons.org/licenses/by/4.0", "reference": 4})
        for part in ("PMID 1", "CC BY 4.0", "Cited in this report as [4]"):
            self.assertIn(part, line)
        self.assertIn("Not discussed in the report text", self.app.credit_markdown({"label": "Fig 1"}))

    def test_conversations_are_saved_listed_reopened_and_deleted(self):
        import tempfile
        from pathlib import Path as P
        app = self.app
        original = app.CHAT_DIR
        app.CHAT_DIR = P(tempfile.mkdtemp()) / "chats"
        try:
            turns = [{"question": "raw question with a name", "saved_question": "scrubbed question " + "x" * 80, "markdown": "## A\nbody", "figures": []}]
            app.save_chat("chat-1", turns)
            app.save_chat("chat-2", [{"question": "second", "markdown": "m"}])
            listed = app.list_chats()
            self.assertEqual({c["id"] for c in listed}, {"chat-1", "chat-2"})
            self.assertTrue(next(c for c in listed if c["id"] == "chat-1")["title"].startswith("scrubbed question"))
            self.assertLessEqual(len(next(c for c in listed if c["id"] == "chat-1")["title"]), app.TITLE_CHARS + 1)
            reopened = app.load_chat("chat-1")
            self.assertEqual(reopened[0]["question"], turns[0]["saved_question"])      # the saved copy holds only the scrubbed question
            self.assertNotIn("raw question", json.dumps(reopened))
            app.delete_chat("chat-1")
            self.assertEqual([c["id"] for c in app.list_chats()], ["chat-2"])
        finally:
            app.CHAT_DIR = original


class CrossEncoderSecondStage(unittest.TestCase):
    def papers(self):
        return {str(i): {"title": f"paper {i}", "abstract": "text"} for i in range(1, 6)}

    def test_it_reorders_the_first_papers_and_keeps_the_rest_after_them(self):
        import hybrid_search as hs
        papers = self.papers()
        trace = {}
        hs.cross_rerank("q", papers, trace, top=3, scorer=lambda q, texts: [0.1, 0.2, 9.0])      # the 3rd paper is judged best by the cross-encoder
        self.assertEqual(list(papers)[0], "3")
        self.assertEqual(list(papers)[3:], ["4", "5"])
        self.assertEqual(papers["3"]["hybrid"]["cross"], 9.0)
        self.assertIn("reranked the first 3", trace["cross_encoder"])

    def test_the_first_stage_order_still_counts(self):
        import hybrid_search as hs
        papers = self.papers()
        hs.cross_rerank("q", papers, {}, top=5, scorer=lambda q, texts: [1.0, 1.01, 1.02, 1.03, 1.04])      # almost no difference between papers: the first-stage order must not be thrown away
        self.assertEqual(list(papers)[0], "1")

    def test_off_or_unavailable_leaves_the_order_alone(self):
        import hybrid_search as hs
        papers = self.papers()
        os.environ["CROSS_ENCODER"] = "off"
        try:
            hs.cross_rerank("q", papers, {}, scorer=lambda q, texts: [0] * len(texts))
        finally:
            del os.environ["CROSS_ENCODER"]
        self.assertEqual(list(papers), ["1", "2", "3", "4", "5"])
        trace = {}
        hs.cross_rerank("q", papers, trace, scorer=lambda q, texts: 1 / 0)
        self.assertIn("skipped", trace["cross_encoder"])
        self.assertEqual(list(papers), ["1", "2", "3", "4", "5"])


class QuestionIsUnderstood(unittest.TestCase):
    QUESTION = "A 7 year old boy with condition X on drug Y has poor sleep. What are the options?"

    def test_only_what_the_question_supports_is_kept_as_understanding(self):
        plan = {"asked": "options for sleep", "purpose": "choose a treatment", "info_needed": ["treatment_options", "made_up", "safety"], "answer_format": "weird",
                "case_facts": ["7 year old boy", "has a fever", "on drug Y"]}
        got = qa_stream.verified_understanding(plan, self.QUESTION)
        self.assertEqual(got["case_facts"], ["7 year old boy", "on drug Y"])      # "has a fever" is not in the question
        self.assertEqual(got["info_needed"], ["treatment_options", "safety"])
        self.assertEqual(got["answer_format"], "summary")


class SkillCatalogForTheQuickCall(unittest.TestCase):
    QUESTION = "A 3 year old child has global delay and regression of milestones. What is the work up and management?"
    PLAN = {"condition": "global delay regression", "understanding": {"case_facts": ["3 year old child", "regression of milestones"]}}

    def test_the_catalog_lists_only_names_and_descriptions_of_the_retrieval_skills(self):
        catalog = qa_stream.retrieval_catalog()
        for name in qa_stream.EXPANDER_SKILLS:
            self.assertIn(f"- {name}:", catalog)
        self.assertNotIn("## Step", catalog)      # the full skill files are not in the catalog

    def test_every_chosen_skill_has_a_query_transformation_section(self):
        for name in qa_stream.EXPANDER_SKILLS:
            self.assertIn("Reply with ONE JSON object", qa_stream.skill_section(name, "Query transformation by intent"))
