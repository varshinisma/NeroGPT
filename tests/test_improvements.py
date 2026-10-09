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
    def test_topics_come_from_config_with_neurodevelopmental_override(self):
        autism = [t["id"] for t in guidelines.topics_for("How to treat autism in children?")]
        other = [t["id"] for t in guidelines.topics_for("What is first-line therapy for hypertension?")]
        for needed in ("speech", "ot", "behavioural", "school", "sleep", "anxiety", "adhd", "gi", "feeding", "assessment"):
            self.assertIn(needed, autism)
        self.assertNotIn("speech", other)
        self.assertIn("first_line", other)

    def test_only_allow_listed_organisations_are_used_with_verbatim_excerpt(self):
        res = guidelines.find_guidelines("How to treat autism in children?", ["autism"], fake_search({"medications": [OFFSITE, GOOD]}))
        rows = {r["topic"]: r for r in res["results"]}
        self.assertEqual(rows["medications"]["organisation"], "NICE (UK)")
        self.assertEqual(rows["medications"]["year"], "2021")
        self.assertIn(rows["medications"]["excerpt"], GOOD["snippet"])       # verbatim (here: the whole snippet)
        self.assertTrue(rows["sleep"].get("none"))                       # nothing retrieved for this topic
        md = guidelines.established_care_markdown(res["results"], True)
        self.assertIn("No guideline retrieved", md)
        self.assertNotIn("random-blog", md)

    def test_off_topic_guideline_is_not_used(self):
        wrong = {"title": "Hypertension guideline", "url": "https://www.nice.org.uk/guidance/ng136", "snippet": "This guideline covers diagnosing and managing hypertension in adults."}
        res = guidelines.find_guidelines("How to treat autism in children?", ["autism"], fake_search({"medications": [wrong]}))
        self.assertTrue(all(not r.get("title") for r in res["results"]))

    def test_failed_search_is_reported_unavailable_not_none(self):
        res = guidelines.find_guidelines("How to treat autism in children?", ["autism"], lambda q: "Web search failed: timeout")
        self.assertIn("error", res)
        md = guidelines.established_care_markdown([], False)
        self.assertIn("search unavailable", md)
        self.assertNotIn("No guideline retrieved", md)

    def test_verifier_rejects_an_established_care_row_without_a_retrieved_guideline(self):
        s = make_state()
        s["guidelines"] = [{"url": GOOD["url"], "title": GOOD["title"], "organisation": "NICE (UK)", "excerpt": GOOD["snippet"], "year": "2021"}]
        table = ("### Established care\n\n| Topic | Guideline | Organisation | Year | Excerpt |\n|---|---|---|---|---|\n"
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
    def test_pubmed_and_pmc_articles_are_not_guidelines(self):
        cfg = guidelines.load_config()
        self.assertIsNone(guidelines.organisation_for("https://pmc.ncbi.nlm.nih.gov/articles/PMC10422951/", cfg))
        self.assertIsNone(guidelines.organisation_for("https://pubmed.ncbi.nlm.nih.gov/123456/", cfg))
        self.assertEqual(guidelines.organisation_for("https://www.nichd.nih.gov/health/topics/autism", cfg), "US National Institute of Child Health and Human Development")

    def test_a_rate_limited_topic_is_retried_once(self):
        calls = []

        def search(q):
            calls.append(q)
            return "Web search failed: rate limit" if len(calls) == 1 else json.dumps([GOOD])
        res = guidelines.find_guidelines("autism", ["autism"], search, guidelines.load_config() | {"default": {"topics": [{"id": "t", "label": "T", "query": "q"}]}, "overrides": {}})
        self.assertEqual(len(calls), 2)
        self.assertEqual(res["results"][0]["organisation"], "NICE (UK)")


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
        out = report.clean_model_key_studies("### Key studies\n\n1. **Restoy et al.\n", [study("100001")], None)
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
        self.assertIn("| Drug | Indication | Dose / route (if sourced) | Approval status / date | Key evidence |", md)      # the specification's six columns
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
                "\n#### Recently approved or updated (regulator announcements retrieved)\n\n| Date | Type | Announcement | Page says | Link |\n|---|---|---|---|---|\n"
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
        header = "| Drug | Indication | Dose / route (if sourced) | Approval status / date | Key evidence |"
        self.assertEqual(md.count(header), 1)      # six columns in the treatment table
        self.assertEqual(md.count("| Drug | Indication | Key evidence |"), 1)      # only three in the other-drugs table
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
        self.assertEqual(topics, {"medications", "sleep", "anxiety", "adhd"})
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
        self.assertIn("| Drug | Indication | Dose / route (if sourced) | Approval status / date | Key evidence |", md)
        self.assertIn("| Drug | Indication | Key evidence |", md)
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

    def test_order_is_relevance_then_date_and_never_study_design(self):
        old_review = study("700011", year="2015", types=("Systematic Review",))
        old_review["triage"] = {"relevance": "indirect", "role": "established", "population": "", "evidence": "x"}
        new_case = study("700012", year="2026", types=("Case Reports",))
        new_case["triage"] = {"relevance": "direct", "role": "emerging", "population": "", "evidence": "x"}
        mid = study("700013", year="2022", types=("Randomized Controlled Trial",))
        mid["triage"] = {"relevance": "direct", "role": "experimental", "population": "", "evidence": "x"}
        order = [p["pmid"] for p in qa_stream.ordered_papers({p["pmid"]: p for p in (old_review, new_case, mid)}, set())]
        self.assertEqual(order, ["700012", "700013", "700011"])

    def test_without_a_triage_decision_no_keyword_rule_removes_a_paper(self):
        p = study("700021", year="2025")
        p["title"] = "A study of something that shares no word with the question"
        self.assertTrue(qa_stream.usable(p, {"zzzzz"}))
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
                return {"papers": [{"id": p["id"], "relevance": "direct", "role": "emerging", "population": "", "sentence": 1} for p in payload["papers"]]}
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
        self.assertTrue(qa_stream.usable(s["papers"]["111111"], set()))
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

    def test_a_sentence_reporting_a_study_result_is_not_a_drug_used_in_practice(self):
        study = self.item(drug="Atomoxetine", sentence=2, statement_type="study_result", indication="improved core symptoms in children", category="Off-label")
        rows, rejected = landscape.verify_drug_decisions([study, self.item(drug="Zzzzol", statement_type=None)], self.SRC, make_state())
        self.assertEqual(rows, [])
        self.assertIn("study_result", rejected[0][1])

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
        got = [p["pmid"] for p in qa_stream.ordered_papers({"710001": a, "710002": b, "710003": c}, set())]
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
