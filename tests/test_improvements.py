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
        res = guidelines.find_guidelines("How to treat autism in children?", ["autism"], fake_search({"speech": [OFFSITE, GOOD]}))
        rows = {r["topic"]: r for r in res["results"]}
        self.assertEqual(rows["speech"]["organisation"], "NICE (UK)")
        self.assertEqual(rows["speech"]["year"], "2021")
        self.assertIn(rows["speech"]["excerpt"], GOOD["snippet"])       # verbatim (here: the whole snippet)
        self.assertTrue(rows["sleep"].get("none"))                       # nothing retrieved for this topic
        md = guidelines.established_care_markdown(res["results"], True)
        self.assertIn("No guideline retrieved", md)
        self.assertNotIn("random-blog", md)

    def test_off_topic_guideline_is_not_used(self):
        wrong = {"title": "Hypertension guideline", "url": "https://www.nice.org.uk/guidance/ng136", "snippet": "This guideline covers diagnosing and managing hypertension in adults."}
        res = guidelines.find_guidelines("How to treat autism in children?", ["autism"], fake_search({"speech": [wrong]}))
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


class UsedInPractice(unittest.TestCase):
    SOURCES = {
        "111111": ("Pharmacotherapy of irritability: risperidone is widely used, and melatonin is commonly prescribed for sleep problems in children with the condition.", "PMID 111111 (Smith et al.)"),
        "222222": ("Dexmedetomidine is being tested in a phase 2 trial in the condition.", "PMID 222222 (Chen et al.)"),
        "333333": ("Prenatal exposure to valproate was associated with the condition.", "PMID 333333 (Wu et al.)"),
    }

    def test_only_drugs_the_source_says_are_used_survive(self):
        s = make_state()
        items = [{"drug": "melatonin", "source": "111111"}, {"drug": "dexmedetomidine", "source": "222222"}, {"drug": "valproate", "source": "333333"},
                 {"drug": "invented-drug", "source": "111111"}, {"drug": "melatonin", "source": "999999"}, {"drug": "risperidone", "source": "111111"}]
        rows = landscape.used_in_practice_rows(items, self.SOURCES, s)
        self.assertEqual([r["drug"] for r in rows], ["melatonin"])       # risperidone already has an FDA-labelled row; the others fail the checks
        self.assertIn("commonly prescribed", rows[0]["excerpt"])

    def test_markdown_has_no_dose_and_names_the_source(self):
        rows = landscape.used_in_practice_rows([{"drug": "melatonin", "source": "111111"}], self.SOURCES, make_state())
        md = landscape.used_in_practice_markdown(rows)
        self.assertIn("PMID 111111", md)
        self.assertIn("Not FDA-labelled for this condition", md)
        self.assertIn("Used for (as the source states)", md)
        self.assertIn("sleep problems", md)
        self.assertNotRegex(md, r"\d+\s?mg")
        self.assertIn("None verified in this run", landscape.used_in_practice_markdown([]))

    def test_a_row_passes_the_verifier(self):
        s = make_state()
        s["papers"]["111111"]["abstract"] = self.SOURCES["111111"][0]
        rows = landscape.used_in_practice_rows([{"drug": "melatonin", "source": "111111"}], self.SOURCES, s)
        text = BASE + landscape.used_in_practice_markdown(rows) + "\n"
        self.assertEqual([i for i in verification.find_issues(text, s) if i["severity"] in ("high", "medium")], [])


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
        s["regulatory"] = [{"generic": "Risperidone", "matches_condition": True, "indication_excerpt": self.RIS, "url": "https://dailymed.nlm.nih.gov/x"}]
        s["source_status"] = {"fda_fast": {"retrieval_status": "success", "evidence_status": "found", "error": None}}
        md = landscape.top_drugs_markdown(s)
        self.assertIn("[DailyMed](https://dailymed.nlm.nih.gov/x)", md)
        self.assertNotIn("...", md)

    def test_a_long_unmatched_excerpt_is_cut_at_a_word_with_a_marker(self):
        out = landscape.condition_clause("word " * 100, ["autis"])
        self.assertTrue(out.endswith(" ..."))
        self.assertLessEqual(len(out), 205)


class UsedInPracticeSelection(unittest.TestCase):
    def test_model_only_sees_sentences_that_say_a_drug_is_used(self):
        text = ("Atomoxetine improved scores compared with placebo in a randomized trial. Melatonin is commonly used to address sleep problems. "
                "Bumetanide is being tested in a phase 3 trial.")
        cand = landscape.candidate_sentences(text)
        self.assertIn("Melatonin is commonly used", cand)
        self.assertNotIn("Atomoxetine", cand)
        self.assertNotIn("Bumetanide", cand)

    def test_search_fragment_marker_never_fuses_two_fragments(self):
        cand = landscape.candidate_sentences("Fluoxetine has shown some effectiveness in [...] Quetiapine is commonly used in clinical practice for irritability.")
        self.assertTrue(cand.startswith("Quetiapine"))

    def test_excerpt_keeps_both_the_drug_and_the_use_wording(self):
        s = ("Beside the evidence of efficacy for the labelled drugs there are few studies of other agents such as quetiapine and ziprasidone, although these are "
             "frequently used off-label in clinical practice for irritability, with variable results across many small samples in the literature and many further words here.")
        out = landscape.excerpt_around(s, "quetiapine")
        self.assertIn("quetiapine", out)
        self.assertIn("used off-label", out)
        self.assertIn(out.replace("... ", "").replace(" ...", "").split(" ")[0], s)


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
        md = landscape.regulatory_markdown(s)
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


class TrialDrugRows(unittest.TestCase):
    def state(self):
        s = make_state()
        s["trials_by_id"] = {
            "NCT01": {"nct_id": "NCT01", "phase": ["PHASE3"], "recruitment_status": "RECRUITING", "last_update": "2026-09-01", "url": "https://clinicaltrials.gov/study/NCT01",
                      "drug_interventions": ["Lumateperone high dose", "Lumateperone low dose", "Placebo"]},
            "NCT02": {"nct_id": "NCT02", "phase": ["PHASE2"], "recruitment_status": "COMPLETED", "last_update": "2026-01-01", "url": "https://clinicaltrials.gov/study/NCT02",
                      "drug_interventions": ["Sertraline", "Placebo", "Comparison of Risperidone and Aripiprazole"]},
            "NCT03": {"nct_id": "NCT03", "phase": ["PHASE1"], "recruitment_status": "NOT_YET_RECRUITING", "last_update": "2026-02-01", "url": "https://clinicaltrials.gov/study/NCT03",
                      "drug_interventions": ["Risperidone"]},
        }
        for trial in s["trials_by_id"].values():
            trial.setdefault("conditions", ["Condition X"])      # the registry records which condition each trial is about
        return s

    def test_names_come_from_the_registry_without_placebo_comparisons_or_non_drugs(self):
        rows = landscape.trial_drug_rows(self.state(), set())
        names = [r["drug"] for r in rows]
        self.assertEqual(names[0], "Lumateperone")                         # the latest phase first, dose labels removed
        self.assertEqual(names.count("Lumateperone"), 1)
        for bad in ("Placebo", "Comparison of Risperidone and Aripiprazole"):
            self.assertNotIn(bad, names)
        self.assertIn("Sertraline", names)

    def test_drugs_that_already_have_a_row_are_not_repeated(self):
        names = [r["drug"].lower() for r in landscape.trial_drug_rows(self.state(), {"Risperidone", "sertraline"})]
        self.assertNotIn("risperidone", names)
        self.assertNotIn("sertraline", names)

    def test_trial_rows_are_labelled_as_being_tested_and_link_to_the_registry(self):
        md = landscape.used_in_practice_markdown([], landscape.trial_drug_rows(self.state(), set()))
        self.assertIn("### 2.3 Drugs being tested in clinical trials (not established treatments)", md)
        self.assertIn("[NCT01](https://clinicaltrials.gov/study/NCT01)", md)
        self.assertIn("Phase 3", md)
        self.assertIn("Condition X", md)
        self.assertNotIn("None found", md.split("### 2.3")[1])      # the trial table is not empty
        self.assertIn("A registry entry does not show that a drug works", md)

    def test_trial_rows_pass_the_verifier(self):
        s = self.state()
        s["trials_by_id"]["NCT01"].update({"title": "T", "interventions": ["Lumateperone"]})
        md = landscape.used_in_practice_markdown([], landscape.trial_drug_rows(s, set()))
        issues = [i for i in verification.find_issues(BASE + md + "\n", s) if i["severity"] in ("high", "medium") and i["kind"] != "unretrieved_nct"]
        self.assertEqual(issues, [])

    def test_clinical_tools_marks_drug_interventions_by_registered_type(self):
        import clinical_tools
        self.assertIn("drug_only", clinical_tools.search_clinical_trials.__code__.co_varnames)


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
        md = landscape.used_in_practice_markdown(rows, [{"drug": "NTI164", "nct": "NCT07257939", "phase": "3", "status": "RECRUITING", "updated": "2026-09-30",
                                                           "url": "https://clinicaltrials.gov/study/NCT07257939", "fda_label": None, "conditions": "Autism", "primary": "ABC score", "ages": "from 6 Years"}])
        self.assertNotIn("FDA label of this drug", md)
        self.assertNotIn("dailymed", md.lower())
        header = md.split("\n\n")[-1].splitlines()[0] if "\n\n*Rows" not in md else md.split("\n\n*Rows")[0].split("\n\n")[-1].splitlines()[0]
        self.assertEqual(header.count("|"), 7)      # six columns
        safety = landscape.safety_alerts_markdown([{"drug": "Quetiapine", "url": "https://dailymed.nlm.nih.gov/q", "boxed_warning": "Increased mortality in elderly patients."}])
        self.assertIn("https://dailymed.nlm.nih.gov/q", safety)      # where the boxed warning came from


class AboutTheCondition(unittest.TestCase):
    def test_a_trial_that_is_not_about_the_condition_gives_no_trial_drug_row(self):
        s = make_state()
        s["core_stems"] = ["autis"]
        s["trials_by_id"] = {
            "NCT10": {"nct_id": "NCT10", "title": "Lumateperone for irritability", "conditions": ["Autism Spectrum Disorder"], "phase": ["PHASE3"], "recruitment_status": "RECRUITING",
                      "last_update": "2026-09-01", "url": "https://clinicaltrials.gov/study/NCT10", "drug_interventions": ["Lumateperone"]},
            "NCT11": {"nct_id": "NCT11", "title": "A trial of Drugzol in schizophrenia", "conditions": ["Schizophrenia"], "phase": ["PHASE3"], "recruitment_status": "RECRUITING",
                      "last_update": "2026-09-01", "url": "https://clinicaltrials.gov/study/NCT11", "drug_interventions": ["Drugzol"]}}
        names = [r["drug"] for r in landscape.trial_drug_rows(s, set())]
        self.assertEqual(names, ["Lumateperone"])

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


class PurposeAndSideEffects(unittest.TestCase):
    MELATONIN = "Melatonin is commonly used to address sleep problems in ASD and appears to be effective."
    LIST = ("Medications, including FDA-approved antipsychotics such as risperidone and aripiprazole, are used to manage comorbid irritability and aggression, stimulants and "
            "non-stimulants (e.g., methylphenidate, atomoxetine) to treat ADHD-like symptoms, and melatonin for sleep disturbances.")

    def test_purpose_is_the_phrase_the_source_gives_for_that_drug(self):
        self.assertEqual(landscape.purpose_phrase(self.MELATONIN, "melatonin"), "sleep problems in ASD")
        self.assertEqual(landscape.purpose_phrase(self.LIST, "methylphenidate"), "ADHD-like symptoms")      # not the antipsychotics' purpose, not melatonin's
        self.assertEqual(landscape.purpose_phrase(self.LIST, "melatonin"), "sleep disturbances")

    def test_a_drug_named_without_a_purpose_is_not_listed(self):
        sentence = "The first-line drugs are selective serotonin reuptake inhibitors (SSRIs) such as sertraline, fluoxetine and fluvoxamine."
        self.assertIsNone(landscape.purpose_phrase(sentence, "sertraline"))
        rows = landscape.used_in_practice_rows([{"drug": "sertraline", "source": "9"}], {"9": (sentence, "[S](https://x.org)")}, make_state())
        self.assertEqual(rows, [])

    def test_side_effect_sentences_are_not_taken_as_a_use(self):
        text = "Quetiapine is commonly used but weight gain and sedation limit its use in children."
        self.assertEqual(landscape.candidate_sentences(text), "")
        rows = landscape.used_in_practice_rows([{"drug": "quetiapine", "source": "9"}], {"9": (text, "[S](https://x.org)")}, make_state())
        self.assertEqual(rows, [])

    def test_row_carries_purpose_and_population(self):
        text = "Quetiapine and ziprasidone are commonly used in clinical practice in the treatment of irritability and aggressive behavior in children and adults with ASD."
        rows = landscape.used_in_practice_rows([{"drug": "quetiapine", "source": "9"}], {"9": (text, "[S](https://x.org)")}, make_state())
        self.assertEqual(rows[0]["purpose"], "irritability and aggressive behavior in children and adults with ASD")
        self.assertEqual(rows[0]["population"], "children and adults with ASD")


class SafetyAndRegulatorUpdates(unittest.TestCase):
    def test_boxed_warning_is_shown_in_the_labels_words_with_a_link(self):
        labels = [{"drug": "Sertraline", "url": "https://dailymed.nlm.nih.gov/s", "boxed_warning": "BOXED WARNING Suicidality and Antidepressant Drugs Antidepressants increased the risk of suicidal thinking in children."},
                  {"drug": "Melatonin", "url": "https://dailymed.nlm.nih.gov/m", "boxed_warning": None}]
        md = landscape.safety_alerts_markdown(labels)
        self.assertIn("Suicidality and Antidepressant Drugs", md)
        self.assertNotIn("BOXED WARNING Suicidality", md)
        self.assertIn("[Sertraline](https://dailymed.nlm.nih.gov/s)", md)
        self.assertNotIn("Melatonin", md)

    def test_no_boxed_warning_says_so_without_claiming_safety(self):
        md = landscape.safety_alerts_markdown([{"drug": "Melatonin", "url": "https://x.org/m", "boxed_warning": None}])
        self.assertIn("No boxed warning was found", md)
        self.assertIn("not a complete safety review", md)

    def test_only_regulator_pages_about_the_condition_are_listed(self):
        s = make_state()
        s["core_stems"] = ["autis"]
        s["web"] = [
            {"source": "web_regulatory", "title": "FDA Takes Action to Make a Treatment Available for Autism Symptoms", "url": "https://www.fda.gov/news-events/press-announcements/fda-takes-action",
             "snippet": "On September 22, 2025, the FDA approved a label update. The agency said the drug may help some autistic children."},
            {"source": "web_regulatory", "title": "FDA Approves Drug for Alzheimer's", "url": "https://www.fda.gov/news-events/press-announcements/alz", "snippet": "A new Alzheimer's drug."},
            {"source": "web_regulatory", "title": "Autism written request", "url": "https://www.fda.gov/media/88437/download", "snippet": "Autism request document."},
            {"source": "web_regulatory", "title": "Autism blog", "url": "https://random-blog.example.com/autism", "snippet": "Autism news."},
            {"source": "web_standard", "title": "Autism guide", "url": "https://www.fda.gov/consumers/autism", "snippet": "Autism."}]
        md = landscape.regulatory_updates_markdown(s)
        self.assertIn("FDA Takes Action", md)
        self.assertIn("September 22, 2025", md)
        self.assertIn("Approval or regulatory action", md)
        for left_out in ("Alzheimer", "written request", "random-blog", "consumers/autism"):
            self.assertNotIn(left_out.lower(), md.lower())

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
        self.assertNotIn("(Smith et al.", out.split("## References")[0])
        refs = out.split("## References")[1]
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
        self.assertIn("### Limitations of this report", md)
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
