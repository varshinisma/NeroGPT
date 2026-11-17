"""Offline tests of the Tavily-sources mode (SOURCES_VIA_TAVILY=on): Tavily finds the records, the sources' own APIs read them by identifier, and the source's own search answers when Tavily gives nothing.
No network, no key: the Tavily search and the identifier lookups are replaced by fakes."""
import asyncio
import importlib
import json
import os
import sys
import unittest
from datetime import date
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import async_core  # noqa: E402
import async_tavily_sources as via  # noqa: E402
import async_web  # noqa: E402

THIS_YEAR = date.today().year


def article(pmid, year, title, abstract, author="Smith", pmc=None, kind="Randomized Controlled Trial"):
    ids = f'<ArticleId IdType="doi">10.1/{pmid}</ArticleId>' + (f'<ArticleId IdType="pmc">{pmc}</ArticleId>' if pmc else "")
    return (f"<PubmedArticle><MedlineCitation><PMID>{pmid}</PMID><Article><Journal><Title>Journal of Tests</Title><JournalIssue><PubDate><Year>{year}</Year></PubDate></JournalIssue></Journal>"
            f"<ArticleTitle>{title}</ArticleTitle><Abstract><AbstractText Label=\"RESULTS\">{abstract}</AbstractText></Abstract><AuthorList><Author><LastName>{author}</LastName><Initials>J</Initials></Author></AuthorList>"
            f"<PublicationTypeList><PublicationType>{kind}</PublicationType></PublicationTypeList></Article></MedlineCitation><PubmedData><ArticleIdList>{ids}</ArticleIdList></PubmedData></PubmedArticle>")


def xml(*articles):
    return ("<PubmedArticleSet>" + "".join(articles) + "</PubmedArticleSet>").encode()


def page(pmid, title="A page title - PubMed", snippet="excerpt of the page"):
    return {"title": title, "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", "snippet": snippet}


class Env:
    """Keys and switches of one test, restored afterwards."""

    def __init__(self, **values):
        self.values = {"TAVILY_API_KEY": "test-key", "TAVILY_SOURCES_SEMANTIC": "on", "TAVILY_SOURCES_ENRICH": "on", "TAVILY_SOURCES_FALLBACK": "on", **values}

    def __enter__(self):
        self.patch = mock.patch.dict(os.environ, self.values)
        self.patch.start()
        self.token = async_web.QUESTION.set("what is the treatment of depression in adolescents")
        self.saved = dict(via.DIRECT)
        return self

    def __exit__(self, *exc):
        async_web.QUESTION.reset(self.token)
        via.DIRECT.clear()
        via.DIRECT.update(self.saved)
        self.patch.stop()


def fake_semantic(items, calls=None):
    async def run(query, count, chars, domains):
        if calls is not None:
            calls.append({"query": query, "count": count, "chars": chars, "domains": domains})
        return items
    return run


def fake_get(body, calls=None, fail=False):
    async def run(url, params=None):
        if calls is not None:
            calls.append((url, params))
        if fail:
            raise RuntimeError("lookup failed")
        return body
    return run


def run(coro):
    return asyncio.run(coro)


class PubMedThroughTavily(unittest.TestCase):
    def search(self, items, body, **kw):
        calls, gets = [], []
        with Env(**kw.pop("env", {})):
            with mock.patch.object(async_web, "_tavily_semantic", fake_semantic(items, calls)), mock.patch.object(async_core, "aget", fake_get(body, gets)):
                out = json.loads(run(via.asearch_pubmed_quick("(depression[tiab]) AND (adolescents[tiab] OR teen*)", **kw)))
        return out, calls, gets

    def test_tavily_finds_and_pubmed_reads_the_records(self):
        body = xml(article("11111111", THIS_YEAR, "Full title one", "Real abstract one.", pmc="PMC1234567"), article("22222222", THIS_YEAR, "Full title two", "Real abstract two.", author="Jones"))
        out, calls, gets = self.search([page("11111111"), page("22222222"), {"title": "x", "url": "https://example.org/a", "snippet": "not pubmed"}], body, max_results=5)
        first = out["results"][0]
        self.assertEqual(first["pmcid"], "PMC1234567")      # figures and full text need this
        self.assertEqual(first["authors"], ["Smith J"])      # the Key studies need the first author
        self.assertIn("Randomized Controlled Trial", first["study_type"])      # and the design
        self.assertEqual(first["journal"], "Journal of Tests")
        self.assertIn("Real abstract one", first["abstract_excerpt"])      # the real abstract, not the page excerpt
        self.assertEqual([r["pmid"] for r in out["results"]], ["11111111", "22222222"])      # the page that is not PubMed is ignored
        self.assertEqual(calls[0]["domains"], via.PUBMED)      # Tavily searches PubMed's own domain
        self.assertGreaterEqual(calls[0]["chars"], 700)
        self.assertEqual(gets[0][1]["id"], "11111111,22222222")      # one lookup by identifier for all of them
        self.assertEqual(gets[0][1]["db"], "pubmed")

    def test_the_years_limit_is_applied_to_the_papers_whose_record_was_read(self):
        body = xml(article("11111111", THIS_YEAR, "New", "a."), article("22222222", THIS_YEAR - 12, "Old", "b."))
        out, _, _ = self.search([page("11111111"), page("22222222")], body, max_results=5, years_back=3)
        self.assertEqual([r["pmid"] for r in out["results"]], ["11111111"])

    def test_sort_by_date_lists_the_newest_first(self):
        body = xml(article("11111111", THIS_YEAR - 1, "A", "a."), article("22222222", THIS_YEAR, "B", "b."))
        out, _, _ = self.search([page("11111111"), page("22222222")], body, max_results=5, years_back=5, sort="date")
        self.assertEqual([r["pmid"] for r in out["results"]], ["22222222", "11111111"])

    def test_a_failed_lookup_keeps_the_tavily_records(self):
        calls = []
        with Env():
            with mock.patch.object(async_web, "_tavily_semantic", fake_semantic([page("11111111", snippet="Excerpt from 2024.")], calls)), mock.patch.object(async_core, "aget", fake_get(b"", fail=True)):
                out = json.loads(run(via.asearch_pubmed_quick("depression adolescents treatment")))
        self.assertEqual(len(out["results"]), 1)
        self.assertIsNone(out["results"][0]["pmcid"])      # only what the page showed
        self.assertIn("Excerpt", out["results"][0]["abstract_excerpt"])

    def test_enrich_off_makes_no_lookup(self):
        out, _, gets = self.search([page("11111111")], xml(), max_results=5, env={"TAVILY_SOURCES_ENRICH": "off"})
        self.assertEqual(gets, [])
        self.assertEqual(len(out["results"]), 1)

    def test_nothing_from_tavily_uses_the_sources_own_search(self):
        async def direct(query, max_results, years_back, sort):
            return json.dumps({"results": [{"pmid": "99999999"}], "via": "direct"})
        with Env():
            via.DIRECT["asearch_pubmed_quick"] = direct
            with mock.patch.object(async_web, "_tavily_semantic", fake_semantic([])):
                out = json.loads(run(via.asearch_pubmed_quick("depression adolescents treatment")))
        self.assertEqual(out["via"], "direct")

    def test_fallback_off_says_no_results(self):
        async def direct(*a, **k):
            raise AssertionError("must not be called")
        with Env(TAVILY_SOURCES_FALLBACK="off"):
            via.DIRECT["asearch_pubmed_quick"] = direct
            with mock.patch.object(async_web, "_tavily_semantic", fake_semantic([])):
                out = json.loads(run(via.asearch_pubmed_quick("depression adolescents treatment")))
        self.assertEqual(out["results"], [])
        self.assertIn("No results through Tavily", out["note"])

    def test_a_full_text_search_of_europe_pmc_is_not_a_web_search(self):
        async def direct(query, max_results, years_back, sort, fulltext=False):
            return json.dumps({"fulltext": fulltext})
        with Env():
            via.DIRECT["asearch_europepmc_quick"] = direct
            with mock.patch.object(async_web, "_tavily_semantic", fake_semantic([page("11111111")])):
                out = json.loads(run(via.asearch_europepmc_quick("depression adolescents", fulltext=True)))
        self.assertTrue(out["fulltext"])

    def test_no_key_means_no_tavily_call(self):
        with Env(TAVILY_API_KEY=""):
            self.assertEqual(run(via._search("anything at all here", via.PUBMED, 5)), [])

    def test_the_query_sent_to_tavily_is_plain_words_short_enough_for_the_sentence_form(self):
        long_boolean = "(" + " OR ".join(f"term{i}[tiab]" for i in range(60)) + ") AND depression"
        self.assertLessEqual(len(via._focus(long_boolean)), 110)
        self.assertNotIn("[", via._focus(long_boolean))


def trial_study(nct, status, drug=True):
    return {"protocolSection": {"identificationModule": {"nctId": nct, "briefTitle": f"Trial {nct}"}, "statusModule": {"overallStatus": status}, "designModule": {"phases": ["PHASE3"], "studyType": "INTERVENTIONAL"},
                                "armsInterventionsModule": {"interventions": [{"name": "Drug X", "type": "DRUG" if drug else "BEHAVIORAL"}]}}}


class TrialsThroughTavily(unittest.TestCase):
    def search(self, items, studies, **kw):
        gets = []

        async def get_json(url, params=None):
            gets.append((url, params))
            return {"studies": studies}
        with Env():
            with mock.patch.object(async_web, "_tavily_semantic", fake_semantic(items)), mock.patch.object(async_core, "aget_json", get_json):
                return json.loads(run(via.asearch_clinical_trials("depression", **kw))), gets

    def pages(self, *ncts):
        return [{"title": f"{n} | ClinicalTrials.gov", "url": f"https://clinicaltrials.gov/study/{n}", "snippet": "summary text"} for n in ncts]

    def test_trial_records_are_read_by_nct_number(self):
        out, gets = self.search(self.pages("NCT00000001", "NCT00000002"), [trial_study("NCT00000001", "RECRUITING"), trial_study("NCT00000002", "COMPLETED")])
        self.assertEqual(gets[0][1]["filter.ids"], "NCT00000001,NCT00000002")
        self.assertEqual(out["results"][0]["phase"], ["PHASE3"])      # not available from a web page
        self.assertEqual(out["results"][0]["recruitment_status"], "RECRUITING")
        self.assertEqual(out["results"][0]["brief_summary"], "summary text")      # the excerpt the page showed is kept

    def test_the_status_filter_is_applied(self):
        out, _ = self.search(self.pages("NCT00000001", "NCT00000002"), [trial_study("NCT00000001", "RECRUITING"), trial_study("NCT00000002", "COMPLETED")], status="RECRUITING")
        self.assertEqual([r["nct_id"] for r in out["results"]], ["NCT00000001"])

    def test_drug_only_keeps_trials_with_a_drug(self):
        out, _ = self.search(self.pages("NCT00000001", "NCT00000002"), [trial_study("NCT00000001", "RECRUITING", drug=False), trial_study("NCT00000002", "RECRUITING")], drug_only=True)
        self.assertEqual([r["nct_id"] for r in out["results"]], ["NCT00000002"])

    def test_nothing_left_uses_the_registrys_own_search(self):
        async def direct(*args):
            return json.dumps({"results": [{"nct_id": "NCT09999999"}], "via": "direct"})
        with Env():
            via.DIRECT["asearch_clinical_trials"] = direct
            with mock.patch.object(async_web, "_tavily_semantic", fake_semantic([])):
                out = json.loads(run(via.asearch_clinical_trials("depression")))
        self.assertEqual(out["via"], "direct")


def label(set_id="12345678-1234-1234-1234-123456789012"):
    return {"set_id": set_id, "effective_time": "20240102", "openfda": {"generic_name": ["SERTRALINE"], "brand_name": ["Zoloft"], "route": ["ORAL"]},
            "indications_and_usage": ["Sertraline is indicated for major depressive disorder in adults."], "dosage_and_administration": ["Major depressive disorder: start 50 mg once daily."],
            "boxed_warning": ["Suicidality and antidepressant drugs."]}


class LabelsThroughTavily(unittest.TestCase):
    SET_ID = "12345678-1234-1234-1234-123456789012"

    def pages(self):
        return [{"title": "SERTRALINE tablet - DailyMed", "url": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={self.SET_ID}", "content": "Sertraline label text", "snippet": "Sertraline label text"}]

    def test_a_label_is_read_from_openfda_by_set_id(self):
        import async_tools as at
        asked = []

        async def fda_get(params):
            asked.append(params)
            return {"results": [label(self.SET_ID)]}

        async def search(query, domains, count, semantic=None):
            return self.pages()
        with Env():
            with mock.patch.object(at, "_fda_get", fda_get), mock.patch.object(via, "_search", search):
                out = json.loads(run(via.asearch_fda_labels(["depression"], ["sertraline"], 8)))
        record = out["results"][0]
        self.assertEqual(asked[0]["search"], f'set_id:"{self.SET_ID}"')
        self.assertEqual(record["brands"], ["Zoloft"])
        self.assertIn("label_sections", record)      # the sections the drug tables are built from
        self.assertEqual(record["label_date"], "2024-01-02")
        self.assertIn("read from openFDA", out["note"])

    def test_a_late_label_lookup_leaves_the_tavily_record(self):
        import async_tools as at

        async def fda_get(params):
            raise RuntimeError("openFDA down")

        async def search(query, domains, count, semantic=None):
            return self.pages()
        with Env():
            with mock.patch.object(at, "_fda_get", fda_get), mock.patch.object(via, "_search", search):
                out = json.loads(run(via.asearch_fda_labels(["depression"], ["sertraline"], 8)))
        self.assertEqual(out["results"][0]["set_id"], self.SET_ID)
        self.assertIsNone(out["results"][0]["dose_statement"])

    def test_the_label_link_carries_the_boxed_warning(self):
        import async_tools as at

        async def fda_get(params):
            return {"results": [label(self.SET_ID)]}

        async def search(query, domains, count, semantic=None):
            self.assertFalse(semantic)      # a drug's name must match: no sentence around it
            return self.pages()
        with Env():
            with mock.patch.object(at, "_fda_get", fda_get), mock.patch.object(via, "_search", search):
                link = run(via.afda_label_link("sertraline"))
        self.assertEqual(link["brand"], "Zoloft")
        self.assertIn("Suicidality", link["boxed_warning"])


class WiringInAsyncTools(unittest.TestCase):
    def test_the_direct_searches_are_kept_and_similar_articles_never_use_tavily(self):
        import async_tools
        with mock.patch.dict(os.environ, {"SOURCES_VIA_TAVILY": "on"}):
            try:
                with mock.patch.object(via, "DIRECT", {}):
                    tools = importlib.reload(async_tools)
                    self.assertIs(tools.asearch_pubmed_quick, via.asearch_pubmed_quick)      # the Tavily version is in place
                    self.assertIsNot(tools._direct_pubmed_quick, via.asearch_pubmed_quick)      # the direct one is kept
                    self.assertEqual(sorted(via.DIRECT), ["afda_label_link", "asearch_clinical_trials", "asearch_europepmc_quick", "asearch_fda_labels", "asearch_pubmed", "asearch_pubmed_quick"])
                    self.assertIsNot(via.DIRECT["asearch_pubmed_quick"], via.asearch_pubmed_quick)
                    queries = []

                    async def direct(query, **kw):
                        queries.append(query)
                        return "{}"

                    async def get_json(url, params=None):
                        return {"linksets": [{"linksetdbs": [{"linkname": "pubmed_pubmed", "links": [{"id": "5", "score": 9}]}]}]}
                    with mock.patch.object(tools, "_direct_pubmed_quick", direct), mock.patch.object(async_core, "aget_json", get_json):
                        run(tools.asimilar_articles(["1"]))
                    self.assertEqual(queries, ["5[uid]"])
            finally:
                os.environ.pop("SOURCES_VIA_TAVILY", None)
                importlib.reload(async_tools)


JUNK_TITLE = ("145 DIVALPROEX DR 250 MG\n105 DIVALPROEX DR 500 MG\n\nthese highlights do not include all the information needed to use divalproex sodium delayed-release tablets safely and effectively. "
              "see full prescribing information for divalproex sodium delayed-release tablets.\n\ndivalproex sodium delayed-release tablets + for oral use\n\ninitial u.s. approval: 1983)")


class DrugNamesAreOneShortLine(unittest.TestCase):
    """A drug name goes into a table cell: a line break, a | or a block of label text in it breaks the row."""

    def test_page_titles_become_clean_names(self):
        self.assertEqual(via._clean_name("Frova®\n\n  (Frovatriptan Succinate) Tablets"), "Frova (Frovatriptan Succinate) Tablets")
        self.assertEqual(via._clean_name("SERTRALINE tablet - DailyMed"), "SERTRALINE tablet")
        self.assertEqual(via._clean_name(JUNK_TITLE), "divalproex sodium delayed-release tablets")      # the product the standard sentence names
        self.assertEqual(via._clean_name("these highlights do not include all the information needed to use this drug"), "")      # boilerplate with no product named
        self.assertEqual(via._clean_name("x" * 200), "")
        self.assertEqual(via._clean_name(None), "")

    def test_the_labels_own_generic_name_is_preferred_and_combinations_keep_their_commas(self):
        self.assertEqual(via._label_generic({"openfda": {"generic_name": ["AVOBENZONE", "HOMOSALATE"]}}, JUNK_TITLE), "AVOBENZONE, HOMOSALATE")
        self.assertEqual(via._label_generic({"openfda": {}}, JUNK_TITLE), "divalproex sodium delayed-release tablets")      # an old label with no openFDA block
        self.assertEqual(via._label_generic({}, ""), "DailyMed label")

    def test_a_label_page_titled_with_label_text_gives_a_one_line_name_in_the_records(self):
        import async_tools as at
        set_id = "12345678-1234-1234-1234-123456789012"
        old_label = {"set_id": set_id, "effective_time": "20170921", "openfda": {}, "indications_and_usage": ["Divalproex is indicated for prophylaxis of migraine headaches."],
                     "dosage_and_administration": ["Migraine: the recommended starting dose is 250 mg twice daily."]}

        async def fda_get(params):
            return {"results": [old_label]}

        async def search(query, domains, count, semantic=None):
            return [{"title": JUNK_TITLE, "url": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}", "content": "label"}]
        for enrich in ("on", "off"):
            with Env(TAVILY_SOURCES_ENRICH=enrich):
                with mock.patch.object(at, "_fda_get", fda_get), mock.patch.object(via, "_search", search):
                    record = json.loads(run(via.asearch_fda_labels(["migraine"], [], 8)))["results"][0]
            self.assertNotIn("\n", record["generic"])
            self.assertNotIn("highlights", record["generic"].lower())
            self.assertLessEqual(len(record["generic"]), 80)

    def test_section_headings_are_not_drug_names(self):
        for heading in ("Highlights Of Prescribing Information", "Complete Boxed Warning", "Boxed Warning", "DailyMed label", "Indications and Usage"):
            self.assertEqual(via._clean_name(heading), "", heading)
        old = {"openfda": {}, "indications_and_usage": ["1.3 Migraine Divalproex sodium delayed-release tablets, USP are indicated for prophylaxis of migraine headaches."]}
        er = {"openfda": {}, "indications_and_usage": ["1.3 Migraine Divalproex sodium extended-release tablets are indicated for prophylaxis of migraine headaches."]}
        self.assertEqual(via._label_generic(old, "Highlights Of Prescribing Information", ["migrain"]), "Divalproex sodium delayed-release tablets")
        self.assertEqual(via._label_generic(er, "Complete Boxed Warning", ["migrain"]), "Divalproex sodium extended-release tablets")

    def test_the_drug_table_never_gets_a_line_break_from_a_name(self):
        import landscape
        for generic in ("Frova®\n\n (Frovatriptan Succinate) Tablets", JUNK_TITLE, "A|B"):
            name = landscape.product_name({"generic": generic, "brands": []})
            self.assertNotIn("\n", name)
            self.assertNotIn("|", name)
        self.assertEqual(landscape.product_name({"generic": "Sumatriptan", "brands": ["Imitrex"]}), "Sumatriptan (Imitrex)")      # an ordinary name is unchanged
        self.assertEqual(landscape.product_name({"generic": "Avobenzone, Homosalate", "brands": []}), "Combination product (one product: avobenzone + homosalate)")


if __name__ == "__main__":
    unittest.main()
