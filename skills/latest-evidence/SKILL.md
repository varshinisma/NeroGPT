---
name: latest-evidence
covers: treatment_options, efficacy, safety, diagnosis, prognosis, mechanism, guideline_recommendation, drug_interaction
description: Finds the newest, most relevant published medical evidence - recent papers, guidelines, systematic reviews, randomized trials and preprints - and records year, journal, study type, key finding and DOI/PMID for each. Use for every clinical question, and especially when the user asks for "latest", "recent", "new", "2025", "2026", "what's new", "current evidence", "newest studies" or "updated guidelines".
---

# Latest Evidence

## Overview

Search the web and literature sources for the newest, most relevant evidence for the question, and return structured, citable records. Every source must have a **publication year** and a **DOI, PMID or URL**. Do not fabricate anything.

## Step 1: Build the search queries

Use the entities from the orchestrator. FAST mode (run by the program): 1-2 queries. DEEP mode: follow "Query transformation by intent" below, which writes as many searches as the question has distinct needs, and none if published evidence does not bear on the question. The guidance here is what a good search contains:
- Core: disease + intervention or drug (use generic and brand names, synonyms)
- Recency: add the year or date range (default last 24 months; "latest/new" means last 12 months)
- DEEP only: add `systematic review`, `meta-analysis`, `randomized`, `guideline`, `phase 3`, `newly approved`, `emerging therapy`
- DEEP only, contradiction probe: add `no benefit`, `failed`, `negative results`, `safety signal`

Search recent first, then add older landmark or still-current guideline evidence.

## Query transformation by intent

You are given the doctor's question, its intent, and what was understood from it (what is asked, why, the kinds of information needed, the case facts). You write the searches for the needs that published literature, guidelines and similar sources can answer.

1. Work out from the INTENT of this question which distinct needs its answer has, and name each in 2 to 4 words of your own. There is no fixed list: the needs depend on what is asked. Examples of the thinking, not a list to copy:
   - how to treat something: the treatment options, their outcomes, their safety, and what guideline organisations recommend (`guideline_sites`);
   - what is new: the newest trials, recent approvals, recent reviews;
   - which of two options is better: head-to-head evidence, indirect comparisons, differences in safety;
   - how something works: the mechanism, the evidence in humans, where it is still unclear;
   - a case study: whatever that case asks for, such as its work-up, the causes to consider, its management and its follow-up;
   - a safety or dosing question: the harms, the risk groups, what labels and guidelines say.
   A simple question has one or two needs. Use as many as the question really has, at most 6. Never add a need the question does not call for.
   **Case-study rule.** When the question describes a patient, `case_facts` holds the case's own findings. The answer must cover the WHOLE case, not one of its findings:
   - Keep the case's own words. Copy each finding from `case_facts` word for word into the searches. Any condition name in a search must be one the question itself names: never rename a finding into a diagnosis the question did not state (a finding described as "tendencies" or "features" stays that finding; do not write it as a named disorder).
   - Work-up and cause: write searches on the broadest problem that frames the case together with its other findings, asking for the causes to consider, the differential diagnosis, the recommended tests (laboratory, genetic, imaging, neurophysiological, sensory) and what guideline organisations recommend for the evaluation (`guideline_sites`).
   - Each associated finding: one search on how it is assessed or screened in this age group, and one on the services or interventions recommended for it.
   - Management and follow-up: one search on the overall management of the whole case (medical, therapy and education services, monitoring and referral), and one on prognosis or follow-up when the question asks for it.
   - Do not let one associated finding take over: at most half of the searches may be about that single finding; the rest are about the broader problem's evaluation and the management of the whole case.
2. For each need write one search query as a natural-language phrase of 6 to 20 words, in standard clinical terms, with one synonym where a second wording is common.
3. When the question asks whether something works or is safe, include one need for evidence that contradicts it (`study_types: ["negative_results"]`).
4. Added concepts are only search terms. They are never statements of fact about the patient.
5. Do not repeat a query, and do not copy the question word for word.

### Sources you may target (each query names ONE `target`; any skill may use any of them when the need calls for it)

- `pubmed` - published studies, reviews, trial reports and guideline papers. Args: `sort` (`relevance` or `date`), `years_back` (1 to 15), `study_types` (any of `rct`, `meta_analysis`, `systematic_review`, `guideline`, `review`, `negative_results`), `open_access` (true to prefer papers with a free full text).
- `europepmc` - a second literature index. Use it for a literature need that PubMed may miss, or as the `alternate` of a `pubmed` query. Args: `sort`, `years_back`, `fulltext` (true searches the WORDS IN THE FULL TEXT of open-access articles, not only the title and abstract: use it for a need whose answer sits in methods or results, such as how a work-up or a procedure is done, and write the query as the words a paper would use there).
- `guideline_sites` - recommendations from guideline organisations.
- `regulator` - regulators' announcements: approvals, new indications, safety communications.
- `fda_label` - official FDA drug labels (indications, dosing, warnings). Args: `condition_terms` (words of the question's condition) and `drug_names` (ONLY medicines the question itself names).
- `web` - anything else, as a targeted search; never the default for everything.
- `trials` - trial registry records. Args: `condition`, `intervention`, `status` (`recruiting`, `completed` or `any`), `drug_only` (true for trials of drugs only).

Choose the source by what the need is. Several queries may go to one source, one skill may write queries for several sources, and an important need may get a query for each of two suitable sources. Do not send a query to every source. But do not stop at one source either: for every ESSENTIAL need, write a query for EACH listed source that can really answer it (two or three sources is normal, each as its own query with the same `need_id`), each phrased the way that source is searched. A `supporting` need may use one source. Skip a source only when it cannot answer the need.

### Output

Reply with ONE JSON object and nothing else:

`{"queries": [{"need_id": "n1", "need": "the need in 2 to 4 words", "target": "one source from the list above", "query": "a natural-language phrase (not used by fda_label or trials)", "args": {}, "serves": "the one information need from the list that this search answers", "priority": "essential or supporting"}]}`

- `need_id` groups queries that serve the same need. `priority` is `essential` for a need that directly answers what was asked and `supporting` for context. Only essential needs are retried elsewhere.
- `serves` names which listed information need (from `info_needed` in the understanding) the search answers; leave it out if none fits. Use it to check, need by need, that each listed information need has a search on a source that can answer it.
- Add an `alternate` (`{"target": ..., "query": ..., "args": {}}`) only for an essential need where a DIFFERENT source would give DIFFERENT evidence (for example the guideline sites beside the literature); most needs need none. Keep the reply short.
- Keep the condition and the patient details from the question in every query. Never name a drug the question does not name or clearly imply.
- **If your source is not needed for this question, reply `{"queries": []}`.** Writing no queries is the correct answer for a skill that has nothing to add. Never write searches only to be thorough.

## Step 2: Run the searches and de-duplicate

Run query variants in parallel with the available search tools. Remove duplicates by DOI, then PMID, then title. If a preprint has a later published version, keep the published version.

## Step 3: Open or check the source for the date

Search snippets often have no date. Open the source or its page to confirm the publication year, journal and identifier. If you cannot confirm the year, do not use the source for a clinical claim.

## Step 4: Screen for relevance

Keep a source only if it matches the disease, population and intervention of the question. Rank by relevance to the exact question first, then quality, then recency. Record notable exclusions with a reason.

## Step 5: Extract the fields

Record title, authors, publication date, journal, study type, population, intervention, comparator, key finding with effect size and confidence interval if given, limitations, DOI/PMID/URL, retrieval timestamp and drugs named. Use `null` for anything not stated. Never guess.

## Step 6: Label the evidence status

Mark each source `established`, `emerging`, `preliminary`, `experimental` or `contradicted`. Label preprints, abstracts and small or interim studies as preliminary. Exclude retracted work.

## Step 7: Rank and return

FAST: return the top 3-5 sources. DEEP: return up to about 25, newest first within each study type, and include at least one guideline or systematic review if one exists. Pass `drugs_named` back so the orchestrator can send them to `drug-intelligence`.

## Rules

- Never fabricate papers, authors, DOIs, PMIDs, dates or findings.
- Do not drop older high-quality evidence only because it is old.
- Include contradictory and negative studies, not only supportive ones.
- If nothing exists in the recent window, say so in `gaps` and fall back to the newest available, with its year.
- Maximum 2 retries per failed search; then return what you have as `partial`.

## Quality check

- [ ] Every source has a year and a DOI, PMID or URL
- [ ] No invented identifiers (null where unknown)
- [ ] Recent window searched first
- [ ] Preprints and preliminary studies labelled
- [ ] Contradictory evidence searched in DEEP mode
- [ ] `drugs_named` extracted

## Example

Question: "Latest evidence for tenecteplase in stroke". FAST returns the most recent guideline update and 1-2 recent trials or meta-analyses with year, PMID, key finding and `drugs_named: [tenecteplase, alteplase]`. DEEP adds late-window trials, safety data on intracranial hemorrhage, and any negative trial.

## References

Load `references/schemas.md` for the exact source record and output fields.
