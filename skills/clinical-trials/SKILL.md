---
name: clinical-trials
covers: trial_status, treatment_options, efficacy
description: Finds current and recent clinical trials from trial registries for a disease, drug, biomarker or intervention, with registry ID, phase, current recruitment status, endpoints and dates. Use when the user asks about "clinical trials", "ongoing trials", "recruiting", "phase 3", "NCT number", "new drugs in trials", "what is being studied", or when ongoing or recent trials bear on the question. Not needed otherwise. Ongoing does not mean effective.
---

# Clinical Trials

## Overview

Find current and recent trials relevant to the question and return structured registry data with the **current status and last-update date**. Keep registry status separate from results. Pass investigational drug names to `drug-intelligence`.

> Ongoing does not mean effective. A recruiting or active trial only shows that research is happening.

## Step 1: Build registry queries

Only when trials bear on the question: combine condition and intervention terms, with synonyms and generic and brand drug names. FAST (run by the program): 1-2 queries. DEEP: follow "Query transformation by intent" below; add biomarker, population and phase variants and search each major emerging drug by name only when the question calls for them.

## Query transformation by intent

You are given the doctor's question, its intent and what was understood from it. You write the searches for needs that registered trials can answer: ongoing or recent trials of an intervention type, trials in the same patient group, trials of emerging treatments. Use at most 3 searches, each for a different need that this question really has.

- A `trials` query needs `args.condition` (the condition words from the question, with its patient group when the question names one: 1 to 12 words) and `args.intervention` (a treatment type, or a drug the question names; empty when the search is about the condition only).
- Always include ONE broad search with only the condition words (and the patient group): narrower registry searches often return nothing, and a broad one keeps the registry covered. Write no qualifiers such as AND or OR in the condition.
- Write nothing when trials do not bear on the question (for example a question about how something works, or a question about one drug's dosing).
- You may also write a query for another source when a trial need is better answered there, for example a `web` search for the registry page of a named trial.

### Sources you may target (each query names ONE `target`; any skill may use any of them when the need calls for it)

- `pubmed` - published studies, reviews, trial reports and guideline papers. Args: `sort` (`relevance` or `date`), `years_back` (1 to 15), `study_types` (any of `rct`, `meta_analysis`, `systematic_review`, `guideline`, `review`, `negative_results`), `open_access` (true to prefer papers with a free full text).
- `europepmc` - a second literature index. Use it for a literature need that PubMed may miss, or as the `alternate` of a `pubmed` query. Args: `sort`, `years_back`.
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

## Step 2: Search the registries

Use ClinicalTrials.gov first, and EU CTIS, WHO ICTRP or CTRI (India) when relevant. If only a web search is available, open the registry page itself to read the real status. Do not rely on a snippet for status. Run queries in parallel and de-duplicate by registry ID.

## Step 3: Screen for relevance

Keep trials that match the condition and population. Drop unrelated ones.

## Step 4: Extract the fields

Record registry ID, title, intervention, comparator, phase, design, condition, population, inclusion and exclusion summary, primary endpoint, secondary endpoints, dose or regimen (only if explicit), recruitment status, locations, start date, estimated completion, last update, sponsor, and source URL. Use `null` for missing values. Never invent an ID, status, phase or date.

## Step 5: Verify the current status

Use the status from the registry record and note its `last_update`. If you cannot confirm it is current, set `status_verified: false` and say so.

## Step 6: Separate results from registry data

If results or a publication exist, attach them in separate fields with the publication year and DOI or PMID. Describe a trial as positive, negative or inconclusive only as the publication reports it. "Completed" does not mean "successful".

## Step 7: Extract investigational drugs

List the investigational drug names so the orchestrator can pass them to `drug-intelligence`. Do not assign approval status here.

## Step 8: Rank and return

FAST: 3-5 of the most relevant current trials (recruiting or active Phase 3 first). DEEP: all relevant trials grouped by status and phase, including terminated and withdrawn trials with the reason, and completed trials with results.

## Rules

- Never present an ongoing trial as evidence that a drug works.
- Never invent trial IDs or statuses.
- Report dose only if explicit in the registry, and label it trial-specific.
- Do not hide terminated trials or negative results.
- Maximum 2 retries; then return `partial` and disclose.

## Quality check

- [ ] Registry IDs come from real registry records
- [ ] Status is current, with `last_update` shown; unverified status flagged
- [ ] Results and publications kept separate from registry data
- [ ] Terminated, withdrawn and negative trials included in DEEP mode
- [ ] Investigational drugs extracted
- [ ] No efficacy implied by "ongoing"

## Example

Question: "Trials for new migraine prevention drugs". Return registry IDs with phase, status, primary endpoint, last update and sponsor. Show completed trials with results separately from recruiting ones. State that these drugs are investigational.

## References

Load `references/schemas.md` for the full trial record and output shape.
