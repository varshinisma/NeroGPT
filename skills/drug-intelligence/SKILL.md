---
name: drug-intelligence
covers: dosing, safety, regulatory_status, drug_interaction, treatment_options, guideline_recommendation
description: Builds a categorized Treatment Drug Landscape when the question concerns medicines - approved and standard drugs, guideline drugs, newly approved drugs, emerging investigational drugs and off-label drugs - with sourced dose, route, approval date and safety. Use for questions about drugs, treatment options, medication, dosage, "first-line", "latest drug", "newly approved", "side effects" or drug interactions. Not needed when the question does not concern medicines.
---

# Drug Intelligence

## Overview

When the question concerns medicines (their use for the condition, doses, safety, approvals, interactions), return a complete and correctly categorized list of the relevant drugs, including the latest approvals. If the condition has no drug treatment, say so explicitly with a source. If the question does not concern medicines, return no searches: the report marks the section not applicable. Never fill the section with unrelated content.

## Step 1: Define the scope

From the entities, set the condition, subtype, line of therapy and population. For mechanism or symptom questions, map to the conditions the drugs treat.

## Step 2: Build the candidate list

Combine drugs from: guidelines, regulator labels (FDA, EMA, and CDSCO when relevant), `drugs_named` from `latest-evidence`, and investigational drugs from `clinical-trials`.

## Query transformation by intent

You are given the doctor's question, its intent and what was understood from it. You write the searches for needs about medicines: which are used for the condition and for its associated problems in THIS patient group, their doses, their safety, approvals and label changes, interactions.

- Official label wording, indications, dosing and warnings: `fda_label` (with `condition_terms` and the `drug_names` the question names).
- Approvals, new indications and safety communications: `regulator`.
- When a source would only name a class of medicines, add a need that asks for the specific medicines of that class, and a need for the medicines used for the condition's associated symptoms or co-occurring conditions when the question concerns them.
- When the question concerns treatment or medicines, include a need for what guideline organisations recommend (first-line medicines and medication recommendations) with target `guideline_sites`, and a need for what the regulators' pages say about the medicines (approvals, safety communications) with target `regulator`.
- When the question concerns treatment, write ALL THREE of these as separate `web` searches (sources that name only a class, or only one medicine, would otherwise leave the landscape short): (1) the medicines used for the condition, by generic name; (2) the medicines used for its associated symptoms or co-occurring conditions; (3) for any class of medicines that sources name, the specific medicines of that class. Each keeps the patient group.
- Write nothing when the question does not concern medicines (for example how a mechanism works, or a diagnostic work-up that asks for no treatment).

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

## Step 3: Categorize every drug

Use this order:
1. Approved for this indication -> `standard` (or `newly_approved` if approved or label-changed in the last ~24 months)
2. Named in a current guideline -> also mark `guideline` (note guideline and year)
3. Evidence supports use outside its approved indication -> `off_label`
4. In late-phase trials or under regulatory review, not approved -> `emerging`
5. Anything else -> leave it out

Never put an investigational drug in an approved category.

## Step 4: Sweep for the latest drugs

Search specifically for approvals, new indications and label changes in the last 12-24 months. Record the approval **date** and regulator. Do not drop an established drug because a newer one exists. Flag withdrawals, restrictions and new safety warnings.

## Step 5: Capture the drug fields

For each drug record generic and brand name, class, mechanism, indication, population, approval status and date, dose, route, frequency, duration, adjustments, contraindications, interactions, major adverse effects, supporting evidence, source and retrieval date.

## Step 6: Keep dosing types separate

Store four separate doses and never merge them: approved label dose, guideline dose, trial dose (with trial ID), investigational dose. Only label and guideline doses may be shown as routine dosing. Label trial and investigational doses "not for routine use".

## Step 7: Cross-check (DEEP only)

Check against guidelines, regulator sources and `clinical-trials`. If sources disagree on a dose, indication or approval date, keep both with sources and set `conflict: true`. Do not silently pick one.

## Step 8: Flag what is missing

Set unknown values to `null` and add a `missing_context` note (for example "pediatric dose not found").

## Rules

- **Never infer or invent a dose.** Retrieve it from a reliable source and cite it with the year, otherwise return `null`.
- Never list an investigational drug as approved.
- Never turn investigational dosing into routine treatment.
- Do not give patient-specific prescribing advice; give sourced, population-level information.
- Never serve approvals or safety alerts from a stale cache.

## FAST and DEEP

FAST: top established drugs plus the single most notable latest drug, each with a source. DEEP: the full categorized list with all four dose types, approval dates, safety alerts and cross-checks.

## Quality check

- [ ] Ran for this query, even if it was not a treatment question
- [ ] All categories searched; empty ones explained
- [ ] Latest approvals searched and dated
- [ ] Every dose sourced, with dose types kept separate
- [ ] No investigational drug in an approved category
- [ ] Safety alerts and conflicts shown, not hidden
- [ ] Every drug has a source and retrieval date

## Example

Question: "New treatments for Alzheimer's". `standard`: cholinesterase inhibitors and memantine with sourced label doses. `newly_approved`: anti-amyloid antibodies with approval dates and safety warnings. `emerging`: late-phase agents labelled "Investigational - not approved". Names and dates must come from retrieved sources at run time.

## References

Load `references/schemas.md` for the per-drug record and landscape output shape.
