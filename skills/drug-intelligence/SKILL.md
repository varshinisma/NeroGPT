---
name: drug-intelligence
description: Builds a categorized Treatment Drug Landscape for every query - approved and standard drugs, guideline drugs, newly approved drugs, emerging investigational drugs and off-label drugs - with sourced dose, route, approval date and safety. Use for any question about drugs, treatment, medication, dosage, "first-line", "latest drug", "newly approved", "side effects", or any clinical query at all, because the drug list is required in every answer.
---

# Drug Intelligence

## Overview

Return, for **every** query, a complete and correctly categorized list of relevant treatment drugs, including the latest approvals. If the question is not about treatment, return the drugs relevant to the condition, symptom, mechanism or population implied. If no drug treatment exists, say so explicitly with a source. Never leave the section empty.

## Step 1: Define the scope

From the entities, set the condition, subtype, line of therapy and population. For mechanism or symptom questions, map to the conditions the drugs treat.

## Step 2: Build the candidate list

Combine drugs from: guidelines, regulator labels (FDA, EMA, and CDSCO when relevant), `drugs_named` from `latest-evidence`, and investigational drugs from `clinical-trials`.

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
