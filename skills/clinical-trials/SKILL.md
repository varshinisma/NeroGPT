---
name: clinical-trials
description: Finds current and recent clinical trials from trial registries for a disease, drug, biomarker or intervention, with registry ID, phase, current recruitment status, endpoints and dates. Use when the user asks about "clinical trials", "ongoing trials", "recruiting", "phase 3", "NCT number", "new drugs in trials", "what is being studied", or for any clinical query, since trial data is always collected. Ongoing does not mean effective.
---

# Clinical Trials

## Overview

Find current and recent trials relevant to the question and return structured registry data with the **current status and last-update date**. Keep registry status separate from results. Pass investigational drug names to `drug-intelligence`.

> Ongoing does not mean effective. A recruiting or active trial only shows that research is happening.

## Step 1: Build registry queries

Combine condition and intervention terms, with synonyms and generic and brand drug names. FAST: 1-2 queries. DEEP: add biomarker, population and phase variants, and search each major emerging drug by name.

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
