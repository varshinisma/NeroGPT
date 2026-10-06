---
name: latest-evidence
description: Finds the newest, most relevant published medical evidence - recent papers, guidelines, systematic reviews, randomized trials and preprints - and records year, journal, study type, key finding and DOI/PMID for each. Use for every clinical question, and especially when the user asks for "latest", "recent", "new", "2025", "2026", "what's new", "current evidence", "newest studies" or "updated guidelines".
---

# Latest Evidence

## Overview

Search the web and literature sources for the newest, most relevant evidence for the question, and return structured, citable records. Every source must have a **publication year** and a **DOI, PMID or URL**. Do not fabricate anything.

## Step 1: Build the search queries

Use the entities from the orchestrator. Write 1-2 queries in FAST mode, 4-6 in DEEP mode:
- Core: disease + intervention or drug (use generic and brand names, synonyms)
- Recency: add the year or date range (default last 24 months; "latest/new" means last 12 months)
- DEEP only: add `systematic review`, `meta-analysis`, `randomized`, `guideline`, `phase 3`, `newly approved`, `emerging therapy`
- DEEP only, contradiction probe: add `no benefit`, `failed`, `negative results`, `safety signal`

Search recent first, then add older landmark or still-current guideline evidence.

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
