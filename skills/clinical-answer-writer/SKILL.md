---
name: clinical-answer-writer
description: Writes the doctor-facing clinical research answer - a fast first response within about 5 seconds, then progressively streamed deeper sections with the Treatment Drug Landscape table, trials, conflicts, limitations and a dated reference list. Use whenever a clinical research answer must be written or updated, such as "answer this clinical question", "summarize the latest research on X", "write up the evidence", or "give me the treatment options with sources".
---

# Clinical Answer Writer

## Overview

Write the final answer from the structured `ResearchState` only. **Answer the exact question first**, show the **year and citation on every finding**, put the **newest evidence first**, and always include the Treatment Drug Landscape. Start writing as soon as the orchestrator says minimum evidence is ready; do not wait for the whole pipeline.

## Citation format (use everywhere)

`Finding statement. (First author et al., Journal, Year) [DOI / PMID / URL]`

Example: "Drug X reduced relapse rate by 30% vs placebo (HR 0.70, 95% CI 0.58-0.85). (Lee et al., Lancet Neurology, 2025) [PMID 12345678]"

If a finding has no year or citation, do not state it; remove it or write "source not retrieved".

## Step 1: Write the fast response (Phase 1, about 5 seconds)

In this order:
1. **Bottom line** answering the exact question, 1-3 sentences, with citation
2. **"Evidence searched on <date>"**
3. **Most important recent finding** with exact date, study type, key result and citation
4. **Immediate context** in 1-3 sentences
5. **Top treatment drugs** (required for every query): established drugs first, then the single most notable latest drug, in a small table with category and source
6. A one-line status: "Preliminary - deeper review (full drug list, trials, contradictions, verification) is continuing."

If something is not ready, say it is still loading. Do not guess.

## Step 2: Stream the deeper sections (Phase 2)

Add these in order, only when their data is in the state, newest studies first in each:
1. Latest findings
2. Current evidence (established, then emerging, then experimental)
3. Treatment Drug Landscape
4. Clinical trials
5. Key studies
6. Conflicting evidence
7. What remains under investigation
8. Evidence limitations
9. Numbered references (authors, title, journal, year, DOI/PMID/URL)

Move the stream state forward as data arrives: `INITIAL -> EVIDENCE_ENRICHING -> DRUGS_ENRICHING -> TRIALS_ENRICHING -> VERIFICATION -> FINAL`.

## Step 3: Build the Treatment Drug Landscape table

Group by category (Approved - standard, Guideline-recommended, Newly approved, Emerging - investigational, Off-label, Safety alerts). Columns:

| Drug (generic / brand) | Category | Indication | Dose / route (if sourced) | Approval status / date | Key evidence (year) |

- Show the dose type: "label" or "guideline". For trial doses write "Trial dose (NCT...) - not for routine use". If not sourced write "Not retrieved".
- Emerging drugs must say **"Investigational - not approved"**. Off-label drugs must say **"Off-label"**.
- Show the approval date for newly approved drugs.
- If no drug treatment exists, say so with a citation.
- Fast response shows the top drugs; the deep answer shows the full list.

## Step 4: Build the trials table

Columns: Trial ID, drug or intervention, phase, status (with "as of" date), primary endpoint, results available. Add: "Ongoing trials show research activity, not proven effectiveness." Keep registry status separate from results.

## Step 5: Apply repairs

When `citation-verification` or `quality-control` returns repair targets, change only those parts (remove, qualify or re-source). Do not rewrite the rest.

## Rules

- Keep each claim next to its citation. Use exact dates for recent findings.
- Label preliminary, experimental and ongoing research clearly.
- Use calibrated wording ("associated with", "reduced X by Y (95% CI ...)").
- Never invent missing values; write "Not retrieved" or "Not reported".
- Never present investigational treatments as established therapy.
- Do not give patient-specific prescribing instructions.
- State your assumption if the question was ambiguous.
- Disclose partial or failed skills (for example "trial registry could not be reached").
- Write for a physician: lead with the answer, no filler.

## Quality check

- [ ] The first line answers the exact question
- [ ] Every finding has a year and a citation
- [ ] "Evidence searched on" date is shown
- [ ] Newest evidence first
- [ ] Phase 1 includes bottom line, recent finding, context and top drugs
- [ ] Full Treatment Drug Landscape table is in the deep answer
- [ ] Investigational, off-label and trial-dose labels present
- [ ] Contradictions and limitations included
- [ ] Gaps from failed skills disclosed

## References

Load `references/schemas.md` for the streamed chunk shape.
