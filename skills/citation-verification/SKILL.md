---
name: citation-verification
description: Checks that every claim in a draft clinical answer is actually supported by its cited source - right disease, drug, population, year, study type, dose, drug category, approval status and trial status - and flags unsupported or fabricated citations. Use before an answer is finalized, or when the user asks to "verify the citations", "check the sources", "fact-check this", "is this claim supported", or "validate the references".
---

# Citation Verification

## Overview

Check each material claim against the retrieved source it cites. Report what is verified and what failed. Never write new clinical content and **never create a citation**.

## Step 1: Extract the material claims

Split the draft into single claims. Material claims are anything a clinician might rely on: efficacy, safety, dose, route, frequency, indication, approval status or date, drug category, guideline recommendation, trial status, phase or ID, effect size, population, and the date of a finding.

## Step 2: Match each claim to its source

Link each claim to its `source_id` and the retrieved record. If none is attached, record a citation gap.

## Step 3: Run the checks

For each claim, check:
1. A source exists
2. The source actually supports the claim (not just the same topic)
3. Disease, drug and population are correct
4. The source is not outdated or superseded; a recent source is used where one was retrieved
5. The study type is described correctly (trial versus observational versus preprint)
6. The citation sits on the claim it supports
7. Dose, route, frequency and duration appear in the source, and it is the right dose type (label, guideline, trial or investigational)
8. Each listed drug is sourced and in the right category (approved, emerging or off-label)
9. Approval status **and date** are backed by a regulatory or guideline source
10. Trial ID, phase and status are backed by current registry data
11. The wording does not exceed the strength of the evidence
12. The DOI, PMID or URL exists in the retrieved record
13. The **publication year shown in the answer matches** the source record
14. The claim is relevant to the user's exact question

## Step 4: Assign severity

- **High:** wrong or unsupported dose; investigational drug shown as approved; wrong trial ID or status; fabricated citation; efficacy claim with no support; population mismatch that affects safety
- **Medium:** citation on the wrong claim; outdated source when newer exists; missing approval date or year; study type mislabelled
- **Low:** formatting or loose wording with correct support

## Step 5: Choose the required action

For each failure choose one, in this order of preference: `retrieve_better_evidence` (only if retries remain), `qualify` (soften to what the source supports), `remove`, or `fix_citation` (only by pointing to an already-retrieved source that really supports it).

## Step 6: Return the result

Return `verified_claims`, `failed_claims`, `citation_gaps`, `overall_severity` and `required_action`. Any high-severity item means the answer must not be finalized until fixed.

## Rules

- Never manufacture or reconstruct citations, DOIs, PMIDs, trial IDs or dates.
- If a claim cannot be verified, remove it, qualify it or retrieve better evidence. Do not leave it as is.
- Check against the retrieved record, not memory.
- Do not accept a trial dose as support for a routine dose.
- Treat missing support for approval status or date as a failure.

## Quality check

- [ ] All material claims extracted and checked
- [ ] Every dose, approval, drug category, trial status and year checked against its source
- [ ] No fabricated identifiers
- [ ] Every failure has a severity and a required action
- [ ] High-severity issues block finalization

## Examples

- "Drug X 50 mg daily is first-line" but the source is a phase 2 trial -> wrong dose type, high; qualify or find the label or guideline.
- "Drug Y is approved" but the source is a Phase 3 trial record -> wrong category, high; qualify as investigational.
- Answer says "(Lee et al., 2025)" but the record says 2023 -> year mismatch, medium; fix the year.

## References

Load `references/schemas.md` for the exact verification output shape.
