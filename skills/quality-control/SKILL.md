---
name: quality-control
description: Final gate before a clinical answer is marked complete - checks that it answers the question, is recent, cited with years, includes the Treatment Drug Landscape, has correct doses and trial statuses, and is internally consistent. Returns PASS or NEEDS_REPAIR with targeted fixes. Use before finalizing any clinical answer, or when the user asks to "review the answer", "quality check", "is this ready", "double-check this response", or "validate the final answer".
---

# Quality Control

## Overview

Inspect the assembled answer and `ResearchState` and decide **PASS** or **NEEDS_REPAIR**. You judge and route repairs; you do not rewrite the answer and you never silently change a clinical fact without evidence.

## Step 1: Check grounding

- Are all material claims supported?
- Are citations present, correct and real (no fabricated IDs)?
- Is anything still marked `block_final` by `citation-verification`?

## Step 2: Check the question and citation format

- Does the bottom line directly answer the user's exact question?
- Does every finding show its **publication year** and a citation (author, journal, year, DOI/PMID/URL)?
- Is "Evidence searched on <date>" shown, and is the newest evidence first?
- Does the reference list match the inline citations?

A missing year or citation on a clinical claim is **high** severity.

## Step 3: Check recency

- Was recent evidence searched, with dates captured?
- Was trial status verified from a registry, with its last-update date?
- Is any time-sensitive item (approval, trial status) stale?

## Step 4: Check drugs

- Is a **Treatment Drug Landscape present for this query**, even a non-treatment one? Missing is always high severity.
- Are established, latest and emerging drugs all covered and separated?
- Were latest approvals searched and dated?
- Is every dose sourced, with route and frequency, and are standard and trial doses clearly separated?
- Is any investigational drug shown as approved, or any major established drug missing?
- If no drug exists, is that stated with a source?

## Step 5: Check trials

- Do trial IDs exist in retrieved registry records?
- Is the status current, and are ongoing and completed trials distinguished?
- Are results separate from registry information, and is no efficacy implied by "ongoing"?

## Step 6: Check evidence and consistency

- Is evidence strength represented correctly, with contradictory evidence included and experimental findings labelled?
- Are there contradictions between sections (bottom line versus drug table versus synthesis)?
- Are there conflicting doses, dates or trial statuses?
- Are failed or partial skills disclosed, and does the answer avoid claiming deep research is complete when it is not?

## Step 7: Decide the status

- **High severity** (blocks FINAL): unsupported clinical claim; wrong or unsourced dose; investigational drug shown as approved; wrong trial ID or status; missing Treatment Drug Landscape; fabricated citation; missing year or citation on a claim; conflicting doses.
- **Medium:** missing approval date; outdated source when newer exists; missing contradiction; unlabelled preliminary study; section inconsistency.
- **Low:** formatting or wording.

Return `PASS` only if no high-severity issue remains. Otherwise return `NEEDS_REPAIR` with `issues`, `severity` and `repair_targets` listing **only the failed parts**.

## Step 8: Handle the repair budget

Allow at most 1 repair cycle. If new evidence is needed, route it through the orchestrator. If issues remain after the budget is spent, finalize only after high-severity claims are removed or explicitly qualified, and add a limitation note. Never finalize with an unqualified high-severity claim.

## Rules

- Do not add new clinical content.
- Judge against the retrieved records, not memory.
- Repair only the failed portion; re-run only the affected checks.

## Quality check

- [ ] All checks in Steps 1-6 evaluated
- [ ] Treatment Drug Landscape confirmed present and categorized
- [ ] No unresolved or unqualified high-severity issue at FINAL
- [ ] Repair targets are minimal and specific
- [ ] Repair budget respected

## Example

The drug table lists "Drug Z 100 mg" as standard, but the only source is a phase 2 trial. Result: `NEEDS_REPAIR`, high severity, repair target `drug_landscape.Z`, action `qualify` -> relabel as "Investigational - trial dose (NCT...) - not for routine use" or remove the dose.

## References

Load `references/schemas.md` for the exact QC output shape.
