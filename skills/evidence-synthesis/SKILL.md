---
name: evidence-synthesis
description: Combines retrieved papers, drugs and trials into one balanced evidence picture - bottom line, what is established versus emerging versus experimental, where studies disagree, and what is unknown. Use after retrieval is complete, or when the user asks to "summarize the evidence", "what does the evidence say", "is it effective", "compare the studies", "any conflicting results", or "how strong is the evidence".
---

# Evidence Synthesis

## Overview

Turn the retrieved sources, drug landscape and trials into an honest summary. Do not retrieve new evidence and do not add facts that are not in the retrieved data. Every statement must point to its sources, and each finding keeps its **year**.

## Step 1: Group the evidence by strength

Sort findings into: **established** (guideline-endorsed or consistent in high-quality studies), **emerging** (promising recent data, not yet adopted), **experimental or preliminary** (preclinical, early phase, small, preprint, interim), and **contradicted** (newer or stronger data disagree).

Do not rank only by age or study type. Weigh design and bias, sample size and precision, population match, endpoint quality (clinical versus surrogate), consistency, recency, and how directly the study answers the question.

## Step 2: Write the bottom line

Write 1-3 sentences that answer the **exact question asked**, using language that matches the evidence strength ("shown in randomized trials" versus "suggested by observational data"). Attach sources and years.

## Step 3: Map agreement and contradiction

For each key point compare direction, size, population and design across studies. If studies disagree, give a reason only if the sources support it; otherwise write "reason not established". Never hide negative results.

## Step 4: Add the drug landscape

Show established, latest and emerging drugs side by side, each tied to its evidence level. Make clear which are approved, off-label or investigational.

## Step 5: Add the trials

Describe what is being tested, clearly **not** as evidence of benefit. Highlight completed trials with negative or mixed results.

## Step 6: Apply the caveats

State where relevant: association is not causation, efficacy (trial) differs from effectiveness (routine care), surrogate versus clinical endpoints, statistical versus clinical significance, and whether trial populations match the question.

## Step 7: List gaps and uncertainty

Note missing head-to-head trials, short follow-up, missing subgroups, reliance on surrogate endpoints, conflicting guidelines, lack of recent data, and any skills that failed or returned partial results.

## Output sections

Bottom line, current evidence, most recent findings, treatment drug landscape, key studies, conflicting evidence, uncertainty, ongoing research.

## Rules

- Do not overstate weak evidence or turn correlation into causation.
- Do not hide contradictory or negative results.
- Do not describe ongoing trials as support for a treatment.
- If evidence was not retrieved, write "evidence not retrieved" instead of using memory.
- Use calibrated wording: "associated with", "reduced X by Y (95% CI ...)", "no significant difference".

## Quality check

- [ ] The bottom line answers the exact question and its strength matches the evidence
- [ ] Established, emerging and experimental evidence are separated
- [ ] Contradictions and negative results are shown
- [ ] Every statement has sources and a year
- [ ] Gaps and pipeline failures are stated
- [ ] No causal or efficacy overreach

## Example

Topic: anti-amyloid antibody in early Alzheimer's. Bottom line (moderate strength): phase 3 trials show a modest slowing of decline with a risk of amyloid-related imaging abnormalities. Conflict: whether the effect size is clinically meaningful is debated, reported only as the cited sources state. Other agents in trials are investigational and imply no efficacy.

## References

Load `references/schemas.md` for the exact synthesis output shape.
