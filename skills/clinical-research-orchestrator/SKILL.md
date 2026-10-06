---
name: clinical-research-orchestrator
description: Controls the full NeuroGPT clinical research workflow - understands the doctor's question, routes it to the retrieval, drug, trial, synthesis, writing and checking skills, and returns a fast first answer followed by deeper research. Use for EVERY clinical or medical research question, such as "latest treatment for multiple sclerosis", "recent trials for migraine prevention", "compare drug A and drug B", "what is new in Alzheimer's", or "give me the evidence for X". It is the only skill allowed to decide what runs next.
---

# Clinical Research Orchestrator

## Overview

Take the doctor's question, run the right skills (in parallel where possible), and return a **recent, cited answer to the exact question asked**. Return a useful first answer in about 5 seconds, then keep researching and stream deeper results. You are the only workflow controller; no other skill calls another skill.

Hard requirements for every answer:
- Answer the **exact question asked** first, not a generic overview.
- Search for **recent evidence** every time (default last 24 months; "latest/new" means last 12 months). Never answer from memory alone.
- Every finding carries its **publication year and a citation**.
- Always run `drug-intelligence` and `clinical-trials`, even if the question is not about treatment, so a Treatment Drug Landscape is always returned.
- Never invent clinical facts. Never treat an ongoing trial as proof.

## Step 1: Start the research state

Create one `ResearchState` object (see `references/schemas.md`). Record `request_started` and set `answer_status = INITIAL`.

## Step 2: Classify the question

Pick one primary intent: `disease_research`, `latest_findings`, `drug_treatment`, `clinical_trial`, `comparison`, `mechanism`, or `comprehensive_research`. If unclear, use `disease_research` + `latest_findings` and state the assumption in the answer. Do not stop to ask clarifying questions.

## Step 3: Extract entities

Extract disease, drug, biomarker, population, intervention, outcome and date range. Normalize names (generic drug names, standard disease names) and keep synonyms. If no drug is named, record the condition or mechanism so `drug-intelligence` can find the relevant drugs.

## Step 4: Plan the routing

Always include: `latest-evidence`, `drug-intelligence`, `clinical-trials`, then `evidence-synthesis`, `clinical-answer-writer`, `citation-verification`, `quality-control`. The intent changes emphasis only, never whether drugs and trials run. For comparisons, run retrieval once per item, in parallel.

## Step 5: Fast path (about 5 seconds)

Run `latest-evidence`, `drug-intelligence` and `clinical-trials` **in parallel** in FAST mode (about 3 s timeout each). As soon as you have at least one relevant recent source and at least one drug entry (or a sourced "no drug treatment exists"), call `clinical-answer-writer` for Phase 1 and record `first_response`. Do not wait for slower skills; mark them `partial` and let them continue.

## Step 6: Deep path

Continue the three retrieval skills in DEEP mode in parallel. Pass data between skills yourself through the state:
- drug names found in `latest-evidence` sources go to `drug-intelligence`
- investigational drugs found in `clinical-trials` go to `drug-intelligence` (emerging category)

Then run `evidence-synthesis`, then let the writer enrich the answer.

## Step 7: Verify and repair

Run `citation-verification`, fix or remove failed claims, then run `quality-control`. If QC returns `NEEDS_REPAIR`, send back only the failed parts. Allow at most **2 research retries** and **1 repair cycle**. Then finalize, even if that means removing or qualifying claims and adding a limitation note.

## Step 8: Finalize and log

Set `answer_status = FINAL` and record `research_completed`. Log first token, first useful answer, retrieval complete, verification complete and final complete times separately.

## Time limits

- FAST retrieval timeout: about 3 seconds per retrieval skill.
- DEEP retrieval timeout: about 25 seconds per retrieval skill.
- Total deep-research budget: about 100 seconds.
- If a skill exceeds its timeout, mark it as `partial` and continue with the other skills.

## Rules for failures

- If one skill fails, continue with the others and tell the doctor what is missing (for example "trial registry unreachable").
- If all retrieval fails, say no evidence was retrieved. Do not answer from memory as fact.
- Never silently choose between contradictory skill outputs; pass both to `evidence-synthesis`.
- Cache stable facts (drug class, mechanism). Never cache trial status, new approvals or recent publications for long. Always keep the original retrieval date.

## Quality check

Before finishing, confirm:
- [ ] The bottom line answers the exact question asked
- [ ] Recent evidence was searched and the search date is shown
- [ ] Every finding has a year and citation
- [ ] `drug-intelligence` and `clinical-trials` ran
- [ ] Retrieval ran in parallel and the first response did not wait for deep research
- [ ] Retries were at most 2 and repairs at most 1
- [ ] Failed or partial skills are disclosed

## Examples

**"Latest treatment for relapsing MS"** -> intent `drug_treatment` + `latest_findings`; all three retrieval skills in parallel; first answer in about 5 s with bottom line, one dated recent finding and top drugs; deep research continues.

**"What causes cluster headache?"** -> intent `mechanism`; drugs and trials still run; the answer still includes acute and preventive drugs, latest approvals and emerging agents.

## References

Load `references/schemas.md` for the `ResearchState` shape and config values.
