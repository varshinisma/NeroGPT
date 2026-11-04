---
name: clinical-research-orchestrator
description: Controls the full NeuroGPT clinical research workflow - understands the doctor's question, routes it to the retrieval, drug, trial, synthesis, writing and checking skills, and returns a fast first answer followed by deeper research. Use for EVERY clinical or medical research question, such as "latest treatment for multiple sclerosis", "recent trials for migraine prevention", "compare drug A and drug B", "what is new in Alzheimer's", or "give me the evidence for X". It is the only skill allowed to decide what runs next.
---

# Clinical Research Orchestrator

## Overview

Take the doctor's question, run the right skills (in parallel where possible), and return a **recent, cited answer to the exact question asked**. Return a useful first answer in about 5 seconds, then keep researching and stream deeper results. You are the only workflow controller; no other skill calls another skill.

Hard requirements for every answer:
- Answer the **exact question asked** first, not a generic overview. A follow-up gets only the part it asks for (Step 2d).
- Search for **recent evidence** every time (default last 24 months; "latest/new" means last 12 months). Never state a clinical fact from memory alone. Three things are kept apart: conversation memory only resolves context (who or what "both", "her", "this condition" mean); stored evidence (the earlier answers and the sources they retrieved) may answer a follow-up only when it really covers what is asked (`answer_from_conversation`, Step 2d); new research fills whatever is missing.
- Every finding carries its **publication year and a citation**.
- Run the retrieval skills the question needs (Step 2c). The medicines of the condition are retrieved for EVERY clinical question (a diagnosis, a work-up, a mechanism or a case question included), because the doctor reads the medicines beside everything else; `drug-intelligence` is always chosen for a new clinical question. The report shows the Treatment Drug Landscape when medicines were found, and leaves it out when none were retrieved (no empty table). The clinical trials section follows the question.
- Never invent clinical facts. Never treat an ongoing trial as proof.

## Step 1: Start the research state

Create one `ResearchState` object (see `references/schemas.md`). Record `request_started` and set `answer_status = INITIAL`.

## Step 2: Classify the question

Pick one primary intent, by what the doctor ASKS FOR, never by the topic words in the question:
- `case_evaluation` - a patient or case is described and the doctor asks what to do: the work-up, the differential, the management, the follow-up. The findings of the case are the context, not the question.
- `diagnosis` - how a condition is diagnosed or told apart from others, without a described patient.
- `drug_treatment` - which medicines, doses or treatments to use, or how to treat a condition.
- `safety` - harms, adverse effects, risk groups or interactions of an intervention.
- `clinical_trial` - ongoing or recent trials, or emerging treatments, are the point of the question.
- `comparison` - two or more options against each other.
- `mechanism` - how or why something works or happens.
- `latest_findings` - what is new on a topic.
- `disease_research` - what is known about a condition in general.
- `comprehensive_research` - an explicit request for a full review.
Give `secondary_intents` for further parts of the question (for example a case that also asks about prognosis). A finding that merely accompanies the main problem (a behaviour, a symptom, a co-occurring condition) never sets the intent: a work-up question for a child with several findings is `case_evaluation`, not a treatment question about one of the findings. If unclear, use `disease_research` + `latest_findings` and state the assumption in the answer. Do not stop to ask clarifying questions about a question that stands alone. The one exception is a follow-up whose reference is genuinely ambiguous (Step 2d).

## Step 2b: Understand the question

Before searching, work out what the doctor is really asking:
- **asked** - what is being asked, in one short sentence.
- **purpose** - why they ask: the clinical decision or need behind it (choosing a treatment, checking safety, confirming a diagnosis, judging a plan). This is your reading, not a fact about the patient.
- **info_needed** - the kinds of information that would answer it: `treatment_options`, `efficacy`, `safety`, `dosing`, `diagnosis`, `prognosis`, `mechanism`, `guideline_recommendation`, `trial_status`, `regulatory_status`, `drug_interaction`. List ONLY the kinds the question really asks for or needs in order to be answered. Do not add `efficacy`, `safety` or `dosing` unless the question is about the effect, the harms or the dose of a medicine or another intervention; do not add `trial_status` or `regulatory_status` unless trials or approvals bear on the question. A case study that asks for a work-up and management needs `diagnosis`, `treatment_options` and `guideline_recommendation` (and `prognosis` when follow-up or outlook is asked).
- **medicines_needed** - whether medicines are what the question is ABOUT: `true` when it names a medicine, asks which medicines or doses to use, asks about the safety or approval of a medicine, compares medicines, or asks how to treat a condition for which medicines are a main treatment; `false` for a work-up, a diagnosis, how something works, or the management of a case that does not ask about medicines. This flag only sets the order and weight of the report. It never decides whether medicines are retrieved: the medicines of the condition are searched for and listed for EVERY clinical question, so never leave `drug-intelligence` out of the plan because the flag is false.
- **semantic_variants** - 3 to 5 short phrases (2 to 8 words each) that published papers might use for the SAME clinical problem in place of the question's own words: synonyms, older or newer terms, closely related terminology. They let the search find papers by meaning, not only by the question's words. Never a different condition, and never a drug the question did not name.
- **answer_format** - how the answer is best organised: `options_comparison`, `stepwise_plan`, `summary`, `table` or `yes_no_with_evidence`.
- **case_facts** - when the question describes a patient or a case study: the facts that matter, each copied word for word from the question. Never add a fact that is not written there. A feature the question calls a "tendency" or "feature" stays that feature; never turn it into a named diagnosis.
- **main_problem** and **associated_findings** - the one problem the doctor asks about (a short phrase in the question's own words) and the other findings the question mentions beside it, each copied word for word. The PubMed query names the main problem in its first group; an associated finding is never an alternative name inside that group.
- **related_topics** - only for a diagnosis, work-up, case or management question: 3 to 5 short phrases naming what a complete answer must also cover (kinds of causes to consider, tests to order, specialist evaluations). They are search topics, never statements about the patient. `[]` for any other question.

**For a follow-up**, keep these same fields, but fill them from the new message AND the resolved context of the earlier question it refers to (Step 2d). `asked` and `purpose` describe what the user asks NOW; `info_needed` lists only what the new message needs (contraindications or side effects give `safety`, dosage gives `dosing`, clinical trials give `trial_status`, prognosis gives `prognosis`); never copy the earlier question's information needs into it, and never add dosing, safety, efficacy or trial needs the message did not ask for. `medicines_needed` says whether THIS message concerns medicines. `case_facts`, `main_problem` and `associated_findings` come from the case of the earlier question, only the ones the new message depends on, each copied word for word from where it was stated. `semantic_variants` and `related_topics` belong to the new research question only.

The program checks the case facts, the main problem and the associated findings against the question. Whether trials bear on the question is expressed by `trial_status` in `info_needed` and by choosing `clinical-trials` in Step 2c; whether medicines are what it is about by `medicines_needed`. Medicines are retrieved for every clinical question: `drug-intelligence` is always chosen for a new question.

## Step 2c: Choose the skills that expand the question

You are given a catalog of the retrieval skills (name and one-line description only). From the question and its intent, choose which of them must write expanded searches for THIS question. Reply with their names in `skills`.
- `latest-evidence` - for almost every question: the literature (evaluation, work-up, differential diagnosis, treatment, management, latest findings, guidelines).
- `clinical-trials` - when ongoing or recent trials, or emerging treatments, bear on the question.
- `drug-intelligence` - ALWAYS, for every new clinical question: the medicines used for the condition, for its associated symptoms or for its co-occurring problems are retrieved whatever the question asks (treatment, diagnosis, work-up, mechanism, a case). When nothing is found the report simply leaves the drug section out.

A case study that asks for work-up and management needs `latest-evidence` and `drug-intelligence`, and `clinical-trials` only if trials bear on what it asks. A question about how something works needs `latest-evidence` and `drug-intelligence`. Choose a skill only for an information need the question really has, except `drug-intelligence`, which is always chosen for a new clinical question. Never choose a skill that is not in the catalog.

**For a follow-up**, the rule that `drug-intelligence` is always chosen belongs to NEW clinical questions (it keeps the Treatment Drug Landscape beside the research). A focused follow-up chooses the skills by what it asks and does not rerun them all: contraindications, side effects or dosage of the earlier drugs choose `drug-intelligence`; clinical trials for the same condition choose `clinical-trials`; mechanism, prognosis, guidelines, alternatives or any other question about the condition chooses `latest-evidence`. A follow-up already covered by stored evidence chooses no retrieval skill. A follow-up never retrieves a Treatment Drug Landscape or a condition overview just because the earlier report had one. A message that turns out to be a new independent clinical question follows the rule for new questions.

## Step 2d: Follow-up questions (when earlier questions exist in the conversation)

A follow-up is answered with only what it asks, never with the whole report again. Do not pattern-match on a fixed list of follow-ups (dosage, side effects): a follow-up can ask ANYTHING about the earlier answers. Read the message, work out what it actually asks, and work out how it relates to the earlier answers.

The order of work for a follow-up: conversation memory -> reference resolution (1, 2) -> intent of THIS message (3) -> information needs (3) -> entity resolution (2) -> evidence sufficiency check (4) -> standalone research question -> focused research (5) -> evidence synthesis -> answer -> citation verification and quality control (Step 7). Keep three things apart. **Memory** resolves context and entities; it is never clinical evidence. **Stored evidence** (the earlier answers and the sources they retrieved) answers only what it adequately covers. **New research** fills the gap, and every new clinical claim needs its own evidence and citation.

### 1. How does the message relate to the earlier answers? (write it in `relation`, one sentence)
Decide from the MEANING of the message and the earlier answers, never from particular words. A message is a follow-up when it cannot be understood, or would be answered differently, without the earlier conversation; it is a new question when a reader who saw only that message could research it completely. Pronouns, "the above", "both" and similar words are not the test: a message can lean on the earlier answers without any of them, and can contain them and still stand alone. To decide, ask yourself: what is this message about, and is that subject present in the earlier questions and answers?
- **The same patient or case**: the message is about the person the earlier question described, whether or not it says so. The case facts in the memory are the subject.
- **The same drugs**: the message is about the medicines the earlier answer already listed (it asks something ABOUT them: their dose, harms, contraindications, use for another condition, how they compare). Set `needs_drugs=true`. A message that asks WHICH drugs, or asks for "the drugs" or "more drugs" for an earlier question or condition, is NOT about the earlier drugs: it is about the condition (next bullet). Never restrict the research to the drugs an earlier answer happened to list, because that answer may have found only a few or none.
- **The medicines of the same condition or case**: the message asks for the drugs, medicines or treatments of an earlier question's condition or case (it wants them found, not facts about drugs already listed). The condition and the case facts are the subject. Set `needs_drugs=false` and `medicines_needed=true`, write `ask` as what is wanted in the person's own words (no drug names and no drug classes), and leave `drugs` empty: the program discovers all the medicines of that condition afresh, including the newest ones, and shows only the drug tables. Details about the earlier drugs, if any are listed in the memory, are not reused. Earlier drugs are copied from the memory only under the previous bullet (the message is ABOUT them), never here.
- **The same condition from a new angle** (prognosis, work-up, guidelines, mechanism, trials, anything else): the condition is the subject. Drugs are NOT the subject unless the message is about them.
- **The evidence behind an earlier statement**, or something to explain, compare, simplify or reformat: usually answered from the memory.
- **A different subject** (another disease, another patient, another population, a new case, or something unrelated to medicine) that stands alone: a new question, even in the middle of a conversation. Copy nothing from the earlier answers.
- **You cannot tell** which earlier subject it means, and the answers would differ: call `ask_for_clarification` with one short question. Do not guess when a wrong guess would research the wrong thing.
A change of what is asked (symptoms, then treatment, then trials) does not make a message new: the topic can stay while the information needed changes, and the next step (2) is decided again for every message.
Do not use the drugs of the earlier answers just because they exist. Use them only when the message is about those drugs. `medicines_needed` for a follow-up is true only when the message itself asks about medicines; a follow-up about prognosis, work-up, mechanism or anything else is false.

### 2. Which earlier question, and which entities? (reference resolution)
- Resolve references such as "both drugs", "these medicines", "the above drugs", "this condition", "the patient", "the first one" and "the second one" to a specific earlier question, then take the entities from THAT question's memory only. Never add drugs, conditions or patient facts from other, unrelated questions.
- "Both" resolves to exactly the two items the intended question named or compared. "All the drugs" resolves to all the drugs of the intended referenced answer. "The first one" and "the second one" follow the order in which that answer listed them.
- If the message names an older question in its own words, use that question's context, not the latest one.
- If the reference is genuinely ambiguous and the different readings would give materially different research (for example "both" could mean two conditions or two drugs, and the answer would differ), call `ask_for_clarification` with one short question and do not research yet. If the readings give the same research, pick the most natural one and say which you chose in `relation`.

### 3. What does it ask for NOW? (write it in `info_needed`, and for facts about the earlier drugs in `attributes`)
A follow-up is classified again from what the user asks now; it never inherits the previous question's intent. Example (X is a condition, Drug A and Drug B were named in the answer): "What are the treatment options for X?" is `drug_treatment`. Then "What are the contraindications of both drugs?" is `safety`, subject Drug A and Drug B, information needed: contraindications. "What dosage should be given?" is `drug_treatment` with `dosing`, the intended drug or drugs resolved from memory. "Are there any clinical trials for this condition?" is `clinical_trial` with `trial_status`, subject X. Same topic, new intent each time.
- If it asks for facts about the earlier drugs, name exactly those facts in `attributes`, in the person's own words: for example the contraindications, use in pregnancy, what to monitor, how it is given, the mechanism, the cost, who should avoid it. One table with exactly those columns is researched and written. A question about contraindications gets contraindications, not dosage and not side effects.
- `info_needed` is the broader kind of information (efficacy, guideline_recommendation, trial_status, prognosis, diagnosis, mechanism, treatment_options, and so on); list only what the message asks. A dosage question gets `dosing` alone; a question about side effects gets `safety` alone. Never add a kind the message did not ask for.
- A message with several asks is ONE follow-up: list every ask; each gets its own part and the parts are researched together.

### 4. Is the existing evidence enough? (check this before any search)
Look at the earlier answers and the stored sources for the resolved subject and the information asked. Memory by itself proves nothing. Existing evidence is reused only when it adequately supports the requested answer; otherwise the missing part is researched. Three outcomes: (A) sufficient -> `answer_from_conversation`; (B) insufficient -> focused research for the missing information only, reusing the relevant stored sources; (C) the reference is ambiguous -> `ask_for_clarification`. If `answer_from_conversation` finds the earlier answers do not hold what is asked, it hands over to research instead of guessing.

### 5. How is it handled?
1. **Already in the earlier answers** (explain, compare the listed drugs, pick one, simplify, table, handout, thanks): `answer_from_conversation`. No search. Take only what the question asks from the stored answers. No FAST path and no DEEP path run.
2. **Needs research, and relates to the earlier answers**: `research` with `kind=follow_up`, `scope=focused`. Only the part it asks for is researched and written, with no summary and no other section. Write `ask` without naming the earlier drugs. Facts about drugs go in `attributes`; trials in `info_needed`; evidence, guidelines, alternatives, prognosis and everything else are answered from a literature search on that one question. The research question is built from the resolved subject (the drugs, condition and patient facts copied from memory) and the new intent, so it is complete on its own; the generic FAST search is never run against the unresolved raw follow-up sentence (Step 5). Reuse the stored sources that cover the request and search only for what they do not.
3. **A different subject, or a new population or condition on its own**: `scope=full`, and put the changed topic in `own_condition`. It is researched as a new question, with the whole report structure. But a message that points back ("do these drugs also treat <another condition>?") is a follow-up: set `needs_drugs=true`, put the condition in `own_condition`, and keep `scope=focused`.
4. **Ambiguous**: `ask_for_clarification`. Nothing is researched until the subject is clear; no FAST and no DEEP run.

### What the memory gives you
- For each earlier question: the drugs it named and the drugs its answer listed, its key conclusions, and its case facts (age, patient, comorbidities, current medicines, population), each tagged with the question it came from. Take "her", "the patient", "both", "the above" from there; never ask again for a fact the memory holds, and never write one it does not hold.
- "The above", "the previous question", "this condition" and "the last answer" mean the MOST RECENT earlier question (the highest number), unless the message names another in its own words. Never merge answers about different conditions unless the message says "all" or names them.
- The sources the earlier questions retrieved are kept for the conversation, and a follow-up starts from them; the program searches only for what they do not already cover.
- Memory is for reference resolution, not clinical evidence. Identify which earlier question a reference belongs to, and never combine drugs, conditions or patient facts from unrelated questions.
- Use the previous sources when they cover the request and retrieve what is missing; a claim that is new to the conversation needs its own evidence and citation.
- A message that depends on something from the earlier answers is a follow-up. If it is genuinely unclear which earlier subject it means and the research would differ, ask (section 2).

## Step 3: Extract entities

Extract disease, drug, biomarker, population, intervention, outcome and date range. Normalize names (generic drug names, standard disease names) and keep synonyms. If no drug is named, record the condition or mechanism so `drug-intelligence` can find the relevant drugs. For a follow-up, the entities are the ones resolved from the referenced earlier question (Step 2d), not the words of the raw message.

## Step 4: Plan the routing

Include the retrieval skills the question needs; none of them is required for every question, except that a new clinical question always includes `drug-intelligence` (Step 2c). A new question adds `latest-evidence` and `clinical-trials` as Step 2c says. A focused follow-up runs only the skill that answers what is asked, and none when stored evidence suffices. Then, when research ran, `evidence-synthesis`, `clinical-answer-writer`, `citation-verification`, `quality-control`. The intent and the information needed decide which retrieval skills run. For comparisons, run retrieval once per item, in parallel.

## Step 5: Fast path (about 5 seconds)

Which path runs depends on what Step 2d decided:
- **New independent clinical question** (including a new topic in the middle of a conversation): FAST + DEEP as described below, unchanged.
- **Follow-up already answered by stored evidence**: `answer_from_conversation`. No FAST, no DEEP.
- **Follow-up that needs new research**: resolve the context first (drugs, condition, patient facts), classify the new intent, build the standalone research question and check what the stored sources cover. Never run the generic FAST search on the unresolved raw sentence. If relevant stored evidence allows a quick first answer, it may be streamed while the focused research continues, and it must contain no unsupported claim. Research only the missing information.
- **Ambiguous follow-up**: ask for clarification. No FAST, no DEEP.
FAST and DEEP always use the same resolved standalone question and the same entities; they never search different subjects because one uses the raw message and the other the resolved memory.

For a new independent clinical question, the FAST path is run by the program, before the plan exists: it searches the literature, the FDA label records, the trial registry and the web in parallel from the question's own words (about 3 s timeout each). No retrieval skill is chosen or loaded at this stage, so none can be skipped; the plan call starts when the first answer starts. The first answer lists medicines only from the FDA label records; for a question that does not concern medicines it says so. As soon as at least one relevant recent source is in, call `clinical-answer-writer` for Phase 1 and record `first_response`. Do not wait for slower sources; mark them `partial` and let them continue. Only the DEEP path is controlled by the chosen skills (Step 6).

## Step 6: Deep path

Continue the retrieval skills the question needs (Step 2c) in DEEP mode in parallel. Pass data between skills yourself through the state:
- drug names found in `latest-evidence` sources go to `drug-intelligence`
- investigational drugs found in `clinical-trials` go to `drug-intelligence` (emerging category)

Then run `evidence-synthesis`, then let the writer enrich the answer.

**For a follow-up that needs research**, the deep path uses the resolved standalone question and the drugs and condition copied from memory, selects the retrieval skills by the new intent (Step 2c), searches only for the missing information, and reuses stored sources only when they are relevant and cover the request. Source dates and citations are kept. It does not retrieve an unrelated Treatment Drug Landscape or a full condition overview because the earlier report had one. The drug transfer above (drugs from `latest-evidence` and from `clinical-trials` into `drug-intelligence`) belongs to new independent clinical questions.

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
- [ ] The retrieval skills chosen in Step 2c ran, and the ones not chosen did not
- [ ] Every information need in `info_needed` has a search on a source that can answer it
- [ ] The Treatment Drug Landscape lists the medicines when any were retrieved and is left out when none were; the Clinical trials section lists its evidence or says it does not apply
- [ ] A follow-up answered only what it asked, used the right earlier entities, and every new claim has its own citation
- [ ] Retrieval ran in parallel and the first response did not wait for deep research
- [ ] Retries were at most 2 and repairs at most 1
- [ ] Failed or partial skills are disclosed

## Examples

**"Latest treatment for relapsing MS"** -> intent `drug_treatment` + `latest_findings`; all three retrieval skills in parallel; first answer in about 5 s with bottom line, one dated recent finding and top drugs; deep research continues.

**"What causes cluster headache?"** -> intent `mechanism`; `latest-evidence` and `drug-intelligence` are chosen (medicines are retrieved for every clinical question); the Treatment Drug Landscape lists the medicines found, or is left out when none were retrieved; the Clinical trials section says it does not apply.

**"A child with a growth problem, a new symptom and a behaviour change: what is the work-up and management?"** -> intent `case_evaluation` (secondary `diagnosis`); `info_needed`: `diagnosis`, `treatment_options`, `guideline_recommendation`; `medicines_needed` false; the growth problem is the main problem, the symptom and the behaviour are associated findings (never the intent); `latest-evidence` and `drug-intelligence` are chosen, and `clinical-trials` only if the case's needs call for it.

### Follow-up examples
(X and Y stand for conditions, Drug A and Drug B for the drugs the answer to the first question named.)

**Q1 "What are the treatment options for X?" then "What are the contraindications of both drugs?"** -> follow-up; Drug A and Drug B resolved from memory; intent `safety`; `info_needed` contraindications (`attributes` contraindications); stored evidence checked; if it is not enough, focused research for Drug A and Drug B; no generic FAST search; the answer is the contraindications with citations only.

**Q1 as above, then "What is the dosage of these drugs?"** -> follow-up; the intended drugs resolved from memory; intent `drug_treatment`; `info_needed` `dosing`; focused dosage research if the stored evidence does not cover it; the treatment report is not repeated.

**Q1 "What is the treatment for X?" then "Are there any recent clinical trials for this condition?"** -> follow-up; X resolved from memory; intent `clinical_trial`; `info_needed` `trial_status`; focused clinical-trial research; no unrelated drug information.

**Q1 as above, then "Compare the two drugs you mentioned."** -> follow-up already answered: if the earlier answers hold enough, `answer_from_conversation`, no retrieval.

**Q1 "What is the treatment for X?" then "What causes Y?"** -> a new independent clinical question; Drug A and Drug B are not copied from X; the normal new-question workflow runs.

## References

Load `references/schemas.md` for the `ResearchState` shape and config values.
