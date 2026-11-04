---
name: drug-intelligence
covers: dosing, safety, regulatory_status, drug_interaction, treatment_options, guideline_recommendation, new_drug_analysis
description: Finds ALL the medicines for the condition a new clinical question is about, whatever the question asks (treatment, diagnosis, work-up, mechanism, a case), and gives each drug its full information. The agent reads the intent and the condition, writes the searches, and the skill returns (1) newly approved drugs with an in-depth analysis, (2) all relevant approved, guideline, off-label, emerging and restricted drugs, (3) tables sorted from most recent to oldest, and (4) mandatory attributes for every drug (dose, route, frequency, duration, adjustments, contraindications, adverse effects, interactions, source). The orchestrator decides when it runs and in which mode (see Modes): the full landscape for a new clinical question, or a focused follow-up on the drugs it resolved from the conversation.
---

# Drug Intelligence

## Overview

When the question concerns medicines (their use for the condition, doses, safety, approvals, interactions), return a complete and correctly categorized list of the relevant drugs, including the latest approvals with an in-depth analysis, presented as tables sorted from most recent to oldest.

This skill has two phases:

- **Phase A: Query planning** (Steps 1-3): decide what to search, in which sources, in two passes.
- **Phase B: Synthesis and output** (Steps 4-11): categorize, deep-dive, fill mandatory attributes, sort, and print tables.

Rules that always apply:

- If the condition has no drug treatment, say so explicitly with a source.
- If the question is not clinical, or no medicine need can be written for it, return no searches (`{"queries": []}`): the report leaves the section out. Never fill the section with unrelated content. For a NEW clinical question the orchestrator always selects this skill and the condition-based discovery of Step 2 runs, even when the question is about a diagnosis, a work-up, a mechanism or a case.
- Today's date comes from the system. Compute every "recent" window from it. Never hardcode a year.

---

## Modes (the orchestrator decides which one; this skill never decides it)

The orchestrator is the only controller. It decides whether the message is a new question or a follow-up, resolves references from the conversation memory, identifies the intent and selects this skill. This skill performs the drug research it is asked for and adds no classifier of its own.

- **New clinical question mode:** the full workflow below (Phase A, Phase B, the Treatment Drug Landscape). It is unchanged.
- **Follow-up mode:** the orchestrator sends a follow-up about drugs already discussed (`plan.focus` contains `drugs`, `safety` or `drug_facts`) together with the resolved drug names, the `attributes` (the specific facts asked), `info_needed` and the scope. This skill researches exactly that and nothing else. See "Follow-up mode" below; the Step 2 condition rules, Pass 1, Step 7 and the landscape tables of Step 9 belong to new-question mode.

---

# PHASE A: QUERY PLANNING

## Step 1: Define the scope

From the entities, set the condition, subtype, line of therapy and population (age group, pregnancy, renal/hepatic status, region). For mechanism or symptom questions, map to the conditions the drugs treat.

Detect the user's region if stated (for example India -> also check CDSCO; EU -> EMA). Default regulators: FDA and EMA; add CDSCO when the user is in or asks about India.

## Step 2: Which condition, and how much to retrieve (new-question mode)

The question's words never decide whether drugs are retrieved. Read the understanding the orchestrator gives you (intent, condition, `main_problem`, `associated_findings`, `case_facts`, `info_needed`, population) and work out WHICH CONDITION or conditions the question is about. Then retrieve every medicine that belongs to that condition, whatever the intent is: treatment, drug choice, dosage, safety, comparison, diagnosis, work-up, mechanism, a described case, or "what is new". A question that names two or more conditions gets the whole discovery for each condition.

- **The condition decides WHICH drugs.** Use the condition, subtype and patient group from the understanding. For a case, use the main problem and the associated findings: the medicines used for the condition itself, for a confirmed cause of it, and for its associated problems and co-occurring conditions as the retrieved sources describe them. For a mechanism or symptom question, map to the conditions the drugs treat. Never use a condition that neither the question nor a retrieved source gives.
- **The intent decides WHAT IS EMPHASISED, not whether drugs are found.** A treatment or drug-choice question gets the full query set (Step 3, Pass 1 and Pass 2) with the approvals and the newest drugs first. A dosage or safety question puts the label dose, adjustments, contraindications and adverse effects first. A comparison runs the discovery for each compared option. A diagnosis, work-up, mechanism or case question still runs the full Pass 1 discovery for the condition, but the drugs are listed after the answer the question asked for and described as "medicines used for this condition or its associated problems", with those used only if a feature is present written as conditional statements ("if present"). Whatever the intent, the newest drugs are always part of the picture (next point).
- **The newest drugs are always retrieved, whatever the question.** For the question's condition, every time: search the regulators (FDA, EMA, and CDSCO when relevant) for approvals, new indications and label changes in the recency window (Step 4), search the literature and the trial registry for drugs that are new or in late-phase development for it, and read what retrieved sources say is newly used. A drug approved or label-expanded for the condition in the window is categorised `newly_approved` and gets the New-Drug Deep Dive (Step 6) and its place at the top of Table A, even when the question is about diagnosis, a work-up or a mechanism and never says "new" or "latest". Drugs still in trials go to Table B as "Investigational - not approved", never mixed with approved ones. If nothing new is found in the window, say "Searched, none found" and name the regulators and sources searched.
- **Every drug found gets its full information:** the mandatory attributes of Step 7 (dose, route, frequency, duration, adjustments, contraindications, adverse effects, interactions, approval status per regulator, source and year), all from retrieved sources.
- **Do not stop at the first list.** Expand classes to specific medicines (Pass 1, item 5), include off-label, emerging and restricted drugs under their own categories, and keep the patient group in every query.
- A search that finds nothing is not proof that no medicine exists. If, after the full discovery, the condition has no drug treatment, say so with a source (see the rules above).

## Step 3: Two-pass search plan

You are given the doctor's question, its intent and what was understood from it. Write searches for needs about medicines: which are used for the condition and for its associated problems in THIS patient group, their doses, their safety, approvals and label changes, interactions. Write the Pass 1 discovery searches for the question's condition whatever its intent (Step 2); the intent changes the priority of the needs, not whether the drugs are discovered.

### Pass 1: DISCOVER (find which drugs exist)

Write ALL of these as separate searches:

1. **Guideline recommendations** (`guideline_sites`): first-line, second-line and adjunct medicines the guideline organisations recommend, with guideline year.
2. **Regulator announcements** (`regulator`): approvals, new indications, label changes and safety communications in the recency window (Step 4), for the question's condition whatever the question asks. Run one query per relevant regulator (FDA, EMA, CDSCO when relevant).
3. **Medicines for the condition by generic name** (`web`).
4. **Medicines for associated symptoms or co-occurring conditions** (`web`): for the condition the question asks about: the symptoms and co-occurring conditions that the retrieved sources describe for it. Never name a symptom or condition that no retrieved source describes for this condition.
5. **Class-to-specific expansion** (`web`): for any class of medicines that a source names, search for the specific medicines of that class.
6. **Emerging drugs** (`trials`): late-phase trials of drugs for the condition (`status: any`, `drug_only: true`), plus `pubmed`/`europepmc` for phase 2-3 reports.
7. **Label basics** (`fda_label`): with `condition_terms` and the `drug_names` the question names.

### Pass 2: DEEP DIVE (analyse each drug found in Pass 1)

After Pass 1, for **every `newly_approved` drug and every drug in the guideline list**, write name-specific searches:

- Official label: dose, route, frequency, duration, adjustments, warnings (`fda_label`; add `regulator` pages for EMA/CDSCO labels when relevant).
- Pivotal trial(s): primary publication, NCT ID, effect sizes (`pubmed`, sort by `relevance`, `study_types: ["rct","meta_analysis"]`; `europepmc` as backup).
- Post-marketing safety and label changes (`regulator`, `pubmed` sorted by `date`).
- Interactions (`fda_label`, `web`).

**Pass 2 naming rule:**
- **New question:** Pass 2 searches may name only drugs the question names or drugs that appeared in retrieved Pass 1 sources. Never name a drug from your own memory.
- **Follow-up:** the previously discussed drug names come from the orchestrator's resolved conversation memory (they are supplied, not guessed). This skill does not resolve references itself.
- **In both modes:** clinical facts, doses, adverse effects, contraindications, approval status and numerical findings come from retrieved evidence, never from memory alone.

### Sources you may target (each query names ONE `target`)

- `pubmed`: published studies, reviews, trial reports, guideline papers. Args: `sort` (`relevance` or `date`), `years_back` (1 to 15), `study_types` (`rct`, `meta_analysis`, `systematic_review`, `guideline`, `review`, `negative_results`), `open_access` (true to prefer free full text).
- `europepmc`: second literature index. Use for needs PubMed may miss, or as the `alternate` of a `pubmed` query. Args: `sort`, `years_back`.
- `guideline_sites`: recommendations from guideline organisations.
- `regulator`: regulators' announcements: approvals, new indications, safety communications.
- `fda_label`: official FDA drug labels. Args: `condition_terms` and `drug_names` (ONLY medicines the question names, medicines found in retrieved sources in Pass 2, or, in follow-up mode, the drugs the orchestrator resolved from the conversation).
- `web`: anything else, as a targeted search; never the default for everything.
- `trials`: trial registry records. Args: `condition`, `intervention`, `status` (`recruiting`, `completed`, `any`), `drug_only` (true for drug trials only).

Choose the source by what the need is. For every ESSENTIAL need, write a query for EACH listed source that can really answer it (two or three sources is normal, each as its own query with the same `need_id`), phrased the way that source is searched. A `supporting` need may use one source. Skip a source only when it cannot answer the need.

### Output (query planning)

Reply with ONE JSON object and nothing else:

`{"queries": [{"need_id": "n1", "need": "the need in 2 to 4 words", "target": "one source from the list above", "query": "a natural-language phrase (not used by fda_label or trials)", "args": {}, "serves": "the one information need from the list that this search answers", "priority": "essential or supporting", "pass": 1}]}`

- `need_id` groups queries that serve the same need. `priority` is `essential` for a need that directly answers what was asked and `supporting` for context. Only essential needs are retried elsewhere.
- `pass` is `1` (discover) or `2` (deep dive).
- `serves` names which listed information need (from `info_needed` in the understanding) the search answers; leave it out if none fits.
- Add an `alternate` (`{"target": ..., "query": ..., "args": {}}`) only for an essential need where a DIFFERENT source would give DIFFERENT evidence (for example guideline sites beside the literature).
- Keep the condition and the patient details from the question in every query.
- **If your source is not needed for this question, reply `{"queries": []}`.** Never write searches only to be thorough.

### Example query set for a treatment question

```json
{"queries": [
 {"need_id":"n1","need":"guideline first-line drugs","target":"guideline_sites","priority":"essential","pass":1},
 {"need_id":"n2","need":"recent approvals","target":"regulator","priority":"essential","pass":1},
 {"need_id":"n3","need":"label dosing","target":"fda_label","priority":"essential","pass":1},
 {"need_id":"n4","need":"drugs by generic name","target":"web","priority":"essential","pass":1},
 {"need_id":"n5","need":"class to specific drugs","target":"web","priority":"essential","pass":1},
 {"need_id":"n6","need":"emerging late-phase","target":"trials","args":{"status":"any","drug_only":true},"priority":"supporting","pass":1},
 {"need_id":"n7","need":"pivotal trial","target":"pubmed","args":{"study_types":["rct"],"sort":"relevance"},"priority":"essential","pass":2}
]}
```

---

# PHASE B: SYNTHESIS AND OUTPUT

## Step 4: Recency window

Recency window = last **24 months from today's date**. If fewer than 2 `newly_approved` drugs are found, extend to **36 months** and say so in the output. Record the window used.

## Step 5: Categorize every drug (multi-label)

A drug may carry several labels. Use `categories` as a list, not a single value.

1. Approved for this indication -> `standard` (add `newly_approved` if approved or label-changed for this indication within the recency window)
2. Named in a current guideline -> add `guideline` (record guideline name and year)
3. Evidence supports use outside its approved indication -> `off_label`
4. In late-phase trials or under regulatory review, not approved -> `emerging`
5. Withdrawn, suspended, restricted, or carrying a new boxed warning or REMS -> `withdrawn_restricted` (keep these visible; do not drop them)
6. Supportive or symptomatic drugs -> keep them when a retrieved source describes them as used for the question's condition, its symptoms or its co-occurring conditions, with relation `treats_associated_symptom_or_comorbidity` (they are shown in a separate table, never as established treatments); leave out a drug whose source says nothing about this condition
7. Anything else -> leave it out

Also set **`line_of_therapy`**: `first_line`, `second_line`, `third_line_or_later`, `adjunct`, `supportive`, or `not_specified`.

Never put an investigational drug in an approved category.

Record approval status **per regulator** (FDA / EMA / CDSCO). Status often differs between them. Do not assume approval in one region means approval in another.

## Step 6: New-drug deep dive (every `newly_approved` drug)

For each drug in the recency window, record all of the following. If a field cannot be found, set it to `null` and add it to `missing_context`.

- **Regulatory:** regulator, approval date, pathway (standard / accelerated / conditional / priority), approved indication wording, boxed warning, REMS.
- **Pivotal trial(s):** trial name, NCT ID, phase, N, comparator, primary endpoint, effect size (HR/OR/absolute difference with CI).
- **Secondary endpoints and subgroups:** include the patient group named in the question.
- **Place in therapy:** first/second line, add-on or replacement, versus existing standard of care.
- **Safety:** top adverse events with rates, monitoring requirements, discontinuation rate.
- **Guideline status:** included or not yet included (name guideline and year).
- **Access:** regional status (CDSCO / EMA if the user is outside the US); generic or biosimilar availability if sources state it.
- **Limitations:** surrogate endpoints, short follow-up, no head-to-head data, small subgroup numbers.

## Step 7: Mandatory attributes for EVERY drug (full landscape report)

In a full Treatment Drug Landscape every field below is required. A focused follow-up report requires only the attributes that were asked, plus any essential clinical context needed to answer them safely (see Follow-up mode); it does not fill the other fields. The fields are **required** for the full landscape. If truly unavailable, set the value to `null` and give a reason. Never leave a cell blank.

| Field | Required |
|---|---|
| Generic name (Brand), class, mechanism | Yes |
| Categories, line of therapy | Yes |
| Indication and population | Yes |
| Approval status per regulator, with date for THIS indication | Yes |
| **Dose, route, frequency, duration** | Yes |
| Renal / hepatic / age adjustments | Yes |
| Contraindications, boxed warning | Yes |
| Major adverse effects | Yes |
| Key interactions | Yes |
| Source, source year, retrieval date | Yes |

**Completeness rule (full landscape):** if dose, route or frequency is `null` after Pass 1, run a second `fda_label` search using the drug's generic name (Pass 2). If still null, show "Not found in retrieved sources" and add to `missing_context`. Never estimate.

## Step 8: Keep dosing types separate

Store four separate doses and never merge them: approved label dose, guideline dose, trial dose (with trial ID), investigational dose.

- The **Dose column** in the main tables shows label and guideline doses only.
- Trial and investigational doses go in a separate column labelled **"Not for routine use"**.

## Step 9: Sort order and tables

Print the **Treatment Drug Landscape** as separate tables. Never mix emerging drugs into the approved table, because they have no approval date and could be read as approved.

**Table A: Approved drugs** (`standard`, `newly_approved`, `guideline`, `off_label`)
- Sorted by `indication_approval_date` **DESC** (most recent first). This is the date the drug was approved or label-expanded for THIS condition. Tie-break: first-ever approval date.
- Drugs with no retrievable date go at the bottom, marked "date not found".
- Off-label drugs (no approval date for this indication) go at the bottom, marked "off-label".

**Table B: Emerging / investigational** (separate; label every row "Investigational - not approved")
- Sorted by phase DESC (phase 3, then 2), then latest trial update date DESC.

**Table C: Withdrawn / restricted** (only if any exist)
- Sorted by action date DESC.

**Table columns (Table A):**
`Approval date (indication)` | `Regulator(s)` | `Generic (Brand)` | `Class / mechanism` | `Line of therapy` | `Categories` | `Dose / route / frequency / duration` | `Adjustments` | `Contraindications / boxed warning` | `Major adverse effects` | `Key interactions` | `Source (year, retrieved)`

After Table A, print a **New-Drug Deep Dive** block for each `newly_approved` drug (Step 6 fields), in the same recent-to-old order.

## Step 10: Cross-check (FAST and DEEP)

This applies to every dose, indication or approval date this skill shows, in both modes; a focused follow-up checks only the facts it shows.

- **Both modes:** verify every dose against the official label or a current guideline before showing it.
- **DEEP only:** also check against regulator sources and `clinical-trials`.
- If sources disagree on a dose, indication or approval date, keep both with sources and set `conflict: true`. Do not silently pick one.

## Step 11: Flag what is missing

Set unknown values to `null` and add a `missing_context` note (for example "pediatric dose not found"). Show "Searched, none found" for any empty category, and say which sources were searched.

---

## Follow-up mode (the drugs come from the orchestrator's resolved conversation memory)

When the orchestrator sends a follow-up about drugs already named in the earlier answers (`plan.focus` contains `drugs`, `safety` or `drug_facts`), this skill does not run Pass 1, does not rediscover the drugs of the condition and does not rebuild the Treatment Drug Landscape. The orchestrator supplies the resolved drugs, `attributes`, `info_needed` and the scope; if they are missing or empty, do not guess them from the wording: research only what was supplied.
1. **Drugs:** use exactly the drugs the orchestrator supplied from the conversation memory (the stored answers). Do not resolve references yourself, do not discover new drugs and do not drop supplied ones.
2. **Research exactly what is asked, nothing more.** Take the requested facts from `attributes` (the person's own words) and `info_needed`. The output holds exactly those attributes: ONE table, one row per drug, one column per requested attribute. Several related asks go into the same table as several columns. Do not add a column just because the full landscape would have it (no dosage, frequency, adverse-effect or interaction column unless asked). The word `safety` in `info_needed` does not by itself mean "side effects": read what was actually asked.
   - **Contraindications only** -> contraindications (and the boxed warning if the label has one) for each drug, from the matching part of its FDA label and from papers or pages searched for exactly that. No dosage, no frequency, no adverse effects, no interactions.
   - **Side effects** -> adverse effects of each drug, common and serious where the sources support it. No dosing and no unrelated attribute.
   - **Monitoring** -> what to monitor, and when, only.
   - **Pregnancy (or lactation, children, kidney or liver impairment)** -> the pregnancy-related safety and label information only.
   - **Dosage** -> dose, route, frequency, titration, duration and the relevant adjustments for the condition and population, verified against the official label and current guidelines. Not the landscape, no safety column.
   - **Interactions** -> the interactions of each drug, only.
   - **Any other fact** (mechanism, how it is given, cost, who should avoid it) -> exactly that, from the label and from papers or pages searched for it.
3. **Start from the stored sources:** the papers, trial records and pages the earlier questions retrieved are already in the state. Decide whether they actually speak of the requested attribute for each drug. Reuse them where they do; search again only for a drug whose stored sources do not cover what is asked. Do not repeat a search just because this is a new follow-up.
4. **Freshness:** the FDA label is always looked up again, and approval status, label changes and safety alerts are never reused from a stored or cached source. Keep the retrieval date and the publication year of everything shown.
5. **What is required in a focused report:** only the requested attributes, plus the essential context needed to read them safely (for example the population a dose applies to). A value no source states says "Not found in retrieved sources" and is added to `missing_context`. Never invent a dose or any clinical fact. All claims are supported by a cited source.
6. **Nothing else is printed:** no summary, no landscape tables, no recently-approved table unless asked, no empty tables.
7. **A follow-up about a different condition:** do not decide the mode from the change of condition or population; follow the orchestrator's routing. If the orchestrator marks the message as a follow-up about the earlier drugs ("can these drugs also treat <another condition>?"), use the supplied drugs and research their evidence and approval status for the new condition only: approved indications, off-label evidence and investigational use, kept clearly apart, and not the landscape of the earlier condition. If the orchestrator marks it as a new question (the user asks about the other condition without referring to the earlier drugs), run the full new-question workflow.
8. **FAST and DEEP:** this skill has no FAST or DEEP controller of its own for a follow-up. Stored evidence that is sufficient -> no retrieval (the orchestrator then answers from the conversation). Missing evidence -> focused research for exactly the missing facts of the supplied drugs. A follow-up never launches the FAST landscape search.
9. Every rule below still applies (no invented doses or numbers, cite with year, no investigational drug presented as approved). The drug *names* come from the orchestrator's resolved memory, and every dose, side effect, contraindication and number still comes from a retrieved source.

---

## Rules

- **Never infer or invent a dose.** Retrieve it from a reliable source and cite it with the year, otherwise return `null`.
- Never list an investigational drug as approved.
- Never turn investigational dosing into routine treatment.
- Pass 2 searches may name only drugs the question names or drugs found in retrieved sources (new question), or the drugs the orchestrator resolved from the conversation (follow-up). Never name a drug from your own memory.
- Doses, adverse effects, contraindications, approval status and numbers come from retrieved evidence in both modes, never from memory alone.
- Do not give patient-specific prescribing advice; give sourced, population-level information.
- Never serve approvals or safety alerts from a stale cache.
- Always end the output with: *"Population-level information from the cited sources, not a prescribing recommendation. Verify against the current label before use."*

## FAST and DEEP

The FAST and DEEP workflow below is for a new clinical question. For a follow-up, the orchestrator's decision applies (Follow-up mode, item 8).

**FAST:** all newly approved drugs found in the recency window (not just one) plus the top established / guideline drugs, each with a source and a label-verified dose. Deep-dive fields limited to regulatory details, pivotal trial and key safety.

**DEEP:** the full categorized list, all four dose types, full new-drug deep dive, approval dates per regulator, safety alerts and cross-checks.

## Quality check

- [ ] Ran for this query, even if it was not a treatment question (new clinical question)
- [ ] Follow-up: used only the drugs the orchestrator supplied, researched only the requested attributes, and printed no other column or table
- [ ] Follow-up: reused the stored sources that cover the request, rechecked the FDA label, and marked every unstated value "Not found in retrieved sources"
- [ ] The condition(s) of the question were identified from the understanding, and Pass 1 discovery ran for each, whatever the intent
- [ ] No drug search was skipped or shrunk because the question's words did not mention treatment
- [ ] Pass 1 and Pass 2 both run; Pass 2 named only drugs found in sources
- [ ] All categories searched; empty ones explained
- [ ] Latest approvals and newest drugs searched, dated, and recency window stated, for every new clinical question and whatever its intent
- [ ] Every `newly_approved` drug has a deep-dive block
- [ ] Table A sorted by indication approval date, newest first; Tables B and C separate
- [ ] Every mandatory attribute filled or `null` with reason
- [ ] Every dose sourced and label-verified, with dose types kept separate
- [ ] No investigational drug in an approved category
- [ ] Per-regulator status shown; withdrawn/restricted drugs shown
- [ ] Safety alerts and conflicts shown, not hidden
- [ ] Every drug has a source and retrieval date
- [ ] Disclaimer line printed

## Example

Question: "New treatments for Alzheimer's".

- **Table A (newest first):** newly approved anti-amyloid antibodies (approval date, pathway, boxed warning, pivotal trial, dose from label), then `standard` cholinesterase inhibitors and memantine with sourced label doses and guideline status.
- **Table B:** late-phase agents labelled "Investigational - not approved".
- **Table C:** any withdrawn or restricted drugs.
- **Deep dive block** for each newly approved antibody.

Names and dates must come from retrieved sources at run time, not from memory.

## References

Load `references/schemas.md` for the per-drug record, deep-dive record and landscape output shape.