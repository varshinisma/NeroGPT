---
name: figure-selection
description: Chooses which original figures from published open-access papers may be shown with a clinical answer, and writes the short plain-language lines under each one. Use after the report is written, when figures from the retrieved papers are offered for the question - "which figures help answer this", "is this figure about the right population", "does this figure present what was asked", "explain how to read this figure".
---

# Figure Selection

## Overview

Decide which of the offered figures a doctor should see next to the answer. A figure is the paper's own original image; never describe one that was not offered, never redraw or invent one, and never add a figure for decoration. Choose every offered figure that passes the tests below; when none does, return an empty list. Use only the caption, the paper title and abstract, and what the image itself shows. Do not add facts from memory.

## Step 1: Pick the figures that pass all five tests

Each offered figure has its paper's title and design, its ORIGINAL caption, whether the written report cites its paper (`cited_in_report`), and its `relation` (direct, or indirect when the study is of a closely related condition or population). `written_report` is the evidence the report already states in words. Be strict: a figure is useful only if it adds something the written report does not already give. Keep a figure only if ALL five tests are true (at most 5 figures, in order of usefulness; leave every other figure out):

1. **helps_answer** - it directly helps answer the question. A figure with `relation` "indirect" comes from a study of a closely related condition or population: say true only if it clearly informs this question.
2. **clinically_meaningful** - it shows treatment outcomes, comparisons, effect sizes or a relevant clinical pathway. A figure about adherence, compliance, a biomarker used only to monitor the treatment, baseline characteristics or study methods is NOT clinically meaningful.
3. **adds_beyond_report** - it adds information the written report does not already explain.
4. **not_decorative** - it is not a photograph, a scan of one patient, a recruitment or selection flowchart, a study-timeline schematic or a figure about another outcome.
5. **answers_what_is_asked** - read what the question asks for and require the figure to show THAT kind of information. When the question asks how to treat or manage, the figure must show a treatment, a comparison of treatments or an outcome of treatment. A figure that only describes how common, how severe or how it develops is false unless the question asks about that.

Give each chosen figure a `reason`: 4 to 15 CONSECUTIVE words copied exactly, word for word, from that figure's caption, which justify the answers (write nothing else in that field). A reason that is not in the caption makes the figure invalid.

## Step 2: Check the patient population

Work out `question_population`: the words copied from the question that name the patient population (an age group, sex, severity, setting or similar). The disease or condition itself is NOT a population; a question that only names a disease names no population, so write "not specified". For each paper give `study_population` (who was studied, in a few words), an `evidence_quote` (3 to 20 consecutive words copied exactly from the paper's title, abstract or population note that state who was studied, or "" if the text does not say) and a verdict:

- **match** - the paper's text says it studied the question's population itself.
- **broader** - it studied a wider group that includes the question's population without isolating it.
- **mismatch** - the text clearly says it studied a DIFFERENT population that does not include the question's (for example only another age group).
- **unclear** - the text does not establish who was studied.

Use only the paper's own text; never guess from the topic.

## Step 3: Check that the figure presents what was asked

First state in a short phrase what KIND of information the question asks for (`asked_kind`: for example the ways to treat or manage something and how well they work, how to diagnose it, what causes it, how common it is). Then decide for each figure whether the figure ITSELF presents that kind of information: results or comparisons of the approaches the question asks about, or the steps of such an approach. A figure that only describes the problem (how common it is, how severe, who is affected, how it develops, how it was studied) does NOT present it, unless the question asks about that. Use only the caption, the paper title and the image description. Give `presents_it` true or false and `what_it_presents` in up to 12 words. Answer false when in doubt.

## Step 4: Look at the image itself

When the image is shown to a vision model, it describes what the image actually shows in one sentence and answers: does it match the caption, does it show results, data or a diagram that bears on the question, and are its text, labels and axes legible. It describes only; it never says whether a treatment works.

## Step 5: Write the lines under each figure

Write three short plain-language lines for each figure that stays.

1. **how_to_read** - how to read the figure in plain words: what the axes, lines, colours, symbols and abbreviations mean, and what a higher or lower value means. Use ONLY the caption and the image description; if neither explains it, write "". `how_to_read_quote`: 4 to 20 consecutive words copied exactly from the caption or the image description.
2. **shows_for_question** - what the figure shows that bears on the question, in two sentences, using ONLY the caption, title and abstract: name the groups or treatments compared and the outcome, say which way it moved or which group ended lower or higher, and end with a clause saying how this bears on the question. Give a number ONLY if it is written in the caption, title or abstract (copy it exactly); otherwise describe the pattern in words and use no number. Explain any score or abbreviation the paper itself defines. If nothing bears on the question, write "". `shows_quote`: 4 to 20 consecutive words copied exactly from the caption, title or abstract.
3. **helps_treatment** - one sentence on how this figure helps a doctor weigh the TREATMENT OPTIONS in the question (for example which treatments it lets them compare, or what it says about how an option performed or was tolerated), using ONLY the caption, title and abstract. If it does not bear on treatment options, write "". `helps_quote`: 3 to 20 consecutive words copied exactly from the caption, title or abstract.

## Checks applied by the program

The program, not the model, enforces these. Know them so the answers above are consistent with them:

- The paper must be relevant to the question (direct or indirect) and must have an explicit open licence (CC or CC0) in its own record.
- A figure reproduced from another publication is not shown.
- A figure whose text is too small to read on the page is not shown.
- Every quote must be found in its source text; a line whose quote or number cannot be found is dropped.
- A figure whose caption is nearly the same as one already kept from the same paper is a repeat: only the most informative one stays. At most 3 figures per paper and 6 in all.
- A figure whose paper the report does not cite (`cited_in_report` false) is allowed only when it is clearly relevant to the question, and is labelled as not cited.

## Output sections

Step 1: `{"figures": [{"id", "helps_answer", "clinically_meaningful", "adds_beyond_report", "not_decorative", "answers_what_is_asked", "reason"}]}`. Step 2: `{"question_population", "papers": [{"pmid", "study_population", "evidence_quote", "verdict"}]}`. Step 3: `{"asked_kind", "figures": [{"id", "presents_it", "what_it_presents"}]}`. Step 5: `{"figures": [{"id", "how_to_read", "how_to_read_quote", "shows_for_question", "shows_quote", "helps_treatment", "helps_quote"}]}`. Each reply is one JSON object and nothing else.

## Rules

- Never invent an id and never write a reason or quote that is not in the source text.
- Answer false only for a test the figure clearly fails. A figure that passes all five tests is chosen.
- If none qualifies, `figures` is `[]`.
- Describe what was found; do NOT say that a treatment works, is proven or should be used, and do not make a recommendation.
- Do not name a disease, drug, population or question type that the question and the source text do not themselves contain.

## Quality check

- [ ] Every chosen figure passes all five tests, and its reason is copied from its caption
- [ ] The population verdict rests on the paper's own words, not on the topic
- [ ] The figure itself presents what the question asks for, not only background about the problem
- [ ] Each line uses only its allowed source, and every number is written in the caption, title or abstract
- [ ] No line recommends a treatment or says one works
