# Before / after: "How to treat autism in children?"

Before = `demo_output/before/how-to-treat-autism-in-children.md` (old pipeline). After = `demo_output/how-to-treat-autism-in-children.md` (this run).
Both were checked with the same independent script (`audit.py`, which does not use the pipeline's verifier).

| Item | Before | After |
|---|---|---|
| Independent audit | 22 of 27 checks pass (V, W, Z, AA, AB fail) | 27 of 27 pass |
| Key studies numbering | starts at "2." and skips (2, 3, 4 shown, item 1 was removed) | 1 to 5, in order; every list is renumbered after any repair |
| Key studies, item 3 | stub: `3. **Restoy et al.` (cut off at "et al.") | complete entry: authors, year, design, verbatim finding sentence, PMID |
| Who writes Key studies | the model | code, from the retrieved records; an entry missing any of authors, year, design, finding, PMID is dropped |
| Stray `**` / cut-off lines | present | none; `lint_report` blocks PASS if any appear |
| Bottom line | model-written, headlined one paper | built in code from counts by evidence tier plus FDA-labelled drugs; no efficacy claim |
| Scope statement | absent | "This is a recent-evidence report, not a complete treatment guideline." under the Bottom line |
| Phase-1 streamed text | looked final | labelled "Preliminary answer" in the stream and the UI; the final report replaces it |
| Source failure shown to the reader | only in the last "Evidence limitations" section | banner at the top naming the source that still failed after retry, with "absence of evidence cannot be concluded"; also a live warning in the browser |
| Retries | deep sources only | one retry with backoff for every fast source, then one more try in the deep stage with the 25 s timeout |
| Established care | none | table by topic from retrieved guideline pages only (title, organisation, year, link, verbatim excerpt) or "No guideline retrieved" / "search unavailable" |
| Disclaimer footer | none | "Research support, not a substitute for clinical judgement." |
| Personal identifiers | sent as typed | scrubbed locally before any search or model call; history stores the scrubbed text |
| Diagnosis / personal advice wording | not checked | flagged and a general-information note added; the sentence is not reworded |

## What this run shows honestly

- The "Established care" table in this particular run says **search unavailable** and the banner is shown at the top. The guideline search failed for all 10 topics (Tavily's plan usage limit is exhausted, and the DuckDuckGo fallback was rate-limited at that moment). An earlier run of the same question retrieved guideline pages for 6 of 10 topics (AAP, AOTA, NICHD, AAP resource page, and 2 PMC journal articles that are now excluded as non-guidelines), so the code path works; the web sources are what is unreliable.
- Timing: first token about 3 s, first useful answer 5.9 s, final verified report 44 s (the deep stage now also runs the guideline search).
