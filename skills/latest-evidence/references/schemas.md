# latest-evidence - schemas

Load this file when you need exact field names or output shapes.

## 5. Per-source record

```json
{
  "id": "src_001",
  "title": "string",
  "authors": ["string"],
  "publication_date": "YYYY-MM-DD | YYYY-MM | YYYY",
  "journal_or_source": "string",
  "study_type": "guideline|systematic_review|meta_analysis|rct|cohort|case_control|case_series|case_report|preclinical|preprint|other",
  "peer_reviewed": true,
  "population": "string|null",
  "intervention_or_exposure": "string|null",
  "comparator": "string|null",
  "sample_size": 0,
  "key_finding": "string (as stated by the source, with the effect size/CI if given)",
  "limitations": ["string"],
  "doi": "string|null",
  "pmid": "string|null",
  "url": "string|null",
  "evidence_status": "established|emerging|preliminary|experimental|contradicted",
  "drugs_named": ["string"],
  "retrieved_at": "ISO-8601 timestamp",
  "relevance_score": 0.0
}
```


## 6. Output

```json
{
  "status": "ok|partial|failed",
  "mode": "FAST|DEEP",
  "sources": [ /* records above */ ],
  "excluded": [{ "title": "", "reason": "" }],
  "queries_run": ["string"],
  "gaps": ["e.g. no RCT found in last 24 months", "no guideline found"],
  "error": null
}
```

