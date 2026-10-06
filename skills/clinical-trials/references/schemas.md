# clinical-trials - schemas

Load this file when you need exact field names or output shapes.

## 6. Per-trial record

```json
{
  "registry_id": "NCT00000000",
  "registry": "ClinicalTrials.gov|EU|CTRI|ICTRP",
  "title": "string",
  "condition": "string",
  "intervention": "string",
  "comparator": "string|null",
  "phase": "Phase 1|Phase 1/2|Phase 2|Phase 2/3|Phase 3|Phase 4|N/A",
  "study_design": "randomized|single-arm|open-label|double-blind|…",
  "population": "string",
  "inclusion_exclusion_summary": "string",
  "primary_endpoint": "string",
  "secondary_endpoints": ["string"],
  "dose_regimen": "string|null (only if explicitly stated in registry)",
  "recruitment_status": "Recruiting|Not yet recruiting|Active, not recruiting|Completed|Suspended|Terminated|Withdrawn",
  "status_why_stopped": "string|null",
  "status_verified": true,
  "locations": ["string"],
  "start_date": "YYYY-MM|null",
  "estimated_completion": "YYYY-MM|null",
  "last_update": "YYYY-MM-DD",
  "sponsor": "string",
  "results_available": false,
  "results_summary": "string|null",
  "publication": { "doi": null, "pmid": null },
  "investigational_drugs": ["string"],
  "source_url": "string",
  "retrieved_at": "ISO-8601"
}
```


## 7. Output

```json
{
  "status": "ok|partial|failed",
  "mode": "FAST|DEEP",
  "trials": [ /* records */ ],
  "investigational_drugs": ["string"],
  "counts_by_status": { "Recruiting": 0, "Completed": 0 },
  "gaps": ["e.g. registry unreachable; status not verified"],
  "error": null
}
```

