# clinical-research-orchestrator - schemas

Load this file when you need exact field names or output shapes.

## 8. Output — `ResearchState`

```json
{
  "query": "string",
  "intent": { "primary": "string", "secondary": ["string"] },
  "entities": [{ "type": "disease|drug|biomarker|population|intervention|date_range|outcome", "text": "", "normalized": "", "confidence": 0.0 }],
  "fast_evidence": [],
  "deep_evidence": [],
  "drugs": [],
  "drug_landscape": { "standard": [], "guideline": [], "newly_approved": [], "emerging": [], "off_label": [] },
  "trials": [],
  "synthesis": {},
  "citations": [],
  "verification": {},
  "qc": {},
  "answer_status": "INITIAL|ENRICHING|FINAL",
  "skill_status": { "<skill>": { "status": "ok|partial|failed", "elapsed_ms": 0, "retries": 0, "error": null } },
  "timestamps": { "request_started": "", "first_response": "", "research_completed": "" }
}
```


## 9. Config contract

```json
{ "fast_timeout_ms": 3000, "deep_timeout_ms": 25000, "total_deep_budget_ms": 60000,
  "max_research_retries": 2, "max_repairs": 1, "default_recency_months": 24 }
```

