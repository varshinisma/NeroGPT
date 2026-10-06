# evidence-synthesis - schemas

Load this file when you need exact field names or output shapes.

## 6. Output schema

```json
{
  "status": "ok|partial",
  "bottom_line": { "text": "", "strength": "strong|moderate|limited|insufficient", "source_ids": [] },
  "current_evidence": [{ "statement": "", "tier": "established|emerging|experimental", "source_ids": [] }],
  "recent_findings": [{ "statement": "", "date": "", "evidence_status": "", "source_ids": [] }],
  "drug_landscape_summary": {
    "established": [{ "drug": "", "note": "", "source_ids": [] }],
    "latest": [{ "drug": "", "note": "", "source_ids": [] }],
    "emerging": [{ "drug": "", "note": "investigational", "source_ids": [] }]
  },
  "key_studies": [{ "source_id": "", "why_it_matters": "" }],
  "conflicts": [{ "topic": "", "positions": [{ "claim": "", "source_ids": [] }], "possible_reason": "string|null" }],
  "uncertainty": [{ "issue": "", "effect_on_confidence": "" }],
  "ongoing_research": [{ "trial_id": "", "note": "being studied; not evidence of efficacy" }],
  "gaps": [""],
  "caveats": ["association not causation", "…"]
}
```

