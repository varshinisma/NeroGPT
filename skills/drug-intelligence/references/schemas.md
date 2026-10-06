# drug-intelligence - schemas

Load this file when you need exact field names or output shapes.

## 6. Per-drug record

```json
{
  "id": "drug_001",
  "generic_name": "string",
  "brand_names": ["string"],
  "category": "standard|guideline|newly_approved|emerging|off_label",
  "flags": ["safety_alert|withdrawn|restricted"],
  "line_of_therapy": "first|second|adjunct|null",
  "class": "string",
  "mechanism": "string",
  "indication": "string",
  "population": "string|null",
  "approval": {
    "status": "approved|under_review|investigational|off_label|withdrawn",
    "regulator": "FDA|EMA|CDSCO|other",
    "approval_date": "YYYY-MM-DD|null",
    "latest_label_update": "YYYY-MM-DD|null"
  },
  "dosing": {
    "approved_label_dose": { "dose": "", "route": "", "frequency": "", "duration": "", "adjustments": "", "source_id": "" },
    "guideline_dose": { "dose": "", "route": "", "frequency": "", "duration": "", "source_id": "" },
    "trial_dose": { "dose": "", "trial_id": "", "source_id": "" },
    "investigational_dose": { "dose": "", "trial_id": "", "source_id": "" }
  },
  "contraindications": ["string"],
  "important_interactions": ["string"],
  "major_adverse_effects": ["string"],
  "supporting_evidence": [{ "source_id": "", "summary": "", "evidence_status": "" }],
  "missing_context": ["string"],
  "conflict": false,
  "source_ids": ["string"],
  "retrieved_at": "ISO-8601"
}
```


## 7. Output

```json
{
  "status": "ok|partial|failed",
  "mode": "FAST|DEEP",
  "drug_landscape": { "standard": [], "guideline": [], "newly_approved": [], "emerging": [], "off_label": [] },
  "safety_alerts": [{ "drug": "", "alert": "", "date": "", "source_id": "" }],
  "no_drug_treatment": { "applies": false, "statement": "", "source_id": "" },
  "gaps": ["string"],
  "error": null
}
```

