# clinical-answer-writer - schemas

Load this file when you need exact field names or output shapes.

## 10. Output schema (per streamed chunk)

```json
{
  "stream_state": "INITIAL|EVIDENCE_ENRICHING|DRUGS_ENRICHING|TRIALS_ENRICHING|VERIFICATION|FINAL",
  "section": "bottom_line|recent_finding|context|top_drugs|latest_findings|current_evidence|drug_landscape|trials|key_studies|conflicts|ongoing_research|limitations|citations",
  "content_markdown": "string",
  "claims": [{ "claim_id": "", "text": "", "source_ids": [] }],
  "complete": false,
  "timestamp": "ISO-8601"
}
```

