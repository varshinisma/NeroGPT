# citation-verification - schemas

Load this file when you need exact field names or output shapes.

## 4. Output schema

```json
{
  "status": "PASS|FAIL",
  "verified_claims": [{ "claim_id": "", "source_ids": [], "notes": "" }],
  "failed_claims": [{
    "claim_id": "",
    "claim_text": "",
    "issue": "unsupported_claim|citation_mismatch|outdated_source|hallucinated_citation|wrong_dosage|wrong_trial_status|wrong_category|population_mismatch|overclaiming",
    "detail": "what the source actually says",
    "severity": "high|medium|low",
    "required_action": "retrieve_better_evidence|qualify|remove|fix_citation",
    "suggested_source_id": "string|null"
  }],
  "citation_gaps": [{ "claim_id": "", "severity": "", "required_action": "" }],
  "coverage": { "material_claims": 0, "verified": 0, "failed": 0, "gaps": 0 },
  "overall_severity": "none|low|medium|high",
  "required_action": "none|repair|block_final"
}
```

`status = PASS` only if there are **no** high-severity failures/gaps. Any high severity → `required_action = "block_final"` until resolved.

