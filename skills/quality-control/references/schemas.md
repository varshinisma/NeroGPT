# quality-control - schemas

Load this file when you need exact field names or output shapes.

## 6. Output schema

```json
{
  "status": "PASS|NEEDS_REPAIR",
  "issues": [{
    "id": "qc_001",
    "category": "grounding|recency|drugs|trials|evidence|consistency|completeness",
    "location": "section or claim_id",
    "description": "string",
    "severity": "high|medium|low"
  }],
  "severity": ["high","medium"],
  "repair_targets": [{
    "location": "section or claim_id",
    "action": "remove|qualify|re_source|retrieve|reconcile|add_section",
    "reason": "string",
    "needs_retrieval": false
  }],
  "checks": { "grounding": "pass|fail", "recency": "pass|fail", "drugs": "pass|fail", "trials": "pass|fail", "evidence": "pass|fail", "consistency": "pass|fail", "completeness": "pass|fail" },
  "repairs_used": 0,
  "final_disposition": "finalize|repair|finalize_with_qualification"
}
```

