"""Compare runs in observability/runs.jsonl before/after a cut-off. Usage: python compare_runs.py "2026-11-17T12:00"  (runs with 'at' >= cut-off are 'after')."""
import json, statistics as st, sys
rows = [json.loads(l) for l in open("observability/runs.jsonl", encoding="utf-8") if l.strip()]
cut = sys.argv[1] if len(sys.argv) > 1 else ""
def stats(rs):
    def med(f):
        v = [f(r) for r in rs]; v = [x for x in v if isinstance(x, (int, float))]
        return round(st.median(v), 1) if v else None
    S = lambda k: (lambda r: (r.get("stages", {}).get(k) or {}).get("seconds"))
    return {"runs": len(rs), "total_s": med(lambda r: r.get("total_s")), "first_useful_s": med(lambda r: r.get("first_useful_s")),
            "deep_search": med(S("deep_search")), "rank": med(S("rank")), "escalate": med(S("escalate")), "write_stream": med(S("write_stream")), "assemble": med(S("assemble")),
            "rate_limit_retries": med(lambda r: (r.get("counters") or {}).get("rate_limit_retries", 0))}
before = [r for r in rows if r.get("at", "") < cut]; after = [r for r in rows if r.get("at", "") >= cut]
print("before", stats(before)); print("after ", stats(after))
