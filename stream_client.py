import json, sys, time, httpx
q = sys.argv[1]
t0 = time.time(); first_tok = None; stages = []; text = ""; final = None
with httpx.Client(timeout=300) as c:
    with c.stream("POST", "http://localhost:8000/api/chat-stream", json={"messages":[{"role":"user","content":q}],"conversation_id":"clienttest"}) as r:
        print("HTTP", r.status_code, r.headers.get("content-type"))
        for line in r.iter_lines():
            if not line: continue
            ev = json.loads(line); now = time.time() - t0
            if ev["type"] == "token":
                if first_tok is None: first_tok = now; print(f"[client {now:5.2f}s] FIRST TOKEN ARRIVED AT BROWSER")
                text += ev["text"]
            elif ev["type"] == "stage": print(f"[client {now:5.2f}s] stage {ev['stage']}")
            elif ev["type"] in ("milestone","verification","pdf","error"): print(f"[client {now:5.2f}s] {ev['type']}: { {k:v for k,v in ev.items() if k not in ('type','t')} }"[:230])
            elif ev["type"] == "final": final = ev; print(f"[client {now:5.2f}s] FINAL ({len(ev['markdown'])} chars)")
print("\nserver timings:", final["timings"] if final else None)
__import__("os").makedirs("demo_output/cli_test_runs", exist_ok=True) or open("demo_output/cli_test_runs/_client_final.md","w",encoding="utf-8").write(final["markdown"] if final else text)
