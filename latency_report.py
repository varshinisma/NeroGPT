"""Latency report for the research agent: percentiles over the recorded runs (observability/runs.jsonl) and a waterfall of one run.

    python latency_report.py                 # tables in the terminal + observability/latency_report.html
    python latency_report.py --run <run id>  # the waterfall of that run (default: the latest)
    python latency_report.py --last 20       # only the 20 newest runs

The same page is served by the API at /latency. Every number comes from the spans the program recorded while it ran (observability.py); nothing is estimated here."""
from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import observability as obs

KIND_COLOR = {"stage": "#cbd5e1", "model": "#2563eb", "tool": "#16a34a"}


def stats(values: list[float]) -> dict:
    values = [v for v in values if isinstance(v, (int, float))]
    return {"n": len(values), "p50": obs.percentile(values, 50), "p95": obs.percentile(values, 95), "p99": obs.percentile(values, 99),
            "mean": round(sum(values) / len(values), 2) if values else None, "max": round(max(values), 2) if values else None}


def collect(runs: list[dict]) -> dict:
    """Distributions over the runs: whole-run time and its three parts, the first answer, each stage, each model purpose (time to first token, generation speed) and each tool."""
    out = {"runs": len(runs), "total": stats([r.get("total_s") for r in runs]), "first_token": stats([r.get("first_token_s") for r in runs]),
           "first_useful": stats([r.get("first_useful_s") for r in runs]), "model_wall": stats([r.get("model_wall_s") for r in runs]), "tool_wall": stats([r.get("tool_wall_s") for r in runs]),
           "orchestration_wait": stats([r.get("orchestration_wait_s") for r in runs]), "stages": {}, "models": {}, "tools": {}, "signals": {}}
    for r in runs:
        for name, row in (r.get("stages") or {}).items():
            out["stages"].setdefault(name, {"seconds": [], "model_s": [], "tool_s": [], "wait_s": []})
            out["stages"][name]["seconds"].append(row.get("seconds"))
            out["stages"][name]["model_s"].append(row.get("model_s"))
            out["stages"][name]["tool_s"].append(row.get("tool_s"))
            out["stages"][name]["wait_s"].append(row.get("orchestration_wait_s"))
        for s in r.get("spans") or []:
            if s.get("kind") == "model":
                m = out["models"].setdefault(s["name"], {"seconds": [], "ttft": [], "tok_per_s": [], "ms_per_token": []})
                m["seconds"].append(s["end"] - s["start"])
                if s.get("ttft") is not None:
                    m["ttft"].append(s["ttft"])
                if s.get("gen_s") and s.get("out_chars"):
                    m["tok_per_s"].append(s["out_chars"] / 4 / max(s["gen_s"], 0.05))
                    m["ms_per_token"].append(1000 * max(s["gen_s"], 0.05) / (s["out_chars"] / 4))      # inter-token latency
            elif s.get("kind") == "tool":
                t = out["tools"].setdefault(s["name"], {"seconds": [], "errors": 0})
                t["seconds"].append(s["end"] - s["start"])
                t["errors"] += 1 if s.get("error") else 0
    for key, label in (("model_calls", "model calls per run"), ("tool_calls", "tool calls per run"), ("errors", "errors per run"), ("retries", "retries per run"),
                       ("input_tokens_max", "largest model input (tokens)"), ("input_tokens_total", "model input tokens per run")):
        out["signals"][label] = stats([r.get(key) for r in runs])
    return out


def _fmt(v) -> str:
    return "-" if v is None else f"{v:g}"


def text_report(runs: list[dict]) -> str:
    c = collect(runs)
    lines = [f"LATENCY OVER {c['runs']} RUN(S), seconds", "", f"{'':34}{'n':>4}{'p50':>8}{'p95':>8}{'p99':>8}{'mean':>8}{'max':>8}"]

    def row(label, s):
        lines.append(f"{label:34}{s['n']:>4}{_fmt(s['p50']):>8}{_fmt(s['p95']):>8}{_fmt(s['p99']):>8}{_fmt(s['mean']):>8}{_fmt(s['max']):>8}")
    row("whole run", c["total"])
    row("time to first token (answer)", c["first_token"])
    row("first useful answer", c["first_useful"])
    row("  model time (wall)", c["model_wall"])
    row("  tool time (wall)", c["tool_wall"])
    row("  orchestration & wait", c["orchestration_wait"])
    lines += ["", "PER STAGE (wall seconds)"]
    for name, d in c["stages"].items():
        row(name, stats(d["seconds"]))
    lines += ["", "PER MODEL PURPOSE: whole call / time to first token / tokens per second"]
    for name, d in c["models"].items():
        row(f"{name}", stats(d["seconds"]))
        row(f"  {name} TTFT", stats(d["ttft"]))
        row(f"  {name} tokens/s", stats(d["tok_per_s"]))
        row(f"  {name} inter-token ms", stats(d["ms_per_token"]))
    lines += ["", "PER TOOL"]
    for name, d in c["tools"].items():
        row(f"{name} ({d['errors']} failed)", stats(d["seconds"]))
    lines += ["", "DIAGNOSTIC SIGNALS"]
    for label, s in c["signals"].items():
        row(label, s)
    return "\n".join(lines)


def waterfall_html(run: dict) -> str:
    total = max(run.get("total_s") or 0.1, 0.1)
    spans = sorted(run.get("spans") or [], key=lambda s: (s["kind"] != "stage", s["start"]))
    rows = []
    for s in spans:
        left, width = 100 * s["start"] / total, max(0.4, 100 * (s["end"] - s["start"]) / total)
        extra = f" · first token {s['ttft']}s" if s.get("ttft") is not None else ""
        extra += " · FAILED" if s.get("error") else ""
        label = f"{s['kind']}: {html.escape(str(s['name']))} — {s['end'] - s['start']:.1f}s{extra}"
        rows.append(f'<div class="row"><div class="lab" title="{label}">{label}</div><div class="track"><div class="bar {s["kind"]}" style="left:{left:.2f}%;width:{width:.2f}%"></div></div></div>')
    return "\n".join(rows)


def run_tables_html(run: dict) -> str:
    """Tools of one run (calls, seconds, failures) and its model calls in time order with the size of each input, so context growth is visible."""
    if not run:
        return ""
    tools = "".join(f"<tr><td>{html.escape(n)}</td><td>{v.get('calls')}</td><td>{v.get('total_s')}</td><td>{v.get('errors')}</td></tr>" for n, v in sorted((run.get("tools") or {}).items(), key=lambda kv: -kv[1].get("total_s", 0)))
    calls = []
    for s in sorted((x for x in run.get("spans") or [] if x.get("kind") == "model"), key=lambda x: x["start"]):
        gen, out = s.get("gen_s"), s.get("out_chars")
        tps = round(out / 4 / max(gen, 0.05), 1) if gen and out else None
        ms = round(1000 * max(gen, 0.05) / (out / 4), 1) if gen and out else None
        calls.append(f"<tr><td>{html.escape(str(s['name']))}</td><td>{s['start']}</td><td>{round(s['end'] - s['start'], 2)}</td><td>{_fmt(s.get('ttft'))}</td><td>{_fmt(tps)}</td><td>{_fmt(ms)}</td>"
                     f"<td>{int((s.get('in_chars') or 0) / 4)}</td><td>{int((out or 0) / 4)}</td><td>{html.escape(str(s.get('error') or ''))}</td></tr>")
    return (f"<h2>Tools in this run</h2><table><tr><th>tool</th><th>calls</th><th>total s</th><th>failed</th></tr>{tools}</table>"
            f"<h2>Model calls in this run (in time order: input size shows the context growing)</h2><table><tr><th>purpose</th><th>started at s</th><th>total s</th><th>first token s</th><th>tokens/s</th>"
            f"<th>ms per token</th><th>input tokens</th><th>output tokens</th><th>error</th></tr>{''.join(calls)}</table>")


def html_report(runs: list[dict], run_id: str | None = None) -> str:
    c = collect(runs)
    chosen = next((r for r in runs if r.get("id") == run_id), None) or (runs[-1] if runs else {})
    ticks = "".join(f"<tr><td>{html.escape(label)}</td><td>{s['n']}</td><td>{_fmt(s['p50'])}</td><td>{_fmt(s['p95'])}</td><td>{_fmt(s['p99'])}</td><td>{_fmt(s['mean'])}</td><td>{_fmt(s['max'])}</td></tr>"
                    for label, s in ([("whole run", c["total"]), ("first token of the answer", c["first_token"]), ("first useful answer", c["first_useful"]), ("model time (wall)", c["model_wall"]),
                                      ("tool time (wall)", c["tool_wall"]), ("orchestration & wait", c["orchestration_wait"])]
                                     + [(f"stage: {n}", stats(d["seconds"])) for n, d in c["stages"].items()]
                                     + [(f"model {n}: whole call", stats(d["seconds"])) for n, d in c["models"].items()] + [(f"model {n}: time to first token", stats(d["ttft"])) for n, d in c["models"].items()]
                                     + [(f"model {n}: tokens per second", stats(d["tok_per_s"])) for n, d in c["models"].items()]
                                     + [(f"model {n}: inter-token latency (ms per token)", stats(d["ms_per_token"])) for n, d in c["models"].items()]
                                     + [(f"tool {n} ({d['errors']} failed)", stats(d["seconds"])) for n, d in c["tools"].items()]
                                     + list(c["signals"].items())))
    stage_rows = "".join(f"<tr><td>{html.escape(n)}</td><td>{v['seconds']}</td><td>{v['model_s']}</td><td>{v['tool_s']}</td><td>{v['orchestration_wait_s']}</td></tr>" for n, v in (chosen.get("stages") or {}).items())
    options = "".join(f'<option value="{html.escape(r.get("id", ""))}"{" selected" if r is chosen else ""}>{html.escape(str(r.get("at", "")))} · {r.get("total_s")}s · {html.escape(str(r.get("question", ""))[:60])}</option>' for r in reversed(runs[-50:]))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Agent latency</title>
<style>:root{{--bg:#fff;--fg:#0f172a;--mut:#64748b;--line:#e2e8f0;--stage:#cbd5e1;--model:#2563eb;--tool:#16a34a}}@media (prefers-color-scheme:dark){{:root{{--bg:#0b1220;--fg:#e2e8f0;--mut:#94a3b8;--line:#1e293b;--stage:#334155}}}}
body{{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;max-width:1100px;margin-inline:auto}}h1{{font-size:20px}}h2{{font-size:16px;margin-top:28px}}
table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid var(--line);padding:4px 8px;text-align:right}}td:first-child,th:first-child{{text-align:left}}
.row{{display:flex;align-items:center;gap:8px;margin:2px 0}}.lab{{width:42%;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:12px;color:var(--mut)}}.track{{position:relative;flex:1;height:12px;background:var(--line);border-radius:3px}}
.bar{{position:absolute;top:0;height:100%;border-radius:3px}}.bar.stage{{background:var(--stage)}}.bar.model{{background:var(--model)}}.bar.tool{{background:var(--tool)}}
.legend span{{display:inline-block;margin-right:14px}}.legend i{{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:4px;vertical-align:-2px}}select{{max-width:100%}}@media(max-width:640px){{.lab{{width:50%}}}}</style></head><body>
<h1>Agent latency</h1><p style="color:var(--mut)">{c['runs']} recorded run(s). A run is a chain of model calls, retrievals and orchestration stages; each is a span. Times are seconds.</p>
<h2>Distribution over runs (p50 / p95 / p99, not only the average)</h2><table><tr><th></th><th>n</th><th>p50</th><th>p95</th><th>p99</th><th>mean</th><th>max</th></tr>{ticks}</table>
<h2>One run</h2><form method="get"><select name="run" onchange="this.form.submit()">{options}</select></form>
<p>Whole run {chosen.get('total_s', '-')}s = model {chosen.get('model_wall_s', '-')}s + tools {chosen.get('tool_wall_s', '-')}s + orchestration &amp; wait {chosen.get('orchestration_wait_s', '-')}s
 · {chosen.get('model_calls', '-')} model calls · {chosen.get('tool_calls', '-')} tool calls · {chosen.get('errors', '-')} errors · {chosen.get('retries', '-')} retries · largest input {chosen.get('input_tokens_max', '-')} tokens</p>
<table><tr><th>stage</th><th>seconds</th><th>model</th><th>tools</th><th>orchestration &amp; wait</th></tr>{stage_rows}</table>
{run_tables_html(chosen)}
<h2>Waterfall</h2><div class="legend"><span><i style="background:var(--stage)"></i>stage</span><span><i style="background:var(--model)"></i>model call</span><span><i style="background:var(--tool)"></i>tool / retrieval</span></div>
{waterfall_html(chosen) if chosen else '<p>No run recorded yet.</p>'}</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=None)
    parser.add_argument("--last", type=int, default=0)
    args = parser.parse_args()
    runs = obs.load_runs()
    if args.last:
        runs = runs[-args.last:]
    if not runs:
        print("No runs recorded yet: ask a question first (runs are written to observability/runs.jsonl).")
        return
    for stream in (sys.stdout,):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    print(text_report(runs))
    target = obs.OBS_DIR / "latency_report.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html_report(runs, args.run), encoding="utf-8")
    print(f"\nWaterfall page: {target}")


if __name__ == "__main__":
    main()
