"""Independent audit of a finished answer against the saved ResearchState. It does NOT call the pipeline's verifier: it re-checks the
output with its own simple rules, so a bug in the verifier cannot hide itself.

    .venv\\Scripts\\python.exe audit.py            (audits demo_output/how-to-treat-autism-in-children.md and .state.json)
    .venv\\Scripts\\python.exe audit.py <slug>
"""
import json
import re
import sys
from pathlib import Path

slug = sys.argv[1] if len(sys.argv) > 1 else "how-to-treat-autism-in-children"
folder = Path(__file__).parent / "demo_output"
presented = (folder / f"{slug}.md").read_text(encoding="utf-8")
state = json.loads((folder / f"{slug}.state.json").read_text(encoding="utf-8"))
_m = re.search(r"\n## Sources\n", presented)
_refs = {int(a): b for a, b in re.findall(r"(?m)^(\d+)\. .*?\[PMID (\d{6,9})\]", presented[_m.end():])} if _m else {}
_note = presented[presented.rfind("\n---\n"):] if "\n---\n" in presented else ""
_main = presented[:_m.start()] if _m else presented.split(_note)[0]
_main = re.sub(r"\[(\d+(?:,\s*\d+)*)\]", lambda g: " ".join(f"[PMID {_refs.get(int(n), 'unknown')}]" for n in re.findall(r"\d+", g.group(1))), _main)
_sources = "\n".join(f"{n}. PMID {pm}" for n, pm in sorted(_refs.items()))
text = _main.rstrip() + "\n\n### Sources\n" + _sources + "\n" + _note        # the old layout the checks below expect
body = text.split("\n---\n*Research support")[0]
claims = body.split("\n### Sources")[0]
papers, trials, reg = state["papers"], state["trials_by_id"], state["regulatory"]
results = []


def check(code, title, ok, detail=""):
    results.append((code, title, bool(ok), detail))


def sentences(t):
    return [s for ln in t.splitlines() if not ln.lstrip().startswith("|") for s in re.split(r"(?<=[.!?])\s+", ln)]


blob = " ".join([(p.get("title") or "") + " " + (p.get("abstract") or "") + str(p.get("year")) + str(p.get("date")) for p in papers.values()] +
                [str(t) for t in trials.values()] + [(w.get("title") or "") + (w.get("snippet") or "") for w in state["web"]] + [r["indication_excerpt"] + " " + (r.get("indication_statement") or "") + " " + (r.get("dose_statement") or "") for r in reg] +
                [d.get("boxed_warning") or "" for d in state.get("drug_labels", [])]).lower()      # the retrieved FDA label text, dosing section included
norm = lambda s: re.sub(r"[\s,\-*_‐-―’']", "", s.lower())

# A. the false statement "no pharmacologic treatments"
bad = [s for s in sentences(claims) if re.search(r"\bno (pharmacolog\w+|drug|medication)\w*\b[^.]{0,70}\b(identified|retrieved|found|exist\w*|available)\b", s, re.I)
       and not re.search(r"timed out|unavailable|could not|not retrieved|incomplete|does not (mean|show)", s, re.I)]
check("A", "no 'no pharmacologic treatments' claim while drug records exist", not (bad and reg), "; ".join(bad)[:150])
# B/C. approved drugs are not called off-label
off = [s for s in sentences(claims) + claims.splitlines() if "off-label" in s.lower() and any(r["generic"].lower().split()[0] in s.lower() for r in reg if r["matches_condition"])]
check("B/C", "drugs with a labelled indication are never 'off-label'", not off, "; ".join(off)[:150])
# D. approval facts come from a regulatory record, quoted
quoted = all(re.sub(r"\s+", " ", r["indication_excerpt"])[40:90] in re.sub(r"\s+", " ", text) or r["generic"] in text for r in reg)
check("D", "regulatory records are present in the answer with the label's own wording", bool(reg) and quoted, f"{len(reg)} FDA record(s)")
# E. outcome scoping: the cited paper must mention the outcome
OUT = ["social communication", "irritability", "anxiety", "depression", "mania", "sleep", "motor", "language", "seizure", "mortality", "relapse"]
scope = []
for ln in claims.splitlines():
    ids = set(re.findall(r"PMID[:\s]*(\d{6,9})", ln))
    if len(ids) == 1 and next(iter(ids)) in papers and not ln.lstrip().startswith("|"):
        p = papers[next(iter(ids))]
        src = ((p.get("title") or "") + " " + (p.get("abstract") or "")).lower()
        scope += [f"{t} / PMID {p['pmid']}" for t in OUT if t in ln.lower() and t not in src]
check("E", "no outcome is attributed to a source that does not report it", not scope, "; ".join(scope)[:150])
# F. wording
over = re.findall(r"(?<!not )(?<!not yet )(?<!not specifically )(?<!never )\b(confirmed|proven|validated|dose[\s-]dependent|stronger effects? at higher doses)\b", claims, re.I)      # 'not validated' states a limit
check("F", "no 'confirmed/validated/dose-dependent' wording", not over, ", ".join(over))
# G. protocols are not efficacy
proto_ids = [p["pmid"] for p in papers.values() if p.get("protocol")]
g = [ln for ln in claims.splitlines() if any(f"PMID {i}" in ln for i in proto_ids) and re.search(r"\b(effective|efficacy|improv\w+|benefit\w*)\b", ln, re.I)
     and not re.search(r"protocol|pending|under investigation|no efficacy|does not|not establish", ln, re.I)]
check("G", "protocol papers are not presented as efficacy", not g, "; ".join(g)[:150])
# H. indirect population
ind = [p["pmid"] for p in papers.values() if any(m in (p.get("title") or "").lower() for m in ("with and without", " traits"))]
h = [ln for ln in claims.splitlines() if any(f"PMID {i}" in ln for i in ind) and not re.search(r"indirect|not (asd|autism)[- ]?specific|related population|not .{0,12}specific", ln, re.I)]
check("H", "indirect-population papers are labelled indirect / not condition-specific", not h, "; ".join(h)[:150])
# I. 'Established' needs a guideline
est = re.findall(r"\*\*Established\*\*", claims)
guide = any("guideline" in str(p.get("type")).lower() for p in papers.values())
check("I", "no 'Established' label without a guideline source", not est or guide, f"{len(est)} label(s)")
# J/K. trial relevance
check("J", "clinical trials section lists treatment trials only (others are counted, not printed)", "## Clinical trials" in text and "Other registry records" not in text)
fox = [t for t in trials.values() if re.search(r"syndrome", t.get("title") or "", re.I)]
check("K", "syndrome-specific trials are flagged as not generalisable", all(f"{t['nct_id']}" not in text or "do not generalise" in text for t in fox), f"{len(fox)} syndrome trial(s)")
# L. the verifier's own summary
_status = state["verification"]["status"]
check("L", "the closing note matches the verification status and shows no internal repair log",
      ("checked against the retrieved sources" in presented if _status == "PASS" else "could not be fully verified" in presented)
      and not re.search(r"Repaired:|\[remaining\]|Repair passes", presented), _status)
# M. failures not turned into absence
failed = [k for k, v in state["source_status"].items() if v["retrieval_status"] != "success"]
m = [s for s in sentences(claims) if re.search(r"\b(there (is|are) no|does not exist|none exist)\b", s, re.I) and failed and re.search(r"drug|approved|treatment", s, re.I)]
check("M", "no non-existence claim while a drug/regulatory source failed", not m, "; ".join(m)[:150])
# N. doses
halves = re.findall(r"\d[\d.]*\s*[–-]\s*\[value", text) + re.findall(r"\[value not in retrieved data\]", text)
unit_nums = re.findall(r"\d+(?:\.\d+)?\s?(?:mg|mcg|µg|mL|IU|g)(?:/(?:kg|day))?\b", claims, re.I)
unsupported = [u for u in unit_nums if norm(u) not in norm(blob)]
check("N", "no half-filled or unsourced dose", not halves and not unsupported, ", ".join(halves + unsupported)[:150])
# O. identifiers
known_urls = {u.rstrip("/").lower() for p in papers.values() for u in [p.get("url") or "", f"https://doi.org/{p['doi']}" if p.get("doi") else ""]} | \
             {(w.get("url") or "").rstrip("/").lower() for w in state["web"]} | {(r.get("url") or "").lower() for r in reg} | {(t.get("url") or "").rstrip("/").lower() for t in trials.values()} | \
             {(g.get("url") or "").rstrip("/").lower() for g in state.get("guidelines", [])} | {(d.get("url") or "").rstrip("/").lower() for d in state.get("drug_labels", [])}
urls = {u.rstrip(".,;:)/").lower() for u in re.findall(r"https?://[^\s\]\)>|,;]+", body)}
dois = {d.rstrip(".,;)]").lower() for d in re.findall(r"10\.\d{4,9}/[^\s\]\)|,;>]+", re.sub(r"https?://(?!(?:dx\.)?doi\.org)[^\s\]\)>|,;]+", " ", body))}
known_dois = {(p.get("doi") or "").lower() for p in papers.values()}
check("O", "every DOI and URL was retrieved", not (urls - known_urls) and not (dois - known_dois), ", ".join(sorted((urls - known_urls) | (dois - known_dois)))[:150])
# P/Q. years and orphans
yr = []
for ln in claims.splitlines():
    for pm in re.findall(r"\[PMID (\d{6,9})\]", ln):
        label = ln[:ln.find(f"[PMID {pm}]")].split("(")[-1]
        ys = re.findall(r"(?:19|20)\d\d", label)
        if pm in papers and ys and papers[pm]["year"] not in ys:
            yr.append(f"PMID {pm}: label {ys} vs record {papers[pm]['year']}")
check("P", "every citation year matches the retrieved record", not yr, "; ".join(yr)[:150])
cited_all = set(re.findall(r"PMID[:\s]*(\d{6,9})", body))   # the answer itself (the footer may name a removed, fabricated ID)
sources = set(re.findall(r"PMID (\d{6,9})", text.split("### Sources")[-1].split("**Verification")[0])) if "### Sources" in text else set()
cited_claims = {x for x in re.findall(r"PMID[:\s]*(\d{6,9})", claims) if x in papers and not papers[x].get("retracted")}   # retracted papers are cited only to say they were excluded
check("Q", "every PMID exists in the ResearchState, and Sources == what the text cites (no orphans)", cited_all <= set(papers) and sources == cited_claims,
      f"unknown={sorted(cited_all - set(papers))} orphan_sources={sorted(sources - cited_claims)} missing_sources={sorted(cited_claims - sources)}")
# R/S. trials
rows = re.findall(r"\| (NCT\d{8}) \|.*?\| ([A-Z_]+) \(", text)
check("R", "every trial row status equals the registry record", all(n in trials and trials[n]["recruitment_status"] == s for n, s in rows), f"{len(rows)} row(s)")
s_bad = [ln for ln in claims.splitlines() if re.search(r"NCT\d{8}", ln) and not ln.lstrip().startswith("|") and re.search(r"\b(effective|efficacy|improv\w+|benefit\w*)\b", ln, re.I)
         and not re.search(r"not|no |cannot|pending|investigation", ln, re.I)]
check("S", "no trial is described as effective", not s_bad, "; ".join(s_bad)[:150])
# T. categories are separate
check("T", "the Treatment Drug Landscape has two tables with the specification's columns: drugs used for treatment, then symptom-directed or still-in-trial drugs",
      text.count("| Drug | Category | Indication | Dose / route (if sourced) | Approval status / date | Key evidence |") == 2
      and 0 <= text.find("### Drugs used for treatment of this condition") < text.find("### Other drugs: used for symptoms"))
check("AG", "no symptom-directed (off-label) or investigational drug sits in the treatment table",
      not re.search(r"\| (?:Off-label|Emerging)", text.split("### Other drugs: used for symptoms")[0].split("### Drugs used for treatment of this condition")[-1]))
# status honesty
check("U", "answer states the drug-landscape status honestly", state["drug_intelligence_status"] == "complete" or "incomplete" in text.lower(), state["drug_intelligence_status"])

# ---- V-AB: formatting, banner, scope, established care, key studies, disclaimer (own regexes; the pipeline's verifier is not used)
lines = text.split("### Sources")[0].splitlines()
runs, expect, bad_runs = 0, 0, []
for ln in lines:
    m = re.match(r"^\s*(\d{1,3})[.)]\s", ln)
    if m:
        expect += 1
        if int(m.group(1)) != expect:
            bad_runs.append(ln[:40])
            expect = int(m.group(1))
    elif ln.startswith(("#", "---")) or (ln.strip() and not ln.startswith((" ", "\t")) and not re.match(r"^\s*[-*] ", ln)):
        expect = 0
check("V", "numbered lists run 1, 2, 3 with no gap", not bad_runs, "; ".join(bad_runs)[:150])
cut = [ln[:50] for ln in lines if ln.strip() and not ln.lstrip().startswith(("#", "|", ">", "---")) and (ln.count("**") % 2 or re.search(r"\bet al\.?\**\s*$", ln.strip()))]
check("W", "no stray ** marker and no list item cut off at 'et al.'", not cut, "; ".join(cut)[:150])
status = state.get("source_status", {})
ok_jobs = {k for k, v in status.items() if v.get("retrieval_status") == "success"}
superseded = {"pubmed_fast": ("pubmed_recent",), "pubmed_fast_q": ("pubmed_quality",), "trials_fast": ("trials_deep",), "fda_fast": ("fda_deep",), "web_fast": ("web_approvals", "web_standard")}
still_bad = [k for k, v in status.items() if not k.startswith("guideline") and v.get("retrieval_status") != "success" and not any(s in ok_jobs for s in superseded.get(k, ()))]
head = "\n".join(text.splitlines()[:4])
limits = text.split("## Evidence limitations")[-1].split("\n## ")[0] if "## Evidence limitations" in text else ""
check("X", "no top banner; a source that failed is named in Evidence limitations with 'absence of evidence cannot be concluded'",
      "Source warning" not in head and (not still_bad or "absence of evidence cannot be concluded" in limits), f"still failed: {still_bad}")
check("Y", "the report opens with a Bottom line and has no fixed scope sentence", "**Bottom line" in text and "not a complete treatment guideline" not in text)
care = text.split("### Established care")[-1].split("\n### ")[0] if "### Established care" in text else ""
gl_urls = {g["url"].rstrip("/").lower() for g in state.get("guidelines", [])}
care_rows = [r for r in care.splitlines() if r.startswith("|") and not re.match(r"^\|[\s:\-|]+$", r) and not r.startswith("| Topic")]
unsourced = [r[:50] for r in care_rows if not ({u.rstrip("/").lower() for u in re.findall(r"\((https?://[^)\s]+)\)", r)} & gl_urls) and "No guideline retrieved" not in r and "search unavailable" not in r]
check("Z", "no Established care section and no model-written 'Other agents' table", "### Established care" not in text and "Other agents named" not in text)
ks = re.split(r"\n#{2,4} ", text.split("## Key studies")[-1])[0] if "## Key studies" in text else ""
ks_items = [ln for ln in ks.splitlines() if re.match(r"^\d+\. ", ln)]
ks_bad = [ln[:50] for ln in ks_items if not (re.search(r"[A-Z][A-Za-z\u00c0-\u017f'\-]+(?: et al\.?| and [A-Z]|,)", ln) and re.search(r"(?:19|20)\d\d", ln)
                                              and re.search(r"meta-analys|systematic review|randomi|trial|cohort|case|review|observational|qualitative|pilot|cross-sectional", ln, re.I)
                                              and re.search(r"PMID \d{6,9}", ln) and ln.count("**") % 2 == 0 and len(ln.split()) >= 12)]
check("AA", "every Key studies entry has authors, year, design, a finding and a PMID", not ks_bad and (bool(ks_items) or "none is listed" in ks), f"{len(ks_items)} entr(ies); bad={ks_bad}")
check("AB", "the research-support disclaimer is in the closing note", "Research support, not a substitute for clinical judgement." in presented)
_order = ["## Clinical summary", "## Latest findings", "## Current evidence", "## Treatment Drug Landscape", "## Clinical trials", "## Key studies",
          "## Conflicting evidence", "## What remains under investigation", "## Evidence limitations", "## Sources"]
_pos = [presented.find(h) for h in _order]
check("AC", "the sections follow the specification's order: summary, latest findings, current evidence, drug landscape, trials, key studies, conflicting evidence, what remains, limitations, sources",
      all(x >= 0 for x in _pos) and _pos == sorted(_pos), ", ".join(h for h, x in zip(_order, _pos) if x < 0))
check("AE", "the fast answer ends with Top treatment drugs (established first, then the most notable latest drug)",
      "**Top treatment drugs (established first, then the most notable latest drug)**" in presented or "**Top treatment drugs:**" in presented)
check("AF", "emerging drugs are late-phase and labelled investigational; none is listed as established",
      all("investigational" in r.lower() and not re.search(r"phase (?:1\b|not applicable)", r, re.I) for r in text.splitlines() if re.match(r"\|[^|]*\|\s*Emerging:", r)))      # drug rows only, not the evidence-level word 'Emerging' 
check("AD", "no internal job names or letter labels in the report", not re.search(r"pubmed_|guidelines_deep|web_fast|trials_deep|fda_fast|\(\s*[A-D]\s*\)", presented))

width = max(len(t) for _, t, _, _ in results)
for code, title, ok, detail in results:
    print(f"{'PASS' if ok else 'FAIL'}  {code:4} {title:{width}}  {detail}")
print(f"\n{sum(r[2] for r in results)}/{len(results)} checks passed")
sys.exit(0 if all(r[2] for r in results) else 1)
