"""Async drug stage (slice 3): the twin of qa_stream.prepare_landscape and its helpers, plus the DrugStages mixin for the async pipeline.

Same rules as the other async files: pure helpers (verification, text building) are imported from the sync modules; only the waiting is async.
Callees still run through asyncio.to_thread and are marked BRIDGE: ma.assess (no model call; only used when SHOW_MEDICATION_ASSESSMENT is on).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import AsyncIterator
from urllib.parse import urlparse

import async_core
import async_tools as at
import async_population
import async_web
import clinical_tools as ct
import conditions as cond_groups
import drug_handoff
import landscape
import medication_assessment as ma
import qa_stream as qs
import retrieval_routing as rr


# ------------------------------------------------------------------ leaf twins
async def apage_date(url: str) -> str | None:
    """Async twin of clinical_tools.page_date: the publication date a web page states about itself; None when it states none."""
    try:
        reply = await async_core.client().get(url, timeout=6)
        if reply.status_code != 200 or "html" not in (reply.headers.get("content-type") or ""):
            return None
        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", re.sub(r"(?is)<(?:script|style).*?</(?:script|style)>", " ", reply.content[:400000].decode("utf-8", "ignore"))))
    except Exception:
        return None
    found = ct._LABELLED_DATE.search(text) or re.search(f"({ct._FULL_DATE})", text[:6000])
    return found.group(1) if found else None


def join_fda_decisions(first: dict | None, second: dict) -> dict:
    """The decisions of two passes over the labels (the labels in hand, then the ones the handoff added). Each record carries its own decision, so the lists are added; what is still undecided is what the LAST pass says."""
    if not first:
        return second
    out = dict(second)
    for key in ("confirmed", "excluded", "reviewer_rejected", "removed"):
        out[key] = list(first.get(key) or []) + list(second.get(key) or [])
    return out


async def ainterpret_fda_labels(question: str, records: list[dict], groups: list[dict] | None = None) -> dict:
    """Async twin of qa_stream.interpret_fda_labels."""
    out = {"confirmed": [], "excluded": [], "unreviewed": [], "reviewer_rejected": []}
    offered = {}
    for r in records:
        text = (r.get("indication_statement") or r.get("indication_excerpt") or "")[:900]
        if r.get("model_relevant") is None and text:
            offered[f"fda:{r['generic']}:{str(r.get('set_id') or len(offered))[:8]}"] = {"indication_text": text, "record": r, "sentences": qs.numbered_sentences(text, 900, 12)}
    if groups:
        order = cond_groups.interleave(list(offered.items()), lambda kv: str(kv[1]["record"].get("generic") or "") + " " + kv[1]["indication_text"], groups)
        offered = dict(order[:min(30, 12 * len(groups))])
    else:
        offered = dict(list(offered.items())[:20])
    for attempt in (1, 2):
        todo = {k: v for k, v in offered.items() if v["record"].get("model_relevant") is None}
        if not todo:
            break
        payload = {"fda_labels": [{"id": k, "sentences": [f"{i}. {t}" for i, t in enumerate(v["sentences"], 1)]} for k, v in todo.items()]}
        if groups:
            payload["question_conditions"] = cond_groups.names(groups)
        answer = await async_core.aask_json(question, qs.FDA_TASK, payload, 900 if groups or len(todo) > 12 else 600)
        part = qs.verify_fda_decisions(answer.get("fda_labels") or [], todo, answer.get("question_asks"))
        out["confirmed"] += part["confirmed"]
        out["excluded"] += part["excluded"]
    out["unreviewed"] = [v["record"]["generic"] for v in offered.values() if v["record"].get("model_relevant") is None]
    out["removed"] = qs.drop_unsuitable_labels(question, records)
    return out


async def adrug_handoff_stage(state: dict, question: str, ask=None, search=None) -> None:
    """Async twin of qa_stream.drug_handoff_stage: the medicines the sources describe get their FDA labels looked up and merged (the label searches run at the same time)."""
    if not drug_handoff.enabled():
        return
    began = time.time()
    try:
        read = drug_handoff.offered(state)
        if not read["sources"]:
            return
        answer = await (ask or async_core.aask_json)(question, drug_handoff.TASK, {"sources": read["sources"]}, 700)
        model_s = round(time.time() - began, 1)
        known = {" ".join(rr._words(r.get("generic") or "")) for r in state.get("regulatory") or []} | {" ".join(rr._words(d)) for d in (state.get("entities") or {}).get("drugs") or []}
        names = drug_handoff.verified(answer.get("drugs"), read["text"], {k for k in known if k})
        kept_keys = {n["name"].lower() for n in names}
        dropped = []      # why each listed name did not get a label search (kept in the saved state, so a short drug table can be explained)
        for item in answer.get("drugs") or []:
            name = " ".join(str((item or {}).get("name") if isinstance(item, dict) else item or "").split()).strip(" .,;")
            if name and name.lower() not in kept_keys:
                key = " ".join(rr._words(name))
                dropped.append({"name": name, "why": "a label for it was already found" if key in known else "no retrieved source writes this name" if name.lower() not in read["text"] else "length or repeat"})
        search_began = time.time()
        raw: dict = {}
        if names:
            chunks = [[n["name"] for n in names[i:i + 6]] for i in range(0, len(names), 6)]
            terms = qs.plan_condition_terms(state.get("entities") or {})
            find = search or at.asearch_fda_labels

            async def one(key: str, chunk: list):
                try:
                    return key, await asyncio.wait_for(find(terms, chunk, 8), timeout=float(os.getenv("DRUG_HANDOFF_SEARCH_S", "10") or 10))
                except Exception:
                    return key, None
            for key, value in await asyncio.gather(*(one(f"fda_handoff_{i}", c) for i, c in enumerate(chunks))):
                if value is not None:
                    raw[key] = value
        before = len(state.get("regulatory") or [])
        qs.ingest_regulatory(state, raw)
        qs.record_source_status(state, raw)
        state["drug_handoff"] = {"listed": len(answer.get("drugs") or []), "kept": [n["name"] for n in names], "dropped": dropped[:30], "labels_added": len(state.get("regulatory") or []) - before, "seconds": round(time.time() - began, 1), "model_s": model_s, "search_s": round(time.time() - search_began, 1)}
    except Exception as error:
        state["drug_handoff"] = {"error": f"{type(error).__name__}: {str(error)[:80]}"}


async def alookup_doses(rows: list[dict], condition: str, question: str, cap: float = 9.0, stems: list[str] | None = None, prefetched: dict | None = None) -> None:
    """Async twin of qa_stream.lookup_doses: one web search per drug without a dose, all at once (at most `cap` seconds); the model picks a sentence and the code verifies it."""
    targets = [r for r in rows if not r.get("dose")][:14]
    if not targets:
        return
    prefetched = prefetched or {}      # searches already started when the drugs were first named: their time is not spent again
    tasks = {(prefetched.get(r["drug"]) or asyncio.ensure_future(async_web.aweb_search(f"{r['drug']} dose and route of administration for {condition}", 1500))): r for r in targets}
    candidates: dict[str, list] = {}
    try:
        done, _ = await asyncio.wait(list(tasks), timeout=cap)
        for task in done:
            try:
                results = qs._json(task.result())
            except Exception:
                continue
            found = qs.dose_candidates(tasks[task]["drug"], results if isinstance(results, list) else [], stems=stems)
            if found:
                candidates[tasks[task]["drug"]] = found
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
    if not candidates:
        return
    payload = {"condition": condition, "drugs": [{"drug": d, "candidates": [{"id": c["id"], "text": c["text"]} for c in cs]} for d, cs in candidates.items()]}
    chosen = qs.verify_dose_picks((await async_core.aask_json(question, qs.DOSE_PICK_TASK, payload, 600)).get("doses"), candidates)
    for row in targets:
        pick = chosen.get(row["drug"])
        if pick:
            row["dose"], row["dose_url"] = pick["text"], pick["url"]


async def apick_statements(question: str, state: dict) -> tuple[list[dict], str | None]:
    """Async twin of medication_assessment.pick_statements."""
    pool = ma.source_pool(state)
    if not pool:
        return [], None
    payload = {"condition": (state.get("entities") or {}).get("condition") or question,
               "sources": [{"id": k, "title": v["label"][:90], "sentences": [f"{i}. {s}" for i, s in enumerate(v["sentences"], 1)]} for k, v in pool.items()]}
    try:
        answer = await async_core.aask_json(question, ma.STATEMENT_TASK, payload, 600)
    except Exception as error:
        return [], f"{type(error).__name__}: {str(error)[:80]}"
    return ma.verify_picks((answer or {}).get("picks"), pool), None


def _finished(task):
    """The result of a task that has finished without error; None otherwise (a lookup that did not finish in time is listed as plain text)."""
    try:
        return task.result() if task.done() and not task.cancelled() and task.exception() is None else None
    except Exception:
        return None


# ------------------------------------------------------------------ the stage
class DrugStages:
    """Methods added to AsyncRun (with DeepStages). `init_drugs()` must be called once, inside the running loop."""

    def init_drugs(self) -> None:
        self.guidelines_ready = asyncio.Event()
        self.landscape_box: dict = {}
        self.landscape_task: asyncio.Task | None = None
        self.landscape_emitted = False
        self.merge_task: asyncio.Task | None = None
        self.parts: dict = {}      # head, landscape, trials, tail: put into the report in the document's order at the end
        self.sections: list = [""]      # slot 0: the opening + top drugs, filled in once the FDA results are in

    # ---- the preparation: no streaming, runs as a task while the writers work
    async def aprepare_landscape(self) -> str:
        state, question, clock, skipped = self.state, self.question, self.clock, self.skipped
        papers, trials = self.papers, self.trials
        marks = {"thread_started": round(clock.now(), 1)}
        light = state.get("medicines_needed") is False

        def time_left():
            """Seconds left before RUN_DEADLINE_S on the run clock; None when no deadline is set (the default). Only the OPTIONAL work of the drug stage uses it: the extra web sources, the second reading,
            the class expansion, the dose lookups and the label wait. The first reading of the sources always finishes, so the drugs the sources name are never cut."""
            try:
                limit = float(os.getenv("RUN_DEADLINE_S", "0") or 0)
            except ValueError:
                limit = 0.0
            return None if limit <= 0 else limit - clock.now()
        agent_mode = bool(getattr(self, "drug_agent_active", False)) and not light      # the drug agent's own sources are read first; the main pipeline's sources follow as a top-up
        agent_box = (getattr(self, "drug_agent_box", None) or {}) if agent_mode else {}

        async def merge_groups() -> None:
            await self.absorb("guidelines")
            self.guidelines_ready.set()
            marks["guidelines_in"] = round(clock.now(), 1)
            await self.absorb("drugs")
            marks["drugs_in"] = round(clock.now(), 1)
            await self.absorb("trials")
            marks["data_in"] = round(clock.now(), 1)
            state["trial_status"] = "complete" if qs.family_status(state, "trials").get("retrieval_status") == "success" else "partial"
            state["trials"] = list(trials.values())

        async def merge_after_ranking() -> None:
            """Agent mode: the groups are merged into the shared pools only when the main pipeline has finished ranking its papers (its ranking thread rewrites the paper pool); the agent's papers and pages join them then."""
            try:
                await asyncio.wait_for(self.papers_ranked.wait(), timeout=self.T.escalation + 10)
            except asyncio.TimeoutError:
                pass
            await merge_groups()
            for pmid, rec in (agent_box.get("papers") or {}).items():
                papers.setdefault(pmid, rec)      # the drug table cites them, so they are papers of the report (its references are numbered from this pool)
            known_urls_now = {x.get("url") for x in state["web"]}
            for w in agent_box.get("web") or []:
                if w.get("url") and w["url"] not in known_urls_now:
                    state["web"].append(w)
                    known_urls_now.add(w["url"])
            marks["merged"] = round(clock.now(), 1)

        merge_task = asyncio.ensure_future(merge_after_ranking()) if agent_mode else None
        if not agent_mode:
            await merge_groups()
        handoff_task = None

        async def handoff_after_merge():
            if merge_task is not None:
                await merge_task
            return await adrug_handoff_stage(state, question)
        if not light:      # the handoff (its model call and the label lookups) runs BESIDE the reading of the sources; only the FDA-label reading waits for it, because it needs the labels it adds
            handoff_task = asyncio.ensure_future(handoff_after_merge())
        fda_recs: list = []
        fda_label_tasks: list = []
        dose_tasks: list = []

        async def condense(rec: dict):
            prompt = f"TASK: {qs.DOSE_TASK}\n\nDRUG: {rec['generic']}\nUSE: {question}\nLABEL DOSING TEXT:\n{rec['dose_statement']}"
            try:
                return rec, (await async_core.acollect(qs.load_skill_text("drug-intelligence"), prompt, 300)).replace("**", "").strip()
            except Exception:
                return rec, ""

        def start_fda_work():
            for rec in state["regulatory"]:
                if qs.fda_relevant(rec) and not any(rec is x for x in fda_recs):
                    fda_recs.append(rec)
                    fda_label_tasks.append(asyncio.ensure_future(at.afda_label_link(rec["generic"])))
                    if len(rec.get("dose_statement") or "") > 350 and not light:
                        dose_tasks.append(asyncio.ensure_future(condense(rec)))
        start_fda_work()

        offered: dict = {}

        def build_sources(src_papers: dict, src_web: list, src_guidelines: list) -> list:
            """Entries of `offered` for these sources; a source that is already offered is kept as it is. Returns the keys that were added."""
            built: dict = {}
            for p in sorted((p for p in list(src_papers.values()) if not p.get("retracted")), key=lambda p: not str(p.get("origin", "")).startswith("pubmed_drugs"))[:qs.DRUG_PAPERS]:
                if not p.get("retracted") and not p.get("protocol") and p.get("abstract"):
                    built[p["pmid"]] = {"title": p["title"][:160], "text": qs.snippet(p["abstract"], 900), "label": f"PMID {p['pmid']} ({p['cite'].split(' [')[0]})", "url": "", "kind": "paper", "study_type": p.get("study_type")}
            for w in src_web[:qs.DRUG_WEB_PAGES]:
                if w.get("url") and qs.source_family(w.get("source") or "") != "regulator" and w.get("snippet"):
                    built[w["url"]] = {"title": (w.get("title") or "web page")[:160], "text": w["snippet"], "label": f"[{(w.get('title') or 'web page')[:70]}]({w['url']})", "url": w["url"], "kind": "web"}
            for g in src_guidelines:
                built[g["url"]] = {"title": g["title"][:160], "text": g.get("text") or g.get("excerpt") or "", "label": f"[{g['organisation']}: {g['title'][:60]}]({g['url']})", "url": g["url"], "kind": "guideline"}
            added = []
            for key, entry in built.items():
                if key not in offered:
                    entry["sentences"] = qs.numbered_sentences(entry["text"], complete_only=entry["kind"] != "paper")
                    offered[key] = entry
                    added.append(key)
            return added

        build_sources((agent_box.get("papers") or {}) if agent_mode else papers,
                      (list(agent_box.get("web") or []) + list(state["web"])) if agent_mode else state["web"],
                      [] if agent_mode else state.get("guidelines", []))
        offered_reg: dict = {}

        def build_offered_reg() -> None:
            for w in state["web"]:
                page = (w.get("url") or "").split("#", 1)[0].split("?", 1)[0].rstrip("/").lower()
                host = (urlparse(w.get("url") or "").hostname or "").lower()
                if qs.source_family(w.get("source") or "") == "regulator" and w.get("snippet") and any(host == h or host.endswith("." + h) for h in qs.REGULATOR_HOSTS) \
                        and not any(v["url"].split("#", 1)[0].split("?", 1)[0].rstrip("/").lower() == page for v in offered_reg.values()) and len(offered_reg) < 8:
                    offered_reg[f"reg:{len(offered_reg) + 1}"] = {"title": re.sub(r"\s*/\s*FDA\s*$", "", re.sub(r"\s+", " ", w.get("title") or w["url"]).strip()), "url": w["url"], "text": w["snippet"][:1200]}
            for v in offered_reg.values():
                v["text"] = v["text"][:1200]
                head = v["title"][:40].lower()
                v["sentences"] = [x for x in qs.numbered_sentences(v["text"], 1200, complete_only=True) if not x.lower().startswith(head)]

        if not agent_mode:      # in agent mode the regulator pages are read after the main pipeline's groups are merged (they come from its searches)
            build_offered_reg()
        condition = (state.get("entities") or {}).get("condition") or question
        dose_prefetch: dict = {}      # drug -> its dose web search, started as soon as the drug interpretation has named the drug (not after the other interpretation calls)
        label_prefetch: dict = {}      # drug -> its own DailyMed label lookup, started at the same moment

        def prefetch(rows: list) -> None:
            for r in rows:
                name = r.get("drug")
                if not name:
                    continue
                if name not in label_prefetch:
                    label_prefetch[name] = asyncio.ensure_future(at.afda_label_link(name))
                if not light and not r.get("dose") and name not in dose_prefetch and len(dose_prefetch) < 14:
                    dose_prefetch[name] = asyncio.ensure_future(async_web.aweb_search(f"{name} dose and route of administration for {condition}", 1500))

        SECOND_READING = (" SECOND READING: a first reading of these sources found few drugs. Read EVERY sentence again and list every specific medicine a sentence says is used, prescribed, given, "
                          "recommended or studied in patients for the condition, its symptoms or its associated problems (antibiotics, immune-directed, anti-inflammatory, psychiatric, symptom-directed). "
                          "When a sentence names a class AND names specific drugs of it, list each named drug. Still never invent a drug, an id or a sentence number.")

        async def interpret_part(keys: list, second: bool = False) -> dict:
            payload = {"condition": condition, "fda_labelled": [r["generic"] for r in state["regulatory"] if qs.fda_relevant(r)],
                       "sources": [{"id": k, "kind": offered[k]["kind"], "title": offered[k]["title"], "sentences": [f"{i}. {t}" for i, t in enumerate(offered[k]["sentences"], 1)]}
                                   for k in keys if offered[k]["sentences"]]}
            return await async_core.aask_json(question, qs.DRUG_INTERPRET_TASK + async_population.reader_note(question, state) + (SECOND_READING if second else ""), payload, 900 if second else 700)

        async def interpret_drugs() -> dict:
            result = {"rows": [], "rejected": [], "items": [], "error": None}
            keys = list(offered)
            parts = [keys[i:i + qs.DRUG_PART_SIZE] for i in range(0, len(keys), qs.DRUG_PART_SIZE)]

            async def safe(part: list, second: bool = False) -> dict:
                try:
                    return await interpret_part(part, second)
                except Exception as error:
                    return {"_error": f"{type(error).__name__}: {str(error)[:80]}"}
            async def topup_read() -> dict:
                """Runs BESIDE the first reading (no added wait): web pages that name the specific medicines for the condition, read by the same reader; every drug is verified against its own sentence like the rest."""
                try:
                    found = await asyncio.gather(*(async_web.aweb_search(q, 1200) for q in (f"medications used to treat {condition}: names of the specific drugs", f"{condition} pharmacological treatment drug options guideline")), return_exceptions=True)
                    new_keys = []
                    for raw in found:
                        pages = json.loads(raw) if isinstance(raw, str) else []
                        for w in pages if isinstance(pages, list) else []:
                            url, text = w.get("url"), w.get("snippet")
                            if url and text and url not in offered and len(new_keys) < 8:
                                offered[url] = {"title": (w.get("title") or "web page")[:160], "text": text, "label": f"[{(w.get('title') or 'web page')[:70]}]({url})", "url": url, "kind": "web"}
                                offered[url]["sentences"] = qs.numbered_sentences(text, complete_only=True)
                                new_keys.append(url)
                    return await asyncio.wait_for(interpret_part(new_keys, False), timeout=float(os.getenv("DRUG_TOPUP_S", "16"))) if new_keys else {}
                except Exception:
                    return {}      # the extra sources only add
            for attempt in ((1,) if light else (1, 2)):
                topup = asyncio.ensure_future(topup_read()) if (attempt == 1 and not light and not agent_mode and os.getenv("DRUG_TOPUP", "on") != "off") else None
                answers = list(await asyncio.gather(*(safe(p, attempt == 2) for p in parts)))
                if topup is not None:      # the extra web sources only add: they run beside the first reading and, with a deadline, get only the time that is left
                    left = time_left()
                    await asyncio.wait([topup], timeout=None if left is None else max(0.5, left))
                    if topup.done() and not topup.cancelled():
                        answers.append(topup.result())
                    else:
                        topup.cancel()
                if answers and all("_error" in a for a in answers):
                    result["error"] = answers[0]["_error"]
                    break
                seen_items = {(str(d.get("drug", "")).lower(), str(d.get("source_id", ""))) for d in result["items"]}
                result["items"] += [d for a in answers for d in (a.get("drugs") or []) if (str(d.get("drug", "")).lower(), str(d.get("source_id", ""))) not in seen_items]
                result["rows"], result["rejected"] = qs.verify_drug_decisions(result["items"], offered, state, limit=qs.MAX_USED_DRUGS)
                prefetch(result["rows"])
                if len(result["rows"]) >= int(os.getenv("DRUG_RETRY_BELOW", "4")) or not offered or clock.now() > 50.0 or (time_left() is not None and time_left() < 0):      # fewer than 4 drugs: the sources are read a second time, with a stronger instruction
                    break
            max_classes = int(os.getenv("DRUG_CLASS_EXPAND", "6"))
            left = time_left()
            if max_classes and not light and not result["error"] and (left is None or left > 2.0):
                try:
                    expand_s = float(os.getenv("DRUG_CLASS_EXPAND_S", "14"))
                    await asyncio.wait_for(expand_classes(result, max_classes), timeout=expand_s if left is None else min(expand_s, left))
                except Exception:
                    pass      # the drugs found so far stay: the expansion only adds
            return result

        async def expand_classes(result: dict, max_classes: int, skip: frozenset = frozenset()) -> None:
            """Class-to-specific expansion (drug-intelligence skill, Pass 1 item 5). The sources named drug CLASSES (penicillins, macrolides, corticosteroids ...) that cannot be a row: each class is searched on the web
            for its specific medicines, the pages found are offered as new sources, the model reads them for sentences that name a specific drug of the class used for the condition, and every drug is verified
            against its sentence exactly like the others. Nothing is added from memory."""
            classes = []
            for name, why in result["rejected"]:
                if ("not one specific drug" in why or why == "not a drug name") and name.lower() not in {c.lower() for c in classes} and name.lower() not in skip and not re.search(r"(?i)\btherap(y|ies)\b|\bphage\b", name):
                    classes.append(name)
            classes = classes[:max_classes]
            if not classes:
                return
            found = await asyncio.gather(*(async_web.aweb_search(f"{c} used for {condition}: names of the specific drugs, for example", 1200) for c in classes), return_exceptions=True)
            new_keys = []
            for c, raw in zip(classes, found):
                try:
                    pages = json.loads(raw) if isinstance(raw, str) else []
                except ValueError:
                    pages = []
                for w in pages if isinstance(pages, list) else []:
                    url, text = w.get("url"), w.get("snippet")
                    if url and text and url not in offered:
                        offered[url] = {"title": (w.get("title") or "web page")[:160], "text": text, "label": f"[{(w.get('title') or 'web page')[:70]}]({url})", "url": url, "kind": "web"}
                        offered[url]["sentences"] = qs.numbered_sentences(text, complete_only=True)
                        new_keys.append(url)
            if not new_keys:
                return
            note = (" CLASS EXPANSION: these sources were found for the drug classes " + ", ".join(classes) + ". List each SPECIFIC drug (one active substance or brand, never a class) that a sentence names as used, "
                    "prescribed, given or studied for the condition, its symptoms or its associated problems, with that sentence's own number. Still never invent a drug, an id or a sentence number.")

            async def read(keys: list) -> list:
                payload = {"condition": condition, "fda_labelled": [r["generic"] for r in state["regulatory"] if qs.fda_relevant(r)],
                           "sources": [{"id": k, "kind": offered[k]["kind"], "title": offered[k]["title"], "sentences": [f"{i}. {t}" for i, t in enumerate(offered[k]["sentences"], 1)]} for k in keys if offered[k]["sentences"]]}
                try:
                    return (await async_core.aask_json(question, qs.DRUG_INTERPRET_TASK + async_population.reader_note(question, state) + note, payload, 900)).get("drugs") or []
                except Exception:
                    return []
            chunks = [new_keys[i:i + qs.DRUG_PART_SIZE] for i in range(0, len(new_keys), qs.DRUG_PART_SIZE)]
            seen_items = {(str(d.get("drug", "")).lower(), str(d.get("source_id", ""))) for d in result["items"]}
            for drugs in await asyncio.gather(*(read(ch) for ch in chunks)):
                for d in drugs:
                    key = (str(d.get("drug", "")).lower(), str(d.get("source_id", "")))
                    if key not in seen_items:
                        seen_items.add(key)
                        result["items"].append(d)
            result["rows"], result["rejected"] = qs.verify_drug_decisions(result["items"], offered, state, limit=qs.MAX_USED_DRUGS)
            result["expanded_from"] = classes
            prefetch(result["rows"])

        async def interpret_regulator() -> dict:
            result = {"items": [], "rejected": [], "error": None}
            if merge_task is not None:      # agent mode: the regulator pages come from the main pipeline's searches, merged after its ranking
                try:
                    await merge_task
                    build_offered_reg()
                except Exception:
                    pass
            if not offered_reg:
                return result
            try:
                answer = await async_core.aask_json(question, qs.REGULATOR_TASK, {"condition": condition, "items": [{"id": k, "title": v["title"], "sentences": [f"{i}. {t}" for i, t in enumerate(v["sentences"], 1)]} for k, v in offered_reg.items()]}, 700)
                result["raw"] = answer.get("items") or []
                result["items"], result["rejected"] = qs.verify_regulator_decisions(result["raw"], offered_reg)
                undated = [x for x in result["items"] if x["date"] == "date not stated"]
                if undated:
                    for x, found in zip(undated, await asyncio.gather(*(apage_date(x["url"]) for x in undated))):
                        if found:
                            x["date"] = found
            except Exception as error:
                result["error"] = f"{type(error).__name__}: {str(error)[:80]}"
            return result

        async def interpret_fda() -> dict:
            if merge_task is not None:      # agent mode: the labels come from the groups that are merged after the main pipeline's ranking
                try:
                    await merge_task
                except Exception:
                    pass
            early = None
            if handoff_task is not None and os.getenv("FDA_INTERPRET_OVERLAP", "on").strip().lower() not in ("off", "0", "false", "no"):
                try:      # the labels already in hand are judged WHILE the handoff still looks up the others; a label keeps its own decision, so the second pass reads only what has none
                    early = await ainterpret_fda_labels(question, state["regulatory"], cond_groups.groups_of(state))
                    marks["fda_first_pass"] = round(clock.now(), 1)
                except Exception:
                    early = None      # the second pass reads everything that has no decision
            try:
                if handoff_task is not None:
                    await asyncio.shield(handoff_task)
            except Exception:
                pass      # a failed handoff is already recorded by the stage itself; the labels in hand are read
            try:
                return join_fda_decisions(early, await ainterpret_fda_labels(question, state["regulatory"], cond_groups.groups_of(state)))
            except Exception as error:
                return join_fda_decisions(early, {"confirmed": [], "excluded": [], "unreviewed": [r["generic"] for r in state["regulatory"] if r.get("model_relevant") is None], "reviewer_rejected": [],
                                                  "error": f"{type(error).__name__}: {str(error)[:80]}"})

        async def interpret_drugs_agent() -> dict:
            """Agent mode: the agent's own sources are read first (reading, second reading and class expansion all run while the main pipeline is still searching and ranking); the sources the main pipeline
            found are then read as a top-up and everything is verified together, so no drug the old stage found is left out."""
            result = await interpret_drugs()
            marks["agent_read"] = round(clock.now(), 1)
            try:
                await merge_task
                fresh = build_sources(papers, state["web"], state.get("guidelines", []))
                cap = int(os.getenv("DRUG_FRESH_MAX", "40"))
                for key in fresh[cap:]:
                    offered.pop(key, None)
                fresh = fresh[:cap]
                marks["fresh_sources"] = len(fresh)

                async def safe_part(part: list) -> dict:
                    try:
                        return await interpret_part(part)
                    except Exception as error:
                        return {"_error": f"{type(error).__name__}: {str(error)[:80]}"}
                if fresh:
                    answers = await asyncio.gather(*(safe_part(fresh[i:i + qs.DRUG_PART_SIZE]) for i in range(0, len(fresh), qs.DRUG_PART_SIZE)))
                    seen_items = {(str(d.get("drug", "")).lower(), str(d.get("source_id", ""))) for d in result["items"]}
                    for a in answers:
                        for d in a.get("drugs") or []:
                            key = (str(d.get("drug", "")).lower(), str(d.get("source_id", "")))
                            if key not in seen_items:
                                seen_items.add(key)
                                result["items"].append(d)
                    if any("_error" not in a for a in answers):
                        result["error"] = None
                    result["rows"], result["rejected"] = qs.verify_drug_decisions(result["items"], offered, state, limit=qs.MAX_USED_DRUGS)
                    prefetch(result["rows"])
                    max_classes, left = int(os.getenv("DRUG_CLASS_EXPAND", "6")), time_left()
                    if max_classes and not result["error"] and (left is None or left > 2.0):
                        expand_s = float(os.getenv("DRUG_CLASS_EXPAND_S", "14"))
                        await asyncio.wait_for(expand_classes(result, max_classes, frozenset(c.lower() for c in result.get("expanded_from") or [])), timeout=expand_s if left is None else min(expand_s, left))
            except Exception as error:
                skipped.append(f"the top-up reading of the main pipeline's sources did not complete ({type(error).__name__}); the drugs of the drug agent's sources are listed")
            marks["topped_up"] = round(clock.now(), 1)
            return result

        # the drug call (itself several calls), the FDA-label call, the regulator call and the medication-statement call run together
        coroutines = [interpret_drugs_agent() if agent_mode else interpret_drugs(), interpret_regulator(), interpret_fda()]
        if qs.SHOW_MEDICATION_ASSESSMENT:
            coroutines.append(apick_statements(question, state))
        results = await asyncio.gather(*coroutines)
        if merge_task is not None:
            try:
                await merge_task
            except Exception as error:
                skipped.append(f"the retrieved groups could not be merged ({type(error).__name__})")
        if handoff_task is not None:
            try:
                await handoff_task
            except Exception:
                pass
        drug_result, reg_result, fda_result = results[0], results[1], results[2]
        if drug_result["items"] and not drug_result["error"]:
            drug_result["rows"], drug_result["rejected"] = qs.verify_drug_decisions(drug_result["items"], offered, state, limit=qs.MAX_USED_DRUGS)
        picks, pick_error = results[3] if len(results) > 3 else ([], None)
        marks["interpreted"] = round(clock.now(), 1)
        async_population.apply_population(state, question)      # a question about children: an FDA label that does not support use in children is not a drug for this patient group
        start_fda_work()
        self.sections[0] = (self.header + self.focus_lead) if self.focused else self.header + self.summary_head + self.opening + "\n\n" + qs.top_drugs_markdown(state)
        fda_names = [r["generic"] for r in fda_recs]
        if drug_result["error"]:
            skipped.append(f"drugs used in practice: the drug interpretation did not complete ({drug_result['error']}); none are listed")
        if reg_result["error"]:
            skipped.append(f"regulator announcements: the interpretation did not complete ({reg_result['error']}); none are listed")
        if fda_result.get("error"):
            skipped.append(f"FDA labels: the interpretation did not complete ({fda_result['error']}); labels without a decision are not listed")
        if fda_result["unreviewed"]:
            skipped.append(f"{len(fda_result['unreviewed'])} retrieved FDA label(s) got no verified relevance decision from the interpretation step and are not listed")
        used_rows, regulator_items = drug_result["rows"], reg_result["items"]
        have = {re.sub(r"\W+", " ", str(r.get("drug") or "")).split(" ")[0].lower() for r in used_rows}
        for item in regulator_items:
            name = str(item.get("drug") or "").strip()
            key = re.sub(r"\W+", " ", name).split(" ")[0].lower() if name else ""
            if (item.get("classification") in {"approval", "new_use", "label_update"} and key and key not in have and str(item.get("quote") or "").strip()
                    and item.get("relation") in {"treats_condition", "treats_associated_symptom_or_comorbidity"} and len(used_rows) < qs.MAX_USED_DRUGS):
                used_rows.append({"dose": "", "drug": name, "relevance": item.get("relation"), "source": item.get("url"), "label": f"[{item.get('title')}]({item.get('url')})",
                                  "excerpt": item.get("quote"), "purpose": item.get("concerns") or name, "population": "not stated in this sentence", "category": "Approved (regulator announcement)"})
                have.add(key)
        label_jobs = {r["drug"]: (label_prefetch.get(r["drug"]) or asyncio.ensure_future(at.afda_label_link(r["drug"]))) for r in used_rows}      # each listed drug's OWN DailyMed page, looked up while the dose searches run
        try:
            if not light:
                doses_call = alookup_doses(used_rows, condition, question, stems=[x for x in (state.get("core_stems") or []) if len(x) >= 4], prefetched=dose_prefetch)
                left = time_left()
                await (doses_call if left is None else asyncio.wait_for(doses_call, timeout=max(3.0, left)))      # a late dose lookup leaves the doses the sources already gave
        except Exception as error:
            skipped.append(f"dose lookup for the listed drugs did not complete ({type(error).__name__}); doses are shown only where a source gave one")
        known_urls = {w.get("url") for w in state["web"]}
        for row in used_rows:
            if row.get("dose_url") and row["dose_url"] not in known_urls:
                state["web"].append({"title": "dose page for " + row["drug"], "url": row["dose_url"], "snippet": row["dose"], "source": "web_dose"})
                known_urls.add(row["dose_url"])
        marks["doses"] = round(clock.now(), 1)

        trial_rows: list = []
        state["trial_drugs"] = []
        marks["reviewed"] = round(clock.now(), 1)
        state["regulator_items"] = regulator_items
        state["used_drugs_trace"] = {"model_items": drug_result["items"], "sources_offered": len(offered), "accepted": len(used_rows), "rejected": drug_result["rejected"],
                                     "fda_decisions": fda_result, "model_regulator_items": reg_result.get("raw"), "regulator_rejected": reg_result["rejected"]}
        counts: dict = {}
        for r in used_rows:
            key = re.sub(r"\W+", " ", str(r.get("excerpt") or "")).strip().lower()
            counts[key] = counts.get(key, 0) + 1
        used_rows[:] = [r for r in used_rows if r.get("relevance") != "treats_associated_symptom_or_comorbidity"
                        or counts.get(re.sub(r"\W+", " ", str(r.get("excerpt") or "")).strip().lower(), 0) < 3 or r.get("category", "").startswith(("Guideline", "Approved"))]
        if not landscape.SHOW_SYMPTOM_DRUGS:
            used_rows[:] = [r for r in used_rows if r.get("relevance") != "treats_associated_symptom_or_comorbidity" or r.get("category", "").startswith(("Guideline", "Approved"))]
        state["used_drugs"] = [r["drug"] for r in used_rows]
        listed = used_rows + trial_rows
        waiting = list(label_jobs.values()) + fda_label_tasks + dose_tasks
        if waiting:      # the label links and dose summaries get a few seconds at most: the answer is never held up by a slow lookup
            left = time_left()
            await asyncio.wait(waiting, timeout=float(os.getenv("LABEL_WAIT_S", "15")) if left is None else min(float(os.getenv("LABEL_WAIT_S", "15")), max(2.0, left)))
        listed_labels = [_finished(label_jobs[r["drug"]]) for r in listed]
        fda_labels = [_finished(t) for t in fda_label_tasks]
        for row, found in zip(listed, listed_labels):
            row["fda_label"] = (found or {}).get("url")
            generic = ((found or {}).get("generic_name") or "").lower()
            if generic and generic != row["drug"].lower():
                row["drug"] = f"{row['drug']} ({generic})"
        used_rows[:], listed_labels = landscape.reconcile_used_rows(state, used_rows, listed_labels[:len(used_rows)])      # brand and generic of one drug are one row; a drug an FDA row already shows is not listed again; 'off-label' only from a retrieved label
        listed = used_rows + trial_rows
        state["used_drugs"] = [r["drug"] for r in used_rows]
        state["used_rows_final"] = list(used_rows)
        shown_fda = [(n, f) for n, f, rec in zip(fda_names, fda_labels, fda_recs) if qs.fda_relevant(rec)]
        safety_labels = [{"drug": n, "url": f["url"], "boxed_warning": f.get("boxed_warning")} for n, f in list(zip([r["drug"] for r in listed], listed_labels)) + shown_fda if f]
        state["drug_labels"] = [{"drug": x["drug"], "url": x["url"], "boxed_warning": x.get("boxed_warning")} for x in safety_labels]
        state["listed_drugs"] = [r["drug"] for r in listed] + [rec["generic"] for rec in fda_recs if qs.fda_relevant(rec)]
        marks["labels"] = round(clock.now(), 1)
        for task in dose_tasks:
            rec, text = _finished(task) or (None, "")
            if rec is None:
                continue
            if qs.dose_summary_ok(text, rec["dose_statement"]):
                rec["dose_summary"] = text
            else:
                rec["dose_summary"] = qs.dose_fallback(rec["dose_statement"])
                skipped.append(f"dose summary for {rec['generic']} was not faithful to the label text; the label's own first sentences are shown")
        for rec in fda_recs:
            if rec.get("dose_statement") and not rec.get("dose_summary"):
                rec["dose_summary"] = qs.dose_fallback(rec["dose_statement"])
        for task in waiting + list(label_prefetch.values()) + list(dose_prefetch.values()):      # lookups that did not finish in time (or were started for a row that was dropped) are not left running
            if not task.done():
                task.cancel()
        marks["dose"] = round(clock.now(), 1)
        state["landscape_marks"] = marks
        landscape_md = qs.landscape_markdown(state, used_rows, trial_rows)
        if qs.SHOW_MEDICATION_ASSESSMENT:
            try:      # builds the assessment from the sources in hand (no model call, no network)
                self.landscape_box["assessment"] = ma.assess(state, question, used_rows, picks, skipped, pick_error)
            except Exception as error:
                self.landscape_box["assessment"] = ma.fallback(f"the medication assessment did not complete ({type(error).__name__})")
        return landscape_md

    async def _run_prepare(self) -> None:
        try:
            ready = getattr(self, "drug_agent_ready", None)
            self.drug_agent_active = False
            if ready is not None:      # the drug agent runs beside the main pipeline: its sources are read as soon as it is done
                try:
                    await asyncio.wait_for(ready.wait(), timeout=float(os.getenv("DRUG_AGENT_WAIT_S", "30")))
                except asyncio.TimeoutError:
                    pass
                self.drug_agent_active = bool((getattr(self, "drug_agent_box", None) or {}).get("ok")) and self.state.get("medicines_needed") is not False
            if not self.drug_agent_active:      # no agent (or it failed): the drug stage starts when the main pipeline's papers are ranked, as before
                gate = self.papers_ranked if qs.DRUG_STAGE_EARLY else self.evidence_ready      # early: starts when the papers are merged and ranked
                try:
                    await asyncio.wait_for(gate.wait(), timeout=self.T.escalation + 10)
                except asyncio.TimeoutError:
                    pass
            self.landscape_box["md"] = await self.aprepare_landscape()
        except Exception as error:
            self.landscape_box["error"] = f"{type(error).__name__}: {str(error)[:120]}"
        finally:
            self.guidelines_ready.set()

    def start_drug_agent(self) -> None:
        """Called by the fast-search stage: the drug agent (async_drug_agent) starts beside the plan and the deep search when DRUG_AGENT=on."""
        import async_drug_agent
        if async_drug_agent.enabled() and not self.focused:
            async_drug_agent.start(self)

    def start_landscape(self) -> None:
        if self.landscape_task is None:
            self.landscape_task = asyncio.ensure_future(self._run_prepare())

    async def amerge_groups(self) -> None:
        """A focused answer without the drug tables: the guideline, drug and trial groups are only merged into the state (the trials table and the figures read them); nothing is interpreted."""
        try:
            await self.absorb("guidelines")
            self.guidelines_ready.set()
            await self.absorb("drugs")
            await self.absorb("trials")
            self.state["trial_status"] = "complete" if qs.family_status(self.state, "trials").get("retrieval_status") == "success" else "partial"
            self.state["trials"] = list(self.trials.values())
        except Exception as error:
            self.skipped.append(f"retrieved sources could not be merged ({type(error).__name__})")
        finally:
            self.guidelines_ready.set()

    def landscape_running(self) -> bool:
        return self.landscape_task is not None and not self.landscape_task.done()

    async def wait_landscape(self, timeout: float) -> None:
        if self.landscape_running():
            await asyncio.wait([self.landscape_task], timeout=max(0.0, timeout))

    # ---- stage: drugs
    async def stage_drugs(self) -> AsyncIterator[dict]:
        if self.focused and "drugs" not in self.focus_sections:      # the drug tables were not asked for: the retrieved groups are only merged, nothing is interpreted
            self.merge_task = asyncio.ensure_future(self.amerge_groups())
            return
        self.start_landscape()      # the landscape reads its own evidence (every retrieved source, each judged by the model): it does not wait for the triage
        yield self.event("token", stage="EVIDENCE_ENRICHING", text="\n\n")

    # ---- the landscape and the trials are shown as soon as they are ready
    async def emit_landscape_and_trials(self, force: bool = False) -> AsyncIterator[dict]:
        state, skipped, box = self.state, self.skipped, self.landscape_box
        if self.landscape_emitted or self.landscape_task is None or (self.landscape_running() and not force):
            return
        self.landscape_emitted = True
        yield self.event("stage", stage="DRUGS_ENRICHING")
        if "md" not in box:      # the preparation failed or timed out: show what is known, never nothing
            skipped.append("drug landscape: " + str(box.get("error", "preparation did not finish in time")))
            box["md"] = qs.landscape_markdown(state, [], [])
        landscape_md = box["md"]
        if self.focused:
            landscape_md = qs.focused_landscape(landscape_md, (self.given_plan or {}).get("info_needed") or [], keep_all=self.discovery)
            box["md"] = landscape_md
        if qs.landscape_is_empty(landscape_md) and not self.focused:      # the section is shown for every new question: when no drug was retrieved it says so
            landscape_md = ("## Treatment Drug Landscape\n\nNo drug that treats this condition, or that this question is about, was identified in the retrieved sources "
                            "(PubMed, ClinicalTrials.gov, FDA labels, guideline and web pages), so no drug table is shown. This does not mean none exists.")
            box["md"] = landscape_md
            state["landscape_empty_note"] = True
            yield self.event("token", stage="DRUGS_ENRICHING", text="\n\n" + landscape_md + "\n\n")
        elif qs.landscape_is_empty(landscape_md):      # a follow-up: no empty section is printed
            landscape_md = ""
            state["landscape_hidden"] = True
            box["md"] = ""
            if self.focused:      # a follow-up must still say what it found
                yield self.event("token", stage="DRUGS_ENRICHING", text="\n\nNo information on this was found in the retrieved sources for the drugs of the earlier answers.\n\n")
        else:
            yield self.event("token", stage="DRUGS_ENRICHING", text="\n\n" + landscape_md + "\n\n")
        state["drug_intelligence_status"] = qs.drug_intelligence_status(state, not any(x.startswith("drugs used in practice") for x in skipped))
        state["drugs"] = landscape_md
        self.parts["landscape"] = landscape_md
        if qs.SHOW_MEDICATION_ASSESSMENT:
            assessment = box.get("assessment")
            if not assessment:
                try:
                    assessment = ma.assess(state, self.question, [], [], skipped + ["drug landscape: " + str(box.get("error", "preparation did not finish in time"))])
                except Exception:
                    assessment = ma.fallback(box.get("error") or "the drug stage did not finish in time")
            self.parts["medication"] = ma.markdown(assessment, primary=state.get("medicines_about") is not False, compact=state.get("medicines_about") is False)
            state["medication_assessment"] = {"outcome": assessment["outcome"], "label": assessment["label"], "medicines": [e["drug"] for e in assessment["entries"]], "needs": assessment["needs"],
                                              "incomplete": assessment["incomplete"], "failures": assessment["failures"], "statements": assessment["picks"], "sources_read": assessment["sources_read"]}
            yield self.event("token", stage="DRUGS_ENRICHING", text="\n\n" + self.parts["medication"] + "\n\n")
        self.clock.mark("drugs_done_s")
        if self.focused:      # the trials were not asked for
            self.clock.mark("trials_done_s")
            return
        yield self.event("stage", stage="TRIALS_ENRICHING")      # clinical-trials: registry records, no model
        self.parts["trials"] = qs.trials_markdown(state)
        yield self.event("token", stage="TRIALS_ENRICHING", text="\n\n" + self.parts["trials"] + "\n\n")      # a blank line after the trials: the next section's heading starts on its own line
        self.clock.mark("trials_done_s")
