"""Async figures (slice 7): the twins of clinical_tools.candidates / search_paper_figures, source_figures.download / _screen / select_figures, qa_stream.vision_ask / vision_inspect / find_figures.

Network and model calls are awaited. The only worker-thread calls left are CPU / file work that cannot be awaited: measuring the text size of an image and saving it (Pillow), and encoding an image
for the vision model. Pure helpers (gates, quote checks, population and strict checks, explanations) are imported from source_figures, never copied.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path

import async_core
import clinical_tools as ct
import qa_stream as qs
import source_figures as sf


# ------------------------------------------------------------------ the figures of the papers (open-access bucket)
async def aoa_article(pmcid: str):
    """Async twin of clinical_tools._article: (article XML text, the file names of that article in the open-access bucket) for the newest version; None when it is not in the bucket."""
    http = async_core.client()
    params = {"list-type": "2", "prefix": pmcid + ".", "max-keys": "200"}
    listing = await http.get(ct.OA_BUCKET, params=params)
    if listing.status_code != 200:
        await asyncio.sleep(1.0)
        listing = await http.get(ct.OA_BUCKET, params=params)
    if listing.status_code != 200:
        return None
    keys = re.findall(r"<Key>(.*?)</Key>", listing.text)
    versions = sorted({k.split("/")[0] for k in keys}, key=lambda v: int(v.rsplit(".", 1)[-1]) if v.rsplit(".", 1)[-1].isdigit() else 0)
    if not versions:
        return None
    newest = versions[-1]
    mine = [k for k in keys if k.startswith(newest + "/")]
    xml_key = f"{newest}/{newest}.xml"
    if xml_key not in mine:
        return None
    response = await http.get(ct.OA_BUCKET + xml_key)
    return (response.text, mine) if response.status_code == 200 else None


async def _abatch(batch: list[dict], cap: float, statuses: list) -> list[dict]:
    """Async twin of clinical_tools._batch: the articles of one batch of papers fetched side by side for at most `cap` seconds."""
    tasks = {asyncio.ensure_future(aoa_article(p["pmcid"])): p for p in batch}
    out = []
    try:
        done, _ = await asyncio.wait(list(tasks), timeout=cap)
        for task in done:
            paper = tasks[task]
            try:
                article = task.result()
                statuses.append(200 if article else 404)
                if not article:
                    continue
                xml_text, keys = article
                licence = ct.open_licence(xml_text)
                for fig in ct.parse_figures(xml_text, paper["pmcid"], keys):
                    out.append({**fig, "pmid": paper.get("pmid"), "pmcid": paper["pmcid"], "doi": paper.get("doi"), "title": paper.get("title") or None, "cite": paper.get("cite") or None,
                                "design": paper.get("design") or None, "licence": licence})
            except Exception:
                continue
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
    return out


async def acandidates(papers: list[dict], want: int = 12, batch_size: int = 8, batches: int = 2, cap: float = 7.0, trace: dict | None = None) -> list[dict]:
    """Async twin of clinical_tools.candidates."""
    chosen, seen = [], set()
    for p in papers:
        if p.get("pmcid") and not p.get("retracted") and p["pmcid"] not in seen:
            seen.add(p["pmcid"])
            chosen.append(p)
    statuses = trace.setdefault("page_status", []) if trace is not None else []
    out = []
    for k in range(batches):
        batch = chosen[k * batch_size:(k + 1) * batch_size]
        if not batch or len(out) >= want:
            break
        out += await _abatch(batch, cap, statuses)
    return out


async def asearch_paper_figures(papers: str, want: int = 12, batches: int = 2) -> str:
    """Async twin of clinical_tools.search_paper_figures."""
    try:
        listed = json.loads(papers) if isinstance(papers, str) else list(papers)
    except Exception:
        listed = []
    trace: dict = {}
    figures = await acandidates(listed, want=want, batches=batches, trace=trace)
    return json.dumps({"figures": figures, "page_status": trace.get("page_status", [])}, ensure_ascii=False)


# ------------------------------------------------------------------ the images
async def _fetch_image(figure: dict) -> bytes | None:
    reply = await async_core.client().get(figure["image"], headers=sf.HEADERS)
    if reply.status_code != 200 or len(reply.content) > sf.MAX_BYTES:
        return None
    return reply.content


async def ascreen(figure: dict):
    """Async twin of source_figures._screen: True/False from the pixels of the downloaded image (nothing is saved); None when it cannot be fetched or is not an image."""
    try:
        content = await _fetch_image(figure)
        if content is None:
            return None
        return await asyncio.to_thread(sf.readable_on_the_page, content)      # CPU (Pillow): kept off the event loop
    except Exception:
        return None


async def adownload(figure: dict, folder: Path) -> dict | None:
    """Async twin of source_figures.download: the fetch is awaited; decoding and saving the file (Pillow, disk) run in a worker thread."""
    try:
        content = await _fetch_image(figure)
        if content is None:
            return None
        return await asyncio.to_thread(sf.save_download, figure, content, folder)
    except Exception:
        return None


# ------------------------------------------------------------------ the vision check
async def avision_ask(text: str, figures: list[dict], max_tokens: int = 400):
    """Async twin of qa_stream.vision_ask: one call to the vision model; the reply text, or None when it could not be reached or refused."""
    try:
        provider, model = qs.vision_settings()
        images = [await asyncio.to_thread(qs.vision_image, f) for f in figures]      # reads and encodes the saved image (CPU / disk)
        if provider == "groq":
            from groq import AsyncGroq
            client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"], timeout=40.0, max_retries=0)
            extra = {"reasoning_effort": "none"} if model.startswith("qwen") else {}
            content = [{"type": "text", "text": text}] + [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}} for mime, data in images]
            waited = 0.0
            for attempt in range(4):
                try:
                    reply = await client.chat.completions.create(model=model, temperature=0, max_tokens=max_tokens, messages=[{"role": "user", "content": content}], **extra)
                    return reply.choices[0].message.content
                except Exception as error:
                    pause = re.search(r"try again in ([\d.]+)s", str(error))
                    if "rate" not in str(error).lower() or not pause or attempt == 3 or waited > 24:
                        raise
                    await asyncio.sleep(min(float(pause.group(1)) + 0.5, 12.0))
                    waited += float(pause.group(1)) + 0.5
        try:
            from mistralai import Mistral
        except ImportError:
            from mistralai.client import Mistral
        client = Mistral(api_key=os.environ["MISTRAL_API_KEY"], timeout_ms=40000)
        content = [{"type": "text", "text": text}] + [{"type": "image_url", "image_url": f"data:{mime};base64,{data}"} for mime, data in images]
        reply = await client.chat.complete_async(model=model, temperature=0, max_tokens=max_tokens, messages=[{"role": "user", "content": content}])
        return reply.choices[0].message.content
    except Exception:
        return None


async def avision_inspect(figure: dict, question: str):
    return await avision_ask(qs.VISION_PROMPT.format(question=question, caption=sf.cut_for_model(figure["caption"], 600)), [figure])


# ------------------------------------------------------------------ the selection
async def aselect_figures(found: list[dict], cited_pmids: set, direct_pmids: set, rubric, inspect, folder: Path, limit: int = sf.MAX_FIGURES, trace: dict | None = None, population=None,
                          indirect_pmids: set | None = None, require_cited: bool = False) -> list[dict]:
    """Async twin of source_figures.select_figures. `rubric`, `population` and `inspect` are async callables."""
    trace = trace if trace is not None else {}
    trace["candidates"] = len(found)
    passed = sf.eligible(found, cited_pmids, direct_pmids, trace, indirect_pmids, require_cited)
    trace["after_gates"] = len(passed)
    if passed and population is not None:
        try:
            passed = await population(passed)
        except Exception:
            passed = []
            trace["population_check_failed"] = True
        trace["after_population_check"] = len(passed)
    if not passed:
        return []
    fates = trace.setdefault("figure_fates", [])
    screened = await asyncio.gather(*(ascreen(fig) for fig in passed))      # every candidate's picture fetched (not saved) and measured BEFORE the model chooses
    readable = []
    for fig, height in zip(passed, screened):
        if height is False:
            fates.append({"id": fig["id"], "label": fig.get("label"), "fate": f"text too small to read on the page (under {sf.MIN_TEXT_POINTS:g} pt), dropped before the relevance test"})
        elif height is not None:
            readable.append(fig)
    trace["unreadable_before_relevance_test"] = sum(1 for h in screened if h is False)
    passed = readable
    if not passed:
        return []
    try:
        raw = await rubric(passed)
        chosen = sf.verify_rubric(raw, passed)
        listed = [item for item in raw if isinstance(item, dict)]
        trace["relevance_test"] = {"offered": len(passed), "model_listed": len(listed), "all_required_yes": sum(1 for i in listed if all(i.get(k) is True for k in sf.REQUIRED_KEYS)), "kept_after_quote_check": len(chosen)}
    except Exception:
        chosen = []
    trace["after_relevance_test"] = len(chosen)
    fates = trace.setdefault("figure_fates", [])
    if not chosen:
        return []
    downloaded = [f for f in await asyncio.gather(*(adownload(fig, folder) for fig in chosen)) if f]
    for fig in chosen:
        if not any(d["id"] == fig["id"] for d in downloaded):
            fates.append({"id": fig["id"], "label": fig.get("label"), "fate": "image could not be downloaded"})
    trace["too_small_to_read"] = sum(1 for f in downloaded if not f["readable"])
    for f in downloaded:
        if not f["readable"]:
            points = await asyncio.to_thread(lambda p=f["path"]: sf.text_height_points(Path(p).read_bytes()))
            fates.append({"id": f["id"], "label": f.get("label"), "fate": f"text too small to read on the page (under {sf.MIN_TEXT_POINTS:g} pt)", "text_points": points})
    downloaded = [f for f in downloaded if f["readable"]]
    per_paper, deduped, removed = {}, [], 0
    for fig in downloaded:
        mates = [k for k in deduped if k.get("pmid") == fig.get("pmid")]
        twin = next((k for k in mates if sf.caption_similarity(fig["caption"], k["caption"]) >= sf.SAME_FIGURE), None)
        if twin is not None:
            removed += 1
            fates.append({"id": fig["id"], "label": fig.get("label"), "fate": f"repeat of {twin.get('label')}: the caption is nearly the same"})
            continue
        per_paper[fig.get("pmid")] = per_paper.get(fig.get("pmid"), 0) + 1
        if per_paper[fig.get("pmid")] <= sf.MAX_PER_PAPER:
            deduped.append(fig)
    trace["repeats_of_one_paper_removed"] = removed
    trace["extra_figures_of_one_paper_removed"] = len(downloaded) - len(deduped) - removed
    downloaded = deduped
    if inspect is None:
        trace["image_check"] = {"mode": "off", "downloaded": len(downloaded), "kept": len(downloaded[:limit])}
        return downloaded[:limit]
    kept, could_not, looked_at = [], 0, 0
    for fig in downloaded:      # one figure at a time, in ranked order (a vision provider's token limit would refuse a burst)
        if len(kept) >= limit or looked_at >= sf.MAX_CHECKS:
            break
        looked_at += 1
        reading = await inspect(fig)
        if reading is None:
            could_not += 1
            fates.append({"id": fig["id"], "label": fig.get("label"), "fate": "the vision check could not run"})
            continue
        keep, shows = sf.accept_image_check(reading)
        if keep:
            kept.append({**fig, "image_check": shows})
        else:
            fates.append({"id": fig["id"], "label": fig.get("label"), "fate": "the vision model rejected the image", "vision_said": shows})
    trace["image_check"] = {"downloaded": len(downloaded), "looked_at": looked_at, "could_not_check": could_not, "kept": len(kept)}
    return kept[:limit]


async def afind_figures(question: str, papers: dict, report: str, found: list, trace: dict, condition: str = "") -> list:
    """Async twin of qa_stream.find_figures: the gates, the population check, the relevance test, the vision check and the two plain-language lines. Any failure only means fewer figures."""
    cut = sf.cut_for_model
    direct = {pmid for pmid, p in papers.items() if (p.get("triage") or {}).get("relevance") == "direct"}
    indirect = {pmid for pmid, p in papers.items() if (p.get("triage") or {}).get("relevance") in ("indirect", "uncertain", "background")}

    async def rubric(available: list) -> list:
        ordered = sorted(available, key=lambda f: (f.get("relation") != "direct", available.index(f)))
        chunks = [ordered[k:k + 10] for k in range(0, min(len(ordered), 40), 10)]

        async def judge(chunk: list) -> list:
            payload = {"question": question, "written_report": report, "figures": [{"id": f["id"], "paper": (f.get("title") or "")[:140], "design": f.get("design") or "not stated",
                                                                                      "relation": f.get("relation") or "direct", "caption": cut(f["caption"], 600)} for f in chunk]}
            try:
                return (await async_core.aask_json(question, qs.with_rules(qs.FIGURE_TASK, 1), payload, 900, skill="evidence-synthesis")).get("figures") or []
            except Exception:
                return []
        return [item for part in await asyncio.gather(*(judge(c) for c in chunks)) for item in part]

    async def population(available: list) -> list:
        related = [{**f, "population_note": "related condition or population: " + (((papers.get(str(f["pmid"])) or {}).get("triage") or {}).get("population") or "see the paper")}
                   for f in available if f.get("relation") == "indirect"]
        available = [f for f in available if f.get("relation") != "indirect"]
        if not available:
            return related
        texts, views = {}, []
        for pmid in sorted({str(f["pmid"]) for f in available}):
            p = papers.get(pmid) or {}
            note = (p.get("triage") or {}).get("population") or ""
            texts[pmid] = " ".join(x for x in (p.get("title"), p.get("abstract"), note) if x)
            views.append({"pmid": pmid, "title": p.get("title"), "abstract": cut(p.get("abstract") or "", 1500), "population_noted_by_relevance_step": note or None})
        answer = await async_core.aask_json(question, qs.with_rules(qs.POPULATION_TASK, 2), {"question": question, "papers": views}, 700, skill="evidence-synthesis")
        return sf.apply_population_check(available, answer, texts, question, trace, condition=condition) + related
    figures: list = []
    try:
        check = (lambda fig: avision_inspect(fig, question)) if os.getenv("FIGURE_IMAGE_CHECK", "off").strip().lower() == "on" else None
        figures = await aselect_figures(found, set(), direct, rubric, check, qs.FIGURE_FOLDER, trace=trace, population=population, indirect_pmids=indirect)
    except Exception as error:
        trace["error"] = f"{type(error).__name__}: {str(error)[:80]}"
    if figures:
        try:      # a second, focused judgement: does each figure itself present the kind of information the question asks for?
            payload = {"question": question, "figures": [{"id": f["id"], "caption": cut(f["caption"], 500), "paper_title": f.get("title"), "image_description": f.get("image_check") or ""} for f in figures]}
            answer = {}
            for attempt in range(2):
                try:
                    answer = await async_core.aask_json(question, qs.with_rules(qs.STRICT_TASK, 3), payload, 700, skill="evidence-synthesis")
                except Exception:
                    answer = {}
                if answer.get("figures"):
                    break
            before = list(figures)
            figures = sf.apply_strict_check(figures, answer, trace)
            for f in before:
                if not any(k["id"] == f["id"] for k in figures):
                    trace.setdefault("figure_fates", []).append({"id": f["id"], "label": f.get("label"), "fate": "dropped by the strict check: it does not itself present what the question asks"})
        except Exception as error:
            trace["strict_check_error"] = f"{type(error).__name__}: {str(error)[:80]}"
    if figures:
        try:      # two plain-language lines per figure, each kept only if it can be verified against the figure's caption and the paper's own text
            texts = {str(f["pmid"]): " ".join(x for x in ((papers.get(str(f["pmid"])) or {}).get("title"), (papers.get(str(f["pmid"])) or {}).get("abstract")) if x) for f in figures}

            async def one(f: dict):      # one figure per call, side by side: a single reply covering several figures can be cut off at the model's output limit
                payload = {"question": question, "figures": [{"id": f["id"], "caption": cut(f["caption"], 900), "paper_title": f.get("title"),
                                                             "paper_abstract": cut(texts[str(f["pmid"])], 1800), "image_description": f.get("image_check") or ""}]}
                for attempt in range(2):
                    try:
                        got = (await async_core.aask_json(question, qs.with_rules(qs.EXPLAIN_TASK, 5), payload, 1300, skill="evidence-synthesis")).get("figures") or []
                    except Exception:
                        got = []
                    if got:
                        return got
                return []
            answers = [item for part in await asyncio.gather(*(one(f) for f in figures)) for item in part]
            figures = sf.explain_figures(figures, {"figures": answers}, texts, trace)
        except Exception as error:
            trace["explanations_error"] = f"{type(error).__name__}: {str(error)[:80]}"
    return figures
