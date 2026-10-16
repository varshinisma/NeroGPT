"""Original figures from published open-access papers, for ANY question, shown only when they help.

Nothing here knows a disease or a drug, and nothing here draws, redraws or edits a figure. The workflow:
1. (background, beside the writing) the PubMed Central Open Access data service (a public bucket of open-access articles, the sanctioned bulk route, not the web pages) is read for each retrieved
   open-access paper: the article XML gives each figure's original caption and the licence, the bucket holds the original image files;
2. (after the report and its references are final) only figures that pass EVERY gate remain:
   - the paper was judged directly relevant to the question, or indirectly relevant (a closely related condition or population: such a figure is labelled as indirect evidence). Any RETRIEVED paper
     can supply a figure, not only the cited ones; a figure from a paper the report does not cite is labelled as such,
   - the page states an explicit open licence, and the caption does not say the figure is reproduced or adapted from another source or used with permission,
   - when the question names a patient population (an age group, sex, severity, setting ...), the paper's OWN population, read from its title, abstract and the relevance step's note, must match it:
     a clear mismatch is rejected, and a population that cannot be established from the paper's text (a verbatim quote is required) is skipped,
   - the text model, shown the question, the written evidence and the caption, answers four yes/no questions (helps answer, clinically meaningful, adds to the written report, not decorative)
     and quotes the caption words that justify it (checked here against the real caption),
   - the image must be readable at the size the PDF shows it: the height of the text lines in the picture, measured from its own pixels (no model), must be at least about 6.5 points on the page;
     a very dense figure is skipped, and at most three figures of one paper are kept (the relevance test is told to judge each on what it adds, so repeats of the same result are not chosen),
   - when the image check is ON (FIGURE_IMAGE_CHECK=on), a vision model looks at the downloaded ORIGINAL image and confirms it matches the caption, bears on the question and is legible; a failed,
     refused or rate-limited check means the figure is skipped. When it is OFF (the default until a vision model is available), the figure is chosen from its paper and caption only and is shown
     with a note that the image itself was not machine-checked;
3. the original image bytes are saved unchanged (with a checksum), together with the paper's own attribution details. Missing details stay missing.
If no figure passes, there is no figure section at all."""
from __future__ import annotations

import hashlib
import html
import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import httpx

from tracing import traced
from clinical_tools import OA_BUCKET, OPEN_LICENCE, PMC_PAGE, candidates, open_licence, parse_figures, search_paper_figures      # the retrieval tools: the figures' captions, licences and image files come from clinical_tools

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36 NeuroGPT-research"}
MAX_BYTES = 4_000_000
EXTENSIONS = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "image/gif": ".gif"}
RUBRIC_KEYS = ("helps_answer", "clinically_meaningful", "adds_beyond_report", "not_decorative", "answers_what_is_asked")
# a caption that says the figure comes from somewhere else, or is used with permission: the article's licence may not cover it
REPRODUCED = re.compile(r"\b(?:reproduced|reprinted|redrawn|adapted|modified|taken|used)\b[^.]{0,80}\b(?:from|with permission|by permission|courtesy)\b|with (?:kind )?permission|courtesy of|©|copyright", re.I)
_http = httpx.Client(timeout=12, follow_redirects=True, headers=HEADERS)


def _text(fragment: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", fragment)).split())


def _words(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def cut_for_model(text: str, limit: int) -> str:
    """Text cut at the end of a sentence (or a word) within `limit` characters. Only used for what is SENT to the model, never for the caption that is shown."""
    if len(text) <= limit:
        return text
    cut = text.rfind(". ", 0, limit)
    return text[:cut + 1] if cut > limit // 2 else text[:limit].rsplit(" ", 1)[0] + "..."


def eligible(figures: list[dict], cited_pmids: set, direct_pmids: set, trace: dict | None = None, indirect_pmids: set | None = None, require_cited: bool = False) -> list[dict]:
    """The gates that need no model: the paper was judged directly (or indirectly) relevant to the question; the page states an explicit open licence; the caption does not say the figure is
    reproduced or adapted from another source or used with permission. With `require_cited` the paper must also be cited in the final report. Each kept figure records whether its paper is cited.
    `trace` counts what each gate removed."""
    counts = {"paper_not_cited": 0, "paper_not_direct": 0, "no_explicit_open_licence": 0, "reproduced_or_permission": 0}
    kept = []
    for f in figures:
        if require_cited and f.get("pmid") not in cited_pmids:
            counts["paper_not_cited"] += 1
        elif f.get("pmid") not in direct_pmids and f.get("pmid") not in (indirect_pmids or set()):
            counts["paper_not_direct"] += 1
        elif not f.get("licence"):
            counts["no_explicit_open_licence"] += 1
        elif REPRODUCED.search(f.get("caption") or ""):
            counts["reproduced_or_permission"] += 1
        else:
            kept.append({**f, "relation": "direct" if f.get("pmid") in direct_pmids else "indirect", "cited": f.get("pmid") in cited_pmids})
    if trace is not None:
        trace["removed_by_gate"] = counts
    return kept


def quote_in_caption(quote: str, caption: str, min_run: int = 4, coverage: float = 0.5) -> bool:
    """The model's justification must rest on words that are really in the text it is checked against (a caption, a paper's abstract, the question): the longest run of consecutive words of the
    justification that also appears, in the same order, in that text must be at least `min_run` words long and cover at least `coverage` (half, unless said otherwise) of the justification."""
    q, c = _words(quote).split(), _words(caption).split()
    best = 0
    previous = [0] * (len(c) + 1)
    for qi in range(1, len(q) + 1):
        current = [0] * (len(c) + 1)
        for ci in range(1, len(c) + 1):
            if q[qi - 1] == c[ci - 1]:
                current[ci] = previous[ci - 1] + 1
                best = max(best, current[ci])
        previous = current
    return best >= min_run and best >= coverage * len(q)


NOT_SPECIFIED = {"", "not specified", "none", "unspecified", "not stated", "n/a", "no population specified"}
POPULATION_VERDICTS = ("match", "broader", "mismatch", "unclear")


def apply_population_check(figures: list[dict], answer, texts: dict, question: str, trace: dict | None = None, condition: str = "") -> list[dict]:
    """The model's reading of populations -> the figures whose paper's own population fits the question.
    - The population the question names must be words really in the question; if the model names one that is not, or none, the question specifies no population and nothing is rejected.
    - For each paper the model gives a verdict and a quote taken from the paper's own text (title, abstract, relevance note); the quote is checked against that text.
    - 'match' needs a verified quote. 'broader' (the paper studied a wider group that includes the question's population without isolating it) also needs a verified quote and is kept with a visible
      caveat that the population is broader than the question. 'mismatch' rejects. 'unclear', a missing paper, or a quote that is not in the paper's text rejects too: when the population cannot be established, the figure is
      skipped, never assumed applicable."""
    trace = trace if trace is not None else {}
    if not isinstance(answer, dict) or "question_population" not in answer:      # no usable reading at all: the population cannot be established, nothing is assumed applicable
        trace["population"] = {"answer_unusable": True}
        return []
    named = " ".join(str(answer.get("question_population") or "").split())
    words = _words(named).split()
    specified = named.lower() not in NOT_SPECIFIED and bool(words) and quote_in_caption(named, question, min_run=len(words), coverage=1.0)      # every word of the named population must be in the question, in order
    if specified and condition and not (content_words(named) - content_words(condition)):
        specified = False      # the model named the disease itself as the population: a disease is not a population, nothing is rejected on population
    trace["population"] = {"question_names_a_population": specified, "question_population": named if specified else None, "mismatch": 0, "not_established": 0}
    if not specified:
        return figures
    verdicts = {str(item.get("pmid")): item for item in (answer.get("papers") or []) if isinstance(item, dict)}
    kept = []
    for f in figures:
        item = verdicts.get(str(f.get("pmid")))
        verdict = (item or {}).get("verdict")
        if verdict == "mismatch":
            trace["population"]["mismatch"] += 1
        elif verdict in ("match", "broader") and quote_in_caption(str(item.get("evidence_quote", "")), texts.get(str(f.get("pmid")), ""), min_run=3):
            studied = " ".join(str(item.get("study_population", "")).split())[:160]
            kept.append({**f, "population_note": (f"broader than the question: {studied}" if verdict == "broader" else studied) or None})
        else:
            trace["population"]["not_established"] += 1
    return kept


def reason_grounded(reason: str, caption: str) -> bool:
    """The relevance test's justification must rest on the caption: either a verbatim run of three caption words, or a paraphrase in which at least three content words are in the caption and at least
    60% of the justification's content words are. A justification with words the caption does not contain (an invented finding) fails."""
    if quote_in_caption(reason, caption, min_run=3, coverage=0.35):
        return True
    mine, theirs = content_words(reason), content_words(caption)
    return len(mine & theirs) >= 3 and len(mine & theirs) >= 0.6 * len(mine)


def verify_rubric(items, available: list[dict], limit: int = 6) -> list[dict]:
    """The text model's four yes/no answers per figure -> the figures that pass all four, each with a justification quoted from its real caption. A figure the model did not mention, answered
    'no' to, or justified with words that are not in the caption is dropped. Order = the model's order (most useful first)."""
    by_id = {f["id"]: f for f in available}
    chosen = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        fig = by_id.get(str(item.get("id", "")).strip())
        if fig is None or any(fig["id"] == c["id"] for c in chosen) or not all(item.get(k) is True for k in RUBRIC_KEYS) or not reason_grounded(str(item.get("reason", "")), fig["caption"]):
            continue
        chosen.append({**fig, "reason": " ".join(str(item["reason"]).split())})
        if len(chosen) >= limit:
            break
    return chosen


NUMBER = re.compile(r"\d+(?:\.\d+)?")
OVERCLAIM = re.compile(r"\b(?:proves?|proven|confirms?|works?|should be (?:used|given|prescribed)|is recommended|are recommended|we recommend|cures?)\b", re.I)


def numbers_in(text: str) -> set[str]:
    """Every number written in a text (thousands separators removed, signs ignored)."""
    return set(NUMBER.findall((text or "").replace(",", "")))


def explain_figures(figures: list[dict], answer, texts: dict, trace: dict | None = None) -> list[dict]:
    """Adds two plain-language lines to each figure, each kept ONLY when it can be verified; a line that cannot be verified is left out (the figure itself stays):
    - how_to_read: how to read the figure, which must rest on what the figure's own caption says, or on what the vision model saw in the picture (a verbatim run of at least four words is required),
    - shows_for_question: what the figure shows for the doctor's question, which must rest on the paper's own words (a verbatim run of at least four words of the caption, title or abstract is
      required), every number in it must be a number written in that caption, title or abstract, and it must not say that a treatment works or should be used."""
    by_id = {str(item.get("id")): item for item in ((answer or {}).get("figures") or [])} if isinstance(answer, dict) else {}
    counts = {"how_to_read_kept": 0, "shows_kept": 0, "dropped": 0}
    out = []
    for f in figures:
        item = by_id.get(f["id"]) or {}
        caption = f.get("caption") or ""
        paper = f"{caption} {texts.get(str(f.get('pmid')), '')}"
        how = " ".join(str(item.get("how_to_read") or "").split())
        shows = " ".join(str(item.get("shows_for_question") or "").split())
        seen = f"{caption} {f.get('image_check') or ''}"      # the caption, and what the vision model saw in the picture itself (axes, legend, notes): the words of the "how to read" line may rest on either
        how_ok = bool(how) and quote_in_caption(str(item.get("how_to_read_quote", "")), seen, min_run=4) and numbers_in(how) <= numbers_in(seen) and not OVERCLAIM.search(how)
        shows_ok = (bool(shows) and quote_in_caption(str(item.get("shows_quote", "")), paper, min_run=3, coverage=0.35) and numbers_in(shows) <= numbers_in(paper) and not OVERCLAIM.search(shows))
        helps = " ".join(str(item.get("helps_treatment") or "").split())
        helps_ok = (bool(helps) and quote_in_caption(str(item.get("helps_quote", "")), paper, min_run=3, coverage=0.35) and numbers_in(helps) <= numbers_in(paper) and not OVERCLAIM.search(helps))
        counts["how_to_read_kept"] += how_ok
        counts["shows_kept"] += shows_ok
        counts["helps_kept"] = counts.get("helps_kept", 0) + helps_ok
        counts["dropped"] += (not how_ok) + (not shows_ok) + (not helps_ok)
        out.append({**f, "how_to_read": how if how_ok else None, "shows_for_question": shows if shows_ok else None, "helps_treatment": helps if helps_ok else None})
    if trace is not None:
        trace["explanations"] = counts
    return out


def accept_image_check(raw) -> tuple[bool, str]:
    """The vision model's reading of the ORIGINAL image -> (keep, what it shows). Kept only when it says the image matches the caption, bears on the question and is legible; any other reply
    (not that JSON object, a refusal, a missing key) keeps nothing."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
        except Exception:
            return False, ""
    if not isinstance(raw, dict):
        return False, ""
    shows = " ".join(str(raw.get("shows") or "").split())[:220]
    keep = raw.get("matches_caption") is True and raw.get("relevant_to_question") is True and raw.get("legible") is True and bool(shows)
    return keep, shows


MIN_TEXT_POINTS = 6.5
PAGE_TEXT_WIDTH_PTS = 495.0      # the picture width inside its frame in the PDF (text width minus the frame's padding)
PAGE_MAX_HEIGHT_PTS = 5.6 * 72


def text_height_points(image_bytes: bytes, width_pts: float = PAGE_TEXT_WIDTH_PTS, max_height_pts: float = PAGE_MAX_HEIGHT_PTS) -> float | None:
    """How tall the text lines of a picture would be on the PDF page, in points, measured from its own pixels (no model): rows of dark ink separated by blank rows are the lines of text; rules, ticks and
    gridlines (rows thinner than 3 pixels) and long vertical lines (columns that are mostly ink) are ignored. None when it cannot be measured (fewer than 3 text rows, or lines taller than 60 points, as in a picture made of coloured blocks)."""
    import statistics
    from PIL import Image
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
    except Exception:
        return None
    w, h = image.size
    scale = min(width_pts / w, max_height_pts / h, 1.6)      # points per pixel, as the PDF draws it
    pixels = image.load()
    column_ink = [sum(1 for y in range(0, h, 2) if pixels[x, y] < 140) for x in range(w)]
    lines = {x for x in range(w) if column_ink[x] > 0.45 * (h / 2)}      # columns that are mostly ink: long vertical (dashed) lines, frames and bars are not text
    xs = [x for x in range(0, w, 2) if x not in lines]
    dark = [sum(1 for x in xs if pixels[x, y] < 140) >= 2 for y in range(h)]
    runs, run = [], 0
    for d in dark + [False]:
        if d:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    runs = [r for r in runs if r >= 3]
    if len(runs) < 3:
        return None
    median = statistics.median(runs) * scale
    return None if median > 60 else median


def readable_on_the_page(image_bytes: bytes) -> bool:
    """False only when the text in the picture would be smaller than MIN_TEXT_POINTS on the page. A picture whose text cannot be measured is kept."""
    height = text_height_points(image_bytes)
    return height is None or height >= MIN_TEXT_POINTS


def download(figure: dict, folder: Path, get=None) -> dict | None:
    """The ORIGINAL image bytes saved unchanged (name from the figure id, extension from the server's content type) with their SHA-256. A copy for the PDF library is made only when the original
    format is one it cannot read (lossless PNG, same pixels). None when the image cannot be fetched, is not an image, or is too large."""
    try:
        response = (get or _http.get)(figure["image"])
        if response.status_code != 200 or len(response.content) > MAX_BYTES:
            return None
        from PIL import Image
        image = Image.open(io.BytesIO(response.content))      # the type comes from the file itself (the bucket labels every file binary/octet-stream)
        content_type = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp", "GIF": "image/gif"}.get(image.format or "")
        if content_type not in EXTENSIONS:
            return None
        width, height = image.size
        folder.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"[^A-Za-z0-9]+", "_", figure["id"])
        original = folder / (stem + EXTENSIONS[content_type])
        original.write_bytes(response.content)
        pdf_copy = original
        if EXTENSIONS[content_type] not in (".jpg", ".png"):
            pdf_copy = folder / (stem + "_pdf.png")
            image.convert("RGBA" if image.mode in ("RGBA", "LA", "P") else "RGB").save(pdf_copy, "PNG")
        return {**figure, "path": str(original), "pdf_path": str(pdf_copy), "sha256": hashlib.sha256(response.content).hexdigest(), "content_type": content_type,
                "width": width, "height": height, "readable": readable_on_the_page(response.content)}
    except Exception:
        return None


MAX_FIGURES = 6
MAX_PER_PAPER = 3
MAX_CHECKS = 6      # at most this many figures are shown to the vision model per question
SAME_FIGURE = 0.7      # two figures of one paper whose captions share at least this fraction of their words show the same thing again (for example the same plot for a subset of the trials)


def caption_similarity(a: str, b: str) -> float:
    """Share of words two captions have in common (words of three letters or more): 1.0 = the same words, 0 = nothing in common."""
    ta = {w for w in re.findall(r"[a-z0-9]+", (a or "").lower()) if len(w) >= 3}
    tb = {w for w in re.findall(r"[a-z0-9]+", (b or "").lower()) if len(w) >= 3}
    return len(ta & tb) / max(len(ta | tb), 1)


def _screen(figure: dict, get=None):
    """Is the picture of this candidate readable on the page? True/False from the pixels of the downloaded image (nothing is saved); None when it cannot be fetched or is not an image."""
    try:
        response = (get or _http.get)(figure["image"])
        if response.status_code != 200 or len(response.content) > MAX_BYTES:
            return None
        return readable_on_the_page(response.content)
    except Exception:
        return None


@traced("select_figures", "chain")
def select_figures(found: list[dict], cited_pmids: set, direct_pmids: set, rubric, inspect, folder: Path, limit: int = MAX_FIGURES, trace: dict | None = None, get=None, population=None,
                   indirect_pmids: set | None = None, require_cited: bool = False) -> list[dict]:
    """The whole selection, after the report is final. `rubric(eligible)` returns the text model's answers; `inspect(figure)` returns the vision model's raw reply for the saved original image, or
    None when it could not be reached. Every failure only removes figures; an empty list means the report has no figure section."""
    trace = trace if trace is not None else {}
    trace["candidates"] = len(found)
    passed = eligible(found, cited_pmids, direct_pmids, trace, indirect_pmids, require_cited)
    trace["after_gates"] = len(passed)
    if passed and population is not None:
        try:
            passed = population(passed)      # the paper's own population must fit the population the question names (apply_population_check)
        except Exception:
            passed = []      # the check could not run: nothing is assumed applicable
            trace["population_check_failed"] = True
        trace["after_population_check"] = len(passed)
    if not passed:
        return []
    fates = trace.setdefault("figure_fates", [])      # what happened to every figure that was dropped, with the reason
    with ThreadPoolExecutor(max_workers=8) as screen_pool:      # the picture of every candidate is fetched (not saved) and its text size measured BEFORE the model chooses: it should choose among figures a doctor can read
        screened = list(screen_pool.map(lambda fig: _screen(fig, get), passed))
    readable = []
    for fig, height in zip(passed, screened):
        if height is False:
            fates.append({"id": fig["id"], "label": fig.get("label"), "fate": "text too small to read on the page (under 6.5 pt), dropped before the relevance test"})
        elif height is not None:
            readable.append(fig)      # None: the image could not be fetched; it is dropped, never assumed readable
    trace["unreadable_before_relevance_test"] = sum(1 for h in screened if h is False)
    passed = readable
    if not passed:
        return []
    try:
        raw = rubric(passed)
        chosen = verify_rubric(raw, passed)
        listed = [item for item in raw if isinstance(item, dict)]
        trace["relevance_test"] = {"offered": len(passed), "model_listed": len(listed), "all_four_yes": sum(1 for i in listed if all(i.get(k) is True for k in RUBRIC_KEYS)), "kept_after_quote_check": len(chosen)}
    except Exception:
        chosen = []
    trace["after_relevance_test"] = len(chosen)
    fates = trace.setdefault("figure_fates", [])      # what happened to every figure the relevance test chose: the reason it was dropped, or that it was kept
    if not chosen:
        return []
    with ThreadPoolExecutor(max_workers=len(chosen)) as pool:
        downloaded = [f for f in pool.map(lambda fig: download(fig, folder, get), chosen) if f]
    for fig in chosen:
        if not any(d["id"] == fig["id"] for d in downloaded):
            fates.append({"id": fig["id"], "label": fig.get("label"), "fate": "image could not be downloaded"})
    trace["too_small_to_read"] = sum(1 for f in downloaded if not f["readable"])
    for f in downloaded:
        if not f["readable"]:
            fates.append({"id": f["id"], "label": f.get("label"), "fate": "text too small to read on the page (under 6.5 pt)", "text_points": text_height_points(Path(f["path"]).read_bytes())})
    downloaded = [f for f in downloaded if f["readable"]]      # text smaller than ~6.5 pt on the page cannot be read by the doctor: the figure is skipped
    per_paper, deduped, removed = {}, [], 0
    for fig in downloaded:      # a paper can have several relevant figures, but not repeats of one another: a figure whose caption is nearly the same as one already kept from its paper is dropped
        mates = [k for k in deduped if k.get("pmid") == fig.get("pmid")]
        twin = next((k for k in mates if caption_similarity(fig["caption"], k["caption"]) >= SAME_FIGURE), None)
        if twin is not None:
            removed += 1
            fates.append({"id": fig["id"], "label": fig.get("label"), "fate": f"repeat of {twin.get('label')}: the caption is nearly the same"})
            continue
        per_paper[fig.get("pmid")] = per_paper.get(fig.get("pmid"), 0) + 1
        if per_paper[fig.get("pmid")] <= MAX_PER_PAPER:
            deduped.append(fig)
    trace["repeats_of_one_paper_removed"] = removed
    trace["extra_figures_of_one_paper_removed"] = len(downloaded) - len(deduped) - removed
    downloaded = deduped
    if inspect is None:      # the image check is off: the figure rests on its paper, licence and caption, and says so
        trace["image_check"] = {"mode": "off", "downloaded": len(downloaded), "kept": len(downloaded[:limit])}
        return downloaded[:limit]
    kept, could_not, looked_at = [], 0, 0
    for fig in downloaded:      # one figure at a time, in ranked order (a vision provider's token limit would refuse a burst); the loop stops once there are enough or MAX_CHECKS were made
        if len(kept) >= limit or looked_at >= MAX_CHECKS:
            break
        looked_at += 1
        reading = inspect(fig)
        if reading is None:
            could_not += 1      # the check could not run (refused, rate-limited, not reachable): the figure is skipped, never assumed correct
            fates.append({"id": fig["id"], "label": fig.get("label"), "fate": "the vision check could not run"})
            continue
        keep, shows = accept_image_check(reading)
        if keep:
            kept.append({**fig, "image_check": shows})
        else:
            fates.append({"id": fig["id"], "label": fig.get("label"), "fate": "the vision model rejected the image", "vision_said": shows})
    trace["image_check"] = {"downloaded": len(downloaded), "looked_at": looked_at, "could_not_check": could_not, "kept": len(kept)}
    return kept[:limit]


SECTION_PRIORITY = ("current evidence", "key studies", "latest findings", "conflicting evidence", "what remains under investigation")
NOT_A_HOME = ("clinical summary", "evidence limitations", "sources", "references", "treatment drug landscape", "clinical trials")


def section_for(markdown: str, number) -> str | None:
    """The heading of the report section where a figure belongs: the section that cites its paper (by reference number), the evidence sections first, otherwise the first other section that cites it.
    None when no section cites it."""
    if not markdown or not number:
        return None
    sections = {}
    for part in re.split(r"(?m)^(?=## )", markdown):
        if part.startswith("## "):
            sections[part.splitlines()[0][3:].strip()] = part
    cites = re.compile(r"\[(?:\d+\s*[,\u2013-]\s*)*" + re.escape(str(number)) + r"(?:\s*[,\u2013-]\s*\d+)*\]")
    by_lower = {name.lower(): name for name in sections}
    for wanted in SECTION_PRIORITY:
        if wanted in by_lower and cites.search(sections[by_lower[wanted]]):
            return by_lower[wanted]
    return next((name for name, text in sections.items() if name.lower() not in NOT_A_HOME and cites.search(text)), None)


HOMES = ("current evidence", "treatment drug landscape", "key studies", "latest findings", "conflicting evidence", "what remains under investigation")
STOPWORDS = frozenset("""about above after again against also among and any are because been before being between both but can could did does doing down during each few for from further had has have
having here his how into its itself just more most not now off once only other our out over own same should some such than that the their them then there these they this those through under
until very was were what when where which while who whom why will with would you your study studies trial trials patient patients result results effect effects analysis versus using used use
based compared comparison figure plot graph forest summary outcome outcomes group groups week weeks month months day days over time trend""".split())


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", (text or "").lower()) if w not in STOPWORDS}


def best_section(markdown: str, figure_text: str, minimum: int = 2) -> str | None:
    """Where a figure belongs when no section cites its paper: the evidence section (or the drug section) whose own words overlap most with the words of the figure's paper title and caption; at
    least `minimum` shared content words are needed, otherwise None. Ties go to the earlier section of HOMES."""
    sections = {}
    for part in re.split(r"(?m)^(?=## )", markdown or ""):
        if part.startswith("## "):
            sections[part.splitlines()[0][3:].strip()] = part
    by_lower = {name.lower(): name for name in sections}
    words = content_words(figure_text)
    best, best_score = None, 0
    for home in HOMES:
        if home in by_lower:
            score = len(words & content_words(sections[by_lower[home]]))
            if score > best_score:
                best, best_score = by_lower[home], score
    return best if best_score >= minimum else None


def apply_strict_check(figures: list[dict], answer, trace: dict | None = None) -> list[dict]:
    """The second, focused judgement of each chosen figure: does the figure ITSELF present the kind of information the question asks for? A figure is kept only when the model says so for that
    figure id; a figure it does not mention, or answers false for, is dropped. When the model gave no usable answer at all, the check is skipped and the figures stay."""
    items = answer.get("figures") if isinstance(answer, dict) else None
    if not items:      # no usable answer (the call failed): this extra check could not run, and the figures that passed every other test are not thrown away because of it
        if trace is not None:
            trace["strict_check"] = {"asked_kind": None, "checked": len(figures), "kept": len(figures), "could_not_run": True}
        return list(figures)
    verdicts = {str(i.get("id")): i for i in items if isinstance(i, dict)}
    kept = []
    for f in figures:
        item = verdicts.get(f["id"])
        if item is not None and item.get("presents_it") is True:
            kept.append({**f, "presents": " ".join(str(item.get("what_it_presents") or "").split())[:140] or None})
    if trace is not None:
        trace["strict_check"] = {"asked_kind": (answer or {}).get("asked_kind") if isinstance(answer, dict) else None, "checked": len(figures), "kept": len(kept)}
    return kept


def why_chosen(f: dict) -> str:
    """Plain words for the reader on why this figure is shown: the reason the model gave (words from the caption), and every check the figure passed. Nothing is claimed that was not checked."""
    relation = ("the paper was judged directly relevant to the question" if f.get("relation") != "indirect"
                else "the paper studies a closely related condition or population (indirect evidence)")
    checks = [relation, "the article states an open licence", "the model's five-part relevance test was passed (helps answer, clinically meaningful, adds to the report, not decorative, "
              "shows the kind of information the question asks for)", "the picture is large enough to read on the page"]
    if f.get("population_note"):
        checks.append("its population fits the question" if not str(f["population_note"]).startswith(("broader", "related")) else f"population: {f['population_note']}")
    if f.get("image_check"):
        checks.append("a vision model confirmed that the image matches its caption and is readable")
    reason = f"The model's reason, from the caption: \u201c{f['reason']}\u201d. " if f.get("reason") else ""
    return reason + "Checks passed: " + "; ".join(checks) + "."


def display_records(figures: list[dict], references: dict | None, markdown: str = "") -> list[dict]:
    """What the web page AND the PDF show for each figure, built once from the saved record: nothing is invented, a missing detail is left out. `section` is where in the report the figure belongs."""
    number_of = {pmid: number for number, pmid in (references or {}).items()}
    records = []
    for f in figures:
        licence = f.get("licence") or {}
        records.append({"url": "demo_output/figures/" + Path(f["path"]).name, "pdf_path": f.get("pdf_path") or f["path"], "width": f.get("width"), "height": f.get("height"),
                        "label": f.get("label"), "caption": f.get("caption"), "title": f.get("title"), "design": f.get("design"), "cite": f.get("cite"), "pmid": f.get("pmid"),
                        "doi": f.get("doi"), "pmcid": f.get("pmcid"), "source_url": f.get("page_url"), "licence_label": licence.get("label"), "licence_url": licence.get("url"),
                        "reference": number_of.get(f.get("pmid")), "image_check": f.get("image_check"), "image_checked": bool(f.get("image_check")), "sha256": f.get("sha256"),
                        "population": f.get("population_note"), "relation": f.get("relation") or "direct", "cited": f.get("pmid") in number_of, "how_to_read": f.get("how_to_read"), "helps_treatment": f.get("helps_treatment"), "why": why_chosen(f), "image_description": f.get("image_check"), "shows_for_question": f.get("shows_for_question"), "section": section_for(markdown, number_of.get(f.get("pmid"))) or best_section(markdown, f"{f.get('title') or ''} {f.get('caption') or ''}")})
    return records
