# Async port of the NeuroGPT pipeline: status

Goal (from the mentor): the pipeline runs on asyncio, no threads in our own code. Work that cannot be awaited (BM25 + fusion, the local ONNX cross-encoder, the blocking DuckDuckGo library,
small file writes, CPU-only verification) goes through `asyncio.to_thread`. Every such call is marked `# BRIDGE` or listed below.

## How to switch it on
- `.env`: `ASYNC_PIPELINE=1` (default is off). Restart the backend. NEW questions AND follow-ups (focused runs, prior sources, safety / drug-facts tables, drug-only answers) use the asyncio pipeline; the old threaded `stream_answer` answers only if a run fails before
  its first event (`async_runner.stream_answer_auto`). To go back: remove the line (or set 0). The old code was not changed (api.py backup: `api.before_async.py`).

## Files (all new, nothing else changed except the three small edits named below)
- async_core.py: limiter, httpx client per loop, aget, astream_llm (Agno arun), acollect, aask_json, run_jobs_async
- async_tools.py / async_web.py / async_support.py: PubMed, Europe PMC, trials, FDA, web, guidelines, regulator, MeSH, embeddings, rerank
- async_routing.py / async_orchestrator.py: AsyncExecution, adecide / aplan_only / arecover
- async_pipeline.py: AsyncRun, fast_search, fast_answer (edit: stage_fast_search now starts `start_deep()` and `start_early_evidence()`; backup async_pipeline.before_wiring.py)
- async_deep.py / async_route.py / async_stages_deep.py: deep retrieval, routing, deep_worker/absorb, rank
- async_stages_evidence.py: read_evidence (full text, triage), escalation
- async_stages_drugs.py: drug stage (handoff, label lookups, 4 concurrent interpretation calls, doses, landscape), emit landscape + trials
- async_stages_writers.py: writers (latest, approach, current in two halves, tail) as tasks + queues; ready-order display
- async_stages_final.py: figures task, assemble, limitations, verify, finalize
- async_runner.py: AsyncPipeline (all mixins), astream_answer, heartbeat, `stream_answer_auto` (the synchronous view used by api.py)
- clinical_tools.py (backup clinical_tools.before_async.py): three shared parsers extracted; api.py: 8-line switch; async_route.py import fix.

## Bridges still using a worker thread (everything else awaits)
- qa_stream.plan_with_skill (only when the plan agent made no plan), qa_stream.fast_drug_decisions (fast answer's FDA/trial decisions)
- figures: search_paper_figures and find_figures (page parsing, image download/measure, vision) -- one thread call each, run BESIDE the writers
- verify_and_repair (CPU only, twice), ma.assess (off by default), BM25/fusion, cross-encoder, DuckDuckGo fallback
- Two adapters back to synchronous code: api.ask_events is still a synchronous generator (Starlette iterates it), so `stream_answer_async` hands events from one long-lived event loop
  (thread `neurogpt-async-loop`); `FigureJob` lets the synchronous PDF step wait for the figures task.

## Not verified (cannot be run from here)
Only syntax, imports (with stubs), name checks and fixture tests of pure twins and of the writers/heartbeat were run. No real network or model call was made. First real run: set ASYNC_PIPELINE=1,
ask one question, compare the `[latency]` log and the report with the threaded run.

## Next
1. Replace the figure bridge (async download + vision), plan_with_skill and fast_drug_decisions (use ainterpret_fda_labels already written in async_stages_drugs.py).
2. Make api.ask_events itself an async generator (adecide exists) so the loop thread adapter can go.
3. Start the drug handoff earlier (as soon as the first papers are ranked) -- the largest remaining latency win.

## Follow-ups (added last)
Same rules as the threaded run, written into the same async stages: focus sections, light_focus (no relevance reading / cross-encoder / full text), ordinary time limits, prior-source seeding with stored_cover,
no summary, focused drug landscape, side-effects and drug-facts tables (safety_answer is reused through `agrounded_call`), no heartbeats. Backups of every file changed: backup_before_followups/.
Not verified on the real services; a mocked run of the safety follow-up passed (fast answer, escalate, writers, assemble, limitations).
