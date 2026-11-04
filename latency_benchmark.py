"""Repeat questions through the pipeline to build the distributions (p50 / p95 / p99) a single run cannot give.

    python latency_benchmark.py questions.txt --repeat 5      # one question per line

Every run is recorded by observability.py like a real one, then `python latency_report.py` shows the percentiles. Use a questions file with a mix of new questions and focused ones."""
from __future__ import annotations

import argparse
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("questions", help="a text file with one question per line")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    questions = [q.strip() for q in open(args.questions, encoding="utf-8").read().splitlines() if q.strip() and not q.startswith("#")]
    import qa_stream
    for round_no in range(1, args.repeat + 1):
        for question in questions:
            began = time.time()
            final = None
            for event in qa_stream.stream_answer(question):
                if event.get("type") == "final":
                    final = event
            took = round(time.time() - began, 1)
            print(f"[{round_no}/{args.repeat}] {took:>6}s  {'ok' if final else 'no answer'}  {question[:80]}", flush=True)
    print("done: run `python latency_report.py` for the percentiles and the waterfall")


if __name__ == "__main__":
    sys.exit(main())
