"""The summary rewrite started early. When every evidence section is written and only the drug stage is still running, the rewrite of the opening summary (a model call that reads the finished sections) can
start at once instead of after the drug stage. It is used ONLY when the text it read is exactly the text the report ends with; otherwise it is cancelled and the rewrite is made as before, so the report is
the same either way and only the waiting differs."""
from __future__ import annotations

import asyncio


class OpeningSpeculation:
    def __init__(self) -> None:
        self.text: str | None = None
        self.task: asyncio.Future | None = None

    def started(self) -> bool:
        return self.task is not None

    def start(self, text: str, make):
        """`make(text)` returns the awaitable that rewrites the opening from `text`. Starts once; empty text starts nothing."""
        if self.task is not None or not str(text or "").strip():
            return None
        self.text = text
        self.task = asyncio.ensure_future(make(text))
        return self.task

    def cancel(self) -> None:
        if self.task is not None and not self.task.done():
            self.task.cancel()
        self.task = self.text = None

    async def result(self, final_text: str, make):
        """The rewrite for `final_text`: the early one when it read exactly this text, else a new one made with `make(final_text)`."""
        task, text = self.task, self.text
        self.task = self.text = None
        if task is not None and text == final_text:
            return await task
        if task is not None and not task.done():
            task.cancel()
        return await make(final_text)
