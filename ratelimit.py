"""One shared limit per data source: legacy and routed searches run together and must not exceed what each source allows (a 429 is a failed source)."""
import os
import threading
import time

# requests per second to each source; PubMed's own limits are 3 without an API key and 10 with one (NCBI), and the defaults stay under them. The others are conservative defaults that RATE_<SOURCE>_PER_S can change.
DEFAULT_PER_SECOND = {"europepmc": 8.0, "clinicaltrials": 8.0, "fda": 4.0, "tavily": 2.0}


def default_rate(source: str) -> float:
    if source == "pubmed":      # read when used: the key may be loaded from .env after this module is imported
        return 8.0 if os.getenv("NCBI_API_KEY") else 2.5      # a margin under NCBI's 10 (with a key) and 3 (without) requests per second: requests that were spaced when sent can still arrive together
    return DEFAULT_PER_SECOND.get(source, 5.0)


def per_second(source: str) -> float:
    try:
        value = float(os.getenv(f"RATE_{source.upper()}_PER_S", default_rate(source)))
    except ValueError:
        value = default_rate(source)
    return max(0.1, value)


class SourceLimiter:
    """Spaces the calls to each source at least 1/rate seconds apart, across all threads (a slot is reserved under the lock, the wait happens outside it)."""

    def __init__(self, clock=time.monotonic, sleep=time.sleep):
        self._lock = threading.Lock()
        self._next: dict[str, float] = {}
        self._clock, self._sleep = clock, sleep

    def acquire(self, source: str) -> float:
        """Block until this call may go; returns the seconds waited."""
        gap = 1.0 / per_second(source)
        with self._lock:
            now = self._clock()
            slot = max(now, self._next.get(source, 0.0))
            self._next[source] = slot + gap
        wait = slot - now
        if wait > 0:
            self._sleep(wait)
        return max(0.0, wait)


limiter = SourceLimiter()


def source_of_url(url: str) -> str:
    if "eutils.ncbi.nlm.nih.gov" in url:
        return "pubmed"
    if "europepmc" in url or "ebi.ac.uk" in url:
        return "europepmc"
    if "clinicaltrials.gov" in url:
        return "clinicaltrials"
    if "api.fda.gov" in url:
        return "fda"
    return "other"
