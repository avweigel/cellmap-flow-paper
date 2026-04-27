from __future__ import annotations

import statistics
import time
from contextlib import contextmanager


class Timer:
    """Records timings in milliseconds. Use either as a context manager (single
    measurement) or by calling .start()/.stop() repeatedly to accumulate samples."""

    def __init__(self, label: str = ""):
        self.label = label
        self.samples_ms: list[float] = []
        self._start: float | None = None

    @contextmanager
    def measure(self):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.samples_ms.append((time.perf_counter() - t0) * 1000.0)

    def start(self) -> None:
        self._start = time.perf_counter()

    def stop(self) -> float:
        if self._start is None:
            raise RuntimeError("Timer.stop() called without matching start()")
        elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        self.samples_ms.append(elapsed_ms)
        self._start = None
        return elapsed_ms


def summarize(samples_ms: list[float]) -> dict:
    """Median / p95 / p99 / mean / count summary. Empty input returns zeros."""
    if not samples_ms:
        return {"count": 0, "median_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0}
    sorted_samples = sorted(samples_ms)
    n = len(sorted_samples)
    return {
        "count": n,
        "median_ms": statistics.median(sorted_samples),
        "p95_ms": sorted_samples[min(int(n * 0.95), n - 1)],
        "p99_ms": sorted_samples[min(int(n * 0.99), n - 1)],
        "mean_ms": statistics.mean(sorted_samples),
    }
