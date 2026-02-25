"""
Latency profiler: measures per-query and batch latency for any index.
"""

import time
import numpy as np
from typing import Dict


class LatencyProfiler:
    """Measure search latency with high precision."""

    def __init__(self, warmup_queries: int = 50):
        self.warmup_queries = warmup_queries

    def profile_batch(self, index, queries: np.ndarray, k: int,
                      repeat: int = 3, **search_params) -> Dict:
        """
        Measure batch search latency (all queries at once).
        Repeats `repeat` times and takes the median.
        """
        # Warmup
        if self.warmup_queries > 0:
            n_warmup = min(self.warmup_queries, len(queries))
            index.search(queries[:n_warmup], k, **search_params)

        times = []
        for _ in range(repeat):
            start = time.perf_counter()
            index.search(queries, k, **search_params)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        total_s = np.median(times)
        nq = len(queries)
        return {
            "total_time_s": total_s,
            "mean_latency_ms": (total_s / nq) * 1000,
            "qps": nq / total_s,
            "num_queries": nq,
            "repeat": repeat,
        }

    def profile_per_query(self, index, queries: np.ndarray, k: int,
                          max_queries: int = 1000, **search_params) -> np.ndarray:
        """
        Measure individual query latencies.
        Returns array of latencies in milliseconds.
        """
        nq = min(len(queries), max_queries)

        # Warmup
        if self.warmup_queries > 0:
            n_warmup = min(self.warmup_queries, nq)
            index.search(queries[:n_warmup], k, **search_params)

        latencies = np.zeros(nq)
        for i in range(nq):
            q = queries[i:i+1]
            start = time.perf_counter()
            index.search(q, k, **search_params)
            elapsed = time.perf_counter() - start
            latencies[i] = elapsed * 1000  # ms

        return latencies
