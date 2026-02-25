"""
Flat (exact brute-force) index using FAISS.

Properties:
  - Recall: always 1.0 (exact search)
  - Latency: O(n * d) — linear scan, slowest for large datasets
  - No tunable search-time parameters
  - Serves as ground-truth baseline and upper-bound on recall
"""

import time
import numpy as np
import faiss

from .base_index import BaseIndex


class FlatIndex(BaseIndex):
    """FAISS IndexFlat — exact nearest neighbor search."""

    def __init__(self, dimension: int, metric: str = "L2"):
        """
        Args:
            dimension: vector dimensionality
            metric: "L2" for Euclidean, "IP" for inner product
        """
        super().__init__(name="Flat", dimension=dimension)
        self.metric = metric
        metric_type = faiss.METRIC_L2 if metric == "L2" else faiss.METRIC_INNER_PRODUCT
        self.index = faiss.IndexFlat(dimension, metric_type)

    def build(self, data: np.ndarray) -> None:
        """Add vectors to the index (no training needed)."""
        data = np.ascontiguousarray(data, dtype=np.float32)
        assert data.shape[1] == self.dimension, \
            f"Data dim {data.shape[1]} != index dim {self.dimension}"

        start = time.perf_counter()
        self.index.add(data)
        self._build_time_s = time.perf_counter() - start

        self.is_built = True
        print(f"[FlatIndex] Built: {self.index.ntotal:,} vectors, "
              f"{self._build_time_s:.2f}s")

    def search(self, queries: np.ndarray, k: int, **params):
        """Exact search — no tunable parameters."""
        assert self.is_built, "Index not built yet"
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        distances, indices = self.index.search(queries, k)
        return distances, indices

    def get_config(self):
        return {
            "type": "Flat",
            "metric": self.metric,
            "dimension": self.dimension,
            "ntotal": self.index.ntotal,
            "tunable_params": {},
        }
