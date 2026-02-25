"""
HNSW (Hierarchical Navigable Small World) graph index using FAISS.

Properties:
  - Graph-based ANN index with logarithmic search complexity
  - Build-time params: M (connections per node), efConstruction
  - Search-time param: efSearch (higher = better recall, slower)
  - Generally offers best recall-latency trade-off for small-to-medium K
  - Memory-intensive: stores graph structure alongside vectors
  - Does NOT support vector removal (append-only)
"""

import time
import numpy as np
import faiss

from .base_index import BaseIndex


class HNSWIndex(BaseIndex):
    """FAISS IndexHNSWFlat — HNSW graph with flat vector storage."""

    def __init__(self, dimension: int, M: int = 32, ef_construction: int = 200,
                 metric: str = "L2"):
        """
        Args:
            dimension: vector dimensionality
            M: number of bi-directional links per node (graph connectivity)
            ef_construction: build-time search width (quality vs build speed)
            metric: "L2" for Euclidean (HNSW in FAISS only supports L2 well)
        """
        super().__init__(name="HNSW", dimension=dimension)
        self.metric = metric
        self.M = M
        self.ef_construction = ef_construction

        self.index = faiss.IndexHNSWFlat(dimension, M)
        self.index.hnsw.efConstruction = ef_construction

    def build(self, data: np.ndarray) -> None:
        """Build the HNSW graph (this can be slow for large datasets)."""
        data = np.ascontiguousarray(data, dtype=np.float32)
        assert data.shape[1] == self.dimension, \
            f"Data dim {data.shape[1]} != index dim {self.dimension}"

        print(f"[HNSWIndex] Building graph (M={self.M}, "
              f"efConstruction={self.ef_construction})...")
        print(f"  This may take several minutes for large datasets...")

        start = time.perf_counter()
        self.index.add(data)
        self._build_time_s = time.perf_counter() - start

        self.is_built = True
        print(f"[HNSWIndex] Built: {self.index.ntotal:,} vectors, "
              f"{self._build_time_s:.2f}s")

    def search(self, queries: np.ndarray, k: int, ef_search: int = 64, **params):
        """
        Search with configurable efSearch.

        Args:
            queries: (nq, d) query vectors
            k: number of neighbors
            ef_search: search-time quality parameter (must be >= k)
        """
        assert self.is_built, "Index not built yet"
        queries = np.ascontiguousarray(queries, dtype=np.float32)

        # efSearch must be >= k for valid results
        ef_search = max(ef_search, k)
        self.index.hnsw.efSearch = ef_search

        distances, indices = self.index.search(queries, k)
        return distances, indices

    def get_config(self):
        return {
            "type": "HNSW",
            "metric": self.metric,
            "dimension": self.dimension,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ntotal": self.index.ntotal,
            "tunable_params": {
                "ef_search": {
                    "description": "Search-time quality parameter",
                    "range": [1, 2048],
                    "default": 64,
                }
            },
        }
