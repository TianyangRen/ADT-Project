"""
IVF (Inverted File) index using FAISS.

Properties:
  - Partitions data into `nlist` Voronoi cells via k-means
  - At search time, only `nprobe` cells are scanned
  - Recall increases with nprobe; latency also increases
  - Tunable search-time parameter: nprobe
  - Good for: medium-to-large K, batch queries, controllable accuracy
"""

import time
import numpy as np
import faiss

from .base_index import BaseIndex


class IVFIndex(BaseIndex):
    """FAISS IndexIVFFlat — inverted file with flat quantizer."""

    def __init__(self, dimension: int, nlist: int = 256, metric: str = "L2"):
        """
        Args:
            dimension: vector dimensionality
            nlist: number of Voronoi cells (clusters)
            metric: "L2" for Euclidean, "IP" for inner product
        """
        super().__init__(name="IVF", dimension=dimension)
        self.metric = metric
        self.nlist = nlist
        metric_type = faiss.METRIC_L2 if metric == "L2" else faiss.METRIC_INNER_PRODUCT
        quantizer = faiss.IndexFlat(dimension, metric_type)
        self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, metric_type)

    def build(self, data: np.ndarray) -> None:
        """Train the quantizer and add vectors."""
        data = np.ascontiguousarray(data, dtype=np.float32)
        assert data.shape[1] == self.dimension, \
            f"Data dim {data.shape[1]} != index dim {self.dimension}"

        start = time.perf_counter()

        # Training requires at least nlist vectors
        print(f"[IVFIndex] Training k-means with nlist={self.nlist}...")
        self.index.train(data)

        print(f"[IVFIndex] Adding {data.shape[0]:,} vectors...")
        self.index.add(data)

        self._build_time_s = time.perf_counter() - start
        self.is_built = True
        print(f"[IVFIndex] Built: {self.index.ntotal:,} vectors, "
              f"nlist={self.nlist}, {self._build_time_s:.2f}s")

    def search(self, queries: np.ndarray, k: int, nprobe: int = 16, **params):
        """
        Search with configurable nprobe.

        Args:
            queries: (nq, d) query vectors
            k: number of neighbors
            nprobe: number of cells to probe (higher = better recall, slower)
        """
        assert self.is_built, "Index not built yet"
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        self.index.nprobe = nprobe
        distances, indices = self.index.search(queries, k)
        return distances, indices

    def get_config(self):
        return {
            "type": "IVF",
            "metric": self.metric,
            "dimension": self.dimension,
            "nlist": self.nlist,
            "ntotal": self.index.ntotal,
            "tunable_params": {
                "nprobe": {
                    "description": "Number of Voronoi cells to probe",
                    "range": [1, self.nlist],
                    "default": 16,
                }
            },
        }
