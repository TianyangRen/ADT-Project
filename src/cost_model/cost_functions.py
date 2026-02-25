"""
Analytical Cost Functions for ANN indexes.

Provides lightweight, interpretable cost estimates based on
index-specific complexity analysis. Useful as:
  1. Baseline cost model before profiling data is available
  2. Interpretable alternative to learned models
  3. Fallback when learned model is unreliable
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict

from src.cost_model.cost_estimator import CostEstimate


class AnalyticalCostModel:
    """
    Analytical cost model based on index complexity formulas.

    Each index has a latency function and a recall function
    parameterized by dataset size, dimension, K, and tunable param.
    """

    def __init__(self, dataset_size: int, dimension: int):
        self.n = dataset_size
        self.d = dimension
        self.is_trained = True  # Always ready, no training needed

    def estimate(self, index_name: str, params: dict,
                 query_features) -> CostEstimate:
        """Estimate cost using analytical formulas."""
        k = query_features.top_k

        if index_name == "Flat":
            lat = self._flat_latency(k)
            rec = self._flat_recall()
        elif index_name == "IVF":
            nprobe = params.get("nprobe", 16)
            lat = self._ivf_latency(k, nprobe)
            rec = self._ivf_recall(k, nprobe)
        elif index_name == "HNSW":
            ef_search = params.get("ef_search", 64)
            lat = self._hnsw_latency(k, ef_search)
            rec = self._hnsw_recall(k, ef_search)
        else:
            lat, rec = 100.0, 0.5

        return CostEstimate(
            index_name=index_name,
            params=params,
            estimated_latency_ms=max(lat, 0.001),
            estimated_recall=min(max(rec, 0.0), 1.0),
            confidence=0.5,  # Lower confidence than learned model
        )

    def estimate_all(self, candidates: list, query_features) -> list:
        return [self.estimate(c.index_name, c.params, query_features) for c in candidates]

    # --- Flat Index: O(n*d) linear scan ---

    def _flat_latency(self, k: int) -> float:
        """Flat: linear scan, proportional to n*d."""
        # Empirical constant calibrated to ~25ms for 1M x 128d
        return 0.0002 * self.n * self.d / 1e6 + 0.5

    def _flat_recall(self) -> float:
        """Flat: always perfect recall (exact search)."""
        return 1.0

    # --- IVF Index: scans nprobe/nlist fraction ---

    def _ivf_latency(self, k: int, nprobe: int, nlist: int = 256) -> float:
        """IVF: scans nprobe out of nlist cells."""
        fraction = nprobe / nlist
        scan_cost = fraction * self.n * self.d / 1e9  # ms scale
        overhead = 0.03 * nprobe  # per-cell overhead
        return scan_cost + overhead + 0.05

    def _ivf_recall(self, k: int, nprobe: int, nlist: int = 256) -> float:
        """IVF recall: sigmoid model of nprobe/nlist ratio."""
        ratio = nprobe / nlist
        # Recall saturates as nprobe increases; harder for larger K
        k_penalty = 1.0 - 0.002 * min(k, 100)
        return k_penalty * (1.0 - math.exp(-8.0 * ratio))

    # --- HNSW Index: O(ef_search * log(n) * d) graph traversal ---

    def _hnsw_latency(self, k: int, ef_search: int) -> float:
        """HNSW: graph traversal proportional to ef_search * log(n)."""
        log_n = math.log2(max(self.n, 2))
        return 0.00001 * ef_search * log_n * self.d / 1e3 + 0.02

    def _hnsw_recall(self, k: int, ef_search: int) -> float:
        """HNSW recall: depends on ef_search relative to k."""
        if ef_search < k:
            # ef_search must be >= k for valid results
            return 0.5 * (ef_search / k)
        ratio = ef_search / max(k, 1)
        return 1.0 - math.exp(-1.5 * ratio)
