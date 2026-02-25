"""
Query Feature Extraction for adaptive execution.

Extracts lightweight features from incoming queries to feed into
the cost model for strategy selection. Features include:
  - Query-level: top_k, dimensionality, vector norm
  - Constraint-level: latency budget, minimum recall requirement
  - System-level: current CPU load, dataset size
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class QueryFeatures:
    """Features extracted from an incoming query for cost-model input."""
    top_k: int
    dimensionality: int
    query_norm: float
    dataset_size: int
    latency_budget_ms: Optional[float] = None   # user-specified max latency
    min_recall: Optional[float] = None           # user-specified min recall
    system_cpu_percent: float = 0.0              # current system load
    concurrency: int = 1                         # simulated concurrent query count

    def to_dict(self) -> dict:
        return {
            "top_k": self.top_k,
            "dimensionality": self.dimensionality,
            "query_norm": self.query_norm,
            "dataset_size": self.dataset_size,
            "latency_budget_ms": self.latency_budget_ms,
            "min_recall": self.min_recall,
            "system_cpu_percent": self.system_cpu_percent,
            "concurrency": self.concurrency,
        }


class QueryAnalyzer:
    """
    Extract features from incoming queries for cost model input.

    Lightweight — designed to add minimal overhead to each query.
    """

    def __init__(self, dataset_size: int, dimensionality: int,
                 monitor_system_load: bool = True):
        """
        Args:
            dataset_size: number of vectors in the index
            dimensionality: vector dimension
            monitor_system_load: whether to check CPU usage (adds ~10ms)
        """
        self.dataset_size = dataset_size
        self.dimensionality = dimensionality
        self.monitor_system_load = monitor_system_load

    def extract_features(self, query_vector: np.ndarray, top_k: int,
                         latency_budget_ms: float = None,
                         min_recall: float = None,
                         concurrency: int = 1) -> QueryFeatures:
        """
        Extract features from a single query.

        Args:
            query_vector: (d,) or (1, d) query vector
            top_k: number of neighbors requested
            latency_budget_ms: optional latency constraint in ms
            min_recall: optional minimum recall requirement [0, 1]
            concurrency: number of concurrent queries (affects cache contention)

        Returns:
            QueryFeatures dataclass
        """
        query_vector = np.asarray(query_vector, dtype=np.float32).ravel()
        query_norm = float(np.linalg.norm(query_vector))

        cpu_pct = 0.0
        if self.monitor_system_load:
            try:
                import psutil
                cpu_pct = psutil.cpu_percent(interval=None)  # non-blocking
            except ImportError:
                pass

        return QueryFeatures(
            top_k=top_k,
            dimensionality=self.dimensionality,
            query_norm=query_norm,
            dataset_size=self.dataset_size,
            latency_budget_ms=latency_budget_ms,
            min_recall=min_recall,
            system_cpu_percent=cpu_pct,
            concurrency=concurrency,
        )
