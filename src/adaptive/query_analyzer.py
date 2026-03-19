from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import time
import numpy as np


@dataclass
class QueryFeatures:
    """Features extracted from an incoming query for cost-model input."""
    # --- Original fields (kept for compatibility) ---
    top_k: int
    dimensionality: int
    query_norm: float
    dataset_size: int
    latency_budget_ms: Optional[float] = None
    min_recall: Optional[float] = None
    system_cpu_percent: float = 0.0
    concurrency: int = 1

    # --- New, cheap, useful features ---
    log2_top_k: float = 0.0
    is_normalized: int = 0               # 1 if ||q|| approx 1.0, else 0
    mean: float = 0.0
    std: float = 0.0
    max_abs: float = 0.0
    zero_frac: float = 0.0              # fraction of elements that are exactly 0

    # Convenience flags (often useful in models/rules)
    has_budget: int = 0
    has_recall_req: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {

            "top_k": self.top_k,
            "dimensionality": self.dimensionality,
            "query_norm": self.query_norm,
            "dataset_size": self.dataset_size,
            "latency_budget_ms": self.latency_budget_ms,
            "min_recall": self.min_recall,
            "system_cpu_percent": self.system_cpu_percent,
            "concurrency": self.concurrency,
   
            "log2_top_k": self.log2_top_k,
            "is_normalized": self.is_normalized,
            "mean": self.mean,
            "std": self.std,
            "max_abs": self.max_abs,
            "zero_frac": self.zero_frac,
            "has_budget": self.has_budget,
            "has_recall_req": self.has_recall_req,
        }


class QueryAnalyzer:

    def __init__(
        self,
        dataset_size: int,
        dimensionality: int,
        monitor_system_load: bool = True,
        cpu_sample_interval_s: float = 0.5,
        normalized_tol: float = 0.05,
    ):
alized_tol: tolerance for treating ||q|| as "normalized" (|norm-1| <= tol)

        Return CPU percent with caching.
        Uses interval=None (non-blocking) but only samples every cpu_sample_interval_s.

        Extract features from a single query.

        Args:
            query_vector: (d,) or (1, d) query vector
            top_k: number of neighbors requested
            latency_budget_ms: optional latency constraint in ms
            min_recall: optional minimum recall requirement [0, 1]
            concurrency: number of concurrent queries (affects cache contention)

        Returns:
            QueryFeatures dataclass
