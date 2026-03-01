"""
Query Feature Extraction for adaptive execution.

Extracts lightweight features from incoming queries to feed into
the cost model for strategy selection.

Optimized version:
- Adds richer but cheap query statistics (mean/std/max_abs/zero_frac)
- Adds log2(top_k) and is_normalized signal
- CPU monitoring is cached with a minimum sampling interval to avoid jitter/overhead
- Keeps backwards compatibility with existing fields used by the engine/selector
"""

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
            # original
            "top_k": self.top_k,
            "dimensionality": self.dimensionality,
            "query_norm": self.query_norm,
            "dataset_size": self.dataset_size,
            "latency_budget_ms": self.latency_budget_ms,
            "min_recall": self.min_recall,
            "system_cpu_percent": self.system_cpu_percent,
            "concurrency": self.concurrency,
            # new
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
    """
    Extract features from incoming queries for cost model input.

    Lightweight — designed to add minimal overhead to each query.

    Key optimization: CPU monitoring is cached with a minimum interval
    to avoid jitter and repeated psutil calls.
    """

    def __init__(
        self,
        dataset_size: int,
        dimensionality: int,
        monitor_system_load: bool = True,
        cpu_sample_interval_s: float = 0.5,
        normalized_tol: float = 0.05,
    ):
        """
        Args:
            dataset_size: number of vectors in the index
            dimensionality: vector dimension
            monitor_system_load: whether to check CPU usage (psutil)
            cpu_sample_interval_s: minimum seconds between CPU samples (cache)
            normalized_tol: tolerance for treating ||q|| as "normalized" (|norm-1| <= tol)
        """
        self.dataset_size = int(dataset_size)
        self.dimensionality = int(dimensionality)
        self.monitor_system_load = bool(monitor_system_load)

        self.cpu_sample_interval_s = float(cpu_sample_interval_s)
        self.normalized_tol = float(normalized_tol)

        self._last_cpu_sample_t = 0.0
        self._last_cpu_percent = 0.0
        self._psutil = None

        if self.monitor_system_load:
            try:
                import psutil  # type: ignore
                self._psutil = psutil
            except Exception:
                self._psutil = None

    def _get_cpu_percent_cached(self) -> float:
        """
        Return CPU percent with caching.
        Uses interval=None (non-blocking) but only samples every cpu_sample_interval_s.
        """
        if (not self.monitor_system_load) or (self._psutil is None):
            return 0.0

        now = time.time()
        if (now - self._last_cpu_sample_t) < self.cpu_sample_interval_s:
            return self._last_cpu_percent

        try:
            # non-blocking instantaneous estimate (can be noisy, so we cache)
            cpu_pct = float(self._psutil.cpu_percent(interval=None))
        except Exception:
            cpu_pct = 0.0

        self._last_cpu_sample_t = now
        self._last_cpu_percent = cpu_pct
        return cpu_pct

    def extract_features(
        self,
        query_vector: np.ndarray,
        top_k: int,
        latency_budget_ms: float = None,
        min_recall: float = None,
        concurrency: int = 1,
    ) -> QueryFeatures:
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
        # ---- sanitize inputs ----
        k = int(top_k)
        if k <= 0:
            k = 1

        conc = int(concurrency)
        if conc <= 0:
            conc = 1

        q = np.asarray(query_vector, dtype=np.float32).ravel()

        # dimensionality mismatch handling (don’t hard crash; still extract stats)
        dim = int(q.shape[0])
        if self.dimensionality <= 0:
            self.dimensionality = dim

        # ---- cheap query stats ----
        # norm
        q_norm = float(np.linalg.norm(q))

        # distribution stats
        # (use float64 for stability; cost is tiny vs ANN search)
        q64 = q.astype(np.float64, copy=False)
        q_mean = float(q64.mean()) if q64.size else 0.0
        q_std = float(q64.std()) if q64.size else 0.0
        q_max_abs = float(np.max(np.abs(q64))) if q64.size else 0.0

        # exact zeros ratio (fast)
        # NOTE: for dense embeddings, this is usually 0.0; for sparse-ish, it’s useful
        zero_frac = float(np.mean(q == 0.0)) if q.size else 0.0

        # log-scaled k is often more model-friendly
        log2_k = float(np.log2(k))

        # normalized signal (useful when metric changes or embeddings are normalized)
        is_norm = 1 if abs(q_norm - 1.0) <= self.normalized_tol else 0

        # ---- constraints flags ----
        has_budget = 1 if latency_budget_ms is not None else 0
        has_recall = 1 if min_recall is not None else 0

        # ---- system ----
        cpu_pct = self._get_cpu_percent_cached()

        return QueryFeatures(
            # original
            top_k=k,
            dimensionality=self.dimensionality,
            query_norm=q_norm,
            dataset_size=self.dataset_size,
            latency_budget_ms=latency_budget_ms,
            min_recall=min_recall,
            system_cpu_percent=cpu_pct,
            concurrency=conc,
            # new
            log2_top_k=log2_k,
            is_normalized=is_norm,
            mean=q_mean,
            std=q_std,
            max_abs=q_max_abs,
            zero_frac=zero_frac,
            has_budget=has_budget,
            has_recall_req=has_recall,
        )
