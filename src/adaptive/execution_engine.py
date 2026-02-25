"""
Adaptive Execution Engine: the top-level entry point.

Wires together:
  QueryAnalyzer → CostModel → StrategySelector → Index.search() → Monitor

This is the main API surface of the adaptive framework.
"""

import time
from dataclasses import dataclass
from typing import Dict, List
import numpy as np

from src.adaptive.query_analyzer import QueryAnalyzer, QueryFeatures
from src.adaptive.strategy_selector import (
    StrategySelector, ExecutionStrategy, SelectionResult,
)
from src.cost_model.cost_estimator import CostModel
from src.cost_model.cost_functions import AnalyticalCostModel
from src.monitor.performance_monitor import PerformanceMonitor


@dataclass
class SearchResult:
    """Result from the adaptive execution engine."""
    indices: np.ndarray       # shape (k,) — neighbor IDs
    distances: np.ndarray     # shape (k,) — distances
    latency_ms: float
    strategy_used: ExecutionStrategy
    selection_result: SelectionResult
    query_features: QueryFeatures


class AdaptiveExecutionEngine:
    """
    Adaptive query-time execution framework for ANN search.

    Usage:
        engine = AdaptiveExecutionEngine(indexes, cost_model)
        result = engine.search(query_vector, top_k=10)
    """

    def __init__(self,
                 indexes: Dict[str, object],
                 cost_model=None,
                 candidates: List[ExecutionStrategy] = None,
                 latency_weight: float = 0.6,
                 recall_weight: float = 0.4,
                 default_latency_budget_ms: float = 5.0,
                 default_min_recall: float = 0.9,
                 dataset_size: int = None,
                 dimension: int = None,
                 monitor: PerformanceMonitor = None):
        """
        Args:
            indexes: dict of {"Flat": flat_index, "IVF": ivf_index, ...}
            cost_model: trained CostModel or AnalyticalCostModel.
                        If None, uses AnalyticalCostModel.
            candidates: execution strategy pool
            latency_weight: weight for latency in selection
            recall_weight: weight for recall in selection
            default_latency_budget_ms: default if not specified per query
            default_min_recall: default if not specified per query
            dataset_size: number of vectors in the index
            dimension: vector dimensionality
            monitor: PerformanceMonitor instance (created if None)
        """
        self.indexes = indexes
        self.dataset_size = dataset_size or 1_000_000
        self.dimension = dimension or 128

        # Cost model
        if cost_model is not None:
            self.cost_model = cost_model
        else:
            self.cost_model = AnalyticalCostModel(self.dataset_size, self.dimension)

        # Strategy selector
        self.selector = StrategySelector(
            candidates=candidates,
            latency_weight=latency_weight,
            recall_weight=recall_weight,
        )

        # Query analyzer
        self.analyzer = QueryAnalyzer(
            dataset_size=self.dataset_size,
            dimensionality=self.dimension,
            monitor_system_load=False,  # skip psutil for speed
        )

        # Performance monitor
        self.monitor = monitor or PerformanceMonitor()

        # Defaults
        self.default_latency_budget_ms = default_latency_budget_ms
        self.default_min_recall = default_min_recall

    def search(self,
               query: np.ndarray,
               top_k: int = 10,
               latency_budget_ms: float = None,
               min_recall: float = None,
               concurrency: int = 1) -> SearchResult:
        """
        Adaptive search: analyze query → estimate costs → select strategy → execute.

        Args:
            query: 1-D query vector
            top_k: number of neighbors to return
            latency_budget_ms: latency constraint (ms)
            min_recall: minimum acceptable recall
            concurrency: simulated concurrent query count (affects index choice)

        Returns:
            SearchResult with neighbors, latency, and decision explanation
        """
        budget = latency_budget_ms or self.default_latency_budget_ms
        recall_req = min_recall or self.default_min_recall

        # Step 1: Analyze query features
        features = self.analyzer.extract_features(
            query_vector=query,
            top_k=top_k,
            latency_budget_ms=budget,
            min_recall=recall_req,
            concurrency=concurrency,
        )

        # Step 2: Estimate costs for all candidates
        estimates = self.cost_model.estimate_all(
            self.selector.candidates, features
        )

        # Step 3: Select best strategy
        selection = self.selector.select(
            estimates=estimates,
            latency_budget_ms=budget,
            min_recall=recall_req,
        )

        strategy = selection.chosen_strategy

        # Step 4: Execute search on the selected index
        index_obj = self.indexes[strategy.index_name]

        # Build search kwargs from strategy params
        search_kwargs = dict(strategy.params)
        if "ef_search" in search_kwargs:
            search_kwargs["ef_search"] = max(search_kwargs["ef_search"], top_k)

        t0 = time.perf_counter()
        D, I = index_obj.search(query.reshape(1, -1), top_k, **search_kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Step 5: Record in monitor
        self.monitor.record(
            index_name=strategy.index_name,
            params=strategy.params,
            actual_latency_ms=elapsed_ms,
            predicted_latency_ms=selection.chosen_estimate.estimated_latency_ms,
        )

        return SearchResult(
            indices=I[0],
            distances=D[0],
            latency_ms=elapsed_ms,
            strategy_used=strategy,
            selection_result=selection,
            query_features=features,
        )

    def batch_search(self,
                     queries: np.ndarray,
                     top_k: int = 10,
                     latency_budget_ms: float = None,
                     min_recall: float = None) -> List[SearchResult]:
        """Run adaptive search per query in a batch."""
        results = []
        for i in range(len(queries)):
            r = self.search(
                queries[i], top_k,
                latency_budget_ms=latency_budget_ms,
                min_recall=min_recall,
            )
            results.append(r)
        return results

    def get_stats(self) -> dict:
        """Return performance monitoring stats."""
        return self.monitor.summary()
