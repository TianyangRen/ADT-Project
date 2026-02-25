"""
Strategy Selector: picks the best (index, params) for each query
given cost estimates, latency budgets, and recall requirements.

Multi-objective selection:
  1. Hard filter: drop candidates violating constraints
  2. Score remaining by weighted objective
  3. Select best, with explainability
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

from src.cost_model.cost_estimator import CostEstimate


@dataclass
class ExecutionStrategy:
    """One candidate strategy to evaluate."""
    index_name: str
    params: dict = field(default_factory=dict)

    def __repr__(self):
        p = ", ".join(f"{k}={v}" for k, v in self.params.items()) if self.params else "default"
        return f"Strategy({self.index_name}({p}))"


@dataclass
class SelectionResult:
    """Result of strategy selection."""
    chosen_strategy: ExecutionStrategy
    chosen_estimate: CostEstimate
    reason: str
    all_estimates: List[CostEstimate] = field(default_factory=list)
    feasible_count: int = 0

    def explain(self) -> str:
        lines = [
            f"Selected: {self.chosen_strategy}",
            f"  Predicted: latency={self.chosen_estimate.estimated_latency_ms:.3f}ms, "
            f"recall={self.chosen_estimate.estimated_recall:.4f}",
            f"  Reason: {self.reason}",
            f"  Feasible candidates: {self.feasible_count}/{len(self.all_estimates)}",
        ]
        return "\n".join(lines)


class StrategySelector:
    """
    Selects the best execution strategy from candidates.

    Default strategy: minimize latency subject to recall constraint.
    Falls back to best-recall if no candidate meets latency budget.
    """

    # Default pool of candidate strategies
    DEFAULT_CANDIDATES = [
        ExecutionStrategy("Flat", {}),
        ExecutionStrategy("IVF", {"nprobe": 1}),
        ExecutionStrategy("IVF", {"nprobe": 4}),
        ExecutionStrategy("IVF", {"nprobe": 16}),
        ExecutionStrategy("IVF", {"nprobe": 64}),
        ExecutionStrategy("HNSW", {"ef_search": 16}),
        ExecutionStrategy("HNSW", {"ef_search": 64}),
        ExecutionStrategy("HNSW", {"ef_search": 256}),
    ]

    def __init__(self,
                 candidates: List[ExecutionStrategy] = None,
                 latency_weight: float = 0.6,
                 recall_weight: float = 0.4):
        """
        Args:
            candidates: pool of strategies to consider
            latency_weight: weight for latency in scoring (higher = prefer speed)
            recall_weight: weight for recall in scoring (higher = prefer quality)
        """
        self.candidates = candidates or self.DEFAULT_CANDIDATES
        self.latency_weight = latency_weight
        self.recall_weight = recall_weight

    def select(self,
               estimates: List[CostEstimate],
               latency_budget_ms: float = None,
               min_recall: float = None) -> SelectionResult:
        """
        Select the best strategy from cost estimates.

        Steps:
          1. Filter by hard constraints (latency budget, min recall)
          2. If feasible set non-empty: pick by weighted score
          3. If empty: relax constraints, pick best recall
        """
        assert len(estimates) > 0, "No candidate estimates"

        # Build (strategy, estimate) pairs
        pairs = list(zip(self.candidates[:len(estimates)], estimates))

        # --- Step 1: Hard constraint filtering ---
        feasible = []
        for strategy, est in pairs:
            ok = True
            if latency_budget_ms is not None and est.estimated_latency_ms > latency_budget_ms:
                ok = False
            if min_recall is not None and est.estimated_recall < min_recall:
                ok = False
            if ok:
                feasible.append((strategy, est))

        # --- Step 2: Selection ---
        if len(feasible) > 0:
            # Pick by weighted objective: minimize latency, maximize recall
            best = min(feasible, key=lambda x: self._score(x[1]))
            reason = f"Best score among {len(feasible)} feasible candidates"
        else:
            # Relaxed: pick the one with best recall
            best = max(pairs, key=lambda x: x[1].estimated_recall)
            reason = "No feasible candidate; chose best recall (constraints relaxed)"

        return SelectionResult(
            chosen_strategy=best[0],
            chosen_estimate=best[1],
            reason=reason,
            all_estimates=estimates,
            feasible_count=len(feasible),
        )

    def _score(self, est: CostEstimate) -> float:
        """
        Compute selection score (lower is better).

        Combines normalized latency (lower better) and recall (higher better).
        """
        # Latency: lower is better → use as-is
        lat_score = est.estimated_latency_ms

        # Recall: higher is better → negate so lower is better
        rec_score = (1.0 - est.estimated_recall) * 100.0  # Scale to ms-like range

        return self.latency_weight * lat_score + self.recall_weight * rec_score

    def select_pareto(self, estimates: List[CostEstimate]) -> List[CostEstimate]:
        """Return Pareto-optimal estimates (no dominated points)."""
        pareto = []
        for i, e in enumerate(estimates):
            dominated = False
            for j, f in enumerate(estimates):
                if j == i:
                    continue
                # f dominates e if f is at least as good on both and strictly better on one
                if (f.estimated_latency_ms <= e.estimated_latency_ms and
                        f.estimated_recall >= e.estimated_recall and
                        (f.estimated_latency_ms < e.estimated_latency_ms or
                         f.estimated_recall > e.estimated_recall)):
                    dominated = True
                    break
            if not dominated:
                pareto.append(e)
        return pareto
