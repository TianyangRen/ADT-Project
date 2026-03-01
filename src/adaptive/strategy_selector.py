"""
Strategy Selector: picks the best (index, params) for each query
given cost estimates, latency budgets, and recall requirements.

Constraint-driven cascading selection:
  1. Filter candidates meeting the recall constraint
  2. Among recall-feasible, pick the fastest that fits the latency budget
  3. Graceful fallback when constraints cannot be jointly satisfied

This produces genuinely different strategies for different SLA profiles:
  - Tight budget + relaxed recall  → fast HNSW / IVF with small params
  - Relaxed budget + strict recall → large params or Flat (exact)
"""

from dataclasses import dataclass, field
from typing import List, Optional

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
    regime: str = "unknown"   # descriptive tag for the selection path taken
    all_estimates: List[CostEstimate] = field(default_factory=list)
    feasible_count: int = 0
    recall_feasible_count: int = 0

    def explain(self) -> str:
        lines = [
            f"Selected: {self.chosen_strategy}",
            f"  Predicted: latency={self.chosen_estimate.estimated_latency_ms:.3f}ms, "
            f"recall={self.chosen_estimate.estimated_recall:.4f}",
            f"  Regime: {self.regime}",
            f"  Reason: {self.reason}",
            f"  Recall-feasible: {self.recall_feasible_count}/{len(self.all_estimates)}, "
            f"Fully-feasible: {self.feasible_count}/{len(self.all_estimates)}",
        ]
        return "\n".join(lines)


class StrategySelector:
    """
    Constraint-driven adaptive strategy selector.

    Core principle: *minimize latency* among candidates that satisfy the
    recall requirement.  The latency budget acts as a secondary filter.

    Selection cascade (first satisfied wins):
      A. Fully feasible  (recall OK + latency OK): pick **fastest**
      B. Recall-only      (recall OK, latency exceeded): pick **fastest**
         — better to miss budget slightly than sacrifice recall
      C. Latency-only     (latency OK, recall missed): pick **highest recall**
      D. Nothing feasible : pick **highest recall** overall
    """

    # Default pool of candidate strategies — fine-grained to allow
    # the selector to pick genuinely different operating points.
    DEFAULT_CANDIDATES = [
        ExecutionStrategy("Flat", {}),
        ExecutionStrategy("IVF", {"nprobe": 1}),
        ExecutionStrategy("IVF", {"nprobe": 4}),
        ExecutionStrategy("IVF", {"nprobe": 16}),
        ExecutionStrategy("IVF", {"nprobe": 64}),
        ExecutionStrategy("IVF", {"nprobe": 128}),
        ExecutionStrategy("HNSW", {"ef_search": 16}),
        ExecutionStrategy("HNSW", {"ef_search": 32}),
        ExecutionStrategy("HNSW", {"ef_search": 64}),
        ExecutionStrategy("HNSW", {"ef_search": 128}),
        ExecutionStrategy("HNSW", {"ef_search": 256}),
        ExecutionStrategy("HNSW", {"ef_search": 512}),
    ]

    def __init__(self,
                 candidates: List[ExecutionStrategy] = None,
                 latency_weight: float = 0.6,
                 recall_weight: float = 0.4):
        self.candidates = candidates or self.DEFAULT_CANDIDATES
        self.latency_weight = latency_weight
        self.recall_weight = recall_weight

    def select(self,
               estimates: List[CostEstimate],
               latency_budget_ms: float = None,
               min_recall: float = None) -> SelectionResult:
        """
        Select the best strategy via constraint-driven cascading.

        Returns the fastest candidate that satisfies the recall floor,
        preferring those that also fit the latency budget.
        """
        assert len(estimates) > 0, "No candidate estimates"
        assert len(estimates) == len(self.candidates), \
            f"Mismatch: {len(estimates)} estimates vs {len(self.candidates)} candidates"

        # Use efficient zipping without list creation if possible, but here we need to partition
        pairs = list(zip(self.candidates, estimates))

        # --- Partition candidates ---
        recall_ok = []    # meets min_recall
        latency_ok = []   # meets latency budget
        fully_ok = []     # meets both

        for strategy, est in pairs:
            r_ok = (min_recall is None) or (est.estimated_recall >= min_recall)
            l_ok = (latency_budget_ms is None) or (est.estimated_latency_ms <= latency_budget_ms)
            if r_ok:
                recall_ok.append((strategy, est))
            if l_ok:
                latency_ok.append((strategy, est))
            if r_ok and l_ok:
                fully_ok.append((strategy, est))

        # --- Cascade ---
        if fully_ok:
            # A: Both constraints met → pick the FASTEST
            best = min(fully_ok, key=lambda x: x[1].estimated_latency_ms)
            reason = (f"Fastest of {len(fully_ok)} fully-feasible candidates "
                      f"(recall>={min_recall}, lat<={latency_budget_ms}ms)")
            regime = "optimal"
        elif recall_ok:
            # B: Recall met, budget missed → pick fastest recall-feasible
            best = min(recall_ok, key=lambda x: x[1].estimated_latency_ms)
            reason = (f"Fastest of {len(recall_ok)} recall-feasible candidates "
                      f"(budget relaxed from {latency_budget_ms}ms)")
            regime = "recall_priority"
        elif latency_ok:
            # C: Budget met, recall missed → pick highest recall within budget
            best = max(latency_ok, key=lambda x: x[1].estimated_recall)
            reason = (f"Best recall of {len(latency_ok)} latency-feasible candidates "
                      f"(recall relaxed from {min_recall})")
            regime = "latency_priority"
        else:
            # D: Nothing feasible → pick highest recall overall
            best = max(pairs, key=lambda x: x[1].estimated_recall)
            reason = "No feasible candidate; chose best recall overall"
            regime = "fallback"

        return SelectionResult(
            chosen_strategy=best[0],
            chosen_estimate=best[1],
            reason=reason,
            regime=regime,
            all_estimates=estimates,
            feasible_count=len(fully_ok),
            recall_feasible_count=len(recall_ok),
        )

    def select_pareto(self, estimates: List[CostEstimate]) -> List[CostEstimate]:
        """Return Pareto-optimal estimates (no dominated points)."""
        pareto = []
        for i, e in enumerate(estimates):
            dominated = False
            for j, f in enumerate(estimates):
                if j == i:
                    continue
                if (f.estimated_latency_ms <= e.estimated_latency_ms and
                        f.estimated_recall >= e.estimated_recall and
                        (f.estimated_latency_ms < e.estimated_latency_ms or
                         f.estimated_recall > e.estimated_recall)):
                    dominated = True
                    break
            if not dominated:
                pareto.append(e)
        return pareto
