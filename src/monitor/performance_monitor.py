"""
Performance Monitor: tracks actual vs predicted performance,
detects drift, and signals when cost model recalibration is needed.
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class MonitorRecord:
    """One observation of actual vs predicted performance."""
    index_name: str
    params: dict
    actual_latency_ms: float
    predicted_latency_ms: float
    prediction_error_ms: float
    timestamp: float


class PerformanceMonitor:
    """
    Sliding-window performance monitor.

    Tracks:
      - Actual latency distribution
      - Prediction error (predicted - actual)
      - Per-index selection frequency
      - Recalibration signal when error exceeds threshold
    """

    def __init__(self, window_size: int = 1000,
                 error_threshold_ms: float = 5.0):
        """
        Args:
            window_size: number of recent records to keep
            error_threshold_ms: mean absolute error threshold for recalibration
        """
        self.window_size = window_size
        self.error_threshold_ms = error_threshold_ms
        self.history: deque = deque(maxlen=window_size)
        self.selection_counts: Dict[str, int] = {}
        self.total_queries = 0

    def record(self, index_name: str, params: dict,
               actual_latency_ms: float,
               predicted_latency_ms: float):
        """Record one search observation."""
        error = predicted_latency_ms - actual_latency_ms

        rec = MonitorRecord(
            index_name=index_name,
            params=params,
            actual_latency_ms=actual_latency_ms,
            predicted_latency_ms=predicted_latency_ms,
            prediction_error_ms=error,
            timestamp=time.time(),
        )
        self.history.append(rec)
        self.total_queries += 1

        # Track selection frequency
        key = self._strategy_key(index_name, params)
        self.selection_counts[key] = self.selection_counts.get(key, 0) + 1

    def should_recalibrate(self) -> bool:
        """Check if cost model should be recalibrated."""
        if len(self.history) < 50:
            return False  # Need enough data

        errors = [abs(r.prediction_error_ms) for r in self.history]
        mae = np.mean(errors)
        return mae > self.error_threshold_ms

    def get_prediction_error(self) -> dict:
        """Get prediction error statistics."""
        if len(self.history) == 0:
            return {"mae": 0, "mean_error": 0, "std_error": 0, "count": 0}

        errors = [r.prediction_error_ms for r in self.history]
        abs_errors = [abs(e) for e in errors]
        return {
            "mae": float(np.mean(abs_errors)),
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors)),
            "count": len(errors),
        }

    def get_latency_stats(self) -> dict:
        """Get actual latency statistics."""
        if len(self.history) == 0:
            return {"mean": 0, "p50": 0, "p95": 0, "p99": 0, "count": 0}

        lats = [r.actual_latency_ms for r in self.history]
        return {
            "mean": float(np.mean(lats)),
            "p50": float(np.percentile(lats, 50)),
            "p95": float(np.percentile(lats, 95)),
            "p99": float(np.percentile(lats, 99)),
            "count": len(lats),
        }

    def get_selection_distribution(self) -> Dict[str, float]:
        """Get the fraction of queries routed to each strategy."""
        total = sum(self.selection_counts.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in sorted(
            self.selection_counts.items(), key=lambda x: -x[1])}

    def summary(self) -> dict:
        """Return complete monitoring summary."""
        return {
            "total_queries": self.total_queries,
            "window_size": len(self.history),
            "prediction_error": self.get_prediction_error(),
            "latency_stats": self.get_latency_stats(),
            "selection_distribution": self.get_selection_distribution(),
            "needs_recalibration": self.should_recalibrate(),
        }

    def _strategy_key(self, index_name: str, params: dict) -> str:
        p = ",".join(f"{k}={v}" for k, v in sorted(params.items())) if params else "default"
        return f"{index_name}({p})"

    def reset(self):
        """Clear all history."""
        self.history.clear()
        self.selection_counts.clear()
        self.total_queries = 0
