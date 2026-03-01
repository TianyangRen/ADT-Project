"""
Query Logging Wrapper for Adaptive Execution Engine.

This module provides a transparent wrapper around the AdaptiveExecutionEngine
to log search queries, decisions, and performance metrics to a file (JSONL/CSV)
without modifying the core engine code.

It is useful for offline analysis of strategy selection quality.
"""

import time
import json
import logging
import hashlib
import numpy as np
from datetime import datetime
from typing import Any, Dict, Optional, List
from threading import Lock

# Import necessary types for type hinting
from src.adaptive.execution_engine import AdaptiveExecutionEngine, SearchResult

class QueryLogger:
    """
    Wrapper for AdaptiveExecutionEngine that logs search events.

    Usage:
        engine = AdaptiveExecutionEngine(...)
        logged_engine = QueryLogger(engine, log_file="query_log.jsonl")
        result = logged_engine.search(query_vector)
    """

    def __init__(self, engine: AdaptiveExecutionEngine, log_file: str = "logs/query_history.jsonl"):
        self.engine = engine
        self.log_file = log_file
        self.lock = Lock()

        # Ensure log directory exists
        import os
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Configure internal logger
        self.logger = logging.getLogger("QueryLogger")
        self.logger.setLevel(logging.INFO)

    def __getattr__(self, name):
        """Delegate unknown attributes/methods to the underlying engine."""
        return getattr(self.engine, name)

    def _hash_vector(self, vector: np.ndarray) -> str:
        """Create a short hash of the query vector for identification."""
        return hashlib.md5(vector.tobytes()).hexdigest()[:8]

    def _serialize(self, obj: Any) -> Any:
        """JSON serializer for objects not serializable by default json code."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        if hasattr(obj, "__dict__"):
            return str(obj)
        return str(obj)

    def search(self,
               query: np.ndarray,
               top_k: int = 10,
               latency_budget_ms: float = None,
               min_recall: float = None,
               concurrency: int = 1) -> SearchResult:
        """
        Wrapped search method.
        Executes search on the engine and logs the result.
        """
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        result = None
        success = False
        error_msg = None

        # Execute the actual search
        try:
            result = self.engine.search(
                query=query,
                top_k=top_k,
                latency_budget_ms=latency_budget_ms,
                min_recall=min_recall,
                concurrency=concurrency
            )
            success = True
        except Exception as e:
            success = False
            error_msg = str(e)
            raise e
        finally:
            # Metadata to log
            duration_ms = (time.time() - start_time) * 1000

            log_entry = {
                "timestamp": timestamp,
                "query_hash": self._hash_vector(query),
                "top_k": top_k,
                "latency_budget_ms": latency_budget_ms,
                "min_recall": min_recall,
                "concurrency": concurrency,
                "success": success,
                "total_duration_ms": duration_ms,
            }

            if success and result:
                # Add result data
                log_entry.update({
                    "strategy_index": result.strategy_used.index_name,
                    "strategy_params": result.strategy_used.params,
                    "predicted_latency_ms": result.selection_result.chosen_estimate.estimated_latency_ms,
                    "actual_latency_ms": result.latency_ms,
                    "num_results": len(result.indices),
                    # We don't log the full vector or full results to save space
                })
            else:
                log_entry["error"] = error_msg

            # Write to file (thread-safe)
            self._write_log(log_entry)

        return result

    def _write_log(self, entry: Dict[str, Any]):
        """Append log entry to file."""
        try:
            with self.lock:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, default=self._serialize) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write query log: {e}")

    def batch_search(self,
                     queries: np.ndarray,
                     top_k: int = 10,
                     latency_budget_ms: float = None,
                     min_recall: float = None) -> List[SearchResult]:
        """Wrapped batch search."""
        # Simple implementation: iterate and log individually or log batch summary?
        # For simplicity, we just delegate to engine.batch_search but we lose per-query logging
        # unless we reimplement the loop.
        # Better to reimplement loop using self.search for full logging.

        results = []
        for i in range(len(queries)):
            r = self.search(
                queries[i], top_k,
                latency_budget_ms=latency_budget_ms,
                min_recall=min_recall,
            )
            results.append(r)
        return results


