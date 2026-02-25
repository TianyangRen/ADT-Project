"""
Recall profiler: measures recall@K for any index against ground truth.
"""

import numpy as np
from typing import Dict

from src.utils.metrics import compute_recall, compute_recall_per_query


class RecallProfiler:
    """Measure recall accuracy of ANN search results."""

    def profile(self, index, queries: np.ndarray, ground_truth: np.ndarray,
                k: int, **search_params) -> Dict:
        """
        Compute recall@K and per-query recall distribution.

        Args:
            index: an index implementing the BaseIndex interface
            queries: (nq, d) query vectors
            ground_truth: (nq, gt_k) true neighbor indices
            k: K for recall computation
            **search_params: passed to index.search()

        Returns:
            Dict with recall stats
        """
        _, predicted = index.search(queries, k, **search_params)
        gt_k = ground_truth[:, :k]

        avg_recall = compute_recall(predicted, gt_k, k)
        per_query = compute_recall_per_query(predicted, gt_k, k)

        return {
            "recall_at_k": avg_recall,
            "recall_min": float(np.min(per_query)),
            "recall_max": float(np.max(per_query)),
            "recall_std": float(np.std(per_query)),
            "recall_p5": float(np.percentile(per_query, 5)),
            "recall_p50": float(np.percentile(per_query, 50)),
            "recall_p95": float(np.percentile(per_query, 95)),
            "queries_with_perfect_recall": float(np.mean(per_query == 1.0)),
            "queries_with_zero_recall": float(np.mean(per_query == 0.0)),
        }
