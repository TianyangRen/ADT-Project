"""
Evaluation metrics for vector similarity search.

Primary metrics:
  - Recall@K: fraction of true top-K neighbors found by ANN search
  - Latency: query execution time
  - QPS: queries per second (throughput)
"""

import numpy as np
from typing import Optional


def compute_recall(predicted: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """
    Compute average Recall@K across all queries.

    Recall@K = |predicted_top_k ∩ true_top_k| / K, averaged over queries.

    Args:
        predicted:    (nq, k') predicted neighbor indices from ANN search
        ground_truth: (nq, k'') ground truth neighbor indices (from exact search)
        k:            K value for recall computation

    Returns:
        Average recall@K as a float in [0, 1]
    """
    nq = predicted.shape[0]
    assert ground_truth.shape[0] == nq, \
        f"Query count mismatch: predicted={nq}, ground_truth={ground_truth.shape[0]}"

    # Truncate to K
    pred_k = predicted[:, :k]
    gt_k = ground_truth[:, :k]

    recall_sum = 0.0
    for i in range(nq):
        # Count how many of the true top-K are in the predicted top-K
        true_set = set(gt_k[i])
        pred_set = set(pred_k[i])
        # Remove -1 entries (FAISS uses -1 for missing results)
        true_set.discard(-1)
        pred_set.discard(-1)
        if len(true_set) == 0:
            continue
        recall_sum += len(true_set & pred_set) / len(true_set)

    return recall_sum / nq


def compute_recall_per_query(predicted: np.ndarray, ground_truth: np.ndarray,
                              k: int) -> np.ndarray:
    """
    Compute Recall@K for each individual query.

    Returns:
        np.ndarray of shape (nq,) with per-query recall values
    """
    nq = predicted.shape[0]
    pred_k = predicted[:, :k]
    gt_k = ground_truth[:, :k]

    recalls = np.zeros(nq, dtype=np.float64)
    for i in range(nq):
        true_set = set(gt_k[i])
        pred_set = set(pred_k[i])
        true_set.discard(-1)
        pred_set.discard(-1)
        if len(true_set) > 0:
            recalls[i] = len(true_set & pred_set) / len(true_set)
    return recalls


def compute_latency_stats(latencies_ms: np.ndarray) -> dict:
    """
    Compute latency statistics from an array of per-query latencies.

    Args:
        latencies_ms: array of latency values in milliseconds

    Returns:
        Dict with mean, median, p50, p95, p99, min, max
    """
    return {
        "mean_ms": float(np.mean(latencies_ms)),
        "median_ms": float(np.median(latencies_ms)),
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
        "min_ms": float(np.min(latencies_ms)),
        "max_ms": float(np.max(latencies_ms)),
        "std_ms": float(np.std(latencies_ms)),
    }


def compute_qps(num_queries: int, total_time_s: float) -> float:
    """Compute queries per second."""
    if total_time_s <= 0:
        return float("inf")
    return num_queries / total_time_s
