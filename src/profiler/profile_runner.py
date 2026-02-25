"""
Profile Runner: orchestrates systematic performance profiling across
all index types, parameter values, and K values.

This is the core of Phase 2 — it produces the profiling data that
demonstrates the failure modes of static index selection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List

from src.profiler.latency_profiler import LatencyProfiler
from src.profiler.recall_profiler import RecallProfiler


class ProfileRunner:
    """
    Sweep over (index, params, K) space and collect latency + recall.
    """

    def __init__(self, warmup: int = 50, repeat: int = 3):
        self.lat_profiler = LatencyProfiler(warmup_queries=warmup)
        self.rec_profiler = RecallProfiler()
        self.repeat = repeat

    def run_sweep(self, indexes: Dict, queries: np.ndarray,
                  ground_truth: np.ndarray, k_values: List[int],
                  param_grids: Dict) -> pd.DataFrame:
        """
        Run a full profiling sweep.

        Args:
            indexes: {"Flat": flat_obj, "IVF": ivf_obj, "HNSW": hnsw_obj}
            queries: (nq, d) query vectors
            ground_truth: (nq, gt_k) ground truth neighbor indices
            k_values: list of K values to test [1, 10, 50, 100]
            param_grids: {
                "Flat": [{}],
                "IVF": [{"nprobe": 1}, {"nprobe": 2}, ...],
                "HNSW": [{"ef_search": 16}, {"ef_search": 32}, ...],
            }

        Returns:
            DataFrame with one row per (index, params, k) combination
        """
        results = []
        total_configs = sum(
            len(param_grids.get(name, [{}])) * len(k_values)
            for name in indexes
        )
        current = 0

        for idx_name, idx_obj in indexes.items():
            params_list = param_grids.get(idx_name, [{}])

            for params in params_list:
                for k in k_values:
                    current += 1
                    param_str = ", ".join(f"{p}={v}" for p, v in params.items()) or "default"
                    print(f"  [{current}/{total_configs}] {idx_name}({param_str}) K={k}")

                    # Measure latency (batch)
                    lat_stats = self.lat_profiler.profile_batch(
                        idx_obj, queries, k, repeat=self.repeat, **params
                    )

                    # Measure recall
                    rec_stats = self.rec_profiler.profile(
                        idx_obj, queries, ground_truth, k, **params
                    )

                    # Get the main tunable parameter value
                    if "nprobe" in params:
                        param_name = "nprobe"
                        param_value = params["nprobe"]
                    elif "ef_search" in params:
                        param_name = "ef_search"
                        param_value = params["ef_search"]
                    else:
                        param_name = "none"
                        param_value = 0

                    result = {
                        "index": idx_name,
                        "param_name": param_name,
                        "param_value": param_value,
                        "k": k,
                        "recall": rec_stats["recall_at_k"],
                        "recall_std": rec_stats["recall_std"],
                        "recall_min": rec_stats["recall_min"],
                        "recall_p5": rec_stats["recall_p5"],
                        "recall_p95": rec_stats["recall_p95"],
                        "latency_ms": lat_stats["mean_latency_ms"],
                        "qps": lat_stats["qps"],
                        "total_time_s": lat_stats["total_time_s"],
                    }
                    results.append(result)

                    print(f"    Recall@{k}={rec_stats['recall_at_k']:.4f}, "
                          f"Latency={lat_stats['mean_latency_ms']:.3f}ms, "
                          f"QPS={lat_stats['qps']:.0f}")

        df = pd.DataFrame(results)
        return df

    @staticmethod
    def build_param_grid(config: dict) -> Dict[str, List[dict]]:
        """
        Build parameter grid from config file settings.

        Returns dict like:
        {
            "Flat": [{}],
            "IVF": [{"nprobe": 1}, {"nprobe": 2}, ...],
            "HNSW": [{"ef_search": 16}, {"ef_search": 32}, ...],
        }
        """
        ivf_nprobes = config["profiling"]["ivf_nprobe_values"]
        hnsw_efs = config["profiling"]["hnsw_ef_search_values"]

        return {
            "Flat": [{}],
            "IVF": [{"nprobe": np} for np in ivf_nprobes],
            "HNSW": [{"ef_search": ef} for ef in hnsw_efs],
        }
