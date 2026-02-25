#!/usr/bin/env python3
"""
Phase 1: Baseline Benchmark
============================
Build Flat, IVF, HNSW indexes on the dataset and measure their performance
under default configurations.

Produces:
  - results/baseline_results.csv
  - Console summary table

Usage:
    python experiments/01_baseline_benchmark.py
    python experiments/01_baseline_benchmark.py --dataset glove-100-angular
"""

import sys
import os
import time
import argparse
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indexes.flat_index import FlatIndex
from src.indexes.ivf_index import IVFIndex
from src.indexes.hnsw_index import HNSWIndex
from src.utils.metrics import compute_recall, compute_latency_stats, compute_qps
from src.utils.io_utils import load_dataset, load_config, ensure_dir, save_results_csv


def benchmark_single(index, queries, ground_truth, k, warmup=50, **search_params):
    """
    Benchmark a single index configuration.

    Returns dict with recall, latency stats, and QPS.
    """
    nq = queries.shape[0]

    # Warmup: run a few queries to stabilize caches
    if warmup > 0:
        index.search(queries[:warmup], k, **search_params)

    # Timed run: measure per-query latency
    latencies = []
    all_indices = []
    for i in range(nq):
        q = queries[i:i+1]
        start = time.perf_counter()
        _, idx = index.search(q, k, **search_params)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed * 1000)  # ms
        all_indices.append(idx[0])

    latencies = np.array(latencies)
    all_indices = np.array(all_indices)

    # Also measure batch throughput
    batch_start = time.perf_counter()
    _, batch_indices = index.search(queries, k, **search_params)
    batch_time = time.perf_counter() - batch_start

    # Compute recall using batch results (more stable)
    gt_k = ground_truth[:, :k]
    recall = compute_recall(batch_indices, gt_k, k)
    lat_stats = compute_latency_stats(latencies)
    qps = compute_qps(nq, batch_time)

    return {
        "recall": recall,
        "qps": qps,
        "batch_time_s": batch_time,
        **lat_stats,
    }


def run_baseline_benchmark(dataset_name="sift-128-euclidean", data_dir="data"):
    """Run the full Phase 1 baseline benchmark."""

    print("=" * 70)
    print("PHASE 1: Baseline Benchmark")
    print("=" * 70)

    # --- Load Dataset ---
    base, queries, ground_truth, gt_distances = load_dataset(dataset_name, data_dir)
    n, dim = base.shape
    nq = queries.shape[0]
    print(f"\nDataset: {dataset_name}")
    print(f"  {n:,} base vectors, {nq:,} queries, {dim}d\n")

    # --- Load Config ---
    config = load_config()
    ivf_nlist = config["indexes"]["ivf"]["nlist"]
    ivf_nprobe = config["indexes"]["ivf"]["default_nprobe"]
    hnsw_M = config["indexes"]["hnsw"]["M"]
    hnsw_ef_con = config["indexes"]["hnsw"]["ef_construction"]
    hnsw_ef_search = config["indexes"]["hnsw"]["default_ef_search"]

    # --- Build All Indexes ---
    print("-" * 70)
    print("Building indexes...")
    print("-" * 70)

    flat = FlatIndex(dim, metric="L2")
    flat.build(base)

    ivf = IVFIndex(dim, nlist=ivf_nlist, metric="L2")
    ivf.build(base)

    hnsw = HNSWIndex(dim, M=hnsw_M, ef_construction=hnsw_ef_con, metric="L2")
    hnsw.build(base)

    # --- Benchmark At Multiple K Values ---
    k_values = config["profiling"]["k_values"]
    results = []

    indexes_to_test = [
        ("Flat", flat, {}),
        ("IVF", ivf, {"nprobe": ivf_nprobe}),
        ("HNSW", hnsw, {"ef_search": hnsw_ef_search}),
    ]

    for k in k_values:
        print(f"\n{'=' * 70}")
        print(f"Benchmarking at K={k}")
        print(f"{'=' * 70}")

        for idx_name, idx_obj, search_params in indexes_to_test:
            param_str = ", ".join(f"{p}={v}" for p, v in search_params.items())
            print(f"\n  [{idx_name}] params=({param_str})")

            stats = benchmark_single(idx_obj, queries, ground_truth, k,
                                     warmup=100, **search_params)

            result = {
                "dataset": dataset_name,
                "index": idx_name,
                "k": k,
                "params": str(search_params),
                "build_time_s": idx_obj.build_time,
                **stats,
            }
            results.append(result)

            print(f"    Recall@{k}: {stats['recall']:.4f}")
            print(f"    Mean latency: {stats['mean_ms']:.3f} ms")
            print(f"    P99 latency:  {stats['p99_ms']:.3f} ms")
            print(f"    QPS (batch):  {stats['qps']:.0f}")

    # --- Save Results ---
    ensure_dir("results")
    df = save_results_csv(results, "results/baseline_results.csv")

    # --- Print Summary Table ---
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    summary_cols = ["index", "k", "recall", "mean_ms", "p99_ms", "qps"]
    print(df[summary_cols].to_string(index=False, float_format="%.4f"))

    print(f"\nResults saved to: results/baseline_results.csv")
    return df


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Baseline Benchmark")
    parser.add_argument("--dataset", type=str, default="sift-128-euclidean",
                        help="Dataset name (default: sift-128-euclidean)")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory containing HDF5 datasets")
    args = parser.parse_args()

    run_baseline_benchmark(args.dataset, args.data_dir)


if __name__ == "__main__":
    main()
