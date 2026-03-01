#!/usr/bin/env python3
"""
Phase 1: Baseline Benchmark (TensorBoard Version)
=================================================
Original script preserved separately.
This version adds full TensorBoard logging.

Produces:
  - results/baseline_results.csv
  - TensorBoard logs in: runs/<run_name>

Usage:
    python experiments/01_baseline_benchmark_tensorboard.py
    python experiments/01_baseline_benchmark_tensorboard.py --dataset glove-100-angular
    tensorboard --logdir runs
"""

import sys
import os
import time
import argparse
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indexes.flat_index import FlatIndex
from src.indexes.ivf_index import IVFIndex
from src.indexes.hnsw_index import HNSWIndex
from src.utils.metrics import compute_recall, compute_latency_stats, compute_qps
from src.utils.io_utils import load_dataset, load_config, ensure_dir, save_results_csv


# -----------------------------
# TensorBoard Writer
# -----------------------------
def get_tb_writer(log_dir):
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir=log_dir)
    except Exception:
        try:
            from tensorboardX import SummaryWriter
            return SummaryWriter(log_dir=log_dir)
        except Exception:
            raise RuntimeError(
                "TensorBoard not installed.\n"
                "Install one of:\n"
                "  pip install torch tensorboard\n"
                "  pip install tensorboardX tensorboard"
            )


def benchmark_single(index, queries, ground_truth, k,
                     warmup=50, per_query_limit=1000,
                     return_latencies=False, **search_params):

    nq = queries.shape[0]

    if warmup > 0:
        index.search(queries[:min(warmup, nq)], k, **search_params)

    # Per-query latency
    n_per_query = min(nq, per_query_limit)
    latencies = []

    for i in range(n_per_query):
        q = queries[i:i+1]
        start = time.perf_counter()
        index.search(q, k, **search_params)
        latencies.append((time.perf_counter() - start) * 1000)

    latencies = np.array(latencies)

    # Batch run
    batch_start = time.perf_counter()
    _, batch_indices = index.search(queries, k, **search_params)
    batch_time = time.perf_counter() - batch_start

    gt_k = ground_truth[:, :k]
    recall = compute_recall(batch_indices, gt_k, k)
    lat_stats = compute_latency_stats(latencies)
    qps = compute_qps(nq, batch_time)

    result = {
        "recall": recall,
        "qps": qps,
        "batch_time_s": batch_time,
        **lat_stats
    }

    if return_latencies:
        result["latencies_ms"] = latencies

    return result


def run_baseline_benchmark(dataset_name="sift-128-euclidean",
                           data_dir="data",
                           log_dir="runs",
                           run_name=None):

    print("=" * 70)
    print("PHASE 1: Baseline Benchmark (TensorBoard Version)")
    print("=" * 70)

    # ---- TensorBoard setup ----
    if run_name is None:
        run_name = f"baseline_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    tb_dir = os.path.join(log_dir, run_name)
    ensure_dir(tb_dir)
    writer = get_tb_writer(tb_dir)

    # ---- Load Dataset ----
    base, queries, ground_truth, gt_distances = load_dataset(dataset_name, data_dir)
    n, dim = base.shape
    nq = queries.shape[0]

    print(f"\nDataset: {dataset_name}")
    print(f"  {n:,} base vectors, {nq:,} queries, {dim}d\n")

    writer.add_text("meta/dataset", dataset_name, 0)
    writer.add_text("meta/shape", f"base={n}x{dim}, queries={nq}", 0)

    # ---- Config ----
    config = load_config()
    ivf_nlist = config["indexes"]["ivf"]["nlist"]
    ivf_nprobe = config["indexes"]["ivf"]["default_nprobe"]
    hnsw_M = config["indexes"]["hnsw"]["M"]
    hnsw_ef_con = config["indexes"]["hnsw"]["ef_construction"]
    hnsw_ef_search = config["indexes"]["hnsw"]["default_ef_search"]
    k_values = config["profiling"]["k_values"]

    # ---- Build Indexes ----
    print("Building indexes...\n")

    flat = FlatIndex(dim, metric="L2")
    flat.build(base)
    writer.add_scalar("build/Flat_time_s", flat.build_time, 0)

    ivf = IVFIndex(dim, nlist=ivf_nlist, metric="L2")
    ivf.build(base)
    writer.add_scalar("build/IVF_time_s", ivf.build_time, 0)

    hnsw = HNSWIndex(dim, M=hnsw_M,
                     ef_construction=hnsw_ef_con,
                     metric="L2")
    hnsw.build(base)
    writer.add_scalar("build/HNSW_time_s", hnsw.build_time, 0)

    # ---- Benchmark ----
    results = []

    indexes = [
        ("Flat", flat, {}, 200),
        ("IVF", ivf, {"nprobe": ivf_nprobe}, 1000),
        ("HNSW", hnsw, {"ef_search": hnsw_ef_search}, 1000),
    ]

    for k in k_values:
        print(f"\nBenchmarking K={k}\n")

        for name, index_obj, params, pq_limit in indexes:

            stats = benchmark_single(
                index_obj,
                queries,
                ground_truth,
                k,
                warmup=100,
                per_query_limit=pq_limit,
                return_latencies=True,
                **params
            )

            print(f"[{name}] Recall@{k}: {stats['recall']:.4f}")
            print(f"[{name}] Mean latency: {stats['mean_ms']:.3f} ms")
            print(f"[{name}] QPS: {stats['qps']:.0f}\n")

            # ---- TensorBoard logging ----
            step = k
            group = name

            writer.add_scalar(f"{group}/recall", stats["recall"], step)
            writer.add_scalar(f"{group}/latency_mean_ms", stats["mean_ms"], step)
            writer.add_scalar(f"{group}/latency_p99_ms", stats["p99_ms"], step)
            writer.add_scalar(f"{group}/qps", stats["qps"], step)

            writer.add_histogram(f"{group}/latency_hist", stats["latencies_ms"], step)

            results.append({
                "dataset": dataset_name,
                "index": name,
                "k": k,
                "params": str(params),
                "build_time_s": index_obj.build_time,
                **stats
            })

    ensure_dir("results")
    save_results_csv(results, "results/baseline_results.csv")

    writer.close()

    print("\nFinished.")
    print(f"TensorBoard logs saved to: {tb_dir}")
    print("Run: tensorboard --logdir runs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="sift-128-euclidean")
    parser.add_argument("--data-dir", type=str,
                        default="data")
    parser.add_argument("--log-dir", type=str,
                        default="runs")
    parser.add_argument("--run-name", type=str,
                        default=None)

    args = parser.parse_args()

    run_baseline_benchmark(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        log_dir=args.log_dir,
        run_name=args.run_name
    )


if __name__ == "__main__":
    main()
