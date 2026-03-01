#!/usr/bin/env python3
"""
Phase 2: Systematic Profiling & Static Index Failure Analysis
==============================================================
Sweep over all (index, params, K) combinations to produce empirical
evidence that no single static index configuration is optimal across
all query regimes.

Produces:
  - results/profiling_sweep.csv        (raw profiling data)
  - results/figures/*.png              (visualization plots)
  - Console analysis of failure modes

Usage:
    python experiments/02_profiling_analysis.py
    python experiments/02_profiling_analysis.py --dataset glove-100-angular
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indexes.flat_index import FlatIndex
from src.indexes.ivf_index import IVFIndex
from src.indexes.hnsw_index import HNSWIndex
from src.profiler.profile_runner import ProfileRunner
from src.utils.io_utils import load_dataset, load_config, ensure_dir, save_results_csv


def build_indexes(base: np.ndarray, dim: int, config: dict, metric: str = "L2"):
    """Build all three index types."""
    print("=" * 70)
    print("Building indexes...")
    print("=" * 70)

    flat = FlatIndex(dim, metric=metric)
    flat.build(base)

    ivf = IVFIndex(dim, nlist=config["indexes"]["ivf"]["nlist"], metric=metric)
    ivf.build(base)

    hnsw = HNSWIndex(
        dim,
        M=config["indexes"]["hnsw"]["M"],
        ef_construction=config["indexes"]["hnsw"]["ef_construction"],
        metric=metric,
    )
    hnsw.build(base)

    return {"Flat": flat, "IVF": ivf, "HNSW": hnsw}


def analyze_failure_modes(df: pd.DataFrame):
    """
    Analyze the profiling data to identify static index failure modes.
    This is the KEY CONTRIBUTION of Phase 2.
    """
    print("\n" + "=" * 70)
    print("STATIC INDEX FAILURE MODE ANALYSIS")
    print("=" * 70)

    # --- Analysis 1: Best index per (K, recall_threshold) ---
    print("\n--- Analysis 1: Best Index Per Query Regime ---")
    print("(Which index has lowest latency while achieving >= recall threshold?)\n")

    for recall_threshold in [0.90, 0.95, 0.99]:
        print(f"  Recall threshold >= {recall_threshold}:")
        for k in sorted(df["k"].unique()):
            subset = df[(df["k"] == k) & (df["recall"] >= recall_threshold)]
            if subset.empty:
                print(f"    K={k:>3d}: NO index achieves recall >= {recall_threshold}")
                continue
            best = subset.loc[subset["latency_ms"].idxmin()]
            print(f"    K={k:>3d}: Best = {best['index']:>4s}"
                  f"(param={best['param_value']:>3.0f}), "
                  f"recall={best['recall']:.4f}, "
                  f"latency={best['latency_ms']:.3f}ms")
        print()

    # --- Analysis 2: No single config wins everywhere ---
    print("--- Analysis 2: Static Config Failure Cases ---")
    print("(For each 'reasonable' static config, show where it fails)\n")

    static_configs = [
        ("IVF", "nprobe", 16),
        ("IVF", "nprobe", 64),
        ("HNSW", "ef_search", 64),
        ("HNSW", "ef_search", 256),
    ]

    for idx_name, pname, pval in static_configs:
        config_df = df[(df["index"] == idx_name) & (df["param_value"] == pval)]
        if config_df.empty:
            continue
        print(f"  Static config: {idx_name}({pname}={pval})")
        for _, row in config_df.iterrows():
            status = "OK" if row["recall"] >= 0.95 else "FAIL (recall < 0.95)"
            print(f"    K={row['k']:>3d}: recall={row['recall']:.4f}, "
                  f"latency={row['latency_ms']:.3f}ms  [{status}]")
        print()

    # --- Analysis 3: Pareto dominance ---
    print("--- Analysis 3: Pareto-Optimal Strategies ---")
    print("(Strategies on the Pareto front of latency vs recall)\n")

    for k in sorted(df["k"].unique()):
        k_df = df[df["k"] == k].copy()
        # Find Pareto front: no other point has both lower latency AND higher recall
        pareto = []
        for _, row in k_df.iterrows():
            dominated = False
            for _, other in k_df.iterrows():
                if (other["latency_ms"] <= row["latency_ms"] and
                    other["recall"] >= row["recall"] and
                    (other["latency_ms"] < row["latency_ms"] or
                     other["recall"] > row["recall"])):
                    dominated = True
                    break
            if not dominated:
                pareto.append(row)

        print(f"  K={k}: Pareto front ({len(pareto)} points):")
        for p in sorted(pareto, key=lambda x: x["latency_ms"]):
            print(f"    {p['index']:>4s}(param={p['param_value']:>3.0f}): "
                  f"recall={p['recall']:.4f}, latency={p['latency_ms']:.3f}ms")
        print()

    # --- Analysis 4: Crossover points ---
    print("--- Analysis 4: Index Crossover Points ---")
    print("(Where does one index become better than another?)\n")

    for k in sorted(df["k"].unique()):
        k_df = df[df["k"] == k]

        # Best IVF config at this K (for recall >= 0.9)
        ivf_good = k_df[(k_df["index"] == "IVF") & (k_df["recall"] >= 0.9)]
        hnsw_good = k_df[(k_df["index"] == "HNSW") & (k_df["recall"] >= 0.9)]

        if not ivf_good.empty and not hnsw_good.empty:
            best_ivf = ivf_good.loc[ivf_good["latency_ms"].idxmin()]
            best_hnsw = hnsw_good.loc[hnsw_good["latency_ms"].idxmin()]
            winner = "IVF" if best_ivf["latency_ms"] < best_hnsw["latency_ms"] else "HNSW"
            print(f"  K={k:>3d} (recall>=0.9): "
                  f"IVF={best_ivf['latency_ms']:.3f}ms vs "
                  f"HNSW={best_hnsw['latency_ms']:.3f}ms → Winner: {winner}")

    return


def run_profiling(dataset_name="sift-128-euclidean", data_dir="data"):
    """Run the full Phase 2 profiling pipeline."""

    print("=" * 70)
    print("PHASE 2: Systematic Performance Profiling")
    print("=" * 70)

    # --- Load ---
    config = load_config()
    dataset_metric = config["dataset"].get("available", {}).get(
        dataset_name, {}).get("metric", "L2")
    base, queries, ground_truth, _ = load_dataset(dataset_name, data_dir)
    n, dim = base.shape

    # Limit query count if configured
    max_queries = config["profiling"]["num_queries"]
    if max_queries > 0 and max_queries < len(queries):
        queries = queries[:max_queries]
        ground_truth = ground_truth[:max_queries]

    # --- Build ---
    indexes = build_indexes(base, dim, config, metric=dataset_metric)

    # --- Sweep ---
    print("\n" + "=" * 70)
    print("Running profiling sweep...")
    print("=" * 70)

    runner = ProfileRunner(
        warmup=config["profiling"]["warmup_queries"],
        repeat=config["profiling"]["repeat"],
    )
    param_grid = runner.build_param_grid(config)
    k_values = config["profiling"]["k_values"]

    df = runner.run_sweep(indexes, queries, ground_truth, k_values, param_grid)

    # Add dataset info
    df["dataset"] = dataset_name
    df["dataset_size"] = n
    df["dimension"] = dim

    # --- Save ---
    ensure_dir("results")
    save_results_csv(df.to_dict("records"), "results/profiling_sweep.csv")

    # --- Analyze ---
    analyze_failure_modes(df)

    # --- Generate Visualizations ---
    print("\nGenerating visualizations...")
    try:
        from experiments.visualize_profiling import generate_all_plots
        generate_all_plots(df)
    except ImportError:
        print("  (Run experiments/visualize_profiling.py separately for plots)")

    print("\n" + "=" * 70)
    print("Phase 2 Complete!")
    print(f"  Profiling data: results/profiling_sweep.csv")
    print(f"  {len(df)} configurations profiled")
    print("=" * 70)

    return df


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Profiling Analysis")
    parser.add_argument("--dataset", type=str, default="sift-128-euclidean")
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    run_profiling(args.dataset, args.data_dir)


if __name__ == "__main__":
    main()
