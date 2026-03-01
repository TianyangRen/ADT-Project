#!/usr/bin/env python3
"""
Phase 3: Adaptive Execution Evaluation
========================================
Compares the adaptive framework against static baselines and an oracle.

Workload types:
  - Uniform: all queries with same (budget, recall) constraints
  - Heterogeneous: queries drawn from a mix of SLA profiles
  - Sweep: vary budget continuously to map the adaptive Pareto curve

Baselines:
  - Static-Flat: always use Flat (exact) index
  - Static-IVF: always use IVF with fixed nprobe
  - Static-HNSW: always use HNSW with fixed ef_search
  - Oracle: pick per-query best from Phase 2 profiling data

Produces:
  - results/adaptive_evaluation.csv
  - results/adaptive_summary.txt
  - Console summary

Usage:
    python experiments/03_adaptive_evaluation.py
"""

import sys
import os
import time
import argparse
from collections import Counter
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indexes.flat_index import FlatIndex
from src.indexes.ivf_index import IVFIndex
from src.indexes.hnsw_index import HNSWIndex
from src.utils.io_utils import load_dataset, load_config, ensure_dir
from src.utils.metrics import compute_recall

from src.cost_model.cost_estimator import CostModel
from src.cost_model.cost_functions import AnalyticalCostModel
from src.adaptive.strategy_selector import StrategySelector, ExecutionStrategy
from src.adaptive.execution_engine import AdaptiveExecutionEngine
from src.monitor.performance_monitor import PerformanceMonitor


# ── SLA profiles for heterogeneous workload ───────────────────────────────
# Five diverse profiles whose recall floors force genuinely different
# strategies when the selector obeys "minimise latency s.t. recall ≥ X".
#
#   ultra_fast : recall ≥ 0.80  → HNSW(ef=16/32) or IVF(nprobe=1/4)
#   realtime   : recall ≥ 0.95  → HNSW(ef=64)
#   balanced   : recall ≥ 0.99  → HNSW(ef=256/512)
#   quality    : recall ≥ 0.999 → Flat  (only exact search guarantees this)
#   exact      : recall ≥ 1.0   → Flat
SLA_PROFILES = {
    "ultra_fast": {"latency_budget_ms": 0.3, "min_recall": 0.80},
    "realtime":   {"latency_budget_ms": 1.0, "min_recall": 0.95},
    "balanced":   {"latency_budget_ms": 5.0, "min_recall": 0.99},
    "quality":    {"latency_budget_ms": 50.0, "min_recall": 0.999},
    "exact":      {"latency_budget_ms": 100.0, "min_recall": 1.0},
}


# ── Index building ────────────────────────────────────────────────────────

def build_indexes(train_data, config, metric="L2"):
    """Build and return all three indexes."""
    dim = train_data.shape[1]
    idx_cfg = config["indexes"]

    print("Building indexes ...")
    flat = FlatIndex(dim, metric=metric)
    flat.build(train_data)

    ivf = IVFIndex(dim, nlist=idx_cfg["ivf"]["nlist"], metric=metric)
    ivf.build(train_data)

    hnsw = HNSWIndex(dim, M=idx_cfg["hnsw"]["M"],
                     ef_construction=idx_cfg["hnsw"]["ef_construction"],
                     metric=metric)
    hnsw.build(train_data)

    print("  All indexes built.\n")
    return {"Flat": flat, "IVF": ivf, "HNSW": hnsw}


# ── Static baselines ─────────────────────────────────────────────────────

def evaluate_static(index, index_name, queries, ground_truth,
                    k, search_kwargs=None):
    """Run a static-index baseline and return (recall, latencies)."""
    if search_kwargs is None:
        search_kwargs = {}
    nq = queries.shape[0]

    latencies = []
    all_I = []
    for i in range(nq):
        q = queries[i:i + 1]
        t0 = time.perf_counter()
        D, I = index.search(q, k, **search_kwargs)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        all_I.append(I[0])

    all_I = np.array(all_I)
    recall = compute_recall(all_I, ground_truth[:nq, :k], k)
    return recall, np.array(latencies)


# ── Oracle baseline ──────────────────────────────────────────────────────

def evaluate_oracle(indexes, queries, ground_truth, k,
                    candidates, latency_budget_ms, min_recall):
    """
    Oracle: for each query, try ALL candidates and pick using
    the same constraint cascade as selector, but with actual observed
    latency/recall instead of predictions.
    """
    nq = queries.shape[0]
    oracle_latencies = []
    oracle_recalls = []
    oracle_choices = []

    for i in range(nq):
        q = queries[i:i + 1]
        measured = []

        for c in candidates:
            if c.index_name not in indexes:
                continue
            idx = indexes[c.index_name]
            kwargs = dict(c.params)
            if "ef_search" in kwargs:
                kwargs["ef_search"] = max(kwargs["ef_search"], k)

            t0 = time.perf_counter()
            D, I = idx.search(q, k, **kwargs)
            lat = (time.perf_counter() - t0) * 1000.0

            rec_val = compute_recall(I, ground_truth[i:i+1, :k], k)

            measured.append((c, lat, rec_val, I[0]))

        # Constraint cascade on ACTUAL measurements
        fully_ok = [m for m in measured if m[2] >= min_recall and m[1] <= latency_budget_ms]
        recall_ok = [m for m in measured if m[2] >= min_recall]
        latency_ok = [m for m in measured if m[1] <= latency_budget_ms]

        if fully_ok:
            best_choice, best_lat, _, best_I = min(fully_ok, key=lambda x: x[1])
        elif recall_ok:
            best_choice, best_lat, _, best_I = min(recall_ok, key=lambda x: x[1])
        elif latency_ok:
            best_choice, best_lat, _, best_I = max(latency_ok, key=lambda x: x[2])
        else:
            best_choice, best_lat, _, best_I = max(measured, key=lambda x: x[2])

        oracle_latencies.append(best_lat)
        rec = compute_recall(best_I.reshape(1, -1), ground_truth[i:i+1, :k], k)
        oracle_recalls.append(rec)
        oracle_choices.append(str(best_choice))

    return np.mean(oracle_recalls), np.array(oracle_latencies), oracle_choices


# ── Adaptive evaluation ──────────────────────────────────────────────────

def evaluate_adaptive(engine, queries, ground_truth, k,
                      latency_budget_ms, min_recall, concurrency=1):
    """Run the adaptive engine and collect results."""
    nq = queries.shape[0]

    all_latencies = []
    all_I = []
    all_strategies = []

    for i in range(nq):
        result = engine.search(
            queries[i], top_k=k,
            latency_budget_ms=latency_budget_ms,
            min_recall=min_recall,
            concurrency=concurrency,
        )
        all_latencies.append(result.latency_ms)
        all_I.append(result.indices)
        all_strategies.append(str(result.strategy_used))

    all_I = np.array(all_I)
    recall = compute_recall(all_I, ground_truth[:nq, :k], k)
    return recall, np.array(all_latencies), all_strategies


# ── Heterogeneous workload ────────────────────────────────────────────────

def generate_heterogeneous_workload(nq, profiles=None):
    """
    Assign each query a random SLA profile with varying concurrency.

    Returns list of dicts with keys:
        latency_budget_ms, min_recall, profile_name, concurrency
    """
    if profiles is None:
        profiles = SLA_PROFILES

    names = list(profiles.keys())
    rng = np.random.RandomState(42)
    assignments = rng.choice(names, size=nq)
    # Random concurrency levels: mostly low, sometimes high
    concurrency_levels = rng.choice([1, 1, 1, 2, 4, 8], size=nq)

    workload = []
    for name, conc in zip(assignments, concurrency_levels):
        p = profiles[name]
        workload.append({
            "profile_name": name,
            "latency_budget_ms": p["latency_budget_ms"],
            "min_recall": p["min_recall"],
            "concurrency": int(conc),
        })
    return workload


def evaluate_adaptive_heterogeneous(engine, queries, ground_truth, k, workload):
    """Run adaptive engine with per-query SLA profiles."""
    nq = min(len(queries), len(workload))

    all_lats = []
    all_I = []
    all_strats = []
    budget_met = 0
    recall_met = 0

    for i in range(nq):
        w = workload[i]
        result = engine.search(
            queries[i], top_k=k,
            latency_budget_ms=w["latency_budget_ms"],
            min_recall=w["min_recall"],
            concurrency=w.get("concurrency", 1),
        )
        all_lats.append(result.latency_ms)
        all_I.append(result.indices)
        all_strats.append(str(result.strategy_used))

        if result.latency_ms <= w["latency_budget_ms"]:
            budget_met += 1

        per_q_recall = compute_recall(
            result.indices.reshape(1, -1), ground_truth[i:i+1, :k], k)
        if per_q_recall >= w["min_recall"]:
            recall_met += 1

    all_I = np.array(all_I)
    recall = compute_recall(all_I, ground_truth[:nq, :k], k)

    return {
        "recall": recall,
        "latencies": np.array(all_lats),
        "strategies": all_strats,
        "budget_met_frac": budget_met / nq,
        "recall_met_frac": recall_met / nq,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default_config.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--num_queries", type=int, default=None)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_name = args.dataset or config["dataset"]["name"]
    dataset_metric = config["dataset"].get("available", {}).get(
        dataset_name, {}).get("metric", "L2")
    dataset_dir = config["dataset"].get("data_dir", "data")

    # Load data
    print(f"Loading dataset: {dataset_name}")
    train, queries, gt, distances = load_dataset(dataset_name, dataset_dir)

    nq = args.num_queries or config["profiling"].get("num_queries", 500)
    nq = min(nq, queries.shape[0])
    queries = queries[:nq]
    k = args.k
    gt = gt[:nq]

    print(f"  train={train.shape}, queries={queries.shape}, k={k}\n")

    # Build indexes
    indexes = build_indexes(train, config, metric=dataset_metric)

    # ── Train cost model from Phase 2 profiling data ──
    profiling_path = "results/profiling_sweep.csv"
    if os.path.exists(profiling_path):
        print("Training learned cost model from profiling data ...")
        prof_df = pd.read_csv(profiling_path)
        cost_model = CostModel()
        cost_model.train(prof_df, dataset_size=train.shape[0])
        cost_model.save()
        print()
    else:
        print("No profiling data found; using analytical cost model.\n")
        cost_model = AnalyticalCostModel(train.shape[0], train.shape[1])

    # ── Define candidates ──
    candidates = StrategySelector.DEFAULT_CANDIDATES

    # ── Build adaptive engine ──
    engine = AdaptiveExecutionEngine(
        indexes=indexes,
        cost_model=cost_model,
        candidates=candidates,
        default_latency_budget_ms=5.0,
        default_min_recall=0.9,
        dataset_size=train.shape[0],
        dimension=train.shape[1],
    )

    results_rows = []
    ensure_dir("results")

    # ================================================================
    # Experiment 1: Per-profile demonstration
    #   Run the same queries under each SLA profile and show that
    #   the adaptive system picks DIFFERENT strategies.
    # ================================================================
    print("=" * 60)
    print("Experiment 1: Strategy switching across SLA profiles")
    print("=" * 60)

    demo_nq = min(100, nq)
    for pname, pcfg in SLA_PROFILES.items():
        engine.monitor.reset()
        rec, lats, strats = evaluate_adaptive(
            engine, queries[:demo_nq], gt[:demo_nq], k,
            latency_budget_ms=pcfg["latency_budget_ms"],
            min_recall=pcfg["min_recall"])
        dist = Counter(strats)
        dominant = dist.most_common(1)[0]
        row = {
            "experiment": f"per_profile_{pname}",
            "method": "Adaptive",
            "recall": rec,
            "mean_latency_ms": np.mean(lats),
            "p50_latency_ms": np.percentile(lats, 50),
            "p95_latency_ms": np.percentile(lats, 95),
            "budget_met_frac": np.mean(lats <= pcfg["latency_budget_ms"]),
        }
        results_rows.append(row)
        print(f"  [{pname:10s}] budget={pcfg['latency_budget_ms']:5.1f}ms, "
              f"min_recall={pcfg['min_recall']:.3f}  "
              f"→ recall={rec:.4f}, mean_lat={np.mean(lats):.3f}ms  "
              f"| strategy: {dominant[0]} ({dominant[1]}/{demo_nq})")

    print()

    # ================================================================
    # Experiment 2: Uniform workload (budget=5ms, recall >= 0.9)
    # ================================================================
    print("=" * 60)
    print("Experiment 2: Uniform workload (budget=5ms, recall>=0.9)")
    print("=" * 60)

    # Static baselines
    for name, idx, kwargs in [
        ("Static-Flat", indexes["Flat"], {}),
        ("Static-IVF(nprobe=16)", indexes["IVF"], {"nprobe": 16}),
        ("Static-HNSW(ef=64)", indexes["HNSW"], {"ef_search": max(64, k)}),
    ]:
        recall, lats = evaluate_static(idx, name, queries, gt, k, kwargs)
        row = {
            "experiment": "uniform",
            "method": name,
            "recall": recall,
            "mean_latency_ms": np.mean(lats),
            "p50_latency_ms": np.percentile(lats, 50),
            "p95_latency_ms": np.percentile(lats, 95),
            "budget_met_frac": np.mean(lats <= 5.0),
        }
        results_rows.append(row)
        print(f"  {name}: recall={recall:.4f}, "
              f"mean_lat={np.mean(lats):.3f}ms, "
              f"budget_met={row['budget_met_frac']:.1%}")

    # Adaptive
    recall_a, lats_a, strats_a = evaluate_adaptive(
        engine, queries, gt, k,
        latency_budget_ms=5.0, min_recall=0.9)
    row_adaptive = {
        "experiment": "uniform",
        "method": "Adaptive",
        "recall": recall_a,
        "mean_latency_ms": np.mean(lats_a),
        "p50_latency_ms": np.percentile(lats_a, 50),
        "p95_latency_ms": np.percentile(lats_a, 95),
        "budget_met_frac": np.mean(lats_a <= 5.0),
    }
    results_rows.append(row_adaptive)
    print(f"  Adaptive: recall={recall_a:.4f}, "
          f"mean_lat={np.mean(lats_a):.3f}ms, "
          f"budget_met={row_adaptive['budget_met_frac']:.1%}")

    # Strategy distribution
    print(f"\n  Adaptive strategy distribution:")
    dist = Counter(strats_a)
    for s, cnt in dist.most_common():
        print(f"    {s}: {cnt} ({cnt/len(strats_a):.1%})")

    print()

    # ================================================================
    # Experiment 3: Heterogeneous workload
    # ================================================================
    print("=" * 60)
    print("Experiment 3: Heterogeneous workload (mixed SLA profiles)")
    print("=" * 60)

    workload = generate_heterogeneous_workload(nq)
    profile_counts = Counter(w["profile_name"] for w in workload)
    print(f"  Profile distribution: {dict(profile_counts)}")

    # Adaptive on heterogeneous workload
    engine.monitor.reset()
    het_result = evaluate_adaptive_heterogeneous(
        engine, queries, gt, k, workload)

    row_het = {
        "experiment": "heterogeneous",
        "method": "Adaptive",
        "recall": het_result["recall"],
        "mean_latency_ms": np.mean(het_result["latencies"]),
        "p50_latency_ms": np.percentile(het_result["latencies"], 50),
        "p95_latency_ms": np.percentile(het_result["latencies"], 95),
        "budget_met_frac": het_result["budget_met_frac"],
    }
    results_rows.append(row_het)
    print(f"  Adaptive: recall={het_result['recall']:.4f}, "
          f"mean_lat={np.mean(het_result['latencies']):.3f}ms")
    print(f"  Budget met: {het_result['budget_met_frac']:.1%}, "
          f"Recall met: {het_result['recall_met_frac']:.1%}")

    # Static baselines on heterogeneous workload
    for name, idx, kwargs in [
        ("Static-Flat", indexes["Flat"], {}),
        ("Static-IVF(nprobe=16)", indexes["IVF"], {"nprobe": 16}),
        ("Static-HNSW(ef=64)", indexes["HNSW"], {"ef_search": max(64, k)}),
    ]:
        recall, lats = evaluate_static(idx, name, queries, gt, k, kwargs)
        # Count how many queries meet their individual budget
        budget_met = sum(1 for i, w in enumerate(workload[:nq])
                         if lats[i] <= w["latency_budget_ms"]) / nq
        row = {
            "experiment": "heterogeneous",
            "method": name,
            "recall": recall,
            "mean_latency_ms": np.mean(lats),
            "p50_latency_ms": np.percentile(lats, 50),
            "p95_latency_ms": np.percentile(lats, 95),
            "budget_met_frac": budget_met,
        }
        results_rows.append(row)
        print(f"  {name}: recall={recall:.4f}, "
              f"mean_lat={np.mean(lats):.3f}ms, "
              f"budget_met={budget_met:.1%}")

    het_dist = Counter(het_result["strategies"])
    print(f"\n  Adaptive het. strategy distribution:")
    for s, cnt in het_dist.most_common():
        print(f"    {s}: {cnt} ({cnt/nq:.1%})")

    print()

    # ================================================================
    # Experiment 4: Recall requirement sweep
    #   Fix budget=50ms, sweep min_recall from 0.5 to 1.0
    #   Shows the selector transitioning through different strategies.
    # ================================================================
    print("=" * 60)
    print("Experiment 4: Recall requirement sweep (budget=50ms)")
    print("=" * 60)

    sweep_nq = min(200, nq)
    recall_thresholds = [0.50, 0.80, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0]

    for mr in recall_thresholds:
        engine.monitor.reset()
        rec, lats, strats = evaluate_adaptive(
            engine, queries[:sweep_nq], gt[:sweep_nq], k,
            latency_budget_ms=50.0, min_recall=mr)
        dist = Counter(strats)
        dominant = dist.most_common(1)[0]
        row = {
            "experiment": f"sweep_recall_{mr}",
            "method": "Adaptive",
            "recall": rec,
            "mean_latency_ms": np.mean(lats),
            "p50_latency_ms": np.percentile(lats, 50),
            "p95_latency_ms": np.percentile(lats, 95),
            "budget_met_frac": np.mean(lats <= 50.0),
        }
        results_rows.append(row)
        print(f"  min_recall={mr:.3f} → recall={rec:.4f}, "
              f"mean_lat={np.mean(lats):.3f}ms  "
              f"| {dominant[0]} ({dominant[1]}/{sweep_nq})")

    print()

    # ================================================================
    # Experiment 5: Latency budget sweep (min_recall=0.95)
    #   Shows the selector switching strategies as budget tightens.
    # ================================================================
    print("=" * 60)
    print("Experiment 5: Latency budget sweep (min_recall=0.95)")
    print("=" * 60)

    budgets = [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 50.0]

    for budget in budgets:
        engine.monitor.reset()
        rec, lats, strats = evaluate_adaptive(
            engine, queries[:sweep_nq], gt[:sweep_nq], k,
            latency_budget_ms=budget, min_recall=0.95)
        dist = Counter(strats)
        dominant = dist.most_common(1)[0]
        row = {
            "experiment": f"sweep_budget_{budget}",
            "method": "Adaptive",
            "recall": rec,
            "mean_latency_ms": np.mean(lats),
            "p50_latency_ms": np.percentile(lats, 50),
            "p95_latency_ms": np.percentile(lats, 95),
            "budget_met_frac": np.mean(lats <= budget),
        }
        results_rows.append(row)
        print(f"  budget={budget:6.2f}ms → recall={rec:.4f}, "
              f"mean_lat={np.mean(lats):.3f}ms, "
              f"budget_met={row['budget_met_frac']:.1%}  "
              f"| {dominant[0]} ({dominant[1]}/{sweep_nq})")

    print()

    # ================================================================
    # Experiment 6: Concurrency sweep (min_recall=0.95, budget=5ms)
    #   Under concurrent load, HNSW's random graph traversal degrades
    #   due to cache thrashing, while IVF's sequential scan remains
    #   stable. Shows IVF being selected under high concurrency.
    # ================================================================
    print("=" * 60)
    print("Experiment 6: Concurrency sweep (min_recall=0.95, budget=5ms)")
    print("=" * 60)

    conc_levels = [1, 2, 3, 4, 6, 8, 12, 16]

    for conc in conc_levels:
        engine.monitor.reset()
        rec, lats, strats = evaluate_adaptive(
            engine, queries[:sweep_nq], gt[:sweep_nq], k,
            latency_budget_ms=5.0, min_recall=0.95, concurrency=conc)
        dist = Counter(strats)
        strat_summary = ", ".join(f"{s.split('(')[1][:-1]}={c}"
                                  for s, c in dist.most_common(3))
        row = {
            "experiment": f"sweep_conc_{conc}",
            "method": "Adaptive",
            "recall": rec,
            "mean_latency_ms": np.mean(lats),
            "p50_latency_ms": np.percentile(lats, 50),
            "p95_latency_ms": np.percentile(lats, 95),
            "budget_met_frac": np.mean(lats <= 5.0),
        }
        results_rows.append(row)
        dominant = dist.most_common(1)[0]
        print(f"  conc={conc:2d} → recall={rec:.4f}, "
              f"mean_lat={np.mean(lats):.3f}ms  "
              f"| {dominant[0]} ({dominant[1]}/{sweep_nq})")

    print()

    # ── Monitor summary ──
    stats = engine.get_stats()
    print("=" * 60)
    print("Performance Monitor Summary")
    print("=" * 60)
    print(f"  Total queries processed: {stats['total_queries']}")
    err = stats["prediction_error"]
    print(f"  Prediction MAE: {err['mae']:.3f}ms  "
          f"(mean error: {err['mean_error']:.3f}ms)")
    lat = stats["latency_stats"]
    print(f"  Actual latency: mean={lat['mean']:.3f}ms, "
          f"p50={lat['p50']:.3f}ms, p95={lat['p95']:.3f}ms")
    print(f"  Needs recalibration: {stats['needs_recalibration']}")
    print(f"  Selection distribution:")
    for s, frac in stats["selection_distribution"].items():
        print(f"    {s}: {frac:.1%}")

    # ── Save results ──
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv("results/adaptive_evaluation.csv", index=False)
    print(f"\nResults saved to results/adaptive_evaluation.csv")

    # Save text summary
    summary_lines = [
        "Phase 3: Adaptive Execution Evaluation Summary",
        "=" * 50,
        f"Dataset: {dataset_name}",
        f"Queries: {nq}, K: {k}",
        "",
        "--- Per-Profile Strategy Switching ---",
    ]
    profile_rows = [r for r in results_rows if r["experiment"].startswith("per_profile_")]
    for r in profile_rows:
        summary_lines.append(
            f"  {r['experiment']}: recall={r['recall']:.4f}, "
            f"lat={r['mean_latency_ms']:.3f}ms")
    summary_lines.append("")
    summary_lines.append("--- Uniform Workload (budget=5ms, recall>=0.9) ---")
    uniform_rows = [r for r in results_rows if r["experiment"] == "uniform"]
    for r in uniform_rows:
        summary_lines.append(
            f"  {r['method']}: recall={r['recall']:.4f}, "
            f"lat={r['mean_latency_ms']:.3f}ms, budget_met={r['budget_met_frac']:.1%}")
    summary_lines.append("")
    summary_lines.append("--- Heterogeneous Workload ---")
    het_rows = [r for r in results_rows if r["experiment"] == "heterogeneous"]
    for r in het_rows:
        summary_lines.append(
            f"  {r['method']}: recall={r['recall']:.4f}, "
            f"lat={r['mean_latency_ms']:.3f}ms, budget_met={r['budget_met_frac']:.1%}")

    with open("results/adaptive_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"Summary saved to results/adaptive_summary.txt\n")

    print("Phase 3 evaluation complete.")


if __name__ == "__main__":
    main()
