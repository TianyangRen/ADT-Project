#!/usr/bin/env python3
"""
Phase 2: Systematic Profiling & Static Index Failure Analysis (TensorBoard)
===========================================================================
TensorBoard-enabled version of experiments/02_profiling_analysis.py.

Produces:
  - results/profiling_sweep.csv
  - TensorBoard logs in: runs/<run_name>
  - Console analysis (same as original)

Usage:
    python experiments/02_profiling_analysis_tensorboard.py
    python experiments/02_profiling_analysis_tensorboard.py --dataset glove-100-angular
    tensorboard --logdir runs
"""

import sys
import os
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indexes.flat_index import FlatIndex
from src.indexes.ivf_index import IVFIndex
from src.indexes.hnsw_index import HNSWIndex
from src.profiler.profile_runner import ProfileRunner
from src.utils.io_utils import load_dataset, load_config, ensure_dir, save_results_csv


# -----------------------------
# TensorBoard Writer (robust)
# -----------------------------
def get_tb_writer(log_dir: str):
    """
    Return a TensorBoard SummaryWriter with best-effort imports:
      1) torch.utils.tensorboard.SummaryWriter
      2) tensorboardX.SummaryWriter
    """
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore
        return SummaryWriter(log_dir=log_dir)
    except Exception:
        try:
            from tensorboardX import SummaryWriter  # type: ignore
            return SummaryWriter(log_dir=log_dir)
        except Exception as e:
            raise RuntimeError(
                "TensorBoard writer not available.\n"
                "Install ONE of the following:\n"
                "  - pip install torch tensorboard\n"
                "  - pip install tensorboardX tensorboard\n"
                f"Original import error: {repr(e)}"
            )


def build_indexes(base: np.ndarray, dim: int, config: dict):
    """Build all three index types."""
    print("=" * 70)
    print("Building indexes...")
    print("=" * 70)

    flat = FlatIndex(dim, metric="L2")
    flat.build(base)

    ivf = IVFIndex(dim, nlist=config["indexes"]["ivf"]["nlist"], metric="L2")
    ivf.build(base)

    hnsw = HNSWIndex(
        dim,
        M=config["indexes"]["hnsw"]["M"],
        ef_construction=config["indexes"]["hnsw"]["ef_construction"],
        metric="L2",
    )
    hnsw.build(base)

    return {"Flat": flat, "IVF": ivf, "HNSW": hnsw}


def _require_cols(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    return missing


def _as_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def log_sweep_to_tensorboard(df: pd.DataFrame, writer, tag_prefix: str = ""):
    """
    Log sweep results to TensorBoard.

    Expected (best effort) columns:
      - index, k, recall, latency_ms
      - param_name (optional), param_value (optional)
      - qps (optional)
    """
    # Core columns we need for meaningful plots
    missing_core = _require_cols(df, ["index", "k", "recall", "latency_ms"])
    if missing_core:
        writer.add_text(
            f"{tag_prefix}errors/missing_columns",
            f"Cannot log sweep: missing columns: {missing_core}. Available: {list(df.columns)}",
            0
        )
        return

    # Log scalar series per index and per K:
    # We'll treat "param_value" as step if present; otherwise use row index.
    has_param_val = "param_value" in df.columns
    has_param_name = "param_name" in df.columns
    has_qps = "qps" in df.columns

    # Make sure numeric
    df2 = df.copy()
    df2["k"] = pd.to_numeric(df2["k"], errors="coerce")
    df2["recall"] = pd.to_numeric(df2["recall"], errors="coerce")
    df2["latency_ms"] = pd.to_numeric(df2["latency_ms"], errors="coerce")
    if has_param_val:
        df2["param_value"] = pd.to_numeric(df2["param_value"], errors="coerce")
    if has_qps:
        df2["qps"] = pd.to_numeric(df2["qps"], errors="coerce")

    # Sort for nicer curves
    sort_cols = ["index", "k"]
    if has_param_val:
        sort_cols.append("param_value")
    df2 = df2.sort_values(sort_cols)

    # Log summary stats per (index, k)
    for idx_name in sorted(df2["index"].dropna().unique()):
        idx_df = df2[df2["index"] == idx_name]
        for k in sorted(idx_df["k"].dropna().unique()):
            sub = idx_df[idx_df["k"] == k].copy()
            if sub.empty:
                continue

            # Choose step axis: param_value if exists, else 0..N-1
            if has_param_val:
                steps = sub["param_value"].values
            else:
                steps = np.arange(len(sub), dtype=np.int64)

            # Tag base
            k_tag = f"{tag_prefix}{idx_name}/K_{int(k)}"

            # Scalars over sweep
            for step, r, lat in zip(steps, sub["recall"].values, sub["latency_ms"].values):
                s = int(step) if np.isfinite(step) else 0
                writer.add_scalar(f"{k_tag}/recall", _as_float(r), s)
                writer.add_scalar(f"{k_tag}/latency_ms", _as_float(lat), s)
                if has_qps:
                    writer.add_scalar(f"{k_tag}/qps", _as_float(sub.loc[sub.index[sub['recall'].index(step) if False else sub.index[0]], "qps"]) if False else _as_float(sub["qps"].iloc[list(steps).index(step)]), s)

            # Best-by-latency under recall thresholds (single-point scalars)
            for thr in [0.90, 0.95, 0.99]:
                good = sub[sub["recall"] >= thr]
                if good.empty:
                    writer.add_scalar(f"{k_tag}/best_latency_ms_recall_ge_{thr}", float("nan"), 0)
                    continue
                best_row = good.loc[good["latency_ms"].idxmin()]
                writer.add_scalar(f"{k_tag}/best_latency_ms_recall_ge_{thr}", _as_float(best_row["latency_ms"]), 0)
                writer.add_scalar(f"{k_tag}/best_recall_recall_ge_{thr}", _as_float(best_row["recall"]), 0)
                if has_param_val:
                    writer.add_scalar(f"{k_tag}/best_param_value_recall_ge_{thr}", _as_float(best_row["param_value"]), 0)

    # Text: show what param is being swept per index (if available)
    if has_param_name:
        for idx_name in sorted(df2["index"].dropna().unique()):
            names = sorted(df2[df2["index"] == idx_name]["param_name"].dropna().unique())
            if names:
                writer.add_text(f"{tag_prefix}{idx_name}/swept_params", ", ".join(names), 0)

    # Text: Pareto front summary per K (global)
    pareto_lines = []
    for k in sorted(df2["k"].dropna().unique()):
        k_df = df2[df2["k"] == k].copy()
        if k_df.empty:
            continue
        pareto = []
        for _, row in k_df.iterrows():
            dominated = False
            for _, other in k_df.iterrows():
                if (other["latency_ms"] <= row["latency_ms"] and
                    other["recall"] >= row["recall"] and
                    (other["latency_ms"] < row["latency_ms"] or other["recall"] > row["recall"])):
                    dominated = True
                    break
            if not dominated:
                pareto.append(row)
        pareto = sorted(pareto, key=lambda x: x["latency_ms"])
        pareto_lines.append(f"K={int(k)} Pareto front ({len(pareto)} pts):")
        for p in pareto[:12]:  # keep text compact
            pv = f", param={p['param_value']:.0f}" if has_param_val and np.isfinite(p.get("param_value", np.nan)) else ""
            pareto_lines.append(f"  {p['index']}({pv.lstrip(', ')}): recall={p['recall']:.4f}, latency={p['latency_ms']:.3f}ms".replace("()", ""))
        pareto_lines.append("")
    if pareto_lines:
        writer.add_text(f"{tag_prefix}analysis/pareto_front", "\n".join(pareto_lines), 0)

    writer.flush()


def analyze_failure_modes(df: pd.DataFrame):
    """
    Analyze the profiling data to identify static index failure modes.
    (Same logic as your original file; kept verbatim style.)
    """
    print("\n" + "=" * 70)
    print("STATIC INDEX FAILURE MODE ANALYSIS")
    print("=" * 70)

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

    print("--- Analysis 3: Pareto-Optimal Strategies ---")
    print("(Strategies on the Pareto front of latency vs recall)\n")

    for k in sorted(df["k"].unique()):
        k_df = df[df["k"] == k].copy()
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

    print("--- Analysis 4: Index Crossover Points ---")
    print("(Where does one index become better than another?)\n")

    for k in sorted(df["k"].unique()):
        k_df = df[df["k"] == k]

        ivf_good = k_df[(k_df["index"] == "IVF") & (k_df["recall"] >= 0.9)]
        hnsw_good = k_df[(k_df["index"] == "HNSW") & (k_df["recall"] >= 0.9)]

        if not ivf_good.empty and not hnsw_good.empty:
            best_ivf = ivf_good.loc[ivf_good["latency_ms"].idxmin()]
            best_hnsw = hnsw_good.loc[hnsw_good["latency_ms"].idxmin()]
            winner = "IVF" if best_ivf["latency_ms"] < best_hnsw["latency_ms"] else "HNSW"
            print(f"  K={k:>3d} (recall>=0.9): "
                  f"IVF={best_ivf['latency_ms']:.3f}ms vs "
                  f"HNSW={best_hnsw['latency_ms']:.3f}ms → Winner: {winner}")


def run_profiling(dataset_name="sift-128-euclidean", data_dir="data",
                  log_dir="runs", run_name=None, tag_prefix="phase2/"):
    print("=" * 70)
    print("PHASE 2: Systematic Performance Profiling (TensorBoard)")
    print("=" * 70)

    # ---- TensorBoard setup ----
    if run_name is None:
        run_name = f"profiling_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tb_dir = os.path.join(log_dir, run_name)
    ensure_dir(tb_dir)
    writer = get_tb_writer(tb_dir)

    # ---- Load ----
    config = load_config()
    base, queries, ground_truth, _ = load_dataset(dataset_name, data_dir)
    n, dim = base.shape

    # Limit query count if configured
    max_queries = config["profiling"]["num_queries"]
    if max_queries > 0 and max_queries < len(queries):
        queries = queries[:max_queries]
        ground_truth = ground_truth[:max_queries]

    # Meta
    writer.add_text(f"{tag_prefix}meta/dataset", dataset_name, 0)
    writer.add_text(f"{tag_prefix}meta/shape", f"base={n}x{dim}, queries={len(queries)}x{dim}", 0)

    # ---- Build ----
    indexes = build_indexes(base, dim, config)

    # ---- Sweep ----
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

    # ---- Save ----
    ensure_dir("results")
    save_results_csv(df.to_dict("records"), "results/profiling_sweep.csv")

    # ---- TensorBoard logging (from the sweep df) ----
    print("\nLogging sweep results to TensorBoard...")
    log_sweep_to_tensorboard(df, writer, tag_prefix=tag_prefix)

    # ---- Analyze (console) ----
    analyze_failure_modes(df)

    # ---- Existing matplotlib figures (optional) ----
    print("\nGenerating visualizations (PNG)...")
    try:
        from experiments.visualize_profiling import generate_all_plots
        generate_all_plots(df)
    except ImportError:
        print("  (Run experiments/visualize_profiling.py separately for plots)")

    writer.flush()
    writer.close()

    print("\n" + "=" * 70)
    print("Phase 2 Complete!")
    print(f"  Profiling data: results/profiling_sweep.csv")
    print(f"  TensorBoard logs: {tb_dir}")
    print(f"  {len(df)} configurations profiled")
    print("=" * 70)

    return df


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Profiling Analysis (TensorBoard)")
    parser.add_argument("--dataset", type=str, default="sift-128-euclidean")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--tag-prefix", type=str, default="phase2/",
                        help="TensorBoard tag prefix (default: phase2/)")
    args = parser.parse_args()

    run_profiling(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        log_dir=args.log_dir,
        run_name=args.run_name,
        tag_prefix=args.tag_prefix
    )


if __name__ == "__main__":
    main()
