#!/usr/bin/env python3
"""
Visualization for Phase 2 profiling data.

Generates key figures that demonstrate static index failure modes:
  1. Recall vs Latency Pareto Front (per K)
  2. Recall Heatmap: (config) x (K)
  3. Best Index Per Query Regime
  4. Latency CDF per index
  5. Parameter Sensitivity Curves

Usage:
    python experiments/visualize_profiling.py
    python experiments/visualize_profiling.py --csv results/profiling_sweep.csv
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.io_utils import ensure_dir

# Style settings
COLORS = {"Flat": "#e74c3c", "IVF": "#3498db", "HNSW": "#2ecc71"}
MARKERS = {"Flat": "s", "IVF": "o", "HNSW": "^"}
FIG_DIR = "results/figures"


def generate_all_plots(df: pd.DataFrame = None, csv_path: str = None):
    """Generate all Phase 2 visualization plots."""
    if df is None:
        if csv_path is None:
            csv_path = "results/profiling_sweep.csv"
        df = pd.read_csv(csv_path)

    ensure_dir(FIG_DIR)

    print("Generating plots...")
    plot_pareto_front(df)
    plot_recall_heatmap(df)
    plot_best_index_per_regime(df)
    plot_parameter_sensitivity(df)
    plot_recall_vs_k(df)
    print(f"All plots saved to: {FIG_DIR}/")


def plot_pareto_front(df: pd.DataFrame):
    """
    Figure 1: Recall vs Latency scatter — the most important plot.
    Shows that different indexes dominate different regions of the
    latency-recall trade-off space.
    """
    k_values = sorted(df["k"].unique())
    fig, axes = plt.subplots(1, len(k_values), figsize=(5 * len(k_values), 5),
                              squeeze=False)

    for i, k in enumerate(k_values):
        ax = axes[0][i]
        k_df = df[df["k"] == k]

        for idx_name in k_df["index"].unique():
            subset = k_df[k_df["index"] == idx_name]
            ax.scatter(
                subset["latency_ms"], subset["recall"],
                c=COLORS.get(idx_name, "gray"),
                marker=MARKERS.get(idx_name, "o"),
                s=60, alpha=0.8, label=idx_name, edgecolors="white", linewidth=0.5
            )

        ax.set_xlabel("Latency (ms)", fontsize=12)
        ax.set_ylabel("Recall@K", fontsize=12)
        ax.set_title(f"K = {k}", fontsize=14, fontweight="bold")
        ax.set_xscale("log")
        ax.set_ylim([0, 1.05])
        ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="95% recall")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Recall vs Latency: No Single Index Dominates All Regimes",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/01_pareto_front.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [1/5] Pareto front plot saved")


def plot_recall_heatmap(df: pd.DataFrame):
    """
    Figure 2: Heatmap of recall values for each (index_config, K) combination.
    Clearly shows where static configs fail.
    """
    # Create a label for each config
    df_copy = df.copy()
    df_copy["config"] = df_copy.apply(
        lambda r: f"{r['index']}({r['param_name']}={int(r['param_value'])})"
        if r["param_name"] != "none" else r["index"],
        axis=1
    )

    # Pivot to matrix
    pivot = df_copy.pivot_table(values="recall", index="config", columns="k")

    fig, ax = plt.subplots(figsize=(8, max(6, len(pivot) * 0.4)))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    # Labels
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"K={k}" for k in pivot.columns], fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    # Annotate cells with recall values
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label="Recall@K")
    ax.set_title("Recall Heatmap: Static Configurations Across Query Regimes",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Query Regime (K value)", fontsize=11)
    ax.set_ylabel("Index Configuration", fontsize=11)

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/02_recall_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [2/5] Recall heatmap saved")


def plot_best_index_per_regime(df: pd.DataFrame):
    """
    Figure 3: Bar chart showing which index is fastest at each K
    while meeting a recall threshold.
    """
    recall_thresholds = [0.90, 0.95, 0.99]
    k_values = sorted(df["k"].unique())

    fig, axes = plt.subplots(1, len(recall_thresholds),
                              figsize=(5 * len(recall_thresholds), 5),
                              squeeze=False)

    for t, threshold in enumerate(recall_thresholds):
        ax = axes[0][t]
        winners = []
        latencies = []
        colors = []

        for k in k_values:
            feasible = df[(df["k"] == k) & (df["recall"] >= threshold)]
            if feasible.empty:
                winners.append("None")
                latencies.append(0)
                colors.append("gray")
            else:
                best = feasible.loc[feasible["latency_ms"].idxmin()]
                winners.append(best["index"])
                latencies.append(best["latency_ms"])
                colors.append(COLORS.get(best["index"], "gray"))

        bars = ax.bar([f"K={k}" for k in k_values], latencies, color=colors)

        # Add winner labels on bars
        for bar, winner in zip(bars, winners):
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        winner, ha="center", va="bottom", fontsize=10,
                        fontweight="bold")

        ax.set_ylabel("Latency (ms)", fontsize=11)
        ax.set_title(f"Recall >= {threshold}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Best Index Per Query Regime (Lowest Latency Meeting Recall Constraint)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/03_best_per_regime.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [3/5] Best index per regime plot saved")


def plot_parameter_sensitivity(df: pd.DataFrame):
    """
    Figure 4: How recall and latency change with the tunable parameter
    for IVF (nprobe) and HNSW (ef_search).
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # IVF: nprobe vs recall
    ivf_df = df[df["index"] == "IVF"]
    for k in sorted(ivf_df["k"].unique()):
        subset = ivf_df[ivf_df["k"] == k].sort_values("param_value")
        axes[0][0].plot(subset["param_value"], subset["recall"],
                        marker="o", label=f"K={k}")
    axes[0][0].set_xlabel("nprobe")
    axes[0][0].set_ylabel("Recall@K")
    axes[0][0].set_title("IVF: nprobe vs Recall")
    axes[0][0].set_xscale("log", base=2)
    axes[0][0].legend()
    axes[0][0].grid(True, alpha=0.3)
    axes[0][0].axhline(y=0.95, color="red", linestyle="--", alpha=0.5)

    # IVF: nprobe vs latency
    for k in sorted(ivf_df["k"].unique()):
        subset = ivf_df[ivf_df["k"] == k].sort_values("param_value")
        axes[0][1].plot(subset["param_value"], subset["latency_ms"],
                        marker="o", label=f"K={k}")
    axes[0][1].set_xlabel("nprobe")
    axes[0][1].set_ylabel("Latency (ms)")
    axes[0][1].set_title("IVF: nprobe vs Latency")
    axes[0][1].set_xscale("log", base=2)
    axes[0][1].legend()
    axes[0][1].grid(True, alpha=0.3)

    # HNSW: ef_search vs recall
    hnsw_df = df[df["index"] == "HNSW"]
    for k in sorted(hnsw_df["k"].unique()):
        subset = hnsw_df[hnsw_df["k"] == k].sort_values("param_value")
        axes[1][0].plot(subset["param_value"], subset["recall"],
                        marker="^", label=f"K={k}")
    axes[1][0].set_xlabel("ef_search")
    axes[1][0].set_ylabel("Recall@K")
    axes[1][0].set_title("HNSW: ef_search vs Recall")
    axes[1][0].set_xscale("log", base=2)
    axes[1][0].legend()
    axes[1][0].grid(True, alpha=0.3)
    axes[1][0].axhline(y=0.95, color="red", linestyle="--", alpha=0.5)

    # HNSW: ef_search vs latency
    for k in sorted(hnsw_df["k"].unique()):
        subset = hnsw_df[hnsw_df["k"] == k].sort_values("param_value")
        axes[1][1].plot(subset["param_value"], subset["latency_ms"],
                        marker="^", label=f"K={k}")
    axes[1][1].set_xlabel("ef_search")
    axes[1][1].set_ylabel("Latency (ms)")
    axes[1][1].set_title("HNSW: ef_search vs Latency")
    axes[1][1].set_xscale("log", base=2)
    axes[1][1].legend()
    axes[1][1].grid(True, alpha=0.3)

    fig.suptitle("Parameter Sensitivity Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/04_parameter_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [4/5] Parameter sensitivity plot saved")


def plot_recall_vs_k(df: pd.DataFrame):
    """
    Figure 5: For fixed parameter configs, show how recall degrades
    as K increases — evidence that static configs break at different K.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Pick representative static configs
    configs = [
        ("IVF", "nprobe", 8),
        ("IVF", "nprobe", 32),
        ("IVF", "nprobe", 128),
        ("HNSW", "ef_search", 32),
        ("HNSW", "ef_search", 128),
        ("HNSW", "ef_search", 512),
    ]

    linestyles = {"IVF": "-", "HNSW": "--"}

    for idx_name, pname, pval in configs:
        subset = df[(df["index"] == idx_name) & (df["param_value"] == pval)]
        if not subset.empty:
            subset = subset.sort_values("k")
            ax.plot(subset["k"], subset["recall"],
                    marker=MARKERS.get(idx_name, "o"),
                    linestyle=linestyles.get(idx_name, "-"),
                    color=COLORS.get(idx_name),
                    alpha=0.8,
                    label=f"{idx_name}({pname}={pval})")

    ax.set_xlabel("K (number of neighbors)", fontsize=12)
    ax.set_ylabel("Recall@K", fontsize=12)
    ax.set_title("Recall Degradation with Increasing K (Static Configs)",
                 fontsize=13, fontweight="bold")
    ax.axhline(y=0.95, color="red", linestyle=":", alpha=0.5, label="95% threshold")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/05_recall_vs_k.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [5/5] Recall vs K plot saved")


def main():
    parser = argparse.ArgumentParser(description="Generate profiling visualizations")
    parser.add_argument("--csv", type=str, default="results/profiling_sweep.csv",
                        help="Path to profiling CSV data")
    args = parser.parse_args()

    generate_all_plots(csv_path=args.csv)


if __name__ == "__main__":
    main()
