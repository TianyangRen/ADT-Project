"""Experiment Results Analysis Module

This module analyzes the experimental results of a vector search system, including:
  - Baseline Analysis: Performance comparison of different indexing methods
  - Profiling Analysis: Impact of parameter tuning on performance
  - Adaptive Evaluation: Effectiveness assessment of adaptive execution strategies
  
Generated outputs include:
  - Summary text reports
  - CSV tabular data (for final report generation)
  - Optional performance comparison charts
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Paths:
    """Storage for all experiment result file paths
    
    Attributes:
        results_dir: Main results directory path
        baseline_csv: Baseline test results CSV file
        profiling_csv: Parameter sweep results CSV file
        adaptive_csv: Adaptive evaluation results CSV file
        summary_txt: Analysis summary text output path
        tables_csv: Unified table format analysis results output path
        plots_dir: Performance comparison charts output directory
    """
    results_dir: Path
    baseline_csv: Path
    profiling_csv: Path
    adaptive_csv: Path
    summary_txt: Path
    tables_csv: Path
    plots_dir: Path


def _safe_literal_eval(s: Any) -> Any:
    """Safely convert string-formatted parameter dictionaries to Python objects
    
    The 'params' column in baseline_results.csv is often stored as a string like "{'nprobe': 16}".
    This function uses ast.literal_eval for safe parsing, avoiding the security risks of eval().
    Returns the original value if parsing fails.
    
    Args:
        s: Value to parse (typically a string)
        
    Returns:
        Parsed Python object, or original value (on parsing failure)
    """
    if not isinstance(s, str):
        return s
    s = s.strip()
    if not s:
        return {}
    try:
        return ast.literal_eval(s)
    except Exception:
        return s



def _ensure_exists(p: Path) -> None:
    """Check if file exists, raise exception if not
    
    Args:
        p: File path to check
        
    Raises:
        FileNotFoundError: Raised when file does not exist
    """
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")


def _format_ms(x: float) -> str:
    """Format numeric value as millisecond string
    
    Args:
        x: Value in milliseconds
        
    Returns:
        Formatted string with 4 decimal places
    """
    return f"{x:.4f} ms"


def _format_pct(x: float) -> str:
    """Convert decimal to percentage string
    
    Args:
        x: Decimal value in range [0, 1]
        
    Returns:
        Formatted percentage string with 2 decimal places
    """
    return f"{100.0 * x:.2f}%"


def _best_under_constraint(
    df: pd.DataFrame,
    recall_col: str,
    latency_col: str,
    min_recall: float,
) -> Optional[pd.Series]:
    """Find configuration with minimum latency under given recall constraint
    
    Given a minimum recall threshold, select the configuration with the smallest
    latency from all configurations that satisfy the constraint. This identifies
    the best performing solution that meets quality requirements.
    
    Args:
        df: DataFrame containing configuration data
        recall_col: Name of recall column
        latency_col: Name of latency column
        min_recall: Minimum recall constraint threshold
        
    Returns:
        Row with minimum latency satisfying constraint (pd.Series), or None
    """
    feasible = df[df[recall_col] >= min_recall].copy()
    if feasible.empty:
        return None
    best_idx = feasible[latency_col].idxmin()
    return feasible.loc[best_idx]


def analyze_baseline(baseline: pd.DataFrame, min_recall: float) -> Tuple[str, pd.DataFrame]:
    """Analyze baseline test results
    
    Analyzes baseline test data with the following steps:
    1. Calculate average performance metrics by indexing method
    2. For each k value, find optimal method under recall constraint
    3. Generate human-readable analysis summary text
    4. Generate structured table data for downstream report generation
    
    Args:
        baseline: Baseline results DataFrame (from baseline_results.csv)
        min_recall: Minimum recall constraint for filtering feasible configs
        
    Returns:
        (Analysis summary text, Structured results table)
    """
    baseline = baseline.copy()
    if "params" in baseline.columns:
        baseline["params_parsed"] = baseline["params"].apply(_safe_literal_eval)

    # Validate existence of required columns
    required_cols = {"dataset", "index", "k", "recall", "mean_ms", "qps"}
    missing = required_cols - set(baseline.columns)
    if missing:
        raise ValueError(f"baseline_results.csv missing columns: {sorted(missing)}")

    lines = []
    lines.append("=== Baseline Analysis ===")
    lines.append(f"Constraint used in this analysis: recall >= {min_recall:.3f}")
    lines.append("")

    # Calculate average performance metrics by indexing method
    agg = (
        baseline.groupby("index", as_index=False)
        .agg(
            recall_mean=("recall", "mean"),
            mean_ms_mean=("mean_ms", "mean"),
            qps_mean=("qps", "mean"),
        )
        .sort_values("mean_ms_mean", ascending=True)
    )

    lines.append("Overall (averaged over k):")
    for _, r in agg.iterrows():
        lines.append(
            f"- {r['index']}: mean latency={_format_ms(r['mean_ms_mean'])}, "
            f"mean recall={r['recall_mean']:.4f}, mean QPS={r['qps_mean']:.1f}"
        )
    lines.append("")

    rows = []
    for k_val in sorted(baseline["k"].unique()):
        sub = baseline[baseline["k"] == k_val].copy()
        best = _best_under_constraint(sub, "recall", "mean_ms", min_recall=min_recall)
        if best is None:
            lines.append(f"For k={k_val}: no method meets recall >= {min_recall:.3f}")
            continue

        lines.append(
            f"For k={int(k_val)}: best feasible method is {best['index']} "
            f"(lat={_format_ms(best['mean_ms'])}, recall={best['recall']:.4f}, QPS={best['qps']:.1f})"
        )

        rows.append(
            {
                "section": "baseline_best_per_k",
                "k": int(k_val),
                "method_or_index": best["index"],
                "mean_latency_ms": float(best["mean_ms"]),
                "recall": float(best["recall"]),
                "qps": float(best["qps"]),
                "params": str(best.get("params", "")),
            }
        )

    lines.append("")
    best_table = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["section", "k", "method_or_index", "mean_latency_ms", "recall", "qps", "params"]
    )

    agg_table = agg.copy()
    agg_table.insert(0, "section", "baseline_overall_avg")
    agg_table["k"] = np.nan
    agg_table["params"] = ""
    agg_table.rename(
        columns={
            "index": "method_or_index",
            "mean_ms_mean": "mean_latency_ms",
            "recall_mean": "recall",
            "qps_mean": "qps",
        },
        inplace=True,
    )
    agg_table = agg_table[["section", "k", "method_or_index", "mean_latency_ms", "recall", "qps", "params"]]

    combined = pd.concat([agg_table, best_table], ignore_index=True)
    return "\n".join(lines), combined


def analyze_profiling(profiling: pd.DataFrame, min_recall: float) -> Tuple[str, pd.DataFrame]:
    """Analyze parameter sweep results
    
    Analyzes parameter sweep data with the following steps:
    1. Find optimal parameter configuration for each index method and k value
    2. Analyze parameter sensitivity (latency and recall range variations)
    3. Generate human-readable analysis summary text
    4. Generate structured table data
    
    Args:
        profiling: Parameter sweep results DataFrame (from profiling_sweep.csv)
        min_recall: Minimum recall constraint
        
    Returns:
        (Analysis summary text, Structured results table)
    """
    profiling = profiling.copy()

    # Validate required columns
    required_cols = {"index", "param_name", "param_value", "k", "recall", "latency_ms", "qps"}
    missing = required_cols - set(profiling.columns)
    if missing:
        raise ValueError(f"profiling_sweep.csv missing columns: {sorted(missing)}")

    lines = []
    lines.append("=== Profiling Analysis ===")
    lines.append(f"Constraint used in this analysis: recall >= {min_recall:.3f}")
    lines.append("")

    rows = []
    # Find optimal configuration for each index/k combination
    for (idx_name, k_val), sub in profiling.groupby(["index", "k"]):
        best = _best_under_constraint(sub, "recall", "latency_ms", min_recall=min_recall)
        if best is None:
            lines.append(f"- {idx_name} @ k={int(k_val)}: no config meets recall >= {min_recall:.3f}")
            continue
        lines.append(
            f"- {idx_name} @ k={int(k_val)}: best config is {best['param_name']}={best['param_value']} "
            f"(lat={_format_ms(best['latency_ms'])}, recall={best['recall']:.4f}, qps={best['qps']:.1f})"
        )
        rows.append(
            {
                "section": "profiling_best_config",
                "k": int(k_val),
                "method_or_index": idx_name,
                "param_name": best["param_name"],
                "param_value": best["param_value"],
                "mean_latency_ms": float(best["latency_ms"]),
                "recall": float(best["recall"]),
                "qps": float(best["qps"]),
            }
        )

    lines.append("")
    # Best configuration table: individual optimal configuration records
    best_table = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["section", "k", "method_or_index", "param_name", "param_value", "mean_latency_ms", "recall", "qps"]
    )

    # Calculate parameter sensitivity for recall and latency ranges
    sens = (
        profiling.groupby(["index", "param_name", "k"], as_index=False)
        .agg(
            recall_min=("recall", "min"),
            recall_max=("recall", "max"),
            latency_min=("latency_ms", "min"),
            latency_max=("latency_ms", "max"),
        )
    )
    sens["recall_range"] = sens["recall_max"] - sens["recall_min"]
    sens["latency_range_ms"] = sens["latency_max"] - sens["latency_min"]
    # Sort by latency range descending to show most unstable parameters
    sens = sens.sort_values(["latency_range_ms", "recall_range"], ascending=False)

    lines.append("Top sensitivity signals (largest latency/recall ranges):")
    for _, r in sens.head(5).iterrows():
        lines.append(
            f"- {r['index']} ({r['param_name']}) @ k={int(r['k'])}: "
            f"latency_range={_format_ms(r['latency_range_ms'])}, recall_range={r['recall_range']:.4f}"
        )

    # Sensitivity analysis table: records performance variations for each parameter
    sens_out = sens.copy()
    sens_out.insert(0, "section", "profiling_sensitivity")
    sens_out.rename(columns={"index": "method_or_index"}, inplace=True)

    # Normalize column names to match final output format (consistent with other analyses)
    for col in ["param_value", "mean_latency_ms", "recall", "qps"]:
        if col not in sens_out.columns:
            sens_out[col] = np.nan
    sens_out["param_value"] = ""
    sens_out["mean_latency_ms"] = np.nan
    sens_out["recall"] = np.nan
    sens_out["qps"] = np.nan

    sens_out = sens_out[
        [
            "section",
            "k",
            "method_or_index",
            "param_name",
            "param_value",
            "mean_latency_ms",
            "recall",
            "qps",
            "recall_min",
            "recall_max",
            "latency_min",
            "latency_max",
            "recall_range",
            "latency_range_ms",
        ]
    ]

    # Convert best config table to same column format (fill missing sensitivity columns)
    best_out = best_table.copy()
    if not best_out.empty:
        for c in ["recall_min", "recall_max", "latency_min", "latency_max", "recall_range", "latency_range_ms"]:
            best_out[c] = np.nan
        best_out = best_out[sens_out.columns]

    # Merge best configs + sensitivity analysis
    combined = pd.concat([best_out, sens_out], ignore_index=True, sort=False)
    return "\n".join(lines), combined


def analyze_adaptive(adaptive: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    """Analyze adaptive execution evaluation results
    
    Analyzes adaptive evaluation data with the following steps:
    1. Find optimal method for each experiment (sorted by budget met, recall, latency)
    2. Calculate average performance of each method across all experiments
    3. Generate human-readable analysis summary
    4. Generate structured table data
    
    Args:
        adaptive: Adaptive evaluation results DataFrame (from adaptive_evaluation.csv)
        
    Returns:
        (Analysis summary text, Structured results table)
    """
    adaptive = adaptive.copy()

    # Validate required columns
    required_cols = {
        "experiment",
        "method",
        "recall",
        "mean_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "budget_met_frac",
    }
    missing = required_cols - set(adaptive.columns)
    if missing:
        raise ValueError(f"adaptive_evaluation.csv missing columns: {sorted(missing)}")

    lines = []
    lines.append("=== Adaptive Evaluation Analysis ===")
    lines.append("Ranking rule per experiment: (1) budget_met_frac, (2) recall, (3) mean_latency_ms")
    lines.append("")

    rows = []
    # Find best method for each experiment using ranking rules
    for exp, sub in adaptive.groupby("experiment"):
        sub2 = sub.sort_values(["budget_met_frac", "recall", "mean_latency_ms"], ascending=[False, False, True])
        best = sub2.iloc[0]
        lines.append(
            f"- {exp}: best={best['method']} "
            f"(budget_met={_format_pct(best['budget_met_frac'])}, recall={best['recall']:.4f}, "
            f"mean_lat={_format_ms(best['mean_latency_ms'])}, p95={_format_ms(best['p95_latency_ms'])})"
        )
        for _, r in sub2.iterrows():
            rows.append(
                {
                    "section": "adaptive_methods",
                    "experiment": exp,
                    "method_or_index": r["method"],
                    "budget_met_frac": float(r["budget_met_frac"]),
                    "recall": float(r["recall"]),
                    "mean_latency_ms": float(r["mean_latency_ms"]),
                    "p50_latency_ms": float(r["p50_latency_ms"]),
                    "p95_latency_ms": float(r["p95_latency_ms"]),
                }
            )

    lines.append("")
    agg = (
        adaptive.groupby("method", as_index=False)
        .agg(
            # Calculate average performance of each method across all experiments
            budget_met_frac_mean=("budget_met_frac", "mean"),
            recall_mean=("recall", "mean"),
            mean_latency_ms_mean=("mean_latency_ms", "mean"),
            p95_latency_ms_mean=("p95_latency_ms", "mean"),
        )
        # Sort by rules: budget met fraction > recall > mean latency
        .sort_values(["budget_met_frac_mean", "recall_mean", "mean_latency_ms_mean"], ascending=[False, False, True])
    )
    lines.append("Overall (averaged over experiments):")
    for _, r in agg.iterrows():
        lines.append(
            f"- {r['method']}: budget_met={_format_pct(r['budget_met_frac_mean'])}, "
            f"mean recall={r['recall_mean']:.4f}, mean latency={_format_ms(r['mean_latency_ms_mean'])}, "
            f"mean p95={_format_ms(r['p95_latency_ms_mean'])}"
        )

    # Best method details per experiment
    out = pd.DataFrame(rows)
    # Convert aggregation table to output format: add "section" column to distinguish data types
    agg_out = agg.copy()
    agg_out.insert(0, "section", "adaptive_overall_avg")
    agg_out.rename(
        columns={
            "method": "method_or_index",
            "budget_met_frac_mean": "budget_met_frac",
            "recall_mean": "recall",
            "mean_latency_ms_mean": "mean_latency_ms",
            "p95_latency_ms_mean": "p95_latency_ms",
        },
        inplace=True,
    )
    agg_out["experiment"] = ""
    agg_out["p50_latency_ms"] = np.nan
    agg_out = agg_out[
        ["section", "experiment", "method_or_index", "budget_met_frac", "recall", "mean_latency_ms", "p50_latency_ms", "p95_latency_ms"]
    ]

    # Merge experiment-level results with overall averages
    combined = pd.concat([out, agg_out], ignore_index=True, sort=False)
    return "\n".join(lines), combined


def maybe_make_plots(paths: Paths, baseline: pd.DataFrame, adaptive: pd.DataFrame) -> None:
    """Generate optional performance comparison charts
    
    Note: Project already has complete chart generation functionality; this is supplementary.
    Skips plotting if matplotlib is not installed.
    
    Generated charts include:
    - Baseline: X-axis latency, Y-axis recall, different colors for index methods
    - Adaptive: X-axis budget met fraction, Y-axis recall
    
    Args:
        paths: File path configuration
        baseline: Baseline test results data
        adaptive: Adaptive evaluation results data
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    paths.plots_dir.mkdir(parents=True, exist_ok=True)

    # Baseline scatter plot: latency vs recall (colored by index method)
    fig1 = paths.plots_dir / "baseline_latency_vs_recall.png"
    plt.figure()
    for idx, sub in baseline.groupby("index"):
        plt.scatter(sub["mean_ms"], sub["recall"], label=str(idx))
    plt.xlabel("Mean latency (ms)")
    plt.ylabel("Recall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig1)
    plt.close()

    # Adaptive evaluation scatter plot: budget met fraction vs recall
    fig2 = paths.plots_dir / "adaptive_budget_vs_recall.png"
    plt.figure()
    plt.scatter(adaptive["budget_met_frac"], adaptive["recall"])
    plt.xlabel("Budget met fraction")
    plt.ylabel("Recall")
    plt.tight_layout()
    plt.savefig(fig2)
    plt.close()


def build_paths(project_root: Path) -> Paths:
    """Build path configuration for all experiment result files
    
    Args:
        project_root: Project root directory
        
    Returns:
        Paths object containing paths to all result files
    """
    results_dir = project_root / "results"
    return Paths(
        results_dir=results_dir,
        baseline_csv=results_dir / "baseline_results.csv",
        profiling_csv=results_dir / "profiling_sweep.csv",
        adaptive_csv=results_dir / "adaptive_evaluation.csv",
        summary_txt=results_dir / "analysis_summary.txt",
        tables_csv=results_dir / "analysis_tables.csv",
        plots_dir=results_dir / "analysis_plots",
    )


def main() -> None:
    """Main program entry point
    
    Workflow:
    1. Load three experiment results (baseline, profiling, adaptive)
    2. Perform detailed analysis on each
    3. Generate human-readable summary report text
    4. Generate unified table format data for downstream report generation
    5. Optional: generate performance comparison charts
    
    Command-line arguments:
        --project-root: Project root directory path (should contain ./results)
        --min-recall: Minimum recall constraint for baseline/profiling analysis (default 0.95)
        --plots: Whether to generate additional performance comparison charts
    """
    parser = argparse.ArgumentParser(
        description="Analyze experiment outputs in ./results and generate report-ready summaries."
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root path (should contain ./results). Default: current directory.",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.95,
        help="Recall constraint for selecting best configs in baseline/profiling summaries.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Also generate minimal extra plots under results/analysis_plots/.",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    paths = build_paths(project_root)

    # Check all required input files
    _ensure_exists(paths.baseline_csv)
    _ensure_exists(paths.profiling_csv)
    _ensure_exists(paths.adaptive_csv)

    # Load experiment result data
    baseline = pd.read_csv(paths.baseline_csv)
    profiling = pd.read_csv(paths.profiling_csv)
    adaptive = pd.read_csv(paths.adaptive_csv)

    # Perform three types of analysis
    baseline_text, baseline_table = analyze_baseline(baseline, min_recall=args.min_recall)
    profiling_text, profiling_table = analyze_profiling(profiling, min_recall=args.min_recall)
    adaptive_text, adaptive_table = analyze_adaptive(adaptive)

    # Generate combined summary text
    summary = "\n\n".join([baseline_text, profiling_text, adaptive_text]).strip() + "\n"
    paths.summary_txt.write_text(summary, encoding="utf-8")

    # Merge all tables into unified format CSV for report generation
    all_tables = []

    # Align baseline table column descriptors with adaptive table (adaptive has fewer cols)
    b = baseline_table.copy()
    b["experiment"] = ""
    b["budget_met_frac"] = np.nan
    b["p50_latency_ms"] = np.nan
    b["p95_latency_ms"] = np.nan
    all_tables.append(b)

    # Align profiling table column descriptors with adaptive table
    p = profiling_table.copy()
    p["experiment"] = ""
    p["budget_met_frac"] = np.nan
    p["p50_latency_ms"] = np.nan
    p["p95_latency_ms"] = np.nan
    all_tables.append(p)

    # Adaptive table: add columns from baseline and profiling
    a = adaptive_table.copy()
    for col in ["k", "param_name", "param_value", "qps", "params"]:
        if col not in a.columns:
            a[col] = "" if col in ("param_name", "param_value", "params") else np.nan
    all_tables.append(a)

    # Combine all tables into one large DataFrame and save as CSV for downstream use
    combined = pd.concat(all_tables, ignore_index=True, sort=False)
    combined.to_csv(paths.tables_csv, index=False)

    # Optional: generate performance comparison charts
    if args.plots:
        maybe_make_plots(paths, baseline=baseline, adaptive=adaptive)

    # Output result paths
    print(f"[OK] Wrote: {paths.summary_txt}")
    print(f"[OK] Wrote: {paths.tables_csv}")
    if args.plots:
        print(f"[OK] Wrote plots under: {paths.plots_dir}")


if __name__ == "__main__":
    main()
