from __future__ import annotations

import glob
import os
import json
import pandas as pd


RUN_DIR = "results/ui_runs"
OUT_SUMMARY_CSV = "results/summary.csv"
OUT_SUMMARY_MD = "results/summary.md"


def load_meta_for(csv_path: str) -> dict:
    """
    Each csv has a matching meta json with the same prefix.
    Example:
      20260301_103512__compare__results.csv
      20260301_103512__compare__meta.json
    """
    meta_path = csv_path.replace("__results.csv", "__meta.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    csv_files = sorted(glob.glob(os.path.join(RUN_DIR, "*__results.csv")))
    if not csv_files:
        print(f"[WARN] No run csv files found under: {RUN_DIR}")
        return

    all_rows = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        meta = load_meta_for(csv_path)

        # Attach key meta columns to every row, so we can group later
        df["run_file"] = os.path.basename(csv_path)
        df["run_type"] = meta.get("run_type", "")
        df["dataset"] = meta.get("dataset", "")
        df["query_id"] = meta.get("query_id", -1)
        df["top_k"] = meta.get("top_k", None)
        df["latency_budget_ms"] = meta.get("latency_budget_ms", None)
        df["min_recall"] = meta.get("min_recall", None)
        df["concurrency"] = meta.get("concurrency", None)

        # static params (optional)
        sp = meta.get("static_params", {}) or {}
        df["ivf_nprobe"] = sp.get("ivf_nprobe", None)
        df["hnsw_ef"] = sp.get("hnsw_ef", None)

        # adaptive info (optional)
        df["adaptive_chosen_index"] = meta.get("adaptive_chosen_index", None)

        all_rows.append(df)

    big = pd.concat(all_rows, ignore_index=True)

    # Expect columns in UI compare table:
    # method, latency_ms, recall@K (column name varies)
    # We'll detect recall column dynamically.
    recall_cols = [c for c in big.columns if c.startswith("recall@")]
    recall_col = recall_cols[0] if recall_cols else None

    group_cols = ["dataset", "method"]
    agg = {"latency_ms": ["mean", "std", "min", "max", "count"]}
    if recall_col:
        agg[recall_col] = ["mean", "std", "min", "max"]

    summary = big.groupby(group_cols).agg(agg)
    # flatten multiindex columns
    summary.columns = ["_".join([c for c in col if c]) for col in summary.columns.values]
    summary = summary.reset_index()

    os.makedirs("results", exist_ok=True)
    summary.to_csv(OUT_SUMMARY_CSV, index=False)

    # Also write a Markdown table for direct report pasting
    md = summary.to_markdown(index=False)
    with open(OUT_SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write(md + "\n")

    print(f"[OK] Wrote: {OUT_SUMMARY_CSV}")
    print(f"[OK] Wrote: {OUT_SUMMARY_MD}")


if __name__ == "__main__":
    main()
