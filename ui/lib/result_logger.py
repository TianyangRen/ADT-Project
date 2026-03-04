from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd


def _now_tag() -> str:
    # Example: 20260301_103512
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@dataclass
class RunMeta:
    run_tag: str
    run_type: str               # "adaptive" or "compare"
    dataset: str
    config_path: str
    query_id: int
    top_k: int
    latency_budget_ms: float
    min_recall: float
    concurrency: int
    static_params: Dict[str, Any]  # e.g., {"ivf_nprobe": 16, "hnsw_ef": 64}
    adaptive_chosen_index: Optional[str] = None
    adaptive_params: Optional[Dict[str, Any]] = None
    adaptive_regime: Optional[str] = None
    adaptive_reason: Optional[str] = None


def log_run(
    *,
    out_dir: str,
    meta: RunMeta,
    results_df: pd.DataFrame,
) -> Dict[str, str]:
    """
    Save:
      - results csv: results/ui_runs/<tag>__results.csv
      - meta json:   results/ui_runs/<tag>__meta.json
    Returns file paths.
    """
    ensure_dir(out_dir)
    tag = meta.run_tag or _now_tag()

    csv_path = os.path.join(out_dir, f"{tag}__{meta.run_type}__results.csv")
    json_path = os.path.join(out_dir, f"{tag}__{meta.run_type}__meta.json")

    # Save results table
    results_df.to_csv(csv_path, index=False)

    # Save metadata
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, ensure_ascii=False, indent=2)

    return {"csv": csv_path, "json": json_path}
