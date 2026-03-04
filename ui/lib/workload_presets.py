from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import yaml


@dataclass
class WorkloadConfig:
    name: str
    description: str
    top_k_choices: List[int]
    latency_budget_ms_range: Tuple[float, float]
    min_recall_range: Tuple[float, float]
    concurrency_choices: List[int]


def load_presets(path: str = "workloads/presets.yaml") -> Dict[str, WorkloadConfig]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    presets = {}
    for name, cfg in raw.get("presets", {}).items():
        presets[name] = WorkloadConfig(
            name=name,
            description=str(cfg.get("description", "")),
            top_k_choices=list(cfg.get("top_k_choices", [10])),
            latency_budget_ms_range=tuple(cfg.get("latency_budget_ms_range", [5.0, 5.0])),
            min_recall_range=tuple(cfg.get("min_recall_range", [0.9, 0.9])),
            concurrency_choices=list(cfg.get("concurrency_choices", [1])),
        )
    return presets


def sample_constraints(preset: WorkloadConfig, rng: random.Random) -> Dict[str, Any]:
    top_k = rng.choice(preset.top_k_choices)
    lat_lo, lat_hi = preset.latency_budget_ms_range
    rec_lo, rec_hi = preset.min_recall_range
    latency_budget_ms = rng.uniform(float(lat_lo), float(lat_hi))
    min_recall = rng.uniform(float(rec_lo), float(rec_hi))
    concurrency = rng.choice(preset.concurrency_choices)

    return {
        "top_k": int(top_k),
        "latency_budget_ms": float(latency_budget_ms),
        "min_recall": float(min_recall),
        "concurrency": int(concurrency),
    }
