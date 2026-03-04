from __future__ import annotations

import argparse
import random
import time
import numpy as np
import pandas as pd

from src.utils.io_utils import load_config, load_dataset
from src.indexes.flat_index import FlatIndex
from src.indexes.ivf_index import IVFIndex
from src.indexes.hnsw_index import HNSWIndex
from src.adaptive.execution_engine import AdaptiveExecutionEngine
from src.adaptive.strategy_selector import ExecutionStrategy, StrategySelector

from ui.lib.workload_presets import load_presets, sample_constraints
from ui.lib.result_logger import RunMeta, log_run


def recall_at_k(result_ids: np.ndarray, gt_ids: np.ndarray, k: int) -> float:
    gt_k = set(gt_ids[:k].tolist())
    res_k = set(result_ids[:k].tolist())
    return len(gt_k.intersection(res_k)) / float(k) if k > 0 else 0.0


def build_candidates_for_available_indexes(available: set) -> list:
    out = []
    for s in StrategySelector.DEFAULT_CANDIDATES:
        if s.index_name in available:
            out.append(ExecutionStrategy(s.index_name, dict(s.params)))
    return out


def build_engine(config_path: str, enable_hnsw: bool):
    cfg = load_config(config_path)
    dataset_name = cfg["dataset"]["name"]
    data_dir = cfg["dataset"].get("data_dir", "data")
    metric = cfg["dataset"].get("metric", "L2")

    base, queries, gt_neighbors, gt_distances = load_dataset(dataset_name, data_dir=data_dir)
    dim = base.shape[1]

    flat = FlatIndex(dimension=dim, metric=metric)
    flat.build(base)

    ivf_cfg = cfg["indexes"]["ivf"]
    ivf = IVFIndex(dimension=dim, nlist=ivf_cfg.get("nlist", 256), metric=metric)
    ivf.build(base)

    indexes = {"Flat": flat, "IVF": ivf}

    if enable_hnsw:
        hnsw_cfg = cfg["indexes"]["hnsw"]
        hnsw = HNSWIndex(dimension=dim, M=hnsw_cfg.get("M", 32), ef_construction=hnsw_cfg.get("ef_construction", 200), metric=metric)
        hnsw.build(base)
        indexes["HNSW"] = hnsw

    candidates = build_candidates_for_available_indexes(set(indexes.keys()))
    engine = AdaptiveExecutionEngine(
        indexes=indexes,
        cost_model=None,
        candidates=candidates,
        dataset_size=base.shape[0],
        dimension=dim,
        default_latency_budget_ms=5.0,
        default_min_recall=0.9,
        latency_weight=0.6,
        recall_weight=0.4,
    )
    return cfg, queries, gt_neighbors, engine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="mixed", help="Preset name in workloads/presets.yaml")
    ap.add_argument("--n", type=int, default=30, help="Number of queries to sample")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--enable_hnsw", action="store_true", help="Build and include HNSW")
    ap.add_argument("--config", default="config/default_config.yaml", help="Config path")
    args = ap.parse_args()

    presets = load_presets()
    if args.preset not in presets:
        raise SystemExit(f"Unknown preset: {args.preset}. Available: {list(presets.keys())}")

    preset = presets[args.preset]
    rng = random.Random(args.seed)

    cfg, queries, gt_neighbors, engine = build_engine(args.config, args.enable_hnsw)

    rows = []
    for _ in range(args.n):
        qid = rng.randrange(0, queries.shape[0])
        q = queries[qid]
        gt = gt_neighbors[qid]

        cons = sample_constraints(preset, rng)
        top_k = cons["top_k"]

        # Static IVF baseline with fixed nprobe=16 (can be changed)
        ivf = engine.indexes["IVF"]
        t0 = time.perf_counter()
        D, I = ivf.search(q.reshape(1, -1), top_k, nprobe=16)
        lat_ivf = (time.perf_counter() - t0) * 1000.0
        rec_ivf = recall_at_k(I[0], gt, top_k)

        # Adaptive
        t0 = time.perf_counter()
        r = engine.search(
            query=q,
            top_k=top_k,
            latency_budget_ms=cons["latency_budget_ms"],
            min_recall=cons["min_recall"],
            concurrency=cons["concurrency"],
        )
        lat_adp = (time.perf_counter() - t0) * 1000.0
        rec_adp = recall_at_k(r.indices, gt, top_k)

        rows.append({
            "preset": preset.name,
            "query_id": qid,
            "top_k": top_k,
            "latency_budget_ms": cons["latency_budget_ms"],
            "min_recall": cons["min_recall"],
            "concurrency": cons["concurrency"],
            "method": "IVF(nprobe=16)",
            "latency_ms": lat_ivf,
            f"recall@{top_k}": rec_ivf,
        })
        rows.append({
            "preset": preset.name,
            "query_id": qid,
            "top_k": top_k,
            "latency_budget_ms": cons["latency_budget_ms"],
            "min_recall": cons["min_recall"],
            "concurrency": cons["concurrency"],
            "method": f"Adaptive=>{r.strategy_used.index_name}",
            "latency_ms": lat_adp,
            f"recall@{top_k}": rec_adp,
            "adaptive_reason": r.selection_result.reason,
        })

    df = pd.DataFrame(rows)
    meta = RunMeta(
        run_tag="",
        run_type="workload",
        dataset=cfg["dataset"]["name"],
        config_path=args.config,
        query_id=-1,
        top_k=-1,
        latency_budget_ms=-1.0,
        min_recall=-1.0,
        concurrency=-1,
        static_params={"preset": preset.name, "n": args.n, "seed": args.seed, "enable_hnsw": bool(args.enable_hnsw)},
        adaptive_chosen_index=None,
        adaptive_params=None,
        adaptive_regime=None,
        adaptive_reason=None,
    )
    paths = log_run(out_dir="results/ui_runs", meta=meta, results_df=df)
    print(f"[OK] Saved workload run: {paths['csv']}")
    print(f"[OK] Saved meta: {paths['json']}")


if __name__ == "__main__":
    main()
