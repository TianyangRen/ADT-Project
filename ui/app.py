import time
import random
import numpy as np
import pandas as pd
import streamlit as st

from src.utils.io_utils import load_config, load_dataset
from src.indexes.flat_index import FlatIndex
from src.indexes.ivf_index import IVFIndex
from src.indexes.hnsw_index import HNSWIndex

from src.adaptive.execution_engine import AdaptiveExecutionEngine
from src.adaptive.strategy_selector import ExecutionStrategy, StrategySelector

from ui.lib.result_logger import RunMeta, log_run
from ui.lib.workload_presets import load_presets, sample_constraints

st.set_page_config(page_title="ADT Adaptive Vector Search Demo", layout="wide")


def recall_at_k(result_ids: np.ndarray, gt_ids: np.ndarray, k: int) -> float:
    k = int(k)
    if k <= 0:
        return 0.0
    gt_k = set(gt_ids[:k].tolist())
    res_k = set(result_ids[:k].tolist())
    return len(gt_k.intersection(res_k)) / float(k)


def build_candidates_for_available_indexes(available: set) -> list:
    out = []
    for s in StrategySelector.DEFAULT_CANDIDATES:
        if s.index_name in available:
            out.append(ExecutionStrategy(s.index_name, dict(s.params)))
    return out


@st.cache_resource
def build_system(config_path: str, enable_hnsw: bool):
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
    return cfg, base, queries, gt_neighbors, engine


st.title("ADT Project – Adaptive Vector Similarity Search (UI Demo)")
st.caption("Adds: workload regimes + explainability panel (query-time decision reasons)")

with st.sidebar:
    st.header("Config")
    config_path = st.text_input("Config path", value="config/default_config.yaml")

    st.divider()
    st.header("Startup Mode")
    enable_hnsw = st.checkbox("Enable HNSW (slow to build)", value=False)

    st.divider()
    st.header("Single Query Constraints")
    top_k = st.slider("top_k", min_value=1, max_value=100, value=10, step=1)
    latency_budget_ms = st.slider("latency_budget_ms", min_value=1.0, max_value=50.0, value=5.0, step=0.5)
    min_recall = st.slider("min_recall", min_value=0.1, max_value=1.0, value=0.9, step=0.05)
    concurrency = st.slider("concurrency (simulated)", min_value=1, max_value=64, value=1, step=1)

    st.divider()
    st.header("Query Source")
    query_index = st.number_input("dataset query index", min_value=0, value=0, step=1)

cfg, base, queries, gt_neighbors, engine = build_system(config_path, enable_hnsw)
dim = base.shape[1]

tabs = st.tabs(["Adaptive Search", "Benchmark Compare", "Workload Scenarios", "Explainability", "Monitor/Stats"])

# -------------------------
# Tab 1: Adaptive Search
# -------------------------
with tabs[0]:
    qid = int(min(max(query_index, 0), queries.shape[0] - 1))
    q = queries[qid]
    gt = gt_neighbors[qid]
    st.write(f"Using query #{qid} from dataset `{cfg['dataset']['name']}`")

    if st.button("Search (Adaptive)", type="primary"):
        r = engine.search(
            query=q,
            top_k=int(top_k),
            latency_budget_ms=float(latency_budget_ms),
            min_recall=float(min_recall),
            concurrency=int(concurrency),
        )
        rec = recall_at_k(r.indices, gt, int(top_k))

        st.success("Search completed.")
        st.write("**Chosen index:**", r.strategy_used.index_name)
        st.write("**Params:**", dict(r.strategy_used.params))
        st.write("**Regime:**", r.selection_result.regime)
        st.write("**Reason:**", r.selection_result.reason)
        st.write(f"**Actual latency:** {r.latency_ms:.3f} ms")
        st.write(f"**Recall@{int(top_k)}:** {rec:.3f}")

        st.dataframe(
            {
                "rank": list(range(1, int(top_k) + 1)),
                "neighbor_id": r.indices.tolist(),
                "distance": r.distances.tolist(),
            },
            width="stretch",
        )

# -------------------------
# Tab 2: Benchmark Compare (Static vs Adaptive)
# -------------------------
with tabs[1]:
    st.subheader("Benchmark Compare: Static vs Adaptive (same query)")

    qid = int(min(max(query_index, 0), queries.shape[0] - 1))
    q = queries[qid]
    gt = gt_neighbors[qid]

    c1, c2, c3 = st.columns(3)
    with c1:
        ivf_nprobe = st.select_slider("IVF nprobe (static)", options=[1, 4, 16, 64, 128, 256], value=16)
    with c2:
        hnsw_ef = st.select_slider("HNSW ef_search (static)", options=[16, 32, 64, 128, 256, 512], value=64)
    with c3:
        warmup = st.checkbox("Warmup each method once", value=True)

    if st.button("Run Compare", type="primary"):
        rows = []

        def run_one(name: str, fn):
            if warmup:
                fn(warmup_only=True)
            t0 = time.perf_counter()
            ids, dists = fn(warmup_only=False)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            rec = recall_at_k(ids, gt, int(top_k))
            rows.append({"method": name, "latency_ms": latency_ms, f"recall@{int(top_k)}": rec})
            return ids, dists, latency_ms, rec

        # Flat
        flat = engine.indexes["Flat"]
        def flat_run(warmup_only=False):
            D, I = flat.search(q.reshape(1, -1), int(top_k))
            if warmup_only:
                return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
            return I[0], D[0]
        run_one("Flat (exact)", flat_run)

        # IVF
        ivf = engine.indexes["IVF"]
        def ivf_run(warmup_only=False):
            D, I = ivf.search(q.reshape(1, -1), int(top_k), nprobe=int(ivf_nprobe))
            if warmup_only:
                return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
            return I[0], D[0]
        run_one(f"IVF (nprobe={ivf_nprobe})", ivf_run)

        # HNSW (optional)
        if "HNSW" in engine.indexes:
            hnsw = engine.indexes["HNSW"]
            def hnsw_run(warmup_only=False):
                D, I = hnsw.search(q.reshape(1, -1), int(top_k), ef_search=int(hnsw_ef))
                if warmup_only:
                    return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
                return I[0], D[0]
            run_one(f"HNSW (ef_search={hnsw_ef})", hnsw_run)
        else:
            st.info("HNSW disabled. Enable it in sidebar if you want HNSW in compare.")

        # Adaptive
        if warmup:
            engine.search(query=q, top_k=int(top_k), latency_budget_ms=float(latency_budget_ms), min_recall=float(min_recall), concurrency=int(concurrency))

        t0 = time.perf_counter()
        rr = engine.search(
            query=q,
            top_k=int(top_k),
            latency_budget_ms=float(latency_budget_ms),
            min_recall=float(min_recall),
            concurrency=int(concurrency),
        )
        lat_adp = (time.perf_counter() - t0) * 1000.0
        rec_adp = recall_at_k(rr.indices, gt, int(top_k))
        rows.append({"method": f"Adaptive => {rr.strategy_used.index_name} {dict(rr.strategy_used.params)}", "latency_ms": lat_adp, f"recall@{int(top_k)}": rec_adp})

        df = pd.DataFrame(rows).sort_values("latency_ms")
        st.dataframe(df, width="stretch")

        st.markdown("### Adaptive Explanation")
        st.write("**Chosen index:**", rr.strategy_used.index_name)
        st.write("**Params:**", dict(rr.strategy_used.params))
        st.write("**Regime:**", rr.selection_result.regime)
        st.write("**Reason:**", rr.selection_result.reason)

        # ---- Save run outputs ----
        save_run = st.checkbox("Save this compare run to results/ui_runs", value=True)
        if save_run:
            meta = RunMeta(
                run_tag="",
                run_type="compare",
                dataset=cfg["dataset"]["name"],
                config_path=config_path,
                query_id=int(qid),
                top_k=int(top_k),
                latency_budget_ms=float(latency_budget_ms),
                min_recall=float(min_recall),
                concurrency=int(concurrency),
                static_params={"ivf_nprobe": int(ivf_nprobe), "hnsw_ef": int(hnsw_ef)},
                adaptive_chosen_index=rr.strategy_used.index_name,
                adaptive_params=dict(rr.strategy_used.params),
                adaptive_regime=rr.selection_result.regime,
                adaptive_reason=rr.selection_result.reason,
            )
            paths = log_run(out_dir="results/ui_runs", meta=meta, results_df=df)
            st.success(f"Saved: {paths['csv']} and {paths['json']}")

# -------------------------
# Tab 3: Workload Scenarios
# -------------------------
with tabs[2]:
    st.subheader("Workload Scenarios (Batch)")
    presets = load_presets()
    preset_name = st.selectbox("Preset", options=list(presets.keys()), index=list(presets.keys()).index("mixed") if "mixed" in presets else 0)
    preset = presets[preset_name]
    st.caption(preset.description)

    n_queries = st.slider("Number of queries to sample", min_value=5, max_value=200, value=30, step=5)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    baseline_nprobe = st.select_slider("Baseline IVF nprobe", options=[1, 4, 16, 64, 128, 256], value=16)

    if st.button("Run Workload Batch", type="primary"):
        rng = random.Random(int(seed))
        rows = []
        for _ in range(int(n_queries)):
            qid = rng.randrange(0, queries.shape[0])
            q = queries[qid]
            gt = gt_neighbors[qid]

            cons = sample_constraints(preset, rng)
            k = cons["top_k"]

            # Baseline IVF
            ivf = engine.indexes["IVF"]
            t0 = time.perf_counter()
            D, I = ivf.search(q.reshape(1, -1), k, nprobe=int(baseline_nprobe))
            lat_ivf = (time.perf_counter() - t0) * 1000.0
            rec_ivf = recall_at_k(I[0], gt, k)

            # Adaptive
            t0 = time.perf_counter()
            r = engine.search(
                query=q,
                top_k=k,
                latency_budget_ms=cons["latency_budget_ms"],
                min_recall=cons["min_recall"],
                concurrency=cons["concurrency"],
            )
            lat_adp = (time.perf_counter() - t0) * 1000.0
            rec_adp = recall_at_k(r.indices, gt, k)

            rows.append({
                "preset": preset.name,
                "query_id": qid,
                "top_k": k,
                "latency_budget_ms": cons["latency_budget_ms"],
                "min_recall": cons["min_recall"],
                "concurrency": cons["concurrency"],
                "method": f"IVF(nprobe={int(baseline_nprobe)})",
                "latency_ms": lat_ivf,
                f"recall@{k}": rec_ivf,
            })
            rows.append({
                "preset": preset.name,
                "query_id": qid,
                "top_k": k,
                "latency_budget_ms": cons["latency_budget_ms"],
                "min_recall": cons["min_recall"],
                "concurrency": cons["concurrency"],
                "method": f"Adaptive=>{r.strategy_used.index_name}",
                "latency_ms": lat_adp,
                f"recall@{k}": rec_adp,
                "adaptive_reason": r.selection_result.reason,
            })

        df = pd.DataFrame(rows)
        st.dataframe(df.head(50), width="stretch")
        st.info("Showing first 50 rows. Full batch is saved if you check 'Save batch' below.")

        save_batch = st.checkbox("Save batch to results/ui_runs", value=True)
        if save_batch:
            meta = RunMeta(
                run_tag="",
                run_type="workload",
                dataset=cfg["dataset"]["name"],
                config_path=config_path,
                query_id=-1,
                top_k=-1,
                latency_budget_ms=-1.0,
                min_recall=-1.0,
                concurrency=-1,
                static_params={"preset": preset.name, "n": int(n_queries), "seed": int(seed), "baseline_nprobe": int(baseline_nprobe), "enable_hnsw": bool(enable_hnsw)},
            )
            paths = log_run(out_dir="results/ui_runs", meta=meta, results_df=df)
            st.success(f"Saved: {paths['csv']} and {paths['json']}")

# -------------------------
# Tab 4: Explainability Panel
# -------------------------
with tabs[3]:
    st.subheader("Explainability Panel (Why this plan?)")

    qid = int(min(max(query_index, 0), queries.shape[0] - 1))
    q = queries[qid]

    if st.button("Explain decision for current query", type="primary"):
        r = engine.search(
            query=q,
            top_k=int(top_k),
            latency_budget_ms=float(latency_budget_ms),
            min_recall=float(min_recall),
            concurrency=int(concurrency),
        )

        st.write("**Chosen index:**", r.strategy_used.index_name)
        st.write("**Chosen params:**", dict(r.strategy_used.params))
        st.write("**Regime:**", r.selection_result.regime)
        st.write("**Reason:**", r.selection_result.reason)

        # Candidate estimate table with constraint checks
        rows = []
        for strat, est in zip(engine.selector.candidates, r.selection_result.all_estimates):
            pred_lat = float(est.estimated_latency_ms)
            pred_rec = float(est.estimated_recall)
            rows.append({
                "index": strat.index_name,
                "params": dict(strat.params),
                "pred_latency_ms": pred_lat,
                "pred_recall": pred_rec,
                "meets_latency": pred_lat <= float(latency_budget_ms),
                "meets_recall": pred_rec >= float(min_recall),
            })

        df = pd.DataFrame(rows).sort_values(["meets_latency", "meets_recall", "pred_latency_ms"], ascending=[False, False, True])

        def highlight_chosen(row):
            chosen = (row["index"] == r.strategy_used.index_name) and (row["params"] == dict(r.strategy_used.params))
            return ["background-color: #E6FFED" if chosen else "" for _ in row]

        st.markdown("### Candidate estimates (cost model)")
        st.dataframe(df, width="stretch")

        st.caption("meets_latency / meets_recall are based on predicted values, matching the proposal’s cost-aware explainability goal.")

# -------------------------
# Tab 5: Monitor/Stats
# -------------------------
with tabs[4]:
    st.subheader("Performance Monitor / Stats")
    st.json(engine.get_stats(), expanded=False)
