import time
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

st.set_page_config(page_title="ADT Adaptive Vector Search Demo", layout="wide")


def recall_at_k(result_ids: np.ndarray, gt_ids: np.ndarray, k: int) -> float:
    """
    result_ids: shape (k,)
    gt_ids: shape (>=k,) ground truth neighbors
    """
    k = int(k)
    if k <= 0:
        return 0.0
    gt_k = set(gt_ids[:k].tolist())
    res_k = set(result_ids[:k].tolist())
    if len(gt_k) == 0:
        return 0.0
    return len(gt_k.intersection(res_k)) / float(k)


def build_candidates_for_available_indexes(available: set) -> list:
    """
    Filter default candidate pool to only include strategies whose index exists.
    This avoids AdaptiveEngine choosing HNSW when HNSW is disabled.
    """
    out = []
    for s in StrategySelector.DEFAULT_CANDIDATES:
        if s.index_name in available:
            out.append(ExecutionStrategy(s.index_name, dict(s.params)))
    return out


@st.cache_resource
def build_system(config_path: str, enable_hnsw: bool):
    """
    Build dataset + indexes + adaptive engine (cached).
    Toggle enable_hnsw to avoid expensive HNSW build on every demo run.
    """
    cfg = load_config(config_path)

    dataset_name = cfg["dataset"]["name"]
    data_dir = cfg["dataset"].get("data_dir", "data")
    metric = cfg["dataset"].get("metric", "L2")

    base, queries, gt_neighbors, gt_distances = load_dataset(dataset_name, data_dir=data_dir)

    dim = base.shape[1]

    # --- Build Flat ---
    flat = FlatIndex(dimension=dim, metric=metric)
    flat.build(base)

    # --- Build IVF ---
    ivf_cfg = cfg["indexes"]["ivf"]
    ivf = IVFIndex(
        dimension=dim,
        nlist=ivf_cfg.get("nlist", 256),
        metric=metric
    )
    ivf.build(base)

    indexes = {"Flat": flat, "IVF": ivf}

    # --- Build HNSW (optional) ---
    if enable_hnsw:
        hnsw_cfg = cfg["indexes"]["hnsw"]
        hnsw = HNSWIndex(
            dimension=dim,
            M=hnsw_cfg.get("M", 32),
            ef_construction=hnsw_cfg.get("ef_construction", 200),
            metric=metric
        )
        hnsw.build(base)
        indexes["HNSW"] = hnsw

    # --- Build adaptive engine with candidates limited to available indexes ---
    available = set(indexes.keys())
    candidates = build_candidates_for_available_indexes(available)

    # config.yaml keys are: dataset, indexes, profiling, output (no "adaptive")
    # so we use safe defaults here.
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


def parse_vector(text: str, dim: int) -> np.ndarray:
    parts = [p.strip() for p in text.replace("\n", " ").replace(",", " ").split() if p.strip()]
    arr = np.array([float(x) for x in parts], dtype=np.float32)
    if arr.shape[0] != dim:
        raise ValueError(f"Vector dim mismatch: expect {dim}, got {arr.shape[0]}")
    return arr


st.title("ADT Project – Adaptive Vector Similarity Search (UI Demo)")
st.caption("Tabs: (1) Adaptive Search  (2) Benchmark Compare (Static vs Adaptive)")

with st.sidebar:
    st.header("Config")
    config_path = st.text_input("Config path", value="config/default_config.yaml")

    st.divider()
    st.header("Startup Mode")
    enable_hnsw = st.checkbox("Enable HNSW (slow to build, better demo completeness)", value=False)

    st.divider()
    st.header("Query Constraints")
    top_k = st.slider("top_k", min_value=1, max_value=100, value=10, step=1)
    latency_budget_ms = st.slider("latency_budget_ms", min_value=1.0, max_value=50.0, value=5.0, step=0.5)
    min_recall = st.slider("min_recall", min_value=0.1, max_value=1.0, value=0.9, step=0.05)
    concurrency = st.slider("concurrency (simulated)", min_value=1, max_value=64, value=1, step=1)

    st.divider()
    st.header("Query Source")
    query_mode = st.radio("Choose input", ["Use a dataset query vector", "Paste a custom vector"])
    query_index = st.number_input("dataset query index", min_value=0, value=0, step=1)

cfg, base, queries, gt_neighbors, engine = build_system(config_path, enable_hnsw)
dim = base.shape[1]

tabs = st.tabs(["Adaptive Search", "Benchmark Compare", "Monitor/Stats"])

# -------------------------
# Tab 1: Adaptive Search
# -------------------------
with tabs[0]:
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.subheader("Run Adaptive Search")

        if query_mode == "Use a dataset query vector":
            qid = int(min(max(query_index, 0), queries.shape[0] - 1))
            q = queries[qid]
            gt = gt_neighbors[qid]
            st.write(f"Using query #{qid} from dataset `{cfg['dataset']['name']}`")
        else:
            default_text = " ".join(["0"] * dim)
            vec_text = st.text_area(f"Paste a {dim}-dim vector (space/comma separated)", value=default_text, height=120)
            q = parse_vector(vec_text, dim)
            gt = None

        if st.button("Search (Adaptive)", type="primary"):
            r = engine.search(
                query=q,
                top_k=int(top_k),
                latency_budget_ms=float(latency_budget_ms),
                min_recall=float(min_recall),
                concurrency=int(concurrency),
            )

            st.success("Search completed.")

            st.markdown("### Strategy Decision")
            st.write("**Chosen index:**", r.strategy_used.index_name)
            st.write("**Params:**", dict(r.strategy_used.params))
            st.write("**Regime:**", r.selection_result.regime)
            st.write("**Reason:**", r.selection_result.reason)

            st.markdown("### Performance")
            st.write(f"**Actual latency:** {r.latency_ms:.3f} ms")
            st.write(f"**Predicted latency:** {r.selection_result.chosen_estimate.estimated_latency_ms:.3f} ms")
            st.write(f"**Predicted recall:** {r.selection_result.chosen_estimate.estimated_recall:.3f}")

            if gt is not None:
                rec = recall_at_k(r.indices, gt, int(top_k))
                st.write(f"**Recall@{int(top_k)} (vs ground truth):** {rec:.3f}")

            st.markdown("### Top-K Results")
            st.dataframe(
                {
                    "rank": list(range(1, int(top_k) + 1)),
                    "neighbor_id": r.indices.tolist(),
                    "distance": r.distances.tolist(),
                },
                width="stretch"
            )

    with col2:
        st.subheader("Decision Debug (All Candidates)")
        st.write("This shows the cost model estimates for each candidate strategy.")

        # For a quick view, run a dry adaptive selection by executing one search with minimal overhead.
        # We do it only when user clicks, to avoid extra work each rerun.
        if st.button("Show candidate estimates table"):
            qid = int(min(max(query_index, 0), queries.shape[0] - 1))
            q = queries[qid]

            # Run search once to get selection_result which includes all_estimates
            r = engine.search(
                query=q,
                top_k=int(top_k),
                latency_budget_ms=float(latency_budget_ms),
                min_recall=float(min_recall),
                concurrency=int(concurrency),
            )

            rows = []
            for strat, est in zip(engine.selector.candidates, r.selection_result.all_estimates):
                rows.append({
                    "index": strat.index_name,
                    "params": dict(strat.params),
                    "pred_latency_ms": float(est.estimated_latency_ms),
                    "pred_recall": float(est.estimated_recall),
                })
            df = pd.DataFrame(rows).sort_values(["pred_latency_ms"])
            st.dataframe(df, width="stretch")

            # ---- Save run outputs (disk logging) ----
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
                    static_params={
                        "ivf_nprobe": int(locals().get("ivf_nprobe", 0) or 0),
                        "hnsw_ef": int(locals().get("hnsw_ef", 0) or 0),
                    },
                    adaptive_chosen_index=rr.strategy_used.index_name,
                    adaptive_params=dict(rr.strategy_used.params),
                    adaptive_regime=rr.selection_result.regime,
                    adaptive_reason=rr.selection_result.reason,
                )

                paths = log_run(out_dir="results/ui_runs", meta=meta, results_df=df)
                st.success(f"Saved: {paths['csv']} and {paths['json']}")

# -------------------------
# Tab 2: Benchmark Compare
# -------------------------
with tabs[1]:
    st.subheader("Benchmark Compare: Static (Flat/IVF/HNSW) vs Adaptive")

    if query_mode != "Use a dataset query vector":
        st.warning("Benchmark Compare needs ground-truth neighbors, so please choose 'Use a dataset query vector'.")
        st.stop()

    qid = int(min(max(query_index, 0), queries.shape[0] - 1))
    q = queries[qid]
    gt = gt_neighbors[qid]

    st.write(f"Query #{qid}  | dataset: `{cfg['dataset']['name']}`  | dim={dim}")

    c1, c2, c3 = st.columns(3)
    with c1:
        ivf_nprobe = st.select_slider("IVF nprobe (static)", options=[1, 4, 16, 64, 128, 256], value=16)
    with c2:
        hnsw_ef = st.select_slider("HNSW ef_search (static)", options=[16, 32, 64, 128, 256, 512], value=64)
    with c3:
        warmup = st.checkbox("Warmup each method once (reduce first-run noise)", value=True)

    if st.button("Run Compare", type="primary"):
        rows = []

        def run_one(name: str, fn):
            if warmup:
                try:
                    fn(warmup_only=True)
                except TypeError:
                    # if fn doesn't support warmup_only, just call once
                    fn()
            t0 = time.perf_counter()
            ids, dists = fn()
            latency_ms = (time.perf_counter() - t0) * 1000.0
            rec = recall_at_k(ids, gt, int(top_k))
            rows.append({
                "method": name,
                "latency_ms": latency_ms,
                f"recall@{int(top_k)}": rec,
                "top1_id": int(ids[0]) if len(ids) > 0 else -1,
            })
            return ids, dists, latency_ms, rec

        # --- Flat ---
        flat = engine.indexes["Flat"]
        def flat_run(warmup_only=False):
            D, I = flat.search(q.reshape(1, -1), int(top_k))
            if warmup_only:
                return
            return I[0], D[0]
        run_one("Flat (exact)", flat_run)

        # --- IVF ---
        ivf = engine.indexes["IVF"]
        def ivf_run(warmup_only=False):
            D, I = ivf.search(q.reshape(1, -1), int(top_k), nprobe=int(ivf_nprobe))
            if warmup_only:
                return
            return I[0], D[0]
        run_one(f"IVF (nprobe={ivf_nprobe})", ivf_run)

        # --- HNSW (optional) ---
        if "HNSW" in engine.indexes:
            hnsw = engine.indexes["HNSW"]
            def hnsw_run(warmup_only=False):
                D, I = hnsw.search(q.reshape(1, -1), int(top_k), ef_search=int(hnsw_ef))
                if warmup_only:
                    return
                return I[0], D[0]
            run_one(f"HNSW (ef_search={hnsw_ef})", hnsw_run)
        else:
            st.info("HNSW is disabled. Enable it in the sidebar to include HNSW in comparison.")

        # --- Adaptive ---
        def adaptive_run(warmup_only=False):
            r = engine.search(
                query=q,
                top_k=int(top_k),
                latency_budget_ms=float(latency_budget_ms),
                min_recall=float(min_recall),
                concurrency=int(concurrency),
            )
            if warmup_only:
                return
            return r.indices, r.distances, r
        # warmup + real run
        if warmup:
            try:
                adaptive_run(warmup_only=True)
            except Exception:
                pass
        t0 = time.perf_counter()
        ids, dists, rr = adaptive_run(warmup_only=False)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        rec = recall_at_k(ids, gt, int(top_k))
        rows.append({
            "method": f"Adaptive => {rr.strategy_used.index_name} {dict(rr.strategy_used.params)}",
            "latency_ms": latency_ms,
            f"recall@{int(top_k)}": rec,
            "top1_id": int(ids[0]) if len(ids) > 0 else -1,
        })

        df = pd.DataFrame(rows).sort_values("latency_ms")
        st.markdown("### Results Table (lower latency is better)")
        st.dataframe(df, width="stretch")

        st.markdown("### Adaptive Explanation")
        st.write("**Chosen index:**", rr.strategy_used.index_name)
        st.write("**Params:**", dict(rr.strategy_used.params))
        st.write("**Regime:**", rr.selection_result.regime)
        st.write("**Reason:**", rr.selection_result.reason)

# -------------------------
# Tab 3: Monitor/Stats
# -------------------------
with tabs[2]:
    st.subheader("Performance Monitor / Stats")
    st.json(engine.get_stats(), expanded=False)

    st.markdown("### Notes")
    st.write(
        "- Your proposal highlights heterogeneous query regimes (K, latency, recall, concurrency) and why static tuning fails."
        " This Compare tab is the cleanest way to demonstrate that claim."
    )
