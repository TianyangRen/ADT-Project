import numpy as np
import streamlit as st

from src.utils.io_utils import load_config, load_dataset
from src.indexes.flat_index import FlatIndex
from src.indexes.ivf_index import IVFIndex
from src.indexes.hnsw_index import HNSWIndex
from src.adaptive.execution_engine import AdaptiveExecutionEngine


st.set_page_config(page_title="ADT Adaptive Vector Search Demo", layout="wide")


@st.cache_resource
def build_engine(config_path: str):
    cfg = load_config(config_path)

    dataset_name = cfg["dataset"]["name"]
    data_dir = cfg["dataset"].get("data_dir", "data")

    base, queries, gt_neighbors, gt_distances = load_dataset(dataset_name, data_dir=data_dir)

    dim = base.shape[1]
    metric = cfg["dataset"].get("metric", "L2")

    flat = FlatIndex(dimension=dim, metric=metric)
    flat.build(base)

    ivf_cfg = cfg["indexes"]["ivf"]
    ivf = IVFIndex(
        dimension=dim,
        nlist=ivf_cfg.get("nlist", 4096),
        metric=metric
    )
    ivf.build(base)

    hnsw_cfg = cfg["indexes"]["hnsw"]
    hnsw = HNSWIndex(
        dimension=dim,
        M=hnsw_cfg.get("M", 32),
        ef_construction=hnsw_cfg.get("ef_construction", 200),
        metric=metric
    )
    hnsw.build(base)

    indexes = {"Flat": flat, "IVF": ivf, "HNSW": hnsw}

    adaptive_cfg = cfg.get("adaptive", {})
    engine = AdaptiveExecutionEngine(
        indexes=indexes,
        cost_model=None,
        dataset_size=base.shape[0],
        dimension=dim,
        default_latency_budget_ms=adaptive_cfg.get("default_latency_budget_ms", 5.0),
        default_min_recall=adaptive_cfg.get("default_min_recall", 0.9),
        latency_weight=adaptive_cfg.get("latency_weight", 0.6),
        recall_weight=adaptive_cfg.get("recall_weight", 0.4),
    )
    return cfg, base, queries, engine


def parse_vector(text: str, dim: int) -> np.ndarray:
    parts = [p.strip() for p in text.replace("\n", " ").replace(",", " ").split() if p.strip()]
    arr = np.array([float(x) for x in parts], dtype=np.float32)
    if arr.shape[0] != dim:
        raise ValueError(f"Vector dim mismatch: expect {dim}, got {arr.shape[0]}")
    return arr


st.title("ADT Project – Adaptive Vector Similarity Search (UI Demo)")
st.caption("Query-time adaptive selection + latency/recall constraints + explanation")

with st.sidebar:
    st.header("Config")
    config_path = st.text_input("Config path", value="config/default_config.yaml")

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


cfg, base, queries, engine = build_engine(config_path)
dim = base.shape[1]

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Run Query")

    if query_mode == "Use a dataset query vector":
        qid = int(min(max(query_index, 0), queries.shape[0] - 1))
        q = queries[qid]
        st.write(f"Using query #{qid} from dataset `{cfg['dataset']['name']}`")
    else:
        default_text = " ".join(["0"] * dim)
        vec_text = st.text_area(f"Paste a {dim}-dim vector", value=default_text, height=120)
        try:
            q = parse_vector(vec_text, dim)
        except Exception as e:
            st.error(str(e))
            st.stop()

    if st.button("Search (Adaptive)", type="primary"):
        result = engine.search(
            query=q,
            top_k=int(top_k),
            latency_budget_ms=float(latency_budget_ms),
            min_recall=float(min_recall),
            concurrency=int(concurrency),
        )

        st.success("Search completed.")

        st.markdown("### Strategy Decision")
        st.write("**Chosen index:**", result.strategy_used.index_name)
        st.write("**Params:**", dict(result.strategy_used.params))
        st.write("**Regime:**", result.selection_result.regime)
        st.write("**Reason:**", result.selection_result.reason)

        st.markdown("### Performance")
        st.write(f"**Actual latency:** {result.latency_ms:.3f} ms")
        st.write(f"**Predicted latency:** {result.selection_result.chosen_estimate.estimated_latency_ms:.3f} ms")
        st.write(f"**Predicted recall:** {result.selection_result.chosen_estimate.estimated_recall:.3f}")

        st.markdown("### Top-K Results")
        st.dataframe(
            {
                "rank": list(range(1, int(top_k) + 1)),
                "neighbor_id": result.indices.tolist(),
                "distance": result.distances.tolist(),
            },
            use_container_width=True
        )

with col2:
    st.subheader("Monitoring / Debug")
    stats = engine.get_stats()
    st.json(stats, expanded=False)
