"""
Quick Test CLI for ADT-Project.

This script allows for quick interaction with the Adaptive Execution Engine
without launching the full Streamlit UI or running long benchmarks.
It helps in validating the configuration and the engine's decision-making logic.

Usage:
    python quick_test_cli.py
"""

import os
import sys
import argparse
import numpy as np
import yaml
import time
from typing import Dict, Any

# Ensure the project root is in the path so "src" can be imported
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from src.adaptive.execution_engine import AdaptiveExecutionEngine
    from src.indexes.flat_index import FlatIndex
    from src.indexes.ivf_index import IVFIndex
    from src.indexes.hnsw_index import HNSWIndex
    from src.utils.io_utils import load_config, load_dataset
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure you are running this script from the project root.")
    sys.exit(1)


def load_local_config(config_path: str = "config/default_config.yaml") -> Dict[str, Any]:
    """Safely load configuration with error handling."""
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def initialize_engine(config: Dict[str, Any]):
    """Initialize index and engine based on config."""
    print(">>> Initializing Indexes and Engine...")

    # Dummy data generation if dataset loading fails or for quick testing
    # In a real scenario, we would load the actual dataset
    dataset_name = config.get("dataset", {}).get("name", "sift1m")
    try:
        # define paths
        data_dir = config.get("dataset", {}).get("data_dir", "data")
        base, queries, _, _ = load_dataset(dataset_name, data_dir=data_dir)
        dim = base.shape[1]
        print(f"    Loaded real dataset '{dataset_name}' with shape {base.shape}")
    except Exception as e:
        print(f"    Warning: Could not load dataset '{dataset_name}' ({e}). Using random data.")
        dim = 128
        base = np.random.random((10000, dim)).astype(np.float32)
        queries = np.random.random((10, dim)).astype(np.float32)

    metric = config.get("dataset", {}).get("metric", "L2")

    # Build Indexes
    indexes = {}

    print("    Building Flat Index...")
    flat = FlatIndex(dimension=dim, metric=metric)
    flat.build(base)
    indexes["Flat"] = flat

    print("    Building IVF Index...")
    ivf_cfg = config.get("indexes", {}).get("ivf", {})
    ivf = IVFIndex(dimension=dim, nlist=ivf_cfg.get("nlist", 100), metric=metric)
    ivf.build(base)
    indexes["IVF"] = ivf

    print("    Building HNSW Index...")
    hnsw_cfg = config.get("indexes", {}).get("hnsw", {})
    hnsw = HNSWIndex(dimension=dim, M=hnsw_cfg.get("M", 16), ef_construction=hnsw_cfg.get("ef_construction", 100),
                     metric=metric)
    hnsw.build(base)
    indexes["HNSW"] = hnsw

    # Initialize Engine
    engine = AdaptiveExecutionEngine(
        indexes=indexes,
        latency_weight=0.5,  # Balanced
        recall_weight=0.5
    )

    return engine, queries


def run_cli_loop(engine, sample_queries):
    """Interactive loop for testing queries."""
    print("\n" + "=" * 50)
    print("ADT Quick Test CLI")
    print("=" * 50)
    print("Commands:")
    print("  test   - Run a test with a random query from the dataset")
    print("  custom - Input a custom vector (simulated)")
    print("  info   - Show engine status")
    print("  exit   - Quit")

    while True:
        try:
            cmd = input("\n(ADT-CLI) > ").strip().lower()
        except KeyboardInterrupt:
            print("\nExiting...")
            break

        if cmd == "exit":
            break

        elif cmd == "info":
            # Safety check if engine initialized correctly
            keys = list(engine.indexes.keys()) if engine and hasattr(engine, 'indexes') else []
            print(f"Loaded Strategies: {keys}")

        elif cmd == "test":
            try:
                idx = np.random.randint(0, len(sample_queries))
                q = sample_queries[idx]

                # Ensure 1-D vector for engine.search (it expects (dim,))
                if q.ndim > 1:
                    q = q.flatten()

                print(f"Running query #{idx}...")

                start_t = time.time()
                result = engine.search(q, top_k=10)
                elapsed = (time.time() - start_t) * 1000

                # Fix: Use index_name instead of name
                strategy_name = "Unknown"
                if hasattr(result.strategy_used, 'index_name'):
                    strategy_name = result.strategy_used.index_name
                elif hasattr(result.strategy_used, 'name'):
                    strategy_name = result.strategy_used.name

                params = getattr(result.strategy_used, 'params', {})

                print(f"    Strategy Selected: {strategy_name}")
                print(f"    Parameters: {params}")
                print(f"    Latency (Internal): {result.latency_ms:.4f} ms")
                print(f"    Wall Clock Time:    {elapsed:.4f} ms")
                print(f"    Neighbors Found:    {len(result.indices)}")

            except Exception as e:
                print(f"(!) Error during test: {e}")

        elif cmd == "custom":
            try:
                print("Generating a random query vector for simulation...")
                dim = sample_queries.shape[1]
                q = np.random.random((dim,)).astype(np.float32)

                start_t = time.time()
                result = engine.search(q, top_k=10)
                elapsed = (time.time() - start_t) * 1000

                # Fix: Use index_name instead of name
                strategy_name = "Unknown"
                if hasattr(result.strategy_used, 'index_name'):
                    strategy_name = result.strategy_used.index_name
                elif hasattr(result.strategy_used, 'name'):
                    strategy_name = result.strategy_used.name

                params = getattr(result.strategy_used, 'params', {})

                print(f"    Strategy Selected: {strategy_name}")
                print(f"    Parameters: {params}")
                print(f"    Latency (Internal): {result.latency_ms:.4f} ms")
                print(f"    Wall Clock Time:    {elapsed:.4f} ms")
                print(f"    Neighbors Found:    {len(result.indices)}")
            except Exception as e:
                print(f"(!) Error during custom query: {e}")

        else:
            if cmd:
                print("Unknown command.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick Test CLI for ADT Project")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to config file")
    args = parser.parse_args()

    cfg = load_local_config(args.config)
    engine, queries = initialize_engine(cfg)
    run_cli_loop(engine, queries)
