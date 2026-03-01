"""
Demo script to showcase the Query Logger functionality.

This script initializes the engine with random data, wraps it with
the QueryLogger util, and runs a series of simulated queries.
The results are saved to 'logs/demo_query_log.jsonl'.
"""

import os
import sys
import time
import numpy as np

# Ensure src is in path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.adaptive.execution_engine import AdaptiveExecutionEngine
from src.indexes.flat_index import FlatIndex
from src.indexes.ivf_index import IVFIndex
from src.indexes.hnsw_index import HNSWIndex
from src.utils.query_logger import QueryLogger

def main():
    print(">>> Initializing Engine with Random Data...")

    # 1. Setup Data and Indexes
    dim = 64
    nb = 10000
    nq = 20

    base = np.random.random((nb, dim)).astype(np.float32)
    queries = np.random.random((nq, dim)).astype(np.float32)

    indexes = {
        "Flat": FlatIndex(dim),
        "IVF": IVFIndex(dim, nlist=100),
        "HNSW": HNSWIndex(dim, M=16, ef_construction=100)
    }

    for name, idx in indexes.items():
        print(f"    Building {name}...")
        idx.build(base)

    engine = AdaptiveExecutionEngine(
        indexes=indexes,
        dataset_size=nb,
        dimension=dim
    )

    # 2. Wrap Engine with Logger
    log_path = "logs/demo_query_log.jsonl"
    if os.path.exists(log_path):
        os.remove(log_path) # Clear old logs for demo

    print(f">>> Wrapping Engine with QueryLogger (log file: {log_path})")
    logged_engine = QueryLogger(engine, log_file=log_path)

    # 3. Run Queries
    print(f">>> Running {nq} queries...")

    # Use different constraints to trigger different strategies
    constraints = [
        (None, None),          # Default
        (1.0, 0.95),           # High Recall, Low Latency (challenging)
        (10.0, 0.5),           # Low Recall, High Latency (easy)
        (0.5, 0.8),            # Very low latency
    ]

    start_t = time.time()
    for i in range(nq):
        lat_budget, min_rec = constraints[i % len(constraints)]

        try:
            # We call the wrapper method, which calls the engine and logs
            result = logged_engine.search(
                queries[i],
                top_k=10,
                latency_budget_ms=lat_budget,
                min_recall=min_rec
            )
            strategy = result.strategy_used.index_name
            print(f"    Query {i+1}: Strategy={strategy}, Latency={result.latency_ms:.2f}ms")
        except Exception as e:
            print(f"    Query {i+1}: Failed ({e})")

    total_t = time.time() - start_t
    print(f">>> Completed in {total_t:.2f}s")

    # 4. Show Log Content
    print("\n>>> Content of Log File (First 3 entries):")
    with open(log_path, "r") as f:
        for i, line in enumerate(f):
            if i >= 3: break
            print(f"    {line.strip()}")

    print(f"\n>>> Full log saved to {log_path}")

if __name__ == "__main__":
    main()

