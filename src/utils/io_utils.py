"""
I/O utilities: dataset loading, configuration, directory management.
"""

import os
import yaml
import numpy as np
from typing import Tuple, Dict, Any


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def load_config(config_path: str = "config/default_config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_dataset(dataset_name: str, data_dir: str = "data") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load an HDF5 dataset from ann-benchmarks format.

    Args:
        dataset_name: e.g. "sift-128-euclidean"
        data_dir: directory containing .hdf5 files

    Returns:
        (base_vectors, query_vectors, ground_truth_neighbors, ground_truth_distances)
        - base_vectors:   (n, d) float32  — the vectors to index
        - query_vectors:  (nq, d) float32 — the query vectors
        - ground_truth_neighbors: (nq, gt_k) int64 — true neighbor indices
        - ground_truth_distances: (nq, gt_k) float32 — true neighbor distances
    """
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "The 'h5py' library is required to load datasets. "
            "Please install it with: pip install h5py"
        )

    filepath = os.path.join(data_dir, f"{dataset_name}.hdf5")
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found: {filepath}\n"
            f"Run: python data/download_datasets.py --dataset {dataset_name}"
        )

    print(f"Loading dataset: {filepath}")
    with h5py.File(filepath, "r") as f:
        base_vectors = np.array(f["train"], dtype=np.float32)
        query_vectors = np.array(f["test"], dtype=np.float32)
        ground_truth_neighbors = np.array(f["neighbors"], dtype=np.int64)
        ground_truth_distances = np.array(f["distances"], dtype=np.float32)

    print(f"  Base vectors:    {base_vectors.shape}")
    print(f"  Query vectors:   {query_vectors.shape}")
    print(f"  Ground truth:    {ground_truth_neighbors.shape}")
    return base_vectors, query_vectors, ground_truth_neighbors, ground_truth_distances


def save_results_csv(results: list, filepath: str):
    """Save a list of result dicts to CSV."""
    import pandas as pd
    ensure_dir(os.path.dirname(filepath))
    df = pd.DataFrame(results)
    df.to_csv(filepath, index=False)
    print(f"Results saved to: {filepath}")
    return df
