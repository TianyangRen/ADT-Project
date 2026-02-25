#!/usr/bin/env python3
"""
Download standard ANN benchmark datasets from ann-benchmarks.com.

Usage:
    python data/download_datasets.py                    # Download default (sift)
    python data/download_datasets.py --dataset all      # Download all datasets
    python data/download_datasets.py --dataset glove-100-angular
"""

import os
import sys
import argparse
import urllib.request
import time

DATASETS = {
    "sift-128-euclidean": {
        "url": "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
        "description": "SIFT1M: 128d, 1M base vectors, 10K queries, L2 distance",
        "size_mb": 501,
    },
    "glove-100-angular": {
        "url": "http://ann-benchmarks.com/glove-100-angular.hdf5",
        "description": "GloVe: 100d, 1.2M base vectors, 10K queries, angular distance",
        "size_mb": 463,
    },
}

DEFAULT_DATASET = "sift-128-euclidean"


def download_with_progress(url: str, filepath: str):
    """Download a file with a simple progress indicator."""
    print(f"  URL: {url}")
    print(f"  Saving to: {filepath}")

    start_time = time.time()

    # Use a proper User-Agent to avoid 403 from some servers
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ADT-Project/1.0"
    })

    response = urllib.request.urlopen(req)
    total_size = int(response.headers.get("Content-Length", 0))

    block_size = 1024 * 1024  # 1 MB
    downloaded = 0

    with open(filepath, "wb") as f:
        while True:
            data = response.read(block_size)
            if not data:
                break
            f.write(data)
            downloaded += len(data)
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                mb_done = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                elapsed = time.time() - start_time
                speed = mb_done / elapsed if elapsed > 0 else 0
                sys.stdout.write(
                    f"\r  Progress: {percent:5.1f}% ({mb_done:.1f}/{mb_total:.1f} MB) "
                    f"@ {speed:.1f} MB/s"
                )
                sys.stdout.flush()

    elapsed = time.time() - start_time
    print(f"\n  Done in {elapsed:.1f}s\n")


def download_dataset(name: str, data_dir: str = "data"):
    """Download a single dataset if not already present."""
    if name not in DATASETS:
        print(f"ERROR: Unknown dataset '{name}'. Available: {list(DATASETS.keys())}")
        return False

    info = DATASETS[name]
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, f"{name}.hdf5")

    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"[SKIP] {name} already exists ({size_mb:.0f} MB): {filepath}")
        return True

    print(f"[DOWNLOAD] {name}")
    print(f"  {info['description']}")
    print(f"  Estimated size: ~{info['size_mb']} MB")

    try:
        download_with_progress(info["url"], filepath)
        return True
    except Exception as e:
        print(f"  ERROR: Download failed: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False


def verify_dataset(name: str, data_dir: str = "data"):
    """Verify a downloaded dataset can be opened and has expected keys."""
    import h5py

    filepath = os.path.join(data_dir, f"{name}.hdf5")
    if not os.path.exists(filepath):
        print(f"[VERIFY] {name}: NOT FOUND")
        return False

    try:
        with h5py.File(filepath, "r") as f:
            keys = list(f.keys())
            required_keys = {"train", "test", "neighbors", "distances"}
            if not required_keys.issubset(set(keys)):
                print(f"[VERIFY] {name}: MISSING KEYS (found {keys})")
                return False

            n_base = f["train"].shape[0]
            dim = f["train"].shape[1]
            n_query = f["test"].shape[0]
            n_neighbors = f["neighbors"].shape[1]

            print(f"[VERIFY] {name}: OK")
            print(f"  Base vectors:  {n_base:,} x {dim}d")
            print(f"  Query vectors: {n_query:,} x {dim}d")
            print(f"  Ground truth:  top-{n_neighbors} neighbors")
            return True
    except Exception as e:
        print(f"[VERIFY] {name}: ERROR - {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download ANN benchmark datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=f"Dataset name or 'all'. Available: {list(DATASETS.keys())}",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to store datasets (default: data/)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing datasets, don't download",
    )
    args = parser.parse_args()

    if args.dataset == "all":
        names = list(DATASETS.keys())
    else:
        names = [args.dataset]

    print("=" * 60)
    print("ANN Benchmark Dataset Downloader")
    print("=" * 60)

    for name in names:
        if not args.verify_only:
            download_dataset(name, args.data_dir)
        verify_dataset(name, args.data_dir)

    print("\nAll done!")


if __name__ == "__main__":
    main()
