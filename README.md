# ADT-Project

**Cost-Aware, Query-Time Adaptive Execution Framework for Vector Similarity Search**

## Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset (~500MB)
```bash
python data/download_datasets.py
```

### 3. Run Phase 1 — Baseline Benchmark
```bash
python experiments/01_baseline_benchmark.py
```
Output: `results/baseline_results.csv`

### 4. Run Phase 2 — Profiling & Failure Analysis
```bash
python experiments/02_profiling_analysis.py
```
Output: `results/profiling_sweep.csv` + `results/figures/*.png`

### 5. Generate Plots Separately (if needed)
```bash
python experiments/visualize_profiling.py
```

## Project Structure
```
ADT-Project/
├── config/default_config.yaml    # All configurable parameters
├── data/                         # Datasets (downloaded, not committed)
├── src/
│   ├── indexes/                  # Flat, IVF, HNSW index wrappers
│   ├── profiler/                 # Latency & recall profiling
│   └── utils/                    # Metrics, I/O helpers
├── experiments/                  # Runnable scripts for each phase
├── results/                      # Output CSV and figures
└── requirements.txt
```

## Datasets
- **SIFT1M**: 1M vectors, 128d, L2 distance (primary)
- **GloVe-100**: 1.2M vectors, 100d, angular distance (secondary)