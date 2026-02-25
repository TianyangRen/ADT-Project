"""
Learned Cost Model: predicts (latency, recall) for each (index, params, query_features).

Trained from Phase 2 profiling data using gradient boosting regression.
This is the core decision-making component of the adaptive system.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import os
import pickle


@dataclass
class CostEstimate:
    """Predicted cost for one execution strategy."""
    index_name: str
    params: dict
    estimated_latency_ms: float
    estimated_recall: float
    confidence: float = 0.8

    def __repr__(self):
        p = ", ".join(f"{k}={v}" for k, v in self.params.items()) if self.params else "default"
        return (f"CostEstimate({self.index_name}({p}): "
                f"lat={self.estimated_latency_ms:.3f}ms, "
                f"recall={self.estimated_recall:.4f})")


class CostModel:
    """
    Learned cost model trained from profiling sweep data.

    Predicts latency and recall given (index_name, params, query_features).
    Uses per-index GradientBoostingRegressor models.
    """

    def __init__(self):
        self.latency_models: Dict[str, object] = {}
        self.recall_models: Dict[str, object] = {}
        self.is_trained = False
        self._training_stats: Dict = {}

    def train(self, profiling_df: pd.DataFrame, dataset_size: int = None):
        """
        Train regression models from Phase 2 profiling sweep data.

        Expected columns: index, param_value, k, recall, latency_ms
        Optionally: dataset_size

        Features per model: [k, param_value]
        """
        from sklearn.ensemble import GradientBoostingRegressor

        if dataset_size is None:
            dataset_size = profiling_df.get("dataset_size", pd.Series([1_000_000])).iloc[0]

        self._training_stats = {
            "n_rows": len(profiling_df),
            "indexes": list(profiling_df["index"].unique()),
            "dataset_size": dataset_size,
        }

        for index_name in profiling_df["index"].unique():
            subset = profiling_df[profiling_df["index"] == index_name].copy()

            # Features: [k, param_value]
            X = subset[["k", "param_value"]].values.astype(np.float64)

            # Latency model
            y_lat = subset["latency_ms"].values
            lat_model = GradientBoostingRegressor(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                random_state=42
            )
            lat_model.fit(X, y_lat)
            self.latency_models[index_name] = lat_model

            # Recall model
            y_rec = subset["recall"].values
            rec_model = GradientBoostingRegressor(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                random_state=42
            )
            rec_model.fit(X, y_rec)
            self.recall_models[index_name] = rec_model

            # Training accuracy
            lat_pred = lat_model.predict(X)
            rec_pred = rec_model.predict(X)
            lat_mae = np.mean(np.abs(lat_pred - y_lat))
            rec_mae = np.mean(np.abs(rec_pred - y_rec))
            print(f"  [CostModel] {index_name}: "
                  f"latency MAE={lat_mae:.4f}ms, recall MAE={rec_mae:.4f}")

        self.is_trained = True
        print(f"  [CostModel] Trained on {len(profiling_df)} data points, "
              f"{len(self.latency_models)} index types")

    def estimate(self, index_name: str, params: dict,
                 query_features) -> CostEstimate:
        """
        Predict latency and recall for a given (index, params, query).

        Args:
            index_name: "Flat", "IVF", or "HNSW"
            params: e.g. {"nprobe": 16} or {"ef_search": 64} or {}
            query_features: QueryFeatures with top_k etc.

        Returns:
            CostEstimate with predicted latency and recall
        """
        assert self.is_trained, "Cost model not trained yet"

        # Extract the tunable parameter value
        if "nprobe" in params:
            param_val = params["nprobe"]
        elif "ef_search" in params:
            param_val = params["ef_search"]
        else:
            param_val = 0  # Flat has no tunable param

        X = np.array([[query_features.top_k, param_val]], dtype=np.float64)

        if index_name in self.latency_models:
            est_latency = float(self.latency_models[index_name].predict(X)[0])
            est_recall = float(self.recall_models[index_name].predict(X)[0])
        else:
            # Fallback for unknown index
            est_latency = 100.0
            est_recall = 0.5

        # Clamp to valid ranges
        est_latency = max(est_latency, 0.001)
        est_recall = np.clip(est_recall, 0.0, 1.0)

        return CostEstimate(
            index_name=index_name,
            params=params,
            estimated_latency_ms=est_latency,
            estimated_recall=est_recall,
            confidence=0.8,
        )

    def estimate_all(self, candidates: list, query_features) -> List[CostEstimate]:
        """Estimate costs for all candidate strategies."""
        return [
            self.estimate(c.index_name, c.params, query_features)
            for c in candidates
        ]

    def save(self, filepath: str = "results/cost_model.pkl"):
        """Save trained model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({
                "latency_models": self.latency_models,
                "recall_models": self.recall_models,
                "training_stats": self._training_stats,
            }, f)
        print(f"  [CostModel] Saved to {filepath}")

    def load(self, filepath: str = "results/cost_model.pkl"):
        """Load trained model from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.latency_models = data["latency_models"]
        self.recall_models = data["recall_models"]
        self._training_stats = data.get("training_stats", {})
        self.is_trained = True
        print(f"  [CostModel] Loaded from {filepath}")

    def get_training_stats(self) -> dict:
        return self._training_stats
