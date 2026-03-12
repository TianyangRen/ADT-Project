from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.ensemble import GradientBoostingRegressor


@dataclass
class CostEstimate:
    """Predicted cost."""
    index_name: str
    params: dict
    estimated_latency_ms: float
    estimated_recall: float
    confidence: float = 0.9  # 优化：基础置信度提高，外推时动态降低

    def __repr__(self):
        p = ", ".join(f"{k}={v}" for k, v in self.params.items()) if self.params else "default"
        return (f"CostEstimate({self.index_name}({p}): "
                f"lat={self.estimated_latency_ms:.3f}ms, "
                f"recall={self.estimated_recall:.4f}, "
                f"conf={self.confidence:.1f})")


class CostModel:
    """
    using different index to train the model to predict latency and recall rate
    with Monotonic Constraints and Extrapolation Detection.
    """

    def __init__(self):
        self.latency_models: Dict[str, object] = {}
        self.recall_models: Dict[str, object] = {}
        self.is_trained = False
        self._training_stats: Dict = {}
        self._feature_bounds: Dict[str, dict] = {} # 新增：记录训练数据的特征边界

    def train(self, profiling_df: pd.DataFrame, dataset_size: int = None):
        """
            phase 2 consider Expected columns: index, param_value, k, recall, latency_ms
        """
        # 1. 更改导入的类
        from sklearn.ensemble import HistGradientBoostingRegressor

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
            
            self._feature_bounds[index_name] = {
                "k_min": subset["k"].min(),
                "k_max": subset["k"].max(),
                "param_min": subset["param_value"].min(),
                "param_max": subset["param_value"].max()
            }

            # 2. 替换为 HistGradientBoostingRegressor，并将 n_estimators 改为 max_iter
            y_lat = subset["latency_ms"].values
            lat_model = HistGradientBoostingRegressor(
                max_iter=100, max_depth=4, learning_rate=0.1, 
                random_state=42,
                monotonic_cst=[1, 1] 
            )
            lat_model.fit(X, y_lat)
            self.latency_models[index_name] = lat_model

            # 3. 同样替换 Recall 模型
            y_rec = subset["recall"].values
            rec_model = HistGradientBoostingRegressor(
                max_iter=100, max_depth=4, learning_rate=0.1,
                random_state=42,
                monotonic_cst=[-1, 1]
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
        predict for HNSW, IVF, LINEAR, FLAT with confidence scoring
        """
        assert self.is_trained, "Cost model not trained yet"

        if "nprobe" in params:
            param_val = params["nprobe"]
        elif "ef_search" in params:
            param_val = params["ef_search"]
        else:
            param_val = 0

        X = np.array([[query_features.top_k, param_val]], dtype=np.float64)
        
        # 新增：动态置信度评估 (外推检测)
        confidence = 0.90
        bounds = self._feature_bounds.get(index_name)
        if bounds:
            # 如果请求的 k 或 param 偏离了训练分布，树模型的预测不再可靠，降低置信度
            if not (bounds["k_min"] <= query_features.top_k <= bounds["k_max"]):
                confidence -= 0.3
            if not (bounds["param_min"] <= param_val <= bounds["param_max"]):
                confidence -= 0.3
        confidence = max(0.1, confidence)

        if index_name in self.latency_models:
            est_latency = float(self.latency_models[index_name].predict(X)[0])
            est_recall = float(self.recall_models[index_name].predict(X)[0])
        else:
            est_latency = 100.0
            est_recall = 0.5
            confidence = 0.1

        # --- Concurrency-aware latency scaling ---
        conc = getattr(query_features, "concurrency", 1)
        if conc > 1:
            est_latency *= self._concurrency_multiplier(index_name, conc)

        est_latency = max(est_latency, 0.001)
        est_recall = np.clip(est_recall, 0.0, 1.0)

        return CostEstimate(
            index_name=index_name,
            params=params,
            estimated_latency_ms=est_latency,
            estimated_recall=est_recall,
            confidence=confidence, # 传入动态计算的置信度
        )

    @staticmethod
    def _concurrency_multiplier(index_name: str, concurrency: int) -> float:
        c = concurrency - 1
        if index_name == "HNSW":
            return 1.0 + 0.5 * c * c
        elif index_name == "IVF":
            return 1.0 + 0.10 * c
        else:
            return 1.0 + 0.05 * c

    def estimate_all(self, candidates: list, query_features) -> List[CostEstimate]:
        return [
            self.estimate(c.index_name, c.params, query_features)
            for c in candidates
        ]

    def save(self, filepath: str = "results/cost_model.pkl"):
        """SAVE MODEL"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({
                "latency_models": self.latency_models,
                "recall_models": self.recall_models,
                "training_stats": self._training_stats,
                "feature_bounds": self._feature_bounds, # 新增：保存边界数据
            }, f)
        print(f"  [CostModel] Saved to {filepath}")

    def load(self, filepath: str = "results/cost_model.pkl"):
        """Load trained model from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.latency_models = data["latency_models"]
        self.recall_models = data["recall_models"]
        self._training_stats = data.get("training_stats", {})
        self._feature_bounds = data.get("feature_bounds", {}) # 新增：读取边界数据
        self.is_trained = True
        print(f"  [CostModel] Loaded from {filepath}")

    def get_training_stats(self) -> dict:
        return self._training_stats
