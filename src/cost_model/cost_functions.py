"""
Analytical Cost Functions for ANN indexes.

Provides lightweight, interpretable cost estimates based on
index-specific complexity analysis. Useful as:
  1. Baseline cost model before profiling data is available
  2. Interpretable alternative to learned models
  3. Fallback when learned model is unreliable
"""

import math

from src.cost_model.cost_estimator import CostEstimate, CostModel


class AnalyticalCostModel:
    """
    Analytical cost model based on index complexity formulas.

    Each index has a latency function and a recall function
    parameterized by dataset size, dimension, K, and tunable param.
    """

    def __init__(self, dataset_size: int, dimension: int, latency_calibration: float = 1.0):
        # 优化4：参数边界保护，防止负数或0导致 math error
        self.n = max(dataset_size, 1)
        self.d = max(dimension, 1)
        
        # 优化3：加入硬件校准因子，方便后续快速适配不同算力的服务器
        self.latency_calibration = latency_calibration
        
        self.is_trained = True  # Always ready, no training needed

        # 优化1：动态估算 nlist (Faiss 经验法则 4 * sqrt(N))，消除 256 的硬编码
        # 限制在 100 到 65536 的合理范围内
        self.estimated_nlist = max(100, min(65536, int(4 * math.sqrt(self.n))))

    def estimate(self, index_name: str, params: dict,
                 query_features) -> CostEstimate:
        """Estimate cost using analytical formulas with concurrency scaling."""
        k = query_features.top_k

        if index_name == "Flat":
            lat = self._flat_latency(k)
            rec = self._flat_recall()
        elif index_name == "IVF":
            nprobe = params.get("nprobe", 16)
            lat = self._ivf_latency(k, nprobe)
            rec = self._ivf_recall(k, nprobe)
        elif index_name == "HNSW":
            ef_search = params.get("ef_search", 64)
            lat = self._hnsw_latency(k, ef_search)
            rec = self._hnsw_recall(k, ef_search)
        else:
            lat, rec = 100.0, 0.5

        # 应用硬件校准因子
        lat *= self.latency_calibration

        # Concurrency-aware latency scaling (same model as CostModel)
        conc = getattr(query_features, "concurrency", 1)
        if conc > 1:
            lat *= CostModel._concurrency_multiplier(index_name, conc)

        return CostEstimate(
            index_name=index_name,
            params=params,
            estimated_latency_ms=max(lat, 0.001),
            estimated_recall=min(max(rec, 0.0), 1.0),
            confidence=0.5,  # Lower confidence than learned model
        )

    def estimate_all(self, candidates: list, query_features) -> list:
        return [self.estimate(c.index_name, c.params, query_features) for c in candidates]

    # --- Flat Index: O(n*d) linear scan ---

    def _flat_latency(self, k: int) -> float:
        """Flat: linear scan, proportional to n*d."""
        return 0.0002 * self.n * self.d / 1e6 + 0.5

    def _flat_recall(self) -> float:
        """Flat: always perfect recall (exact search)."""
        return 1.0

    # --- IVF Index: scans nprobe/nlist fraction ---

    def _ivf_latency(self, k: int, nprobe: int) -> float:
        """IVF: scans nprobe out of nlist cells."""
        nlist = self.estimated_nlist
        nprobe = min(nprobe, nlist) # 保护：扫描数量不可能超过总桶数
        fraction = nprobe / nlist
        scan_cost = fraction * self.n * self.d / 1e9  # ms scale
        overhead = 0.03 * nprobe  # per-cell overhead
        return scan_cost + overhead + 0.05

    def _ivf_recall(self, k: int, nprobe: int) -> float:
        """IVF recall: sigmoid model of nprobe/nlist ratio."""
        nlist = self.estimated_nlist
        nprobe = min(nprobe, nlist)
        ratio = nprobe / nlist
        
        # 优化2：维度惩罚 (维度越高，聚类边界越模糊，相同 ratio 下召回率越低)
        dim_penalty = 128.0 / max(self.d, 128.0)
        
        # Recall saturates as nprobe increases; harder for larger K
        k_penalty = 1.0 - 0.002 * min(k, 100)
        
        return k_penalty * (1.0 - math.exp(-8.0 * ratio * dim_penalty))

    # --- HNSW Index: O(ef_search * log(n) * d) graph traversal ---

    def _hnsw_latency(self, k: int, ef_search: int) -> float:
        """HNSW: graph traversal proportional to ef_search * log(n)."""
        log_n = math.log2(self.n)
        return 0.00001 * ef_search * log_n * self.d / 1e3 + 0.02

    def _hnsw_recall(self, k: int, ef_search: int) -> float:
        """HNSW recall: depends on ef_search relative to k."""
        if ef_search < k:
            # ef_search must be >= k for valid results
            return 0.5 * (ef_search / k)
            
        ratio = ef_search / max(k, 1)
        
        # 优化2：维度惩罚 (高维度空间中，图路由更容易陷入局部最优，需要更大的 ef_search)
        dim_factor = math.sqrt(self.d / 128.0) 
        
        return 1.0 - math.exp(-1.5 * (ratio / dim_factor))
