"""
Abstract base class for all ANN index implementations.

Every index must implement build(), search(), and get_config().
This provides a uniform interface for the profiler, benchmark, and
adaptive execution engine to interact with any index type.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np


class BaseIndex(ABC):
    """Abstract base class for ANN indexes."""

    def __init__(self, name: str, dimension: int):
        self.name = name
        self.dimension = dimension
        self.is_built = False
        self._build_time_s = 0.0

    @abstractmethod
    def build(self, data: np.ndarray) -> None:
        """
        Build the index from a dataset of base vectors.

        Args:
            data: np.ndarray of shape (n, d), float32
        """
        ...

    @abstractmethod
    def search(self, queries: np.ndarray, k: int, **params) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.

        Args:
            queries: np.ndarray of shape (nq, d), float32
            k: number of neighbors to return
            **params: index-specific runtime parameters (e.g. nprobe, ef_search)

        Returns:
            distances: np.ndarray of shape (nq, k)
            indices:   np.ndarray of shape (nq, k)
        """
        ...

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return a dict describing the index configuration."""
        ...

    @property
    def build_time(self) -> float:
        """Time taken to build the index in seconds."""
        return self._build_time_s

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r}, dim={self.dimension}, built={self.is_built})"
