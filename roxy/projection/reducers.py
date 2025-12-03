
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from sklearn.decomposition import PCA


class BaseReducer(ABC):
    """Abstract base class for dimensionality reduction methods."""

    name: str

    @abstractmethod
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the reducer on X and return the projected data."""
        raise NotImplementedError


@dataclass
class PCAReducer(BaseReducer):
    """PCA-based dimensionality reducer."""

    n_components: int = 2
    kwargs: Dict[str, Any] = None

    def __post_init__(self) -> None:
        self.name = "pca"
        self._model = PCA(n_components=self.n_components, **(self.kwargs or {}))

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self._model.fit_transform(X)


def project(
    X: np.ndarray,
    method: str = "pca",
    n_components: int = 2,
    **kwargs: Any,
) -> np.ndarray:
    """Project the dataset into a low-dimensional space."""
    if method == "pca":
        reducer = PCAReducer(n_components=n_components, kwargs=kwargs)
    else:
        raise ValueError(f"Unknown projection method: {method!r}")
    return reducer.fit_transform(X)
