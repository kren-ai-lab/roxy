from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None  # type: ignore

try:
    import umap  # type: ignore
except ImportError:  # pragma: no cover
    umap = None  # type: ignore


ArrayLike = Union[np.ndarray, "pd.DataFrame"]


def _to_numpy(X: ArrayLike) -> np.ndarray:
    """Convert input to a NumPy array, preserving row order.

    Parameters
    ----------
    X :
        Input data, either a NumPy array or a pandas DataFrame.

    Returns
    -------
    numpy.ndarray
        2D array with shape (n_samples, n_features).
    """
    if pd is not None and isinstance(X, pd.DataFrame):
        return X.to_numpy()
    return np.asarray(X)


class BaseReducer(ABC):
    """Abstract base class for dimensionality reduction methods.

    Concrete reducers should wrap a specific model implementation
    (e.g. PCA, t-SNE, UMAP) and provide a consistent interface
    for fitting and projecting data into a low-dimensional space.
    """

    name: str

    @abstractmethod
    def fit(self, X: ArrayLike) -> "BaseReducer":
        """Fit the reducer on the provided data.

        Parameters
        ----------
        X :
            Input data with shape (n_samples, n_features).

        Returns
        -------
        BaseReducer
            The fitted reducer instance.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: ArrayLike) -> np.ndarray:
        """Project new data into the low-dimensional space.

        Parameters
        ----------
        X :
            Input data with shape (n_samples, n_features).

        Returns
        -------
        numpy.ndarray
            Projected data with shape (n_samples, n_components).
        """
        raise NotImplementedError

    def fit_transform(self, X: ArrayLike) -> np.ndarray:
        """Fit the reducer on the data and return the projected points.

        This is a convenience method mirroring the scikit-learn API.

        Parameters
        ----------
        X :
            Input data with shape (n_samples, n_features).

        Returns
        -------
        numpy.ndarray
            Projected data with shape (n_samples, n_components).
        """
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------


@dataclass
class PCAReducer(BaseReducer):
    """PCA-based dimensionality reducer.

    This wrapper uses :class:`sklearn.decomposition.PCA` internally and
    supports both fitting and transforming new data.

    Parameters
    ----------
    n_components :
        Number of principal components to retain.
    kwargs :
        Additional keyword arguments forwarded to
        :class:`sklearn.decomposition.PCA`, e.g. ``random_state``,
        ``svd_solver``, etc.
    """

    n_components: int = 2
    kwargs: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.name = "pca"
        if self.kwargs is None:
            self.kwargs = {}
        self._model = PCA(n_components=self.n_components, **self.kwargs)

    def fit(self, X: ArrayLike) -> "PCAReducer":
        X_arr = _to_numpy(X)
        self._model.fit(X_arr)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        X_arr = _to_numpy(X)
        return self._model.transform(X_arr)


# ---------------------------------------------------------------------------
# t-SNE
# ---------------------------------------------------------------------------


@dataclass
class TSNEReducer(BaseReducer):
    """t-SNE-based dimensionality reducer.

    This wrapper uses :class:`sklearn.manifold.TSNE` internally. Note that
    scikit-learn's t-SNE implementation does not provide a reliable
    ``transform`` method for new points; therefore, this reducer only
    implements :meth:`fit_transform`. Calling :meth:`transform` will
    raise :class:`NotImplementedError`.

    Parameters
    ----------
    n_components :
        Number of dimensions in the embedded space (typically 2 or 3).
    kwargs :
        Additional keyword arguments forwarded to
        :class:`sklearn.manifold.TSNE`, e.g. ``perplexity``,
        ``learning_rate``, ``n_iter``, ``metric``, ``random_state``.
    """

    n_components: int = 2
    kwargs: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.name = "tsne"
        if self.kwargs is None:
            self.kwargs = {}
        # We delay model construction to fit_transform, because TSNE's
        # constructor requires the full parameter set up front.
        self._model: Optional[TSNE] = None

    def fit(self, X: ArrayLike) -> "TSNEReducer":
        # t-SNE in sklearn does not support a separate `fit` that can be
        # followed by `transform` reliably. We keep this for API symmetry.
        X_arr = _to_numpy(X)
        self._model = TSNE(
            n_components=self.n_components,
            **self.kwargs,
        )
        self._embedding_ = self._model.fit_transform(X_arr)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:  # pragma: no cover - behaviour is explicit
        raise NotImplementedError(
            "t-SNE does not support a reliable transform for new data in "
            "the scikit-learn implementation. Use `fit_transform` on the "
            "full dataset you wish to embed."
        )

    def fit_transform(self, X: ArrayLike) -> np.ndarray:
        X_arr = _to_numpy(X)
        self._model = TSNE(
            n_components=self.n_components,
            **self.kwargs,
        )
        self._embedding_ = self._model.fit_transform(X_arr)
        return self._embedding_


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------


@dataclass
class UMAPReducer(BaseReducer):
    """UMAP-based dimensionality reducer.

    This wrapper uses :mod:`umap-learn` internally. It supports both
    ``fit`` and ``transform`` to embed new data points after fitting.

    Parameters
    ----------
    n_components :
        Number of dimensions in the embedded space (typically 2 or 3).
    kwargs :
        Additional keyword arguments forwarded to ``umap.UMAP``,
        e.g. ``n_neighbors``, ``min_dist``, ``metric``, ``random_state``.
    """

    n_components: int = 2
    kwargs: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.name = "umap"
        if self.kwargs is None:
            self.kwargs = {}
        if umap is None:  # pragma: no cover
            raise ImportError(
                "UMAPReducer requires the 'umap-learn' package. "
                "Install it with `pip install umap-learn` to use this reducer."
            )
        self._model = umap.UMAP(
            n_components=self.n_components,
            **self.kwargs,
        )

    def fit(self, X: ArrayLike) -> "UMAPReducer":
        X_arr = _to_numpy(X)
        self._model.fit(X_arr)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        X_arr = _to_numpy(X)
        return self._model.transform(X_arr)

    def fit_transform(self, X: ArrayLike) -> np.ndarray:
        X_arr = _to_numpy(X)
        return self._model.fit_transform(X_arr)


# ---------------------------------------------------------------------------
# Front-end helper
# ---------------------------------------------------------------------------


def project(
    X: ArrayLike,
    method: str = "pca",
    n_components: int = 2,
    **kwargs: Any,
) -> np.ndarray:
    """Project a dataset into a low-dimensional space.

    This is a convenience function that instantiates the appropriate
    reducer (PCA, t-SNE, or UMAP) and returns the embedding.

    Parameters
    ----------
    X :
        Input data, either a NumPy array or pandas DataFrame with shape
        (n_samples, n_features).
    method :
        Name of the projection method. One of ``"pca"``, ``"tsne"``,
        or ``"umap"`` (requires ``umap-learn``).
    n_components :
        Number of output dimensions for the embedding.
    **kwargs :
        Additional keyword arguments forwarded to the underlying reducer
        (PCA, TSNE, or UMAP).

    Returns
    -------
    numpy.ndarray
        Embedded coordinates with shape (n_samples, n_components).

    Raises
    ------
    ValueError
        If an unknown projection method is requested.
    ImportError
        If ``method="umap"`` is requested but ``umap-learn`` is not
        installed.
    """
    method = method.lower()

    if method == "pca":
        reducer: BaseReducer = PCAReducer(n_components=n_components, kwargs=kwargs)
    elif method == "tsne":
        reducer = TSNEReducer(n_components=n_components, kwargs=kwargs)
    elif method == "umap":
        reducer = UMAPReducer(n_components=n_components, kwargs=kwargs)
    else:
        raise ValueError(
            f"Unknown projection method: {method!r}. "
            "Supported methods are: 'pca', 'tsne', 'umap'."
        )

    return reducer.fit_transform(X)
