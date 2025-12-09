from __future__ import annotations

"""Dimensionality reduction utilities for Roxy.

This module wraps common projection methods into a small, consistent API:

- :class:`PCAReducer`  – PCA via :mod:`sklearn.decomposition`.
- :class:`TSNEReducer` – t-SNE via :mod:`sklearn.manifold`.
- :class:`UMAPReducer` – UMAP via :mod:`umap-learn` (optional dependency).

The main entry point for quick use is :func:`project`, which selects
the appropriate reducer based on a string identifier and returns the
embedded coordinates as a NumPy array.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from roxy.core.exceptions import ProjectionError
from roxy.core.logging_utils import get_logger

logger = get_logger(__name__)

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

    Raises
    ------
    ProjectionError
        If the input cannot be converted to a 2D NumPy array.
    """
    try:
        if pd is not None and isinstance(X, pd.DataFrame):
            arr = X.to_numpy()
        else:
            arr = np.asarray(X)
    except Exception as exc:  # pragma: no cover - very rare
        msg = f"Failed to convert input to NumPy array: {exc}"
        logger.error("_to_numpy: %s", msg)
        raise ProjectionError(msg) from exc

    if arr.ndim != 2:
        msg = (
            f"Projection methods expect a 2D array, got shape {arr.shape!r}. "
            "Ensure X has shape (n_samples, n_features)."
        )
        logger.error("_to_numpy: %s", msg)
        raise ProjectionError(msg)

    logger.debug("_to_numpy: converted input to array with shape %s.", arr.shape)
    return arr


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
        logger.info(
            "%s.fit_transform: starting fit+transform on data.",
            self.__class__.__name__,
        )
        X_emb = self.fit(X).transform(X)
        logger.info(
            "%s.fit_transform: finished embedding with shape %s.",
            self.__class__.__name__,
            X_emb.shape,
        )
        return X_emb


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
        logger.debug(
            "PCAReducer initialised with n_components=%d, kwargs=%r.",
            self.n_components,
            self.kwargs,
        )

    def fit(self, X: ArrayLike) -> "PCAReducer":
        X_arr = _to_numpy(X)
        logger.info("PCAReducer.fit: fitting PCA on shape %s.", X_arr.shape)
        self._model.fit(X_arr)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        X_arr = _to_numpy(X)
        logger.info(
            "PCAReducer.transform: transforming data with shape %s.", X_arr.shape
        )
        emb = self._model.transform(X_arr)
        logger.debug(
            "PCAReducer.transform: output embedding shape %s.", emb.shape
        )
        return emb


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
        # Model is constructed in fit/fit_transform.
        self._model: Optional[TSNE] = None
        self._embedding_: Optional[np.ndarray] = None
        logger.debug(
            "TSNEReducer initialised with n_components=%d, kwargs=%r.",
            self.n_components,
            self.kwargs,
        )

    def fit(self, X: ArrayLike) -> "TSNEReducer":
        """Fit t-SNE and store the embedding.

        Notes
        -----
        For practical use, :meth:`fit_transform` is preferred, as t-SNE
        is typically run on the full dataset one time.
        """
        X_arr = _to_numpy(X)
        logger.info("TSNEReducer.fit: fitting t-SNE on shape %s.", X_arr.shape)
        self._model = TSNE(
            n_components=self.n_components,
            **self.kwargs,
        )
        self._embedding_ = self._model.fit_transform(X_arr)
        logger.info(
            "TSNEReducer.fit: finished, embedding shape %s.",
            self._embedding_.shape,
        )
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:  # pragma: no cover
        """Not implemented for t-SNE in scikit-learn.

        Raises
        ------
        NotImplementedError
            Always raised, since scikit-learn's t-SNE does not provide a
            robust transform for new samples.
        """
        raise NotImplementedError(
            "t-SNE does not support a reliable transform for new data in "
            "the scikit-learn implementation. Use `fit_transform` on the "
            "full dataset you wish to embed."
        )

    def fit_transform(self, X: ArrayLike) -> np.ndarray:
        X_arr = _to_numpy(X)
        logger.info("TSNEReducer.fit_transform: running t-SNE on %s.", X_arr.shape)
        self._model = TSNE(
            n_components=self.n_components,
            **self.kwargs,
        )
        self._embedding_ = self._model.fit_transform(X_arr)
        logger.info(
            "TSNEReducer.fit_transform: finished, embedding shape %s.",
            self._embedding_.shape,
        )
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
            msg = (
                "UMAPReducer requires the 'umap-learn' package. "
                "Install it with `pip install umap-learn` or "
                "`pip install roxy[umap]` to use this reducer."
            )
            logger.error("UMAPReducer.__post_init__: %s", msg)
            raise ImportError(msg)
        self._model = umap.UMAP(
            n_components=self.n_components,
            **self.kwargs,
        )
        logger.debug(
            "UMAPReducer initialised with n_components=%d, kwargs=%r.",
            self.n_components,
            self.kwargs,
        )

    def fit(self, X: ArrayLike) -> "UMAPReducer":
        X_arr = _to_numpy(X)
        logger.info("UMAPReducer.fit: fitting UMAP on shape %s.", X_arr.shape)
        self._model.fit(X_arr)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        X_arr = _to_numpy(X)
        logger.info(
            "UMAPReducer.transform: transforming data with shape %s.", X_arr.shape
        )
        emb = self._model.transform(X_arr)
        logger.debug(
            "UMAPReducer.transform: output embedding shape %s.", emb.shape
        )
        return emb

    def fit_transform(self, X: ArrayLike) -> np.ndarray:
        X_arr = _to_numpy(X)
        logger.info(
            "UMAPReducer.fit_transform: running UMAP on data shape %s.", X_arr.shape
        )
        emb = self._model.fit_transform(X_arr)
        logger.info(
            "UMAPReducer.fit_transform: finished, embedding shape %s.", emb.shape
        )
        return emb


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
    ProjectionError
        If an unknown method is requested or if the projection fails.
    ImportError
        If ``method="umap"`` is requested but ``umap-learn`` is not
        installed.
    """
    method_norm = method.lower()
    logger.info(
        "project: method=%r, n_components=%d, kwargs=%r.",
        method_norm,
        n_components,
        kwargs,
    )

    try:
        if method_norm == "pca":
            reducer: BaseReducer = PCAReducer(
                n_components=n_components,
                kwargs=kwargs,
            )
        elif method_norm == "tsne":
            reducer = TSNEReducer(
                n_components=n_components,
                kwargs=kwargs,
            )
        elif method_norm == "umap":
            reducer = UMAPReducer(
                n_components=n_components,
                kwargs=kwargs,
            )
        else:
            msg = (
                f"Unknown projection method: {method!r}. "
                "Supported methods are: 'pca', 'tsne', 'umap'."
            )
            logger.error("project: %s", msg)
            raise ProjectionError(msg)

        emb = reducer.fit_transform(X)
        logger.info(
            "project: successfully embedded data, shape %s.", emb.shape
        )
        return emb

    except ProjectionError:
        # Already logged, just propagate
        raise
    except ImportError:
        # UMAP-specific import errors are already descriptive
        raise
    except Exception as exc:  # pragma: no cover - defensive
        msg = f"Projection failed for method {method!r}: {exc}"
        logger.error("project: %s", msg)
        raise ProjectionError(msg) from exc
