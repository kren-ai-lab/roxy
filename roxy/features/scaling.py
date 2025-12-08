"""Scaling and transformation utilities for feature matrices.

This module provides thin wrappers around scikit-learn transformers
to apply scaling and normalisation to pandas DataFrames while
preserving column names and indices.

Supported strategies include:

- ``"standard"`` : :class:`sklearn.preprocessing.StandardScaler`
- ``"minmax"``   : :class:`sklearn.preprocessing.MinMaxScaler`
- ``"robust"``   : :class:`sklearn.preprocessing.RobustScaler`
- ``"maxabs"``   : :class:`sklearn.preprocessing.MaxAbsScaler`
- ``"power"``    : :class:`sklearn.preprocessing.PowerTransformer`
- ``"quantile"`` : :class:`sklearn.preprocessing.QuantileTransformer`
- ``"normalizer"`` : :class:`sklearn.preprocessing.Normalizer`
- ``"none"``     : identity / no scaling
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)


ScalerLike = Union[str, BaseEstimator]


_SCALER_REGISTRY: Dict[str, Any] = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
    "maxabs": MaxAbsScaler,
    "power": PowerTransformer,
    "quantile": QuantileTransformer,
    "normalizer": Normalizer,
    "none": None,  # identity
}


def create_scaler(name: str, **kwargs: Any) -> Optional[BaseEstimator]:
    """Create a scikit-learn scaler instance from a strategy name.

    Parameters
    ----------
    name :
        Strategy name. One of ``"standard"``, ``"minmax"``, ``"robust"``,
        ``"maxabs"``, ``"power"``, ``"quantile"``, ``"normalizer"``,
        or ``"none"``.
    **kwargs :
        Additional keyword arguments forwarded to the scaler constructor.

    Returns
    -------
    BaseEstimator or None
        Instantiated scaler, or ``None`` for the ``"none"`` strategy.

    Raises
    ------
    KeyError
        If an unknown strategy name is provided.
    """
    name = name.lower()
    if name not in _SCALER_REGISTRY:
        raise KeyError(
            f"Unknown scaler strategy {name!r}. "
            f"Available: {', '.join(sorted(_SCALER_REGISTRY))}"
        )
    cls = _SCALER_REGISTRY[name]
    if cls is None:
        return None
    return cls(**kwargs)


@dataclass
class ColumnScaler(BaseEstimator, TransformerMixin):
    """Apply a scaling strategy to selected columns of a DataFrame.

    This transformer wraps a scikit-learn scaler and operates on
    pandas DataFrames, preserving column names and indices.

    Parameters
    ----------
    strategy :
        Scaling strategy name (see :func:`create_scaler`). If a
        scikit-learn transformer instance is provided instead of a
        string, it is used directly.
    columns :
        Optional list of column names to scale. If ``None``, all numeric
        columns are scaled.
    scaler_kwargs :
        Additional keyword arguments forwarded to the scaler constructor
        when ``strategy`` is provided as a string.
    """

    strategy: ScalerLike = "standard"
    columns: Optional[Iterable[str]] = None
    scaler_kwargs: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.scaler_kwargs is None:
            self.scaler_kwargs = {}
        self._scaler: Optional[BaseEstimator] = None
        self._columns_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Any = None) -> "ColumnScaler":
        """Fit the underlying scaler on the selected columns."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ColumnScaler expects a pandas DataFrame as input.")

        if self.columns is None:
            cols = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols = list(self.columns)

        if not cols:
            # Nothing to scale
            self._columns_ = []
            self._scaler = None
            return self

        self._columns_ = cols

        if isinstance(self.strategy, str):
            self._scaler = create_scaler(self.strategy, **self.scaler_kwargs)
        else:
            self._scaler = self.strategy

        if self._scaler is not None:
            self._scaler.fit(X[self._columns_].to_numpy(), y)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the selected columns and return a new DataFrame."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ColumnScaler expects a pandas DataFrame as input.")

        if not self._columns_ or self._scaler is None:
            # Identity transform if no columns or no scaler
            return X.copy()

        X_out = X.copy()
        arr = self._scaler.transform(X_out[self._columns_].to_numpy())
        X_out[self._columns_] = arr
        return X_out

    def fit_transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Fit to data, then transform it."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Return output feature names (identical to input column names)."""
        if input_features is None:
            return list(self._columns_)
        return list(input_features)
