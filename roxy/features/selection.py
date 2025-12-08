"""Feature selection utilities.

This module provides a unified interface over several scikit-learn
feature selection strategies for use with pandas DataFrames.

Supported strategies include:

- ``"variance"``    : :class:`sklearn.feature_selection.VarianceThreshold`
- ``"kbest"``       : :class:`sklearn.feature_selection.SelectKBest`
- ``"percentile"``  : :class:`sklearn.feature_selection.SelectPercentile`
- ``"mutual_info"`` : :class:`sklearn.feature_selection.SelectKBest` with
                      mutual information score functions.
- ``"model_l1"``    : :class:`sklearn.feature_selection.SelectFromModel`
                      with L1-regularised linear models.
- ``"model_tree"``  : :class:`sklearn.feature_selection.SelectFromModel`
                      with tree-based ensembles.
- ``"rfe"``         : :class:`sklearn.feature_selection.RFE` with a
                      linear estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    SelectPercentile,
    VarianceThreshold,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.linear_model import Lasso, LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


TaskType = str  # "classification" or "regression"


def _get_score_func(name: str, task_type: TaskType):
    """Return a score function for SelectKBest/SelectPercentile."""
    name = name.lower()
    if name == "f":
        return f_classif if task_type == "classification" else f_regression
    if name == "mutual_info":
        return mutual_info_classif if task_type == "classification" else mutual_info_regression
    raise ValueError(f"Unknown score function {name!r}.")


def _build_selector(
    strategy: str,
    task_type: TaskType,
    selector_kwargs: Dict[str, Any],
) -> BaseEstimator:
    """Instantiate a scikit-learn feature selector based on a strategy name."""
    strategy = strategy.lower()

    if strategy == "variance":
        return VarianceThreshold(**selector_kwargs)

    if strategy == "kbest":
        score_name = selector_kwargs.pop("score", "f")
        score_func = _get_score_func(score_name, task_type)
        return SelectKBest(score_func=score_func, **selector_kwargs)

    if strategy == "percentile":
        score_name = selector_kwargs.pop("score", "f")
        score_func = _get_score_func(score_name, task_type)
        return SelectPercentile(score_func=score_func, **selector_kwargs)

    if strategy == "mutual_info":
        score_func = _get_score_func("mutual_info", task_type)
        return SelectKBest(score_func=score_func, **selector_kwargs)

    if strategy == "model_l1":
        if task_type == "classification":
            base_estimator: BaseEstimator = LogisticRegression(
                penalty="l1",
                solver="liblinear",
                max_iter=500,
            )
        else:
            base_estimator = Lasso(
                alpha=selector_kwargs.pop("alpha", 0.001),
                max_iter=5000,
            )
        # NOTE: in recent scikit-learn versions, the keyword is `estimator`
        return SelectFromModel(estimator=base_estimator, **selector_kwargs)

    if strategy == "model_tree":
        if task_type == "classification":
            base_estimator = RandomForestClassifier(
                n_estimators=selector_kwargs.pop("n_estimators", 200),
                random_state=selector_kwargs.pop("random_state", 42),
                n_jobs=selector_kwargs.pop("n_jobs", -1),
            )
        else:
            base_estimator = RandomForestRegressor(
                n_estimators=selector_kwargs.pop("n_estimators", 200),
                random_state=selector_kwargs.pop("random_state", 42),
                n_jobs=selector_kwargs.pop("n_jobs", -1),
            )
        return SelectFromModel(estimator=base_estimator, **selector_kwargs)

    if strategy == "rfe":
        if task_type == "classification":
            base_estimator = LogisticRegression(max_iter=1000)
        else:
            base_estimator = LinearRegression()
        return RFE(estimator=base_estimator, **selector_kwargs)

    raise ValueError(
        f"Unknown feature selection strategy {strategy!r}. "
        "Supported: variance, kbest, percentile, mutual_info, "
        "model_l1, model_tree, rfe."
    )

@dataclass
class FeatureSelector(BaseEstimator, TransformerMixin):
    """Unified feature selection wrapper with DataFrame support.

    This transformer wraps several scikit-learn feature selection
    strategies and operates on pandas DataFrames, preserving column
    names during transformation.

    Parameters
    ----------
    strategy :
        Name of the selection strategy (see module docstring).
    task_type :
        Task type, one of ``"classification"`` or ``"regression"``.
        Determines which score functions or base estimators are used.
    columns :
        Optional iterable of column names to consider for selection.
        If ``None``, all numeric columns are considered.
    selector_kwargs :
        Additional keyword arguments forwarded to the underlying selector
        constructor. For example:

        - ``k`` for ``"kbest"`` (number of features to select),
        - ``percentile`` for ``"percentile"``,
        - ``threshold`` for ``"model_l1"`` or ``"model_tree"``.
    """

    strategy: str = "variance"
    task_type: TaskType = "classification"
    columns: Optional[Iterable[str]] = None
    selector_kwargs: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.selector_kwargs is None:
            self.selector_kwargs = {}
        self._selector: Optional[BaseEstimator] = None
        self._columns_: List[str] = []
        self.support_mask_: Optional[np.ndarray] = None
        self.selected_columns_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureSelector":
        """Fit the feature selector on the provided data."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureSelector expects a pandas DataFrame as input.")

        if self.task_type not in {"classification", "regression"}:
            raise ValueError("`task_type` must be 'classification' or 'regression'.")

        if self.columns is None:
            cols = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols = list(self.columns)

        if not cols:
            # Nothing to select from
            self._columns_ = []
            self._selector = None
            self.support_mask_ = None
            self.selected_columns_ = []
            return self

        self._columns_ = cols

        selector = _build_selector(
            strategy=self.strategy,
            task_type=self.task_type,
            selector_kwargs=dict(self.selector_kwargs),
        )
        selector.fit(X[self._columns_].to_numpy(), None if y is None else y.to_numpy())

        self._selector = selector

        # Determine selected columns
        if hasattr(selector, "get_support"):
            mask = selector.get_support()
            self.support_mask_ = np.asarray(mask, dtype=bool)
            self.selected_columns_ = [
                col for col, keep in zip(self._columns_, self.support_mask_) if keep
            ]
        else:
            # Fallback: assume all columns kept
            self.support_mask_ = np.ones(len(self._columns_), dtype=bool)
            self.selected_columns_ = list(self._columns_)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reduce the feature matrix to the selected columns."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureSelector expects a pandas DataFrame as input.")

        if not self._columns_ or self._selector is None:
            # No-op if selector was not fitted or no columns were available
            return X.copy()

        if not self.selected_columns_:
            # All dropped; return empty DataFrame with same index
            return pd.DataFrame(index=X.index)

        return X[self.selected_columns_].copy()

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Fit to data, then return the reduced DataFrame."""
        return self.fit(X, y).transform(X)
