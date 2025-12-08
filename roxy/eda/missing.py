from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def missingness_summary(X: pd.DataFrame) -> Dict[str, Any]:
    """Compute basic missingness diagnostics for a feature matrix.

    Parameters
    ----------
    X :
        Feature matrix with samples in rows and features in columns.

    Returns
    -------
    dict
        Dictionary with:
        - ``per_column``: Series with the number of missing values per column.
        - ``per_column_ratio``: Series with the fraction of missing values per column.
        - ``per_row``: Series with the number of missing values per row.
        - ``per_row_ratio``: Series with the fraction of missing values per row.
        - ``global_fraction``: Overall fraction of missing values in the matrix.
    """
    n_samples, n_features = X.shape

    per_column = X.isna().sum()
    per_column_ratio = per_column / n_samples if n_samples > 0 else per_column * np.nan

    per_row = X.isna().sum(axis=1)
    per_row_ratio = per_row / n_features if n_features > 0 else per_row * np.nan

    total_missing = float(X.isna().sum().sum())
    global_fraction = total_missing / (n_samples * n_features) if n_samples * n_features > 0 else np.nan

    return {
        "per_column": per_column,
        "per_column_ratio": per_column_ratio,
        "per_row": per_row,
        "per_row_ratio": per_row_ratio,
        "global_fraction": global_fraction,
    }


def missingness_by_label(
    X: pd.DataFrame,
    y: Optional[pd.Series],
) -> Optional[Dict[Any, Dict[str, Any]]]:
    """Summarise missingness patterns stratified by labels.

    This can be useful to detect label-dependent missingness.

    Parameters
    ----------
    X :
        Feature matrix.
    y :
        Target labels with the same index as ``X``. If ``None``, the
        function returns ``None``.

    Returns
    -------
    dict or None
        Mapping from label value to a missingness summary dictionary
        as returned by :func:`missingness_summary`, or ``None`` if
        ``y`` is ``None``.
    """
    if y is None:
        return None

    if not X.index.equals(y.index):
        raise ValueError("Index of X and y must match for missingness_by_label.")

    summaries: Dict[Any, Dict[str, Any]] = {}
    for label, idx in y.groupby(y).groups.items():
        X_sub = X.loc[idx]
        summaries[label] = missingness_summary(X_sub)

    return summaries
