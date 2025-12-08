"""Utilities for automatic feature type inference.

This module provides a lightweight abstraction over pandas dtypes to
classify columns into a small set of semantic categories (numeric,
categorical, boolean, datetime, text, other). This is useful for
downstream tasks such as encoding, scaling, and feature selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)


class FeatureKind(str, Enum):
    """Semantic kind of a feature/column."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    OTHER = "other"


@dataclass
class FeatureTypeInfo:
    """Summary of the inferred type for a single feature.

    Attributes
    ----------
    name :
        Column name in the feature matrix.
    kind :
        Inferred semantic kind, one of :class:`FeatureKind`.
    dtype :
        Original pandas dtype string representation.
    n_unique :
        Number of unique non-missing values.
    is_constant :
        Whether the column is effectively constant (one unique value
        ignoring NaNs).
    """

    name: str
    kind: FeatureKind
    dtype: str
    n_unique: int
    is_constant: bool


def infer_feature_kinds(
    X: pd.DataFrame,
    *,
    max_categories: int = 50,
    treat_bool_as_categorical: bool = False,
) -> Dict[str, FeatureTypeInfo]:
    """Infer semantic kinds for all columns in a feature matrix.

    Parameters
    ----------
    X :
        Input feature matrix with samples in rows and features in columns.
    max_categories :
        Maximum number of unique values for an object/string column to be
        considered ``CATEGORICAL``. Columns with more unique values will be
        labelled as ``TEXT``.
    treat_bool_as_categorical :
        If ``True``, boolean columns are labelled as ``CATEGORICAL`` instead
        of ``BOOLEAN``.

    Returns
    -------
    dict
        Mapping from column name to :class:`FeatureTypeInfo`.
    """
    info: Dict[str, FeatureTypeInfo] = {}

    for col in X.columns:
        s = X[col]
        dtype_str = str(s.dtype)
        n_unique = int(s.nunique(dropna=True))
        is_const = n_unique <= 1

        if is_bool_dtype(s):
            kind = FeatureKind.CATEGORICAL if treat_bool_as_categorical else FeatureKind.BOOLEAN
        elif is_numeric_dtype(s):
            kind = FeatureKind.NUMERIC
        elif is_datetime64_any_dtype(s):
            kind = FeatureKind.DATETIME
        else:
            # Object/string-like
            if n_unique <= max_categories:
                kind = FeatureKind.CATEGORICAL
            else:
                kind = FeatureKind.TEXT

        info[col] = FeatureTypeInfo(
            name=col,
            kind=kind,
            dtype=dtype_str,
            n_unique=n_unique,
            is_constant=is_const,
        )

    return info


def get_columns_of_kind(
    info: Dict[str, FeatureTypeInfo],
    kind: FeatureKind,
) -> List[str]:
    """Return a list of column names with the requested feature kind."""
    return [name for name, meta in info.items() if meta.kind == kind]


def summarise_feature_kinds(info: Dict[str, FeatureTypeInfo]) -> pd.DataFrame:
    """Return a compact summary table of inferred feature kinds.

    Parameters
    ----------
    info :
        Mapping from column name to :class:`FeatureTypeInfo`.

    Returns
    -------
    pandas.DataFrame
        Table with one row per feature and columns:
        ``kind``, ``dtype``, ``n_unique``, ``is_constant``.
    """
    rows = []
    for name, meta in info.items():
        rows.append(
            {
                "name": name,
                "kind": meta.kind.value,
                "dtype": meta.dtype,
                "n_unique": meta.n_unique,
                "is_constant": meta.is_constant,
            }
        )
    return pd.DataFrame(rows).set_index("name").sort_index()
