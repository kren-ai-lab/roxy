"""Roxy features package.

This subpackage provides utilities for working with feature matrices,
including:

- automatic feature type inference (:mod:`roxy.features.typing`),
- scaling and transformation (:mod:`roxy.features.scaling`),
- feature selection (:mod:`roxy.features.selection`).

The focus is on integrating scikit-learn transformers with pandas
DataFrames while preserving column names and indices.
"""

from .typing import FeatureKind, FeatureTypeInfo, infer_feature_kinds, get_columns_of_kind, summarise_feature_kinds
from .scaling import ColumnScaler, create_scaler
from .selection import FeatureSelector

__all__ = [
    # typing
    "FeatureKind",
    "FeatureTypeInfo",
    "infer_feature_kinds",
    "get_columns_of_kind",
    "summarise_feature_kinds",
    # scaling
    "ColumnScaler",
    "create_scaler",
    # selection
    "FeatureSelector",
]
