from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Dict, Any

import numpy as np
import pandas as pd

from roxy.core.dataset import RoxyDataset


@dataclass
class BasicEDAReport:
    """Container for basic exploratory data analysis results.

    This report is intentionally lightweight and focuses on tabular
    summaries that can be reused in notebooks, scripts or higher-level
    visualisation utilities.

    Attributes
    ----------
    shape :
        Tuple (n_samples, n_features) for the analysed feature table.
    feature_keys :
        List of feature block keys that were combined to obtain the
        analysed feature table.
    numeric_summary :
        Summary statistics for numeric columns (similar to ``DataFrame.describe()``).
    categorical_summary :
        Mapping from column name to value-counts Series for categorical columns.
    missing_per_column :
        Number of missing values per column in the analysed feature table.
    correlations :
        Correlation matrix for numeric features (method depends on the engine
        configuration). May be ``None`` if no numeric columns are present or
        correlation computation was disabled.
    metadata :
        Arbitrary metadata about the dataset or analysis (e.g. dataset name,
        task type, notes).
    """

    shape: tuple[int, int]
    feature_keys: List[str]
    numeric_summary: pd.DataFrame
    categorical_summary: Dict[str, pd.Series]
    missing_per_column: pd.Series
    correlations: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary of the main report components."""
        return {
            "shape": self.shape,
            "feature_keys": self.feature_keys,
            "numeric_summary": self.numeric_summary,
            "categorical_summary": self.categorical_summary,
            "missing_per_column": self.missing_per_column,
            "correlations": self.correlations,
            "metadata": self.metadata,
        }


class BasicEDA:
    """Basic exploratory data analysis engine for Roxy datasets.

    This engine operates on one or more feature blocks of a
    :class:`RoxyDataset` and returns a :class:`BasicEDAReport` with
    summary statistics, missingness, and simple correlations.

    The goal is to provide a standard, reusable EDA entry point that
    can be extended later with plotting helpers or more advanced
    diagnostics.
    """

    def __init__(
        self,
        feature_keys: Optional[Iterable[str]] = None,
        *,
        how: str = "inner",
        corr_method: Optional[str] = "pearson",
        max_categories: int = 20,
    ) -> None:
        if how not in {"inner", "left"}:
            raise ValueError("`how` must be 'inner' or 'left'.")

        self.feature_keys = list(feature_keys) if feature_keys is not None else None
        self.how = how
        self.corr_method = corr_method
        self.max_categories = max_categories

    def run(self, dataset: RoxyDataset) -> BasicEDAReport:
        """Run basic EDA on the given dataset."""
        # 1. Determine which feature blocks to use
        if self.feature_keys is None:
            feature_keys = dataset.feature_blocks
        else:
            feature_keys = self.feature_keys

        if not feature_keys:
            raise ValueError(
                "No feature blocks available for EDA. "
                "Make sure you have added features to the dataset."
            )

        # 2. Retrieve combined feature table
        X = dataset.get_feature_table(keys=feature_keys, how=self.how)

        # 3. Separate numeric / non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = [c for c in X.columns if c not in numeric_cols]

        # 4. Numeric summary
        if numeric_cols:
            numeric_summary = X[numeric_cols].describe().T
        else:
            numeric_summary = pd.DataFrame()

        # 5. Categorical summary
        categorical_summary: Dict[str, pd.Series] = {}
        for col in non_numeric_cols:
            vc = X[col].value_counts(dropna=False)
            if len(vc) > self.max_categories:
                vc = vc.head(self.max_categories)
            categorical_summary[col] = vc

        # 6. Missingness per column
        missing_per_column = X.isna().sum()

        # 7. Correlations
        if self.corr_method is not None and numeric_cols:
            correlations = X[numeric_cols].corr(method=self.corr_method)
        else:
            correlations = None

        # 8. Metadata
        metadata: Dict[str, Any] = {
            "dataset_name": dataset.name,
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "y_available": dataset.y is not None,
        }
        if dataset.metadata is not None:
            metadata.update(dataset.metadata)

        return BasicEDAReport(
            shape=X.shape,
            feature_keys=list(feature_keys),
            numeric_summary=numeric_summary,
            categorical_summary=categorical_summary,
            missing_per_column=missing_per_column,
            correlations=correlations,
            metadata=metadata,
        )
