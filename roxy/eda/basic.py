from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, Iterable, List, Optional
from dataclasses import dataclass, field

from roxy.core.dataset import RoxyDataset
from roxy.core.exceptions import EDAError
from roxy.core.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class BasicEDAReport:
    """Container for basic exploratory data analysis results."""

    shape: tuple[int, int]
    feature_keys: List[str]
    numeric_summary: pd.DataFrame
    categorical_summary: Dict[str, pd.Series]
    missing_per_column: pd.Series
    correlations: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation suitable for serialisation."""
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
    """Basic exploratory data analysis engine for Roxy.

    Provides summary statistics, missingness diagnostics,
    correlations, and categorical summaries.
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
            raise EDAError("`how` must be 'inner' or 'left'.")

        self.feature_keys = list(feature_keys) if feature_keys is not None else None
        self.how = how
        self.corr_method = corr_method
        self.max_categories = max_categories

    def run(self, dataset: RoxyDataset) -> BasicEDAReport:
        """Run EDA on one or more feature blocks of a RoxyDataset."""
        logger.info("BasicEDA: beginning EDA on dataset '%s'.", dataset.name)

        if self.feature_keys is None:
            feature_keys = dataset.feature_blocks
            logger.debug("BasicEDA: using all feature blocks: %s.", feature_keys)
        else:
            feature_keys = self.feature_keys
            logger.debug("BasicEDA: using selected feature blocks: %s.", feature_keys)

        if not feature_keys:
            raise EDAError("No feature blocks available for EDA.")

        X = dataset.get_feature_table(keys=feature_keys, how=self.how)
        logger.info("BasicEDA: combined feature table shape = %s.", X.shape)

        # Separate numeric and categorical
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = [c for c in X.columns if c not in numeric_cols]

        # Numeric summary
        numeric_summary = (
            X[numeric_cols].describe().T if numeric_cols else pd.DataFrame()
        )

        # Categorical summary
        categorical_summary: Dict[str, pd.Series] = {}
        for col in non_numeric_cols:
            vc = X[col].value_counts(dropna=False)
            if len(vc) > self.max_categories:
                vc = vc.head(self.max_categories)
            categorical_summary[col] = vc

        # Missingness
        missing_per_column = X.isna().sum()

        # Correlations
        correlations = None
        if self.corr_method is not None and numeric_cols:
            try:
                correlations = X[numeric_cols].corr(method=self.corr_method)
            except Exception as e:
                logger.warning("BasicEDA: correlation computation failed: %s.", e)

        metadata: Dict[str, Any] = {
            "dataset_name": dataset.name,
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "y_available": dataset.y is not None,
        }
        if dataset.metadata:
            metadata.update(dataset.metadata)

        logger.info("BasicEDA: finished building report for dataset '%s'.", dataset.name)

        return BasicEDAReport(
            shape=X.shape,
            feature_keys=list(feature_keys),
            numeric_summary=numeric_summary,
            categorical_summary=categorical_summary,
            missing_per_column=missing_per_column,
            correlations=correlations,
            metadata=metadata,
        )
