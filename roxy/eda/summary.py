
from __future__ import annotations

from typing import Optional, Dict

import numpy as np
import pandas as pd

from roxy.core.report import DatasetReport, FeatureSummary


def build_report(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    dataset_name: Optional[str] = None,
    task_type: Optional[str] = None,
) -> DatasetReport:
    """Build a basic exploratory analysis report from a feature matrix."""
    n_samples, n_features = X.shape
    class_distribution: Optional[Dict[str, int]] = None
    if task_type == "classification" and y is not None:
        class_distribution = y.value_counts().to_dict()

    feature_summaries: Dict[str, FeatureSummary] = {}

    for col in X.columns:
        s = X[col]
        n_missing = int(s.isna().sum())
        missing_ratio = float(n_missing) / n_samples
        n_unique = int(s.nunique(dropna=True))
        dtype = str(s.dtype)

        if np.issubdtype(s.dtype, np.number):
            desc = s.describe()
            mean = float(desc.get("mean", np.nan))
            std = float(desc.get("std", np.nan))
            _min = float(desc.get("min", np.nan))
            _max = float(desc.get("max", np.nan))
            skewness = float(s.skew())
            kurtosis = float(s.kurt())
        else:
            mean = std = _min = _max = skewness = kurtosis = None

        fs = FeatureSummary(
            name=col,
            dtype=dtype,
            n_missing=n_missing,
            missing_ratio=missing_ratio,
            n_unique=n_unique,
            mean=mean,
            std=std,
            min=_min,
            max=_max,
            skewness=skewness,
            kurtosis=kurtosis,
            target_association=None,
        )
        feature_summaries[col] = fs

    corr = X.corr(numeric_only=True)

    return DatasetReport(
        dataset_name=dataset_name,
        n_samples=n_samples,
        n_features=n_features,
        task_type=task_type,
        class_distribution=class_distribution,
        feature_summaries=feature_summaries,
        correlation_matrix=corr,
        notes=[],
    )
