
from __future__ import annotations

from typing import Optional, Dict

import numpy as np
import pandas as pd

from roxy.core.report import DatasetReport, FeatureSummary
from roxy.eda.target_relations import compute_feature_target_associations


def build_report(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    dataset_name: Optional[str] = None,
    task_type: Optional[str] = None,
) -> DatasetReport:
    """Build a basic exploratory analysis report from a feature matrix.

    This function summarises each feature and, when a suitable target
    is provided for a classification task, computes univariate
    feature–target associations (ANOVA/Kruskal + optional post hoc).

    Parameters
    ----------
    X :
        Feature matrix with samples in rows and features in columns.
    y :
        Optional target vector. For classification tasks, this should be
        a categorical Series with the same index as ``X``.
    dataset_name :
        Optional dataset name for reporting.
    task_type :
        Optional task type (e.g. ``"classification"`` or ``"regression"``).
        Currently, feature–target association tests are only computed for
        ``"classification"``.
    """
    n_samples, n_features = X.shape

    # Class distribution for classification tasks
    class_distribution: Optional[Dict[str, int]] = None
    if task_type == "classification" and y is not None:
        class_distribution = y.value_counts().to_dict()

    # Compute feature–target associations if applicable
    if y is not None and task_type == "classification":
        associations = compute_feature_target_associations(
            X, y, task_type=task_type, numeric_only=True
        )
    else:
        associations = {col: None for col in X.columns}

    feature_summaries: Dict[str, FeatureSummary] = {}

    for col in X.columns:
        s = X[col]
        n_missing = int(s.isna().sum())
        missing_ratio = float(n_missing) / n_samples if n_samples > 0 else 0.0
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
            target_association=associations.get(col),
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
