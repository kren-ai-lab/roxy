
from __future__ import annotations

"""Lightweight report structures used across Roxy.

These dataclasses provide a structured representation of dataset-level
summaries produced by the EDA routines. They are intentionally kept
simple and serialisable so they can be:

- converted to dicts,
- dumped to JSON/YAML,
- rendered into markdown or HTML reports.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List

import pandas as pd


@dataclass
class FeatureSummary:
    """Statistical summary for a single feature.

    This structure captures basic univariate statistics, including
    missingness and simple distribution properties. More advanced
    associations (e.g. with the target variable) can be stored in
    :attr:`target_association`.

    All numeric fields are optional so that the same container can
    be used for categorical and numeric features.
    """

    name: str
    dtype: str
    n_missing: int
    missing_ratio: float
    n_unique: int
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    target_association: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return the summary as a plain dictionary."""
        return asdict(self)


@dataclass
class DatasetReport:
    """Comprehensive report of a dataset used in exploratory analysis."""

    dataset_name: Optional[str]
    n_samples: int
    n_features: int
    task_type: Optional[str]
    class_distribution: Optional[Dict[str, int]]
    feature_summaries: Dict[str, FeatureSummary]
    correlation_matrix: Optional[pd.DataFrame] = None
    notes: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the report to a nested dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "task_type": self.task_type,
            "class_distribution": self.class_distribution,
            "feature_summaries": {
                name: fs.to_dict() for name, fs in self.feature_summaries.items()
            },
            "notes": self.notes or [],
        }