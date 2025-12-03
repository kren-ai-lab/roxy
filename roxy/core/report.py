
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List

import pandas as pd


@dataclass
class FeatureSummary:
    """Statistical summary for a single feature."""

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
