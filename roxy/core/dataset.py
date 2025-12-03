
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np


@dataclass
class RoxyDataset:
    """Multimodal container for molecular data used in exploratory analysis."""

    samples: pd.DataFrame
    y: Optional[pd.Series | np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    name: Optional[str] = None
    features: Dict[str, pd.DataFrame] = field(default_factory=dict)

    def add_features(self, key: str, X: pd.DataFrame) -> None:
        """Register a feature table under a given key (e.g. 'seq_desc')."""
        self.features[key] = X

    def get_merged_features(self) -> pd.DataFrame:
        """Return a single DataFrame with all registered feature tables merged on index."""
        if not self.features:
            raise ValueError("No feature tables have been registered in this dataset.")
        dfs = list(self.features.values())
        merged = dfs[0]
        for df in dfs[1:]:
            merged = merged.join(df, how="inner")
        return merged
