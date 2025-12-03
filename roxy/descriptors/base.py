
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseDescriptorEngine(ABC):
    """Base class for descriptor calculators operating on a samples table."""

    name: str

    @abstractmethod
    def compute(self, samples: pd.DataFrame) -> pd.DataFrame:
        """Compute descriptors for the given samples table."""
