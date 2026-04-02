from __future__ import annotations

"""Core engine abstractions for sequence descriptor computation.

This module defines the small DataFrame-oriented engine contract used by
the active sequence architecture. It is kept separate from the modular
block abstractions so sequence-family logic can stay lightweight while
high-level orchestration still has a stable tabular interface.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    import pandas as pd


class BaseDescriptorEngine(ABC):
    """Abstract base class for DataFrame-oriented descriptor engines."""

    #: Optional engine name used for registration, logging, or display.
    name: str = "base_descriptor_engine"

    @abstractmethod
    def compute(self, samples: "pd.DataFrame") -> "pd.DataFrame":
        """Compute descriptors for a samples table."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - trivial
        name = getattr(self, "name", self.__class__.__name__)
        return f"{self.__class__.__name__}(name={name!r})"
