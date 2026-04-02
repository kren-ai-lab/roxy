from __future__ import annotations

"""Base interface for descriptor engines in Roxy.

All descriptor engines operate on a samples table (pandas DataFrame) and
return a feature table with one row per sample. Engines should be
stateless or only store configuration (e.g. pH, list of AAIndex codes).
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    import pandas as pd


class BaseDescriptorEngine(ABC):
    """
    Abstract base class for descriptor calculators.

    Subclasses must implement :meth:`compute`, which receives a samples
    DataFrame and returns a feature table (DataFrame) indexed like
    ``samples``.
    """

    #: Optional engine name used for registration, logging or display.
    name: str = "base_descriptor_engine"

    @abstractmethod
    def compute(self, samples: "pd.DataFrame") -> "pd.DataFrame":
        """
        Compute descriptors for the given samples table.

        Parameters
        ----------
        samples :
            DataFrame with at least the columns required by the engine
            (e.g. ``'sequence'``, ``'pdb_path'`` or ``'smiles'``).

        Returns
        -------
        pandas.DataFrame
            Feature table indexed like ``samples``.
        """
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - trivial
        name = getattr(self, "name", self.__class__.__name__)
        return f"{self.__class__.__name__}(name={name!r})"
