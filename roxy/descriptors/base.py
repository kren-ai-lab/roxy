"""Base class for Roxy descriptor families."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

import pandas as pd


class BaseDescriptor(ABC):
    """Abstract base for sequence descriptor families.

    Subclasses declare ``name`` (registry key) and ``family`` (grouping
    label), then implement :meth:`compute_one`. The :meth:`compute` method
    is provided and applies ``compute_one`` to each sequence, returning a
    DataFrame with columns prefixed by ``{name}_``.
    """

    name: str = "base"
    family: str = "misc"

    @abstractmethod
    def compute_one(self, sequence: str) -> dict[str, float]:
        """Compute descriptors for a single amino-acid sequence.

        Returns
        -------
        dict
            Mapping of unprefixed feature name → value.

        """
        raise NotImplementedError

    def compute(
        self,
        sequences: Iterable[str],
        *,
        ids: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Compute descriptors for multiple sequences.

        Parameters
        ----------
        sequences:
            Iterable of amino-acid strings.
        ids:
            Optional row labels for the resulting DataFrame.

        Returns
        -------
        pandas.DataFrame
            One row per sequence; columns are ``{name}_{feature}``.

        """
        seqs = list(sequences)
        rows = [self.compute_one(s) for s in seqs]
        df = pd.DataFrame(rows, index=list(ids) if ids is not None else None)
        return df.add_prefix(f"{self.name}_")

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(name={self.name!r}, family={self.family!r})"


__all__ = ["BaseDescriptor"]
