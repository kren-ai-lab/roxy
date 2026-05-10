"""Base class for Roxy descriptor families."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterable


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

        Returns:
            Mapping of unprefixed feature name to value.

        """
        raise NotImplementedError

    def compute(
        self,
        sequences: Iterable[str],
        *,
        ids: Iterable[str] | None = None,
    ) -> pl.DataFrame:
        """Compute descriptors for multiple sequences.

        Args:
            sequences: Iterable of amino-acid strings.
            ids: Optional sequence identifiers stored in an ``id`` column.

        Returns:
            One row per sequence; columns are ``{name}_{feature}``.
            If ``ids`` is provided, an ``id`` column is prepended.

        """
        seqs = list(sequences)
        rows = [self.compute_one(s) for s in seqs]
        df = pl.DataFrame(rows).rename(lambda col: f"{self.name}_{col}")
        if ids is not None:
            df = df.with_columns(pl.Series("id", list(ids))).select(["id", *df.columns])
        return df

    def __repr__(self) -> str:  # pragma: no cover
        """Return a string representation of the descriptor."""
        return f"{self.__class__.__name__}(name={self.name!r}, family={self.family!r})"


__all__ = ["BaseDescriptor"]
