"""Base abstractions for sequence descriptor components.

This module introduces a lightweight foundation for modular sequence
descriptor families without forcing an immediate rewrite of the
current codebase.

The design keeps three concerns small and explicit:
- a protocol for "sequence descriptor blocks" that describe one sequence,
- a reusable base class that materializes tabular outputs,
- an optional bridge to the existing descriptor-engine interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterable, Mapping, Optional, Protocol, Sequence, runtime_checkable

from roxy.descriptors.base import BaseDescriptorEngine

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    import pandas as pd

SequenceFeatureMap = Mapping[str, float]


@runtime_checkable
class SequenceDescriptorBlock(Protocol):
    """Protocol for a modular block that computes descriptors per sequence.

    Implementations should be stateless or configuration-only and return a
    mapping from feature name to numeric value for a single input sequence.
    """

    name: str

    def describe_sequence(self, sequence: str) -> SequenceFeatureMap:
        """Return descriptors for a single protein sequence."""
        ...


class BaseSequenceDescriptorBlock(ABC):
    """Abstract base class for modular sequence descriptor families.

    Subclasses implement :meth:`describe_sequence`. The base class
    provides convenience methods for batching and for converting feature
    mappings into pandas DataFrames.
    """

    #: Optional block name used in registries and logs.
    name: str = "sequence_descriptor_block"

    @abstractmethod
    def describe_sequence(self, sequence: str) -> SequenceFeatureMap:
        """Compute descriptors for a single protein sequence."""
        raise NotImplementedError

    def feature_names(self) -> Optional[Sequence[str]]:
        """Return a stable feature order when known in advance.

        Returning ``None`` means the order is inferred from the emitted
        dictionaries during batching.
        """
        return None

    def describe_sequences(
        self,
        sequences: Iterable[str],
        *,
        index: Optional[Iterable[object]] = None,
    ) -> "pd.DataFrame":
        """Compute descriptors for an iterable of sequences."""
        import pandas as pd

        rows = [dict(self.describe_sequence(sequence)) for sequence in sequences]
        df = pd.DataFrame(rows)

        feature_names = self.feature_names()
        if feature_names is not None and not df.empty:
            ordered = [name for name in feature_names if name in df.columns]
            remainder = [name for name in df.columns if name not in ordered]
            df = df[ordered + remainder]

        if index is not None:
            df.index = list(index)

        return df

    def describe_dataframe(
        self,
        samples: "pd.DataFrame",
        *,
        sequence_column: str = "sequence",
    ) -> "pd.DataFrame":
        """Compute descriptors from a DataFrame containing sequences."""
        if sequence_column not in samples.columns:
            raise KeyError(
                f"Sequence column {sequence_column!r} not found in samples."
            )

        return self.describe_sequences(
            samples[sequence_column].tolist(),
            index=samples.index,
        )

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{self.__class__.__name__}(name={self.name!r})"


class SequenceDescriptorEngine(BaseSequenceDescriptorBlock, BaseDescriptorEngine):
    """Bridge base class for sequence blocks that also act as engines.

    This keeps compatibility with the current descriptor-engine registry
    while allowing future sequence families to implement the smaller
    per-sequence abstraction first.
    """

    def compute(self, samples: "pd.DataFrame") -> "pd.DataFrame":
        """Compute descriptors for a samples table with a sequence column."""
        return self.describe_dataframe(samples, sequence_column="sequence")


# TODO:
# - Introduce a small config/metadata pattern only when multiple modular
#   sequence families need richer self-description.
# - Revisit whether some migrated helpers should become concrete block
#   classes after `roxy.sequence.api` is the primary public surface.

__all__ = [
    "BaseSequenceDescriptorBlock",
    "SequenceDescriptorBlock",
    "SequenceDescriptorEngine",
    "SequenceFeatureMap",
]
