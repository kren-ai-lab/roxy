"""Shared descriptor block contracts and low-level sequence helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Callable, Protocol, TypeAlias, runtime_checkable

from roxy.core.exceptions import (
    DescriptorBlockContractError,
    SequenceCollectionError,
    SequenceInputError,
)
from roxy.sequence.cleaning import (
    normalize_sequence_value,
    remove_terminal_stop_marker,
)

DescriptorValue: TypeAlias = float | int | bool | None
DescriptorRecord: TypeAlias = dict[str, DescriptorValue]
SequenceFeatureMap: TypeAlias = Mapping[str, DescriptorValue]


def _is_valid_name_fragment(value: str) -> bool:
    """Return whether a family or prefix name is explicit and identifier-like."""
    return bool(value) and all(char.isalnum() or char == "_" for char in value)


def _validate_name_fragment(value: object, *, field_name: str) -> str:
    """Validate and normalize a block metadata value."""
    if not isinstance(value, str):
        raise DescriptorBlockContractError(
            f"`{field_name}` must be a string, received {type(value).__name__}."
        )

    normalized = value.strip()
    if not normalized:
        raise DescriptorBlockContractError(f"`{field_name}` cannot be empty.")
    if not _is_valid_name_fragment(normalized):
        raise DescriptorBlockContractError(
            f"`{field_name}` must contain only letters, digits, or underscores."
        )
    return normalized


def _validate_feature_value(name: str, value: object) -> DescriptorValue:
    """Validate a descriptor value and return it in its original scalar form."""
    if value is None or isinstance(value, (bool, int, float)):
        return value

    raise DescriptorBlockContractError(
        f"Feature {name!r} returned unsupported value type "
        f"{type(value).__name__}. Expected float, int, bool, or None."
    )


@runtime_checkable
class SequenceDescriptorBlock(Protocol):
    """Structural contract for a sequence descriptor block."""

    family_name: str
    column_prefix: str

    def transform_sequence(self, sequence: str) -> DescriptorRecord:
        """Transform one protein sequence into a flat descriptor record."""
        ...


class DescriptorBlock(ABC):
    """Abstract base class for explicit sequence descriptor blocks."""

    family_name: str = ""
    column_prefix: str = ""

    def __init__(self) -> None:
        self.family_name = _validate_name_fragment(
            self.family_name,
            field_name="family_name",
        )
        self.column_prefix = _validate_name_fragment(
            self.column_prefix,
            field_name="column_prefix",
        )

    @property
    def name(self) -> str:
        """Return the stable block name."""
        return self.family_name

    def feature_name(self, suffix: str) -> str:
        """Return a normalized output feature name for this block."""
        normalized_suffix = _validate_name_fragment(
            suffix,
            field_name="feature name",
        )
        expected_prefix = f"{self.column_prefix}_"
        if normalized_suffix.startswith(expected_prefix):
            return normalized_suffix
        return f"{expected_prefix}{normalized_suffix}"

    def transform_sequence(self, sequence: str) -> DescriptorRecord:
        """Transform a single protein sequence into a descriptor record."""
        if not isinstance(sequence, str):
            raise SequenceInputError(
                "Descriptor blocks expect `sequence` to be a string, "
                f"received {type(sequence).__name__}."
            )

        raw_features = self._transform_sequence(sequence)
        return self._normalize_output(raw_features)

    def transform_sequences(self, sequences: Iterable[str]) -> list[DescriptorRecord]:
        """Transform an iterable of sequences using the same block instance."""
        if isinstance(sequences, str):
            raise SequenceCollectionError(
                "Expected an iterable of sequences, not a single string. "
                "Use `transform_sequence(...)` for one sequence."
            )

        try:
            values = list(sequences)
        except TypeError as exc:
            raise SequenceCollectionError(
                "Expected an iterable of sequence strings."
            ) from exc

        return [self.transform_sequence(sequence) for sequence in values]

    def describe_sequence(self, sequence: str) -> DescriptorRecord:
        """Backward-compatible alias for :meth:`transform_sequence`."""
        return self.transform_sequence(sequence)

    def feature_names(self) -> tuple[str, ...] | None:
        """Return a stable feature order when the block can declare one."""
        return None

    @abstractmethod
    def _transform_sequence(self, sequence: str) -> Mapping[str, DescriptorValue]:
        """Compute the raw descriptor mapping for one sequence."""
        raise NotImplementedError

    def _normalize_output(
        self,
        features: Mapping[str, DescriptorValue],
    ) -> DescriptorRecord:
        """Validate and normalize block output into a plain dictionary."""
        if not isinstance(features, Mapping):
            raise DescriptorBlockContractError(
                "Descriptor block output must be a mapping from feature names "
                "to scalar values."
            )

        normalized: DescriptorRecord = {}
        for raw_name, raw_value in features.items():
            if not isinstance(raw_name, str):
                raise DescriptorBlockContractError(
                    "Descriptor feature names must be strings."
                )
            feature_name = self.feature_name(raw_name)
            normalized[feature_name] = _validate_feature_value(
                feature_name,
                raw_value,
            )
        return normalized

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"{self.__class__.__name__}("
            f"family_name={self.family_name!r}, "
            f"column_prefix={self.column_prefix!r})"
        )


BaseSequenceDescriptorBlock = DescriptorBlock


class FunctionDescriptorBlock(DescriptorBlock):
    """Descriptor block backed by a plain transformation function."""

    def __init__(
        self,
        family_name: str,
        column_prefix: str,
        transform_function: Callable[..., Mapping[str, DescriptorValue]],
        *,
        preserve_feature_names: bool = False,
        known_feature_names: Iterable[str] | None = None,
        **default_kwargs: object,
    ) -> None:
        self.family_name = family_name
        self.column_prefix = column_prefix
        self._transform_function = transform_function
        self._preserve_feature_names = preserve_feature_names
        self._known_feature_names = (
            tuple(known_feature_names) if known_feature_names is not None else None
        )
        self._default_kwargs = dict(default_kwargs)
        super().__init__()

    def feature_names(self) -> tuple[str, ...] | None:
        """Return predeclared feature names when available."""
        return self._known_feature_names

    def _transform_sequence(self, sequence: str) -> Mapping[str, DescriptorValue]:
        """Call the wrapped descriptor function for one sequence."""
        return self._transform_function(sequence, **self._default_kwargs)

    def _normalize_output(
        self,
        features: Mapping[str, DescriptorValue],
    ) -> DescriptorRecord:
        """Optionally keep function-emitted feature names unchanged."""
        if not self._preserve_feature_names:
            return super()._normalize_output(features)

        if not isinstance(features, Mapping):
            raise DescriptorBlockContractError(
                "Descriptor block output must be a mapping from feature names "
                "to scalar values."
            )

        normalized: DescriptorRecord = {}
        for raw_name, raw_value in features.items():
            if not isinstance(raw_name, str):
                raise DescriptorBlockContractError(
                    "Descriptor feature names must be strings."
                )
            normalized[raw_name] = _validate_feature_value(raw_name, raw_value)
        return normalized


def prepare_descriptor_sequence(sequence: object) -> str:
    """Normalize a sequence for low-level descriptor computation."""
    normalized = normalize_sequence_value(
        sequence,
        remove_internal_whitespace=False,
    )
    if normalized is None:
        return ""
    normalized = remove_terminal_stop_marker(normalized)
    return normalized.replace("*", "")


def mean_scale(sequence: str, scale: Mapping[str, float]) -> float:
    """Return the mean value of a residue-level scale over a sequence."""
    values = [scale[aa] for aa in prepare_descriptor_sequence(sequence) if aa in scale]
    if not values:
        return math.nan
    return float(sum(values) / len(values))


def count_residues(sequence: str, residues: Iterable[str]) -> int:
    """Count the total number of residues from a set within a sequence."""
    cleaned = prepare_descriptor_sequence(sequence)
    return int(sum(cleaned.count(residue) for residue in residues))


def sum_residue_values(
    sequence: str,
    values: Mapping[str, int | float],
) -> float:
    """Sum per-residue values over a cleaned sequence."""
    cleaned = prepare_descriptor_sequence(sequence)
    return float(sum(values.get(residue, 0) for residue in cleaned))


def safe_divide(
    numerator: int | float,
    denominator: int | float,
    *,
    default: float = 0.0,
) -> float:
    """Safely divide two numbers, returning a default for zero denominator."""
    if denominator == 0:
        return float(default)
    return float(numerator / denominator)


__all__ = [
    "BaseSequenceDescriptorBlock",
    "DescriptorBlock",
    "DescriptorRecord",
    "DescriptorValue",
    "FunctionDescriptorBlock",
    "SequenceDescriptorBlock",
    "SequenceFeatureMap",
    "count_residues",
    "mean_scale",
    "prepare_descriptor_sequence",
    "safe_divide",
    "sum_residue_values",
]
