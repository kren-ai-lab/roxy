"""Explicit sequence cleaning helpers for protein descriptor workflows.

Cleaning is responsible only for deterministic normalization and residue
policy application. It does not decide whether a cleaned sequence is
acceptable for a given downstream task; that belongs to validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isnan
from typing import Iterable, Literal, Optional, Sequence

from roxy.core.constants import AA20, SEQUENCE_STOP_MARKER
from roxy.core.exceptions import (
    InvalidSequenceError,
    SequenceCollectionError,
    SequenceInputError,
)

InvalidResiduePolicy = Literal["strict", "drop", "keep_unknown"]


@dataclass(frozen=True)
class SequenceCleaningConfig:
    """Configuration for deterministic sequence cleaning."""

    remove_internal_whitespace: bool = True
    remove_stop: bool = True
    remove_terminal_stop: Optional[bool] = None
    invalid_policy: InvalidResiduePolicy = "strict"
    allowed_residues: frozenset[str] = field(
        default_factory=lambda: frozenset(AA20)
    )

    def __post_init__(self) -> None:
        if self.remove_terminal_stop is None:
            object.__setattr__(self, "remove_terminal_stop", self.remove_stop)
            return
        object.__setattr__(self, "remove_stop", self.remove_terminal_stop)


def is_missing_sequence_value(value: object) -> bool:
    """Return whether a value should be treated as a missing sequence."""
    if value is None:
        return True
    if isinstance(value, float):
        try:
            return isnan(value)
        except TypeError:  # pragma: no cover - defensive numeric guard
            return False
    return False


def require_sequence_string(sequence: object) -> str:
    """Return a string sequence or raise a clear input error."""
    if is_missing_sequence_value(sequence):
        raise SequenceInputError("Sequence input cannot be missing.")
    if not isinstance(sequence, str):
        raise SequenceInputError(
            "Sequence input must be a string; "
            f"received {type(sequence).__name__}."
        )
    return sequence


def normalize_sequence(
    sequence: str,
    *,
    remove_internal_whitespace: bool = True,
) -> str:
    """Normalize a sequence string by trimming, uppercasing, and spacing policy."""
    normalized = require_sequence_string(sequence).strip().upper()
    if remove_internal_whitespace:
        normalized = "".join(char for char in normalized if not char.isspace())
    return normalized


def normalize_sequence_value(
    sequence: object,
    *,
    remove_internal_whitespace: bool = True,
) -> Optional[str]:
    """Normalize a possibly-missing sequence value for pipeline reuse."""
    if is_missing_sequence_value(sequence):
        return None
    return normalize_sequence(
        require_sequence_string(sequence),
        remove_internal_whitespace=remove_internal_whitespace,
    )


def remove_terminal_stop_marker(
    sequence: str,
    *,
    stop_marker: str = SEQUENCE_STOP_MARKER,
) -> str:
    """Remove exactly one terminal stop marker when present."""
    if sequence.endswith(stop_marker):
        return sequence[:-1]
    return sequence


def find_noncanonical_residues(
    sequence: str,
    *,
    allowed_residues: Iterable[str] = AA20,
) -> list[str]:
    """Return sorted non-canonical residue symbols present in a sequence."""
    allowed = set(allowed_residues)
    return sorted({char for char in sequence if char not in allowed})


def apply_residue_policy(
    sequence: str,
    *,
    policy: InvalidResiduePolicy = "strict",
    allowed_residues: Iterable[str] = AA20,
) -> str:
    """Apply the configured invalid-residue policy to a normalized sequence."""
    invalid_residues = find_noncanonical_residues(
        sequence,
        allowed_residues=allowed_residues,
    )
    if not invalid_residues:
        return sequence

    if policy == "keep_unknown":
        return sequence

    if policy == "drop":
        allowed = set(allowed_residues)
        return "".join(char for char in sequence if char in allowed)

    if policy != "strict":
        raise SequenceInputError(
            f"Unsupported invalid residue policy: {policy!r}."
        )

    invalid_text = ", ".join(invalid_residues)
    raise InvalidSequenceError(
        "Sequence contains non-canonical residues after cleaning: "
        f"{invalid_text}."
    )


def clean_sequence(
    sequence: str,
    *,
    policy: InvalidResiduePolicy = "strict",
    remove_stop: bool = True,
    remove_internal_whitespace: bool = True,
    allowed_residues: Iterable[str] = AA20,
    config: Optional[SequenceCleaningConfig] = None,
    invalid_policy: Optional[InvalidResiduePolicy] = None,
) -> str:
    """Clean a protein sequence according to an explicit residue policy.

    The cleaning steps are:
    1. trim external whitespace,
    2. convert to uppercase,
    3. remove internal whitespace when requested,
    4. remove a single terminal ``*`` when requested,
    5. apply the invalid-residue policy.

    Internal ``*`` characters are never silently removed. When present,
    they are treated as non-canonical symbols and therefore follow the
    configured residue policy.
    """
    if config is not None:
        policy = config.invalid_policy
        remove_stop = config.remove_stop
        remove_internal_whitespace = config.remove_internal_whitespace
        allowed_residues = config.allowed_residues
    elif invalid_policy is not None:
        policy = invalid_policy

    normalized = normalize_sequence(
        require_sequence_string(sequence),
        remove_internal_whitespace=remove_internal_whitespace,
    )

    cleaned = normalized
    if remove_stop:
        cleaned = remove_terminal_stop_marker(cleaned)

    return apply_residue_policy(
        cleaned,
        policy=policy,
        allowed_residues=allowed_residues,
    )


def clean_sequence_value(
    sequence: object,
    *,
    policy: InvalidResiduePolicy = "strict",
    remove_stop: bool = True,
    remove_internal_whitespace: bool = True,
    allowed_residues: Iterable[str] = AA20,
    config: Optional[SequenceCleaningConfig] = None,
    invalid_policy: Optional[InvalidResiduePolicy] = None,
) -> Optional[str]:
    """Clean a possibly-missing sequence value for batch-oriented pipelines."""
    if is_missing_sequence_value(sequence):
        return None
    return clean_sequence(
        require_sequence_string(sequence),
        policy=policy,
        remove_stop=remove_stop,
        remove_internal_whitespace=remove_internal_whitespace,
        allowed_residues=allowed_residues,
        config=config,
        invalid_policy=invalid_policy,
    )


def clean_sequences(
    sequences: Sequence[object] | Iterable[object],
    *,
    policy: InvalidResiduePolicy = "strict",
    remove_stop: bool = True,
    remove_internal_whitespace: bool = True,
    allowed_residues: Iterable[str] = AA20,
    config: Optional[SequenceCleaningConfig] = None,
    invalid_policy: Optional[InvalidResiduePolicy] = None,
) -> list[Optional[str]]:
    """Clean an iterable of sequences using one shared cleaning policy."""
    if isinstance(sequences, str):
        raise SequenceCollectionError(
            "Expected an iterable of sequence values, not a single string. "
            "Use `clean_sequence(...)` for single-sequence inputs."
        )

    try:
        values = list(sequences)
    except TypeError as exc:
        raise SequenceCollectionError(
            "Expected an iterable of sequence values."
        ) from exc

    return [
        clean_sequence_value(
            sequence,
            policy=policy,
            remove_stop=remove_stop,
            remove_internal_whitespace=remove_internal_whitespace,
            allowed_residues=allowed_residues,
            config=config,
            invalid_policy=invalid_policy,
        )
        for sequence in values
    ]


__all__ = [
    "InvalidResiduePolicy",
    "SequenceCleaningConfig",
    "apply_residue_policy",
    "clean_sequence",
    "clean_sequence_value",
    "clean_sequences",
    "find_noncanonical_residues",
    "is_missing_sequence_value",
    "normalize_sequence",
    "normalize_sequence_value",
    "remove_terminal_stop_marker",
    "require_sequence_string",
]
