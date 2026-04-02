"""Sequence cleaning helpers for protein descriptor workflows.

This module provides a small, explicit preprocessing layer for protein
sequence inputs. The focus is deterministic normalization rather than
descriptor-specific business logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isnan
from typing import Iterable, List, Literal, Optional, Sequence

from roxy.core.constants import AA20
from roxy.core.exceptions import (
    InvalidSequenceError,
    SequenceCollectionError,
    SequenceInputError,
)

InvalidResiduePolicy = Literal["strict", "drop", "keep_unknown"]


@dataclass(frozen=True)
class SequenceCleaningConfig:
    """Configuration for sequence normalization and invalid-residue handling."""

    uppercase: bool = True
    strip_external_whitespace: bool = True
    remove_internal_whitespace: bool = True
    remove_terminal_stop: bool = True
    invalid_policy: InvalidResiduePolicy = "strict"
    allowed_residues: frozenset[str] = field(
        default_factory=lambda: frozenset(AA20)
    )


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


def normalize_sequence(
    sequence: object,
    *,
    uppercase: bool = True,
    strip_external_whitespace: bool = True,
    remove_internal_whitespace: bool = True,
) -> Optional[str]:
    """Normalize a raw sequence-like value into a cleaned string or ``None``.

    Normalization includes optional external trimming, uppercasing, and
    removal of internal whitespace characters.
    """
    if is_missing_sequence_value(sequence):
        return None
    if not isinstance(sequence, str):
        raise SequenceInputError(
            "Sequence input must be a string, `None`, or NaN-like missing "
            f"value; received {type(sequence).__name__}."
        )

    normalized = sequence
    if strip_external_whitespace:
        normalized = normalized.strip()
    if uppercase:
        normalized = normalized.upper()
    if remove_internal_whitespace:
        normalized = "".join(ch for ch in normalized if not ch.isspace())
    return normalized


def remove_terminal_stop_marker(sequence: str) -> str:
    """Remove a single terminal ``*`` stop marker when present."""
    if sequence.endswith("*"):
        return sequence[:-1]
    return sequence


def find_noncanonical_residues(
    sequence: str,
    *,
    allowed_residues: Iterable[str] = AA20,
) -> List[str]:
    """Return sorted non-canonical residue symbols present in a sequence."""
    allowed = set(allowed_residues)
    return sorted({char for char in sequence if char not in allowed})


def clean_sequence(
    sequence: object,
    *,
    config: Optional[SequenceCleaningConfig] = None,
    uppercase: bool = True,
    strip_external_whitespace: bool = True,
    remove_internal_whitespace: bool = True,
    remove_terminal_stop: bool = True,
    invalid_policy: InvalidResiduePolicy = "strict",
    allowed_residues: Iterable[str] = AA20,
) -> Optional[str]:
    """Clean a protein sequence according to an explicit policy.

    Parameters are available directly for convenience. When ``config`` is
    provided, it takes precedence over the per-argument defaults.

    Invalid-residue policies
    ------------------------
    ``strict``:
        Raise :class:`InvalidSequenceError` when non-canonical residues
        remain.
    ``drop``:
        Remove non-canonical residues after normalization.
    ``keep_unknown``:
        Keep non-canonical residues in the returned sequence unchanged.
    """
    if config is not None:
        uppercase = config.uppercase
        strip_external_whitespace = config.strip_external_whitespace
        remove_internal_whitespace = config.remove_internal_whitespace
        remove_terminal_stop = config.remove_terminal_stop
        invalid_policy = config.invalid_policy
        allowed_residues = config.allowed_residues

    normalized = normalize_sequence(
        sequence,
        uppercase=uppercase,
        strip_external_whitespace=strip_external_whitespace,
        remove_internal_whitespace=remove_internal_whitespace,
    )
    if normalized is None:
        return None

    cleaned = normalized
    if remove_terminal_stop:
        cleaned = remove_terminal_stop_marker(cleaned)

    invalid_residues = find_noncanonical_residues(
        cleaned,
        allowed_residues=allowed_residues,
    )
    if not invalid_residues:
        return cleaned

    if invalid_policy == "keep_unknown":
        return cleaned

    if invalid_policy == "drop":
        allowed = set(allowed_residues)
        return "".join(char for char in cleaned if char in allowed)

    if invalid_policy != "strict":
        raise SequenceInputError(
            f"Unsupported invalid residue policy: {invalid_policy!r}."
        )

    invalid_text = ", ".join(invalid_residues)
    raise InvalidSequenceError(
        "Sequence contains non-canonical residues after normalization: "
        f"{invalid_text}."
    )


def clean_sequences(
    sequences: Sequence[object] | Iterable[object],
    *,
    config: Optional[SequenceCleaningConfig] = None,
    uppercase: bool = True,
    strip_external_whitespace: bool = True,
    remove_internal_whitespace: bool = True,
    remove_terminal_stop: bool = True,
    invalid_policy: InvalidResiduePolicy = "strict",
    allowed_residues: Iterable[str] = AA20,
) -> List[Optional[str]]:
    """Clean an iterable of sequences using the same policy for each item."""
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
        clean_sequence(
            sequence,
            config=config,
            uppercase=uppercase,
            strip_external_whitespace=strip_external_whitespace,
            remove_internal_whitespace=remove_internal_whitespace,
            remove_terminal_stop=remove_terminal_stop,
            invalid_policy=invalid_policy,
            allowed_residues=allowed_residues,
        )
        for sequence in values
    ]


__all__ = [
    "InvalidResiduePolicy",
    "SequenceCleaningConfig",
    "clean_sequence",
    "clean_sequences",
    "find_noncanonical_residues",
    "is_missing_sequence_value",
    "normalize_sequence",
    "remove_terminal_stop_marker",
]
