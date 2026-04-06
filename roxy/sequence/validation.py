"""Sequence validation helpers independent from descriptor computation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from roxy.core.constants import AA20
from roxy.core.exceptions import (
    EmptySequenceError,
    InvalidSequenceError,
    MissingSequenceError,
    SequenceCollectionError,
    SequenceInputError,
    SequenceTooShortError,
)
from roxy.sequence.cleaning import (
    InvalidResiduePolicy,
    SequenceCleaningConfig,
    clean_sequence_value,
    find_noncanonical_residues,
    is_missing_sequence_value,
    normalize_sequence_value,
    remove_terminal_stop_marker,
    require_sequence_string,
)


@dataclass(frozen=True)
class SequenceValidationResult:
    """Structured validation report for one sequence value."""

    original: object
    normalized: Optional[str]
    cleaned: Optional[str]
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    invalid_residues: list[str] = field(default_factory=list)
    is_missing: bool = False
    is_empty: bool = False
    length: int = 0
    min_length: Optional[int] = None


class SequenceValidationError(InvalidSequenceError):
    """Raised when validation fails without a more specific exception."""


def _resolve_cleaning_config(
    *,
    cleaning_config: Optional[SequenceCleaningConfig],
    invalid_policy: InvalidResiduePolicy,
    allowed: Iterable[str],
    remove_internal_whitespace: bool = True,
    remove_terminal_stop: bool = True,
) -> SequenceCleaningConfig:
    """Return the effective cleaning configuration used for validation."""
    if cleaning_config is not None:
        return cleaning_config
    return SequenceCleaningConfig(
        remove_internal_whitespace=remove_internal_whitespace,
        remove_terminal_stop=remove_terminal_stop,
        invalid_policy=invalid_policy,
        allowed_residues=frozenset(allowed),
    )


def _normalized_for_validation(
    sequence: object,
    *,
    cleaning_config: SequenceCleaningConfig,
) -> Optional[str]:
    """Return the normalized sequence before residue-policy application."""
    normalized = normalize_sequence_value(
        sequence,
        remove_internal_whitespace=cleaning_config.remove_internal_whitespace,
    )
    if normalized is None:
        return None
    if cleaning_config.remove_terminal_stop:
        return remove_terminal_stop_marker(normalized)
    return normalized


def find_invalid_residues(
    sequence: object,
    *,
    cleaning_config: Optional[SequenceCleaningConfig] = None,
    allowed: Iterable[str] = AA20,
    invalid_policy: InvalidResiduePolicy = "strict",
    remove_internal_whitespace: bool = True,
    remove_terminal_stop: bool = True,
) -> list[str]:
    """Return sorted invalid residue symbols for a sequence-like value."""
    effective_config = _resolve_cleaning_config(
        cleaning_config=cleaning_config,
        invalid_policy=invalid_policy,
        allowed=allowed,
        remove_internal_whitespace=remove_internal_whitespace,
        remove_terminal_stop=remove_terminal_stop,
    )
    normalized = _normalized_for_validation(
        sequence,
        cleaning_config=effective_config,
    )
    if normalized is None:
        return []
    return find_noncanonical_residues(
        normalized,
        allowed_residues=effective_config.allowed_residues,
    )


def inspect_sequence(
    sequence: object,
    *,
    cleaning_config: Optional[SequenceCleaningConfig] = None,
    invalid_policy: InvalidResiduePolicy = "strict",
    allowed: Iterable[str] = AA20,
    allow_empty: bool = False,
    min_length: Optional[int] = None,
    remove_internal_whitespace: bool = True,
    remove_terminal_stop: bool = True,
) -> SequenceValidationResult:
    """Inspect a sequence value and return a structured validation report."""
    if min_length is not None and min_length < 0:
        raise SequenceInputError("`min_length` must be non-negative.")

    effective_config = _resolve_cleaning_config(
        cleaning_config=cleaning_config,
        invalid_policy=invalid_policy,
        allowed=allowed,
        remove_internal_whitespace=remove_internal_whitespace,
        remove_terminal_stop=remove_terminal_stop,
    )

    if is_missing_sequence_value(sequence):
        return SequenceValidationResult(
            original=sequence,
            normalized=None,
            cleaned=None,
            is_valid=False,
            errors=["Sequence value is missing."],
            is_missing=True,
            min_length=min_length,
        )

    try:
        require_sequence_string(sequence)
    except SequenceInputError as exc:
        return SequenceValidationResult(
            original=sequence,
            normalized=None,
            cleaned=None,
            is_valid=False,
            errors=[str(exc)],
            min_length=min_length,
        )

    normalized = _normalized_for_validation(
        sequence,
        cleaning_config=effective_config,
    )
    invalid_residues = find_invalid_residues(
        sequence,
        cleaning_config=effective_config,
    )

    cleaned = None
    try:
        cleaned = clean_sequence_value(
            sequence,
            config=effective_config,
        )
    except InvalidSequenceError:
        cleaned = normalized

    cleaned = "" if cleaned is None else cleaned
    length = len(cleaned)
    is_empty = length == 0

    errors: list[str] = []
    if is_empty and not allow_empty:
        errors.append("Sequence is empty after cleaning.")
    if (
        min_length is not None
        and not is_empty
        and length < min_length
    ):
        errors.append(
            f"Sequence length {length} is shorter than the required minimum "
            f"length {min_length}."
        )
    if effective_config.invalid_policy == "strict" and invalid_residues:
        invalid_text = ", ".join(invalid_residues)
        errors.append(
            "Sequence contains non-canonical residues: "
            f"{invalid_text}."
        )

    return SequenceValidationResult(
        original=sequence,
        normalized=normalized,
        cleaned=cleaned,
        is_valid=not errors,
        errors=errors,
        invalid_residues=invalid_residues,
        is_empty=is_empty,
        length=length,
        min_length=min_length,
    )


def validate_sequence(
    sequence: object,
    *,
    cleaning_config: Optional[SequenceCleaningConfig] = None,
    invalid_policy: InvalidResiduePolicy = "strict",
    allowed: Iterable[str] = AA20,
    allow_empty: bool = False,
    min_length: Optional[int] = None,
    remove_internal_whitespace: bool = True,
    remove_terminal_stop: bool = True,
) -> None:
    """Validate one sequence value and raise a clear exception on failure."""
    result = inspect_sequence(
        sequence,
        cleaning_config=cleaning_config,
        invalid_policy=invalid_policy,
        allowed=allowed,
        allow_empty=allow_empty,
        min_length=min_length,
        remove_internal_whitespace=remove_internal_whitespace,
        remove_terminal_stop=remove_terminal_stop,
    )
    if result.is_valid:
        return

    if result.is_missing:
        raise MissingSequenceError("Sequence value is missing.")
    if result.normalized is None and not result.errors:
        raise SequenceInputError("Sequence input is invalid.")
    if result.errors:
        if any("must be a string" in error for error in result.errors):
            raise SequenceInputError(result.errors[0])
        if result.is_empty:
            raise EmptySequenceError(result.errors[0])
        if any("shorter than the required minimum" in error for error in result.errors):
            raise SequenceTooShortError(result.errors[0])
        if result.invalid_residues:
            raise InvalidSequenceError(result.errors[0])
        raise SequenceValidationError(" ".join(result.errors))

    raise SequenceValidationError("Sequence validation failed.")


def assert_valid_sequence(
    sequence: object,
    *,
    cleaning_config: Optional[SequenceCleaningConfig] = None,
    invalid_policy: InvalidResiduePolicy = "strict",
    allowed: Iterable[str] = AA20,
    allow_empty: bool = False,
    min_length: Optional[int] = None,
    remove_internal_whitespace: bool = True,
    remove_terminal_stop: bool = True,
) -> str:
    """Validate one sequence and return the cleaned string."""
    result = inspect_sequence(
        sequence,
        cleaning_config=cleaning_config,
        invalid_policy=invalid_policy,
        allowed=allowed,
        allow_empty=allow_empty,
        min_length=min_length,
        remove_internal_whitespace=remove_internal_whitespace,
        remove_terminal_stop=remove_terminal_stop,
    )
    if not result.is_valid:
        validate_sequence(
            sequence,
            cleaning_config=cleaning_config,
            invalid_policy=invalid_policy,
            allowed=allowed,
            allow_empty=allow_empty,
            min_length=min_length,
            remove_internal_whitespace=remove_internal_whitespace,
            remove_terminal_stop=remove_terminal_stop,
        )
    return result.cleaned or ""


def validate_sequences_input(
    sequences: Sequence[object] | Iterable[object],
) -> list[object]:
    """Validate collection input shape and materialize it as a list."""
    if isinstance(sequences, str):
        raise SequenceCollectionError(
            "Expected an iterable of sequences, not a single string."
        )

    try:
        values = list(sequences)
    except TypeError as exc:
        raise SequenceCollectionError(
            "Expected an iterable of sequence values."
        ) from exc

    return values


def inspect_sequences(
    sequences: Sequence[object] | Iterable[object],
    *,
    cleaning_config: Optional[SequenceCleaningConfig] = None,
    invalid_policy: InvalidResiduePolicy = "strict",
    allowed: Iterable[str] = AA20,
    allow_empty: bool = False,
    min_length: Optional[int] = None,
    remove_internal_whitespace: bool = True,
    remove_terminal_stop: bool = True,
) -> list[SequenceValidationResult]:
    """Inspect a sequence collection and return one report per item."""
    values = validate_sequences_input(sequences)
    return [
        inspect_sequence(
            sequence,
            cleaning_config=cleaning_config,
            invalid_policy=invalid_policy,
            allowed=allowed,
            allow_empty=allow_empty,
            min_length=min_length,
            remove_internal_whitespace=remove_internal_whitespace,
            remove_terminal_stop=remove_terminal_stop,
        )
        for sequence in values
    ]


def validate_sequence_collection(
    sequences: Sequence[object] | Iterable[object],
    *,
    cleaning_config: Optional[SequenceCleaningConfig] = None,
    invalid_policy: InvalidResiduePolicy = "strict",
    allowed: Iterable[str] = AA20,
    allow_empty: bool = False,
    min_length: Optional[int] = None,
    remove_internal_whitespace: bool = True,
    remove_terminal_stop: bool = True,
) -> list[SequenceValidationResult]:
    """Validate a collection and return structured reports."""
    return inspect_sequences(
        sequences,
        cleaning_config=cleaning_config,
        invalid_policy=invalid_policy,
        allowed=allowed,
        allow_empty=allow_empty,
        min_length=min_length,
        remove_internal_whitespace=remove_internal_whitespace,
        remove_terminal_stop=remove_terminal_stop,
    )


def cleaned_valid_sequences(
    sequences: Sequence[object] | Iterable[object],
    *,
    cleaning_config: Optional[SequenceCleaningConfig] = None,
    invalid_policy: InvalidResiduePolicy = "strict",
    allowed: Iterable[str] = AA20,
    allow_empty: bool = False,
    min_length: Optional[int] = None,
    remove_internal_whitespace: bool = True,
    remove_terminal_stop: bool = True,
) -> list[str]:
    """Return cleaned sequences or raise on the first invalid item."""
    values = validate_sequences_input(sequences)
    return [
        assert_valid_sequence(
            sequence,
            cleaning_config=cleaning_config,
            invalid_policy=invalid_policy,
            allowed=allowed,
            allow_empty=allow_empty,
            min_length=min_length,
            remove_internal_whitespace=remove_internal_whitespace,
            remove_terminal_stop=remove_terminal_stop,
        )
        for sequence in values
    ]


def is_valid_protein_sequence(
    sequence: object,
    *,
    cleaning_config: Optional[SequenceCleaningConfig] = None,
    invalid_policy: InvalidResiduePolicy = "strict",
    allowed: Iterable[str] = AA20,
    allow_empty: bool = False,
    min_length: Optional[int] = None,
    remove_internal_whitespace: bool = True,
    remove_terminal_stop: bool = True,
) -> bool:
    """Return whether one sequence passes validation."""
    result = inspect_sequence(
        sequence,
        cleaning_config=cleaning_config,
        invalid_policy=invalid_policy,
        allowed=allowed,
        allow_empty=allow_empty,
        min_length=min_length,
        remove_internal_whitespace=remove_internal_whitespace,
        remove_terminal_stop=remove_terminal_stop,
    )
    return result.is_valid


__all__ = [
    "SequenceValidationError",
    "SequenceValidationResult",
    "assert_valid_sequence",
    "cleaned_valid_sequences",
    "find_invalid_residues",
    "inspect_sequence",
    "inspect_sequences",
    "is_valid_protein_sequence",
    "validate_sequence",
    "validate_sequence_collection",
    "validate_sequences_input",
]
