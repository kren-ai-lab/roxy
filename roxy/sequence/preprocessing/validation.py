"""Sequence validation helpers.

This module provides a small validation layer for protein sequence
inputs. It is designed to stay independent from any one descriptor
family while remaining practical for composition, grouped, k-mer,
terminal, CTD, pattern, and order-related workflows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence

from roxy.core.constants import AA20
from roxy.core.exceptions import (
    EmptySequenceError,
    InvalidSequenceError,
    SequenceCollectionError,
    SequenceInputError,
)

from .cleaning import (
    InvalidResiduePolicy,
    SequenceCleaningConfig,
    clean_sequence,
    find_noncanonical_residues,
    is_missing_sequence_value,
    normalize_sequence,
    remove_terminal_stop_marker,
)

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    import pandas as pd


@dataclass(frozen=True)
class SequenceValidationResult:
    """Structured validation result for a single sequence input."""

    original: object
    normalized: Optional[str]
    cleaned: Optional[str]
    is_missing: bool
    is_empty: bool
    became_empty_after_cleaning: bool
    invalid_residues: List[str] = field(default_factory=list)
    length: int = 0
    min_length: Optional[int] = None
    is_too_short: bool = False
    is_valid: bool = False
    errors: List[str] = field(default_factory=list)


class SequenceValidationError(InvalidSequenceError):
    """Raised when strict validation assertions fail."""


def _require_pandas() -> "type[pd]":
    """Import pandas lazily for table-oriented validation helpers."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "pandas is required for DataFrame-based sequence validation."
        ) from exc
    return pd


def _normalize_for_validation(
    sequence: object,
    *,
    strip_internal_whitespace: bool = True,
    remove_terminal_stop: bool = True,
) -> Optional[str]:
    """Normalize a value for validation without dropping invalid residues."""
    normalized = normalize_sequence(
        sequence,
        uppercase=True,
        strip_external_whitespace=True,
        remove_internal_whitespace=strip_internal_whitespace,
    )
    if normalized is None:
        return None
    if remove_terminal_stop:
        normalized = remove_terminal_stop_marker(normalized)
    return normalized


def find_invalid_residues(
    sequence: object,
    *,
    allowed: Iterable[str] = AA20,
    strip_internal_whitespace: bool = True,
    remove_terminal_stop: bool = True,
) -> List[str]:
    """Return sorted non-canonical residue symbols present in a sequence."""
    normalized = _normalize_for_validation(
        sequence,
        strip_internal_whitespace=strip_internal_whitespace,
        remove_terminal_stop=remove_terminal_stop,
    )
    if normalized is None:
        return []
    return find_noncanonical_residues(
        normalized,
        allowed_residues=allowed,
    )


def validate_sequence(
    sequence: object,
    *,
    cleaning_config: Optional[SequenceCleaningConfig] = None,
    invalid_policy: InvalidResiduePolicy = "strict",
    allowed: Iterable[str] = AA20,
    allow_empty: bool = False,
    min_length: Optional[int] = None,
) -> SequenceValidationResult:
    """Validate a single sequence value.

    Validation uses the shared cleaning rules for normalization so
    descriptor families can rely on a consistent preprocessing policy.
    """
    if min_length is not None and min_length < 0:
        raise SequenceInputError(
            "`min_length` must be non-negative when provided."
        )

    if cleaning_config is None:
        cleaning_config = SequenceCleaningConfig(
            invalid_policy=invalid_policy,
            allowed_residues=frozenset(allowed),
        )
    else:
        invalid_policy = cleaning_config.invalid_policy
        allowed = cleaning_config.allowed_residues

    raw_normalized = normalize_sequence(
        sequence,
        uppercase=cleaning_config.uppercase,
        strip_external_whitespace=cleaning_config.strip_external_whitespace,
        remove_internal_whitespace=cleaning_config.remove_internal_whitespace,
    )
    is_missing = is_missing_sequence_value(sequence)

    if is_missing:
        errors = ["Sequence value is missing."]
        return SequenceValidationResult(
            original=sequence,
            normalized=None,
            cleaned=None,
            is_missing=True,
            is_empty=False,
            became_empty_after_cleaning=False,
            invalid_residues=[],
            length=0,
            min_length=min_length,
            is_too_short=False,
            is_valid=False,
            errors=errors,
        )

    normalized = raw_normalized
    if normalized is not None and cleaning_config.remove_terminal_stop:
        normalized = remove_terminal_stop_marker(normalized)

    invalid_residues = find_invalid_residues(
        sequence,
        allowed=allowed,
        strip_internal_whitespace=cleaning_config.remove_internal_whitespace,
        remove_terminal_stop=cleaning_config.remove_terminal_stop,
    )

    try:
        cleaned = clean_sequence(
            sequence,
            config=cleaning_config,
        )
    except InvalidSequenceError:
        cleaned = normalized

    cleaned = "" if cleaned is None else cleaned
    is_empty = cleaned == ""
    became_empty_after_cleaning = bool(raw_normalized) and is_empty
    length = len(cleaned)
    is_too_short = min_length is not None and length < min_length

    errors: List[str] = []
    if is_empty and not allow_empty:
        if became_empty_after_cleaning:
            errors.append("Sequence became empty after cleaning.")
        else:
            errors.append("Sequence is empty after normalization.")
    if invalid_policy == "strict" and invalid_residues:
        invalid_text = ", ".join(invalid_residues)
        errors.append(
            "Sequence contains non-canonical residues: "
            f"{invalid_text}."
        )
    if is_too_short:
        errors.append(
            f"Sequence length {length} is shorter than the required minimum "
            f"length {min_length}."
        )

    return SequenceValidationResult(
        original=sequence,
        normalized=normalized,
        cleaned=cleaned,
        is_missing=False,
        is_empty=is_empty,
        became_empty_after_cleaning=became_empty_after_cleaning,
        invalid_residues=invalid_residues,
        length=length,
        min_length=min_length,
        is_too_short=is_too_short,
        is_valid=not errors,
        errors=errors,
    )


def is_valid_protein_sequence(
    sequence: object,
    *,
    allowed: Iterable[str] = AA20,
    allow_empty: bool = False,
    min_length: Optional[int] = None,
    invalid_policy: InvalidResiduePolicy = "strict",
) -> bool:
    """Return whether a sequence passes the default protein checks."""
    result = validate_sequence(
        sequence,
        allowed=allowed,
        allow_empty=allow_empty,
        min_length=min_length,
        invalid_policy=invalid_policy,
    )
    return result.is_valid


def validate_sequence_collection(
    sequences: Sequence[object] | Iterable[object],
    *,
    cleaning_config: Optional[SequenceCleaningConfig] = None,
    invalid_policy: InvalidResiduePolicy = "strict",
    allowed: Iterable[str] = AA20,
    allow_empty: bool = False,
    min_length: Optional[int] = None,
) -> List[SequenceValidationResult]:
    """Validate an iterable of sequence values and return structured results."""
    if isinstance(sequences, str):
        raise SequenceCollectionError(
            "Expected an iterable of sequence values, not a single string. "
            "Use `validate_sequence(...)` for single-sequence checks."
        )

    try:
        values = list(sequences)
    except TypeError as exc:
        raise SequenceCollectionError(
            "Expected an iterable of sequence values."
        ) from exc

    return [
        validate_sequence(
            sequence,
            cleaning_config=cleaning_config,
            invalid_policy=invalid_policy,
            allowed=allowed,
            allow_empty=allow_empty,
            min_length=min_length,
        )
        for sequence in values
    ]


def assert_valid_sequence(
    sequence: object,
    *,
    cleaning_config: Optional[SequenceCleaningConfig] = None,
    allowed: Iterable[str] = AA20,
    allow_empty: bool = False,
    min_length: Optional[int] = None,
    invalid_policy: InvalidResiduePolicy = "strict",
) -> str:
    """Return the cleaned sequence or raise a validation error."""
    result = validate_sequence(
        sequence,
        cleaning_config=cleaning_config,
        allowed=allowed,
        allow_empty=allow_empty,
        min_length=min_length,
        invalid_policy=invalid_policy,
    )
    if not result.is_valid:
        if result.is_missing:
            raise SequenceInputError("Sequence value is missing.")
        if result.is_empty:
            if result.became_empty_after_cleaning:
                raise EmptySequenceError("Sequence became empty after cleaning.")
            raise EmptySequenceError("Sequence is empty after normalization.")
        if result.invalid_residues and invalid_policy == "strict":
            invalid_text = ", ".join(result.invalid_residues)
            raise InvalidSequenceError(
                f"Sequence contains non-canonical residues: {invalid_text}."
            )
        raise SequenceValidationError(" ".join(result.errors))
    return result.cleaned or ""


def _coerce_sequence_series(
    data: object,
    *,
    sequence_column: str = "sequence",
) -> tuple["pd.Series", Optional["pd.Index"]]:
    """Normalize collection input into a pandas Series plus optional index."""
    pd = _require_pandas()

    if isinstance(data, str):
        raise SequenceCollectionError(
            "Expected an iterable of sequences or a DataFrame, not a single "
            "string. Use `validate_sequence(...)` for single-sequence checks."
        )

    if isinstance(data, pd.DataFrame):
        if sequence_column not in data.columns:
            raise SequenceCollectionError(
                f"Sequence column {sequence_column!r} not found in input DataFrame."
            )
        return data[sequence_column], data.index

    try:
        values = list(data)  # type: ignore[arg-type]
    except TypeError as exc:
        raise SequenceCollectionError(
            "Expected an iterable of sequence values or a pandas DataFrame."
        ) from exc

    return pd.Series(values, dtype="object"), None


def validate_sequences(
    data: object,
    *,
    sequence_column: str = "sequence",
    cleaning_config: Optional[SequenceCleaningConfig] = None,
    invalid_policy: InvalidResiduePolicy = "strict",
    allowed: Iterable[str] = AA20,
    allow_empty: bool = False,
    min_length: Optional[int] = None,
    check_duplicates: bool = False,
    require_unique: bool = False,
) -> "pd.DataFrame":
    """Validate a collection of sequences and return a tabular report."""
    pd = _require_pandas()
    series, original_index = _coerce_sequence_series(
        data,
        sequence_column=sequence_column,
    )

    results = validate_sequence_collection(
        series.tolist(),
        cleaning_config=cleaning_config,
        invalid_policy=invalid_policy,
        allowed=allowed,
        allow_empty=allow_empty,
        min_length=min_length,
    )

    rows = []
    for result in results:
        row = asdict(result)
        row["invalid_count"] = len(result.invalid_residues)
        row["error_count"] = len(result.errors)
        rows.append(row)

    df = pd.DataFrame(rows)
    if original_index is not None:
        df.index = original_index

    if check_duplicates or require_unique:
        cleaned_series = df["cleaned"].fillna("<missing>")
        duplicate_counts = cleaned_series.map(cleaned_series.value_counts())
        df["duplicate_count"] = duplicate_counts
        df["is_duplicate"] = duplicate_counts > 1

        if require_unique:
            duplicate_mask = df["is_duplicate"] & df["cleaned"].notna()
            df.loc[duplicate_mask, "errors"] = df.loc[
                duplicate_mask, "errors"
            ].apply(lambda errors: list(errors) + ["Sequence is duplicated."])
            df.loc[duplicate_mask, "error_count"] = df.loc[
                duplicate_mask, "errors"
            ].apply(len)
            df.loc[duplicate_mask, "is_valid"] = False

    return df


__all__ = [
    "SequenceValidationError",
    "SequenceValidationResult",
    "assert_valid_sequence",
    "find_invalid_residues",
    "is_valid_protein_sequence",
    "validate_sequence_collection",
    "validate_sequence",
    "validate_sequences",
]
