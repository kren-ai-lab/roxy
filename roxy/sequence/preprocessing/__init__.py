"""Preprocessing entrypoints for sequence input handling.

This package hosts the canonical sequence cleaning and validation logic
used before descriptor computation.
"""

from roxy.sequence.preprocessing.cleaning import (
    InvalidResiduePolicy,
    SequenceCleaningConfig,
    clean_sequence,
    clean_sequences,
    normalize_sequence,
)
from roxy.sequence.preprocessing.validation import (
    SequenceValidationResult,
    validate_sequence,
    validate_sequences,
)

__all__ = [
    "InvalidResiduePolicy",
    "SequenceCleaningConfig",
    "SequenceValidationResult",
    "clean_sequence",
    "clean_sequences",
    "normalize_sequence",
    "validate_sequence",
    "validate_sequences",
]
