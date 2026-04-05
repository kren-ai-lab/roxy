"""Public API entrypoints for sequence descriptor workflows."""

from .main import (
    describe_fasta,
    describe_sequences,
    list_available_descriptors,
    validate_sequences,
)

__all__ = [
    "describe_fasta",
    "describe_sequences",
    "list_available_descriptors",
    "validate_sequences",
]
