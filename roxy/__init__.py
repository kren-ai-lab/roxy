"""Public API for sequence descriptor workflows."""

from roxy.sequence.api import (
    describe_fasta,
    describe_sequences,
    list_available_descriptors,
    validate_sequences,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "describe_fasta",
    "describe_sequences",
    "list_available_descriptors",
    "validate_sequences",
]
