"""Roxy — protein sequence descriptors for machine learning."""

from __future__ import annotations

__version__ = "0.2.0"

from .core import (
    AAIndexError,
    DescriptorError,
    RoxyError,
    RoxyIOError,
    SequenceValidationError,
    read_fasta,
    read_sequences,
    write_table,
)
from .descriptors import DESCRIPTOR_REGISTRY, BaseDescriptor, register

__all__ = [
    "DESCRIPTOR_REGISTRY",
    "AAIndexError",
    "BaseDescriptor",
    "DescriptorError",
    "RoxyError",
    "RoxyIOError",
    "SequenceValidationError",
    "__version__",
    "read_fasta",
    "read_sequences",
    "register",
    "write_table",
]
