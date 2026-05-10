"""Roxy — protein sequence descriptors for machine learning."""

from __future__ import annotations

__version__ = "0.2.0"

from .core import (
    AAIndexError,
    DescriptorError,
    RoxyError,
    RoxyIOError,
    SequenceValidationError,
    get_cache_root,
    read_fasta,
    read_sequences,
    set_cache_root,
    temporary_cache_root,
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
    "get_cache_root",
    "read_fasta",
    "read_sequences",
    "register",
    "set_cache_root",
    "temporary_cache_root",
    "write_table",
]
