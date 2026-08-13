"""Roxy — protein sequence descriptors for machine learning."""

from __future__ import annotations

__version__ = "0.2.0"

from .core import (
    RoxyError,
    RoxyIOError,
    read_fasta,
    read_sequences,
    write_table,
)
from .descriptors import DESCRIPTOR_REGISTRY, BaseDescriptor, register

__all__ = [
    "DESCRIPTOR_REGISTRY",
    "BaseDescriptor",
    "RoxyError",
    "RoxyIOError",
    "__version__",
    "read_fasta",
    "read_sequences",
    "register",
    "write_table",
]
