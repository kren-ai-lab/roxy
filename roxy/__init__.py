"""Roxy — protein sequence descriptors for machine learning."""

from __future__ import annotations

__version__ = "0.3.0"

from .core import (
    RoxyError,
    RoxyIOError,
    read_fasta,
    read_sequences,
    write_table,
)
from .descriptors import DESCRIPTOR_REGISTRY, BaseDescriptor, register

from .cli._config_file import ConfigIssue  # isort: skip
from .cli.validate import validate_config  # isort: skip

__all__ = [
    "DESCRIPTOR_REGISTRY",
    "BaseDescriptor",
    "ConfigIssue",
    "RoxyError",
    "RoxyIOError",
    "__version__",
    "read_fasta",
    "read_sequences",
    "register",
    "validate_config",
    "write_table",
]
