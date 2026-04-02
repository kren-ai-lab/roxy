"""Roxy.

Roxy is a Python package for modular extraction of protein sequence
descriptors.

The package-level namespace is intentionally small and exposes only the
main sequence-facing entry points.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

from . import sequence

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from roxy.sequence.api import (
        describe_fasta,
        describe_sequences,
        list_available_descriptors,
        validate_sequences,
    )


try:
    __version__ = version("roxy")
except PackageNotFoundError:  # pragma: no cover - local source tree fallback
    __version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    """Resolve the small public API lazily from ``roxy.sequence.api``."""
    if name in {
        "describe_fasta",
        "describe_sequences",
        "list_available_descriptors",
        "validate_sequences",
    }:
        from roxy.sequence.api import (
            describe_fasta,
            describe_sequences,
            list_available_descriptors,
            validate_sequences,
        )

        return {
            "describe_fasta": describe_fasta,
            "describe_sequences": describe_sequences,
            "list_available_descriptors": list_available_descriptors,
            "validate_sequences": validate_sequences,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "describe_fasta",
    "describe_sequences",
    "list_available_descriptors",
    "sequence",
    "validate_sequences",
]
