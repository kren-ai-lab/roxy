"""Sequence-focused public surface for Roxy.

This package exposes the conservative sequence-first API and keeps the
modular descriptor-family implementation under ``roxy.sequence.*``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from .api import (
        describe_fasta,
        describe_sequences,
        list_available_descriptors,
        validate_sequences,
    )


def __getattr__(name: str) -> Any:
    """Resolve the public sequence API lazily."""
    if name in {
        "describe_fasta",
        "describe_sequences",
        "list_available_descriptors",
        "validate_sequences",
    }:
        from .api import (
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
    "describe_fasta",
    "describe_sequences",
    "list_available_descriptors",
    "validate_sequences",
]
