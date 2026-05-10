"""Shared CLI utilities for Roxy."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from pathlib import Path

HELP_CONTEXT_SETTINGS: dict = {"help_option_names": ["-h", "--help"]}

EXPORT_CHOICES = ("csv", "parquet")

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def level_from_str(name: str) -> int:
    """Map a string log-level name to the stdlib logging constant."""
    return getattr(logging, (name or "INFO").upper(), logging.INFO)


def validate_choice(value: str, choices: tuple[str, ...], opt: str) -> str:
    """Validate a CLI option against a list of case-insensitive choices."""
    normalized = (value or "").strip().lower()
    allowed = {c.lower(): c for c in choices}
    if normalized not in allowed:
        msg = f"Invalid {opt}: {value!r}. Allowed: {', '.join(choices)}"
        raise typer.BadParameter(msg)
    return allowed[normalized]


def ensure_ext(path: Path, fmt: str) -> Path:
    """Append the requested extension only when the user omitted one."""
    fmt = fmt.lower().lstrip(".")
    return path if path.suffix else path.with_suffix(f".{fmt}")


def load_sequences(
    path: Path,
    seq_col: str = "sequence",
) -> list[tuple[str, str]]:
    """Load sequences from FASTA, CSV or Parquet; return (id, seq) pairs."""
    from roxy.core.io import read_sequences  # noqa: PLC0415

    return read_sequences(path, seq_col=seq_col)
