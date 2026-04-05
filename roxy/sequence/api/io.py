"""Small IO and input-coercion helpers for the sequence API."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

from roxy.core.exceptions import SequenceCollectionError

from .types import FastaRecord, SequenceInput

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    import pandas as pd


def require_pandas() -> "type[pd]":
    """Import pandas lazily for DataFrame-based API operations."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "pandas is required for the sequence API DataFrame outputs."
        ) from exc
    return pd


def coerce_sequence_series(
    data: SequenceInput,
    *,
    sequence_column: str = "sequence",
) -> Tuple["pd.Series", Optional["pd.Index"]]:
    """Normalize API input into a sequence Series plus an optional index."""
    pd = require_pandas()

    if isinstance(data, str):
        raise SequenceCollectionError(
            "Expected an iterable of sequences or a DataFrame, not a single "
            "string. Use sequence-level validation helpers for one sequence."
        )

    if isinstance(data, pd.DataFrame):
        if sequence_column not in data.columns:
            raise SequenceCollectionError(
                f"Sequence column {sequence_column!r} not found in input DataFrame."
            )
        series = data[sequence_column]
        return series, data.index

    try:
        values = list(data)
    except TypeError as exc:
        raise SequenceCollectionError(
            "Expected an iterable of sequence values or a pandas DataFrame."
        ) from exc

    return pd.Series(values, dtype="object"), None


def read_fasta_records(fasta_path: str) -> List[FastaRecord]:
    """Read FASTA records as ``(id, sequence)`` pairs."""
    records: List[FastaRecord] = []
    current_id: Optional[str] = None
    current_seq: List[str] = []

    with open(fasta_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records.append((current_id, "".join(current_seq)))
                current_id = line[1:].strip()
                current_seq = []
                continue
            current_seq.append(line)

    if current_id is not None:
        records.append((current_id, "".join(current_seq)))

    return records


# TODO:
# - Consider consolidating collection coercion with
#   `roxy.sequence.preprocessing.validation` once the shared table-input
#   contract is stable across API and validation layers.

__all__ = [
    "coerce_sequence_series",
    "read_fasta_records",
    "require_pandas",
]
