"""Sequence input and output helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from roxy.core.exceptions import SequenceCollectionError
from roxy.core.optional_deps import require_pandas

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    import pandas as pd

FastaRecord = tuple[str, str]


def read_fasta_records(fasta_path: str) -> list[FastaRecord]:
    """Read FASTA records as ``(id, sequence)`` pairs."""
    records: list[FastaRecord] = []
    current_id: Optional[str] = None
    current_sequence: list[str] = []

    with open(fasta_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records.append((current_id, "".join(current_sequence)))
                current_id = line[1:].strip()
                current_sequence = []
                continue
            current_sequence.append(line)

    if current_id is not None:
        records.append((current_id, "".join(current_sequence)))

    return records


def coerce_sequence_series(
    data: object,
    *,
    sequence_column: str = "sequence",
) -> tuple["pd.Series", Optional["pd.Index"]]:
    """Normalize API input into a sequence Series plus an optional index."""
    pd = require_pandas(purpose="sequence API DataFrame outputs")

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
        return data[sequence_column], data.index

    try:
        values = list(data)
    except TypeError as exc:
        raise SequenceCollectionError(
            "Expected an iterable of sequence values or a pandas DataFrame."
        ) from exc

    return pd.Series(values, dtype="object"), None


__all__ = ["FastaRecord", "coerce_sequence_series", "read_fasta_records"]
