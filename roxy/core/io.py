"""Sequence and feature-table I/O utilities for Roxy."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from .exceptions import RoxyIOError

_FASTA_EXTENSIONS = {".fasta", ".fa", ".faa"}
_CSV_EXTENSIONS = {".csv"}
_PARQUET_EXTENSIONS = {".parquet", ".pq"}


def read_fasta(path: Path | str) -> list[tuple[str, str]]:
    """Parse a FASTA file into (header, sequence) pairs.

    Raises
    ------
    RoxyIOError
        If the file does not exist or cannot be parsed.

    """
    path = Path(path)
    if not path.exists():
        msg = f"FASTA file not found: {path}"
        raise RoxyIOError(msg)

    records: list[tuple[str, str]] = []
    header = ""
    seq_parts: list[str] = []

    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()  # noqa: PLW2901
            if not line:
                continue
            if line.startswith(">"):
                if header:
                    records.append((header, "".join(seq_parts)))
                header = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line.upper())
        if header:
            records.append((header, "".join(seq_parts)))
    except OSError as exc:
        msg = f"Could not read FASTA file {path}: {exc}"
        raise RoxyIOError(msg) from exc

    return records


def read_csv(path: Path | str, seq_col: str | None = None) -> pd.DataFrame:
    """Load a CSV into a DataFrame.

    Raises
    ------
    RoxyIOError
        If the file does not exist or ``seq_col`` is missing.

    """
    import pandas as pd  # noqa: PLC0415

    path = Path(path)
    if not path.exists():
        msg = f"CSV file not found: {path}"
        raise RoxyIOError(msg)

    df = pd.read_csv(path)
    if seq_col is not None and seq_col not in df.columns:
        msg = f"Column {seq_col!r} not found. Available: {list(df.columns)}"
        raise RoxyIOError(msg)
    return df


def read_parquet(path: Path | str, seq_col: str | None = None) -> pd.DataFrame:
    """Load a Parquet file into a DataFrame.

    Raises
    ------
    RoxyIOError
        If the file does not exist, the extra is not installed, or ``seq_col`` is missing.

    """
    import pandas as pd  # noqa: PLC0415

    path = Path(path)
    if not path.exists():
        msg = f"Parquet file not found: {path}"
        raise RoxyIOError(msg)

    try:
        df = pd.read_parquet(path)
    except ImportError as exc:
        msg = (
            "Reading Parquet requires the 'parquet' extra: "
            "pip install roxy[parquet]"
        )
        raise RoxyIOError(msg) from exc

    if seq_col is not None and seq_col not in df.columns:
        msg = f"Column {seq_col!r} not found. Available: {list(df.columns)}"
        raise RoxyIOError(msg)
    return df


def read_sequences(
    path: Path | str,
    seq_col: str | None = None,
) -> list[tuple[str, str]]:
    """Dispatch-by-extension loader returning (id, sequence) pairs.

    Accepts FASTA, CSV and Parquet. For CSV/Parquet, ``seq_col`` must be
    provided unless the file has a column named ``"sequence"``.

    Raises
    ------
    RoxyIOError
        If the extension is unsupported, the file is missing, or ``seq_col``
        is absent.

    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext in _FASTA_EXTENSIONS:
        return read_fasta(path)

    _col = seq_col or "sequence"

    if ext in _CSV_EXTENSIONS:
        df = read_csv(path, seq_col=_col)
    elif ext in _PARQUET_EXTENSIONS:
        df = read_parquet(path, seq_col=_col)
    else:
        msg = (
            f"Unsupported file extension {ext!r}. "
            f"Supported: {sorted(_FASTA_EXTENSIONS | _CSV_EXTENSIONS | _PARQUET_EXTENSIONS)}"
        )
        raise RoxyIOError(msg)

    # Use index as id if available, else positional
    ids = [str(i) for i in df.index]
    seqs = df[_col].astype(str).str.strip().str.upper().tolist()
    return list(zip(ids, seqs, strict=True))


def write_table(df: pd.DataFrame, path: Path | str) -> None:
    """Write a DataFrame to CSV or Parquet (dispatched by extension).

    Raises
    ------
    RoxyIOError
        If the extension is unsupported or write fails.

    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()

    if ext in _CSV_EXTENSIONS:
        df.to_csv(path, index=True)
    elif ext in _PARQUET_EXTENSIONS:
        try:
            df.to_parquet(path)
        except ImportError as exc:
            msg = "Writing Parquet requires the 'parquet' extra: pip install roxy[parquet]"
            raise RoxyIOError(msg) from exc
    else:
        msg = f"Unsupported output extension {ext!r}. Use .csv or .parquet."
        raise RoxyIOError(msg)


__all__ = [
    "read_csv",
    "read_fasta",
    "read_parquet",
    "read_sequences",
    "write_table",
]
