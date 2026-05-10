"""Tests for roxy.core.io — FASTA / CSV / Parquet round-trips."""

from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl
import pytest

from roxy.core.exceptions import RoxyIOError
from roxy.core.io import (
    read_fasta,
    read_sequences,
    write_table,
)

FASTA_CONTENT = """>seq1
ACDEFGHIKLMNPQRSTVWY
>seq2
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL
"""

CSV_CONTENT = """id,sequence,label
s1,ACDEF,1
s2,GHIKL,0
"""


def _write_tmp(suffix: str, content: str) -> Path:
    with tempfile.NamedTemporaryFile(suffix=suffix, mode="w", delete=False) as f:
        f.write(content)
        return Path(f.name)


def test_read_fasta_basic():
    path = _write_tmp(".fasta", FASTA_CONTENT)
    records = read_fasta(path)
    assert len(records) == 2
    assert records[0][0] == "seq1"
    assert records[0][1] == "ACDEFGHIKLMNPQRSTVWY"
    path.unlink()


def test_read_fasta_missing_raises():
    with pytest.raises(RoxyIOError):
        read_fasta(Path("/nonexistent/file.fasta"))


def test_read_sequences_fasta():
    path = _write_tmp(".fasta", FASTA_CONTENT)
    pairs = read_sequences(path)
    assert pairs[0] == ("seq1", "ACDEFGHIKLMNPQRSTVWY")
    path.unlink()


def test_read_sequences_csv():
    path = _write_tmp(".csv", CSV_CONTENT)
    pairs = read_sequences(path, seq_col="sequence")
    assert len(pairs) == 2
    assert pairs[0][1] == "ACDEF"
    path.unlink()


def test_read_sequences_bad_ext():
    path = _write_tmp(".xyz", "")
    with pytest.raises(RoxyIOError):
        read_sequences(path)
    path.unlink()


def test_write_table_csv(tmp_path):
    df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    out = tmp_path / "out.csv"
    write_table(df, out)
    loaded = pl.read_csv(out)
    assert loaded.columns == ["a", "b"]


def test_write_table_bad_ext(tmp_path):
    df = pl.DataFrame({"a": [1]})
    with pytest.raises(RoxyIOError):
        write_table(df, tmp_path / "out.xyz")
