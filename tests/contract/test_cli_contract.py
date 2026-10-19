"""Shared ecosystem CLI contract: record ids and exit codes."""

from __future__ import annotations

import sys

import polars as pl
import pytest

from roxy.cli.main import app, run
from tests.cli._helpers import runner


def _compute(inp, out, *extra):
    return runner.invoke(app, ["compute", str(inp), "-d", "aac", "-o", str(out), "--no-progress", *extra])


def test_fasta_id_is_first_token(tmp_path):
    inp = tmp_path / "s.fasta"
    inp.write_text(">p1 some description\nACD\n>p2\nWYY\n")
    out = tmp_path / "o.csv"
    assert _compute(inp, out).exit_code == 0
    assert pl.read_csv(out)["id"].to_list() == ["p1", "p2"]


def test_table_without_id_column_exits_2(tmp_path):
    inp = tmp_path / "s.csv"
    pl.DataFrame({"name": ["a", "b"], "sequence": ["ACD", "WYY"]}).write_csv(inp)
    out = tmp_path / "o.csv"
    result = _compute(inp, out)
    assert result.exit_code == 2
    assert not out.exists()
    assert _compute(inp, out, "--id-col", "name").exit_code == 0


def test_duplicate_ids_exit_2(tmp_path):
    inp = tmp_path / "s.fasta"
    inp.write_text(">a x\nACD\n>a y\nWYY\n")
    assert _compute(inp, tmp_path / "o.csv").exit_code == 2


def _exit_code(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["roxy", *argv])
    with pytest.raises(SystemExit) as exc:
        run()
    return exc.value.code


def test_runtime_error_exits_3(monkeypatch, tmp_path):
    missing = tmp_path / "missing.fasta"
    assert _exit_code(monkeypatch, "compute", str(missing), "-d", "aac", "-o", str(tmp_path / "o.csv")) == 3


def test_usage_error_exits_2(monkeypatch):
    assert _exit_code(monkeypatch, "compute") == 2


def test_internal_error_exits_4(monkeypatch, tmp_path):
    inp = tmp_path / "s.fasta"
    inp.write_text(">a\nACD\n")

    def boom(*_a, **_k):
        raise KeyError

    monkeypatch.setattr("roxy.cli.compute._run_descriptors", boom)
    assert _exit_code(monkeypatch, "compute", str(inp), "-d", "aac", "-o", str(tmp_path / "o.csv")) == 4
