"""Integration tests for ``roxy compute``."""

from __future__ import annotations

import polars as pl

from roxy.cli.main import app

from ._helpers import FASTA, runner


def test_compute_no_selector(tmp_path):
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text(FASTA)
    result = runner.invoke(app, ["compute", str(fasta), "-o", str(tmp_path / "out.csv")])
    assert result.exit_code == 1
    assert "Specify" in result.output or "Specify" in (result.stderr or "")


def test_compute_config_and_all_exclusive(tmp_path):
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text(FASTA)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("aac:\n")
    result = runner.invoke(
        app, ["compute", str(fasta), "--config", str(cfg), "--all", "-o", str(tmp_path / "out.csv")]
    )
    assert result.exit_code == 1


def test_compute_invalid_descriptor(tmp_path):
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text(FASTA)
    result = runner.invoke(
        app, ["compute", str(fasta), "-d", "not_a_real_descriptor", "-o", str(tmp_path / "out.csv")]
    )
    assert result.exit_code == 1


def test_compute_happy_path_csv(tmp_path):
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text(FASTA)
    out = tmp_path / "out.csv"
    result = runner.invoke(
        app, ["compute", str(fasta), "-d", "aac", "-o", str(out), "--no-progress"]
    )
    assert result.exit_code == 0, result.output
    df = pl.read_csv(out)
    assert df.shape[0] == 2
    assert "length" in df.columns
    assert "aac_freq_A" in df.columns


def test_compute_with_config(tmp_path):
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text(FASTA)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("aac:\n  include_counts: false\ncharge:\n")
    out = tmp_path / "out.parquet"
    result = runner.invoke(
        app, ["compute", str(fasta), "--config", str(cfg), "-o", str(out), "--no-progress"]
    )
    assert result.exit_code == 0, result.output
    df = pl.read_parquet(out)
    assert df.shape[0] == 2
    assert "aac_count_A" not in df.columns
    assert "charge_positive_fraction" in df.columns
