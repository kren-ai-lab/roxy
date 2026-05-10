"""Tests for roxy compute, init-config, and supporting functions."""

from __future__ import annotations

import polars as pl
import pytest
import typer
from typer.testing import CliRunner

from roxy.cli._utils import _resolve_names
from roxy.cli.compute import _concat_frames, _load_config
from roxy.cli.main import app
from roxy.descriptors import DESCRIPTOR_REGISTRY

runner = CliRunner()

_FASTA = ">seq1\nACDEFGHIKLMNPQRSTVWY\n>seq2\nAAAACCCC\n"

# ---------------------------------------------------------------------------
# _resolve_names
# ---------------------------------------------------------------------------


def test_resolve_names_all():
    result = _resolve_names(True, [], [])
    assert result == sorted(DESCRIPTOR_REGISTRY)


def test_resolve_names_descriptors():
    result = _resolve_names(False, ["charge", "aac"], [])
    assert result == ["aac", "charge"]


def test_resolve_names_family():
    result = _resolve_names(False, [], ["composition"])
    composition = sorted(n for n, c in DESCRIPTOR_REGISTRY.items() if c.family == "composition")
    assert result == composition


def test_resolve_names_dedup():
    # aac appears via both -d and -f composition
    result = _resolve_names(False, ["aac"], ["composition"])
    assert result.count("aac") == 1


# ---------------------------------------------------------------------------
# _concat_frames
# ---------------------------------------------------------------------------


def _make_frame(name: str, n: int = 2) -> pl.DataFrame:
    desc = DESCRIPTOR_REGISTRY[name]()
    seqs = ["ACDEFGHIKLMNPQRSTVWY"] * n
    ids = [f"s{i}" for i in range(n)]
    return desc.compute(seqs, ids=ids)


def test_concat_single_frame_renames_base_cols():
    df = _make_frame("aac")
    result = _concat_frames([df], ["aac"])
    assert "length" in result.columns
    assert "valid_residue_count" in result.columns
    assert "aac_length" not in result.columns


def test_concat_two_frames_no_duplicate_base():
    df1 = _make_frame("aac")
    df2 = _make_frame("charge")
    result = _concat_frames([df1, df2], ["aac", "charge"])
    assert result.columns.count("length") == 1
    assert result.columns.count("id") == 1
    assert "charge_length" not in result.columns
    assert "charge_positive_fraction" in result.columns


def test_concat_three_frames_no_duplicate_base():
    df1 = _make_frame("aac")
    df2 = _make_frame("charge")
    df3 = _make_frame("dpc")
    result = _concat_frames([df1, df2, df3], ["aac", "charge", "dpc"])
    assert result.columns.count("length") == 1
    assert result.columns.count("id") == 1
    assert "charge_length" not in result.columns
    assert "dpc_length" not in result.columns
    assert "dpc_freq_AA" in result.columns


# ---------------------------------------------------------------------------
# _load_config
# ---------------------------------------------------------------------------


def test_load_config_valid(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("aac:\n  include_counts: true\n")
    result = _load_config(cfg)
    assert result == {"aac": {"include_counts": True}}


def test_load_config_empty_params(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("aac:\n")
    result = _load_config(cfg)
    assert result == {"aac": {}}


def test_load_config_unknown_descriptor(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("nonexistent_descriptor:\n")
    with pytest.raises(typer.BadParameter, match="Unknown descriptor"):
        _load_config(cfg)


def test_load_config_unknown_param(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("aac:\n  totally_fake_param: 42\n")
    with pytest.raises(typer.BadParameter, match="Unknown param"):
        _load_config(cfg)


def test_load_config_invalid_yaml(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("key: [unclosed\n")
    with pytest.raises(typer.BadParameter, match="Invalid YAML"):
        _load_config(cfg)


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_compute_no_selector(tmp_path):
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text(_FASTA)
    result = runner.invoke(app, ["compute", str(fasta), "-o", str(tmp_path / "out.csv")])
    assert result.exit_code == 1
    assert "Specify" in result.output or "Specify" in (result.stderr or "")


def test_compute_config_and_all_exclusive(tmp_path):
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text(_FASTA)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("aac:\n")
    result = runner.invoke(
        app, ["compute", str(fasta), "--config", str(cfg), "--all", "-o", str(tmp_path / "out.csv")]
    )
    assert result.exit_code == 1


def test_compute_invalid_descriptor(tmp_path):
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text(_FASTA)
    result = runner.invoke(
        app, ["compute", str(fasta), "-d", "not_a_real_descriptor", "-o", str(tmp_path / "out.csv")]
    )
    assert result.exit_code == 1


def test_compute_happy_path_csv(tmp_path):
    fasta = tmp_path / "seqs.fasta"
    fasta.write_text(_FASTA)
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
    fasta.write_text(_FASTA)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("aac:\n  include_counts: false\ncharge:\n")
    out = tmp_path / "out.parquet"
    result = runner.invoke(
        app, ["compute", str(fasta), "--config", str(cfg), "-o", str(out), "--no-progress"]
    )
    assert result.exit_code == 0, result.output
    df = pl.read_parquet(out)
    assert df.shape[0] == 2
    assert "aac_count_A" not in df.columns   # include_counts=false
    assert "charge_positive_fraction" in df.columns
