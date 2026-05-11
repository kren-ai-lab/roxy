"""Tests for descriptor frame concatenation."""

from __future__ import annotations

from roxy.cli.compute import _concat_frames
from roxy.descriptors import DESCRIPTOR_REGISTRY


def _make_frame(name: str, n: int = 2):
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
