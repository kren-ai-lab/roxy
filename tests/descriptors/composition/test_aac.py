"""Tests for AACDescriptor."""

from __future__ import annotations

import math

import pytest

from roxy.descriptors.composition.aac import AACDescriptor

SEQ_ALL20 = "ACDEFGHIKLMNPQRSTVWY"
SEQ_BIASED = "AAACCCDDDD"


def test_freq_sums_to_one():
    d = AACDescriptor()
    out = d.compute_one(SEQ_ALL20)
    s = sum(out[f"freq_{aa}"] for aa in "ACDEFGHIKLMNPQRSTVWY")
    assert math.isclose(s, 1.0, abs_tol=1e-9)


def test_count_freq_consistency():
    d = AACDescriptor()
    out = d.compute_one(SEQ_BIASED)
    n = out["length"]
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        assert math.isclose(out[f"count_{aa}"] / n, out[f"freq_{aa}"], abs_tol=1e-9)


def test_empty_sequence():
    d = AACDescriptor()
    out = d.compute_one("")
    assert out["length"] == 0
    assert out["valid_residue_count"] == 0
    assert math.isnan(out["freq_A"])
    assert math.isnan(out["frequency_sum"])


def test_consistent_schema_across_seqs():
    d = AACDescriptor()
    df = d.compute([SEQ_ALL20, SEQ_BIASED, ""])
    assert df.shape[0] == 3
    assert len(df.columns) == len(d.compute_one(SEQ_ALL20))


@pytest.mark.parametrize("seq", ["*ACDEF*", "acdef", "  ACDEF  "])
def test_cleaning(seq):
    d = AACDescriptor()
    out = d.compute_one(seq)
    assert out["length"] > 0
