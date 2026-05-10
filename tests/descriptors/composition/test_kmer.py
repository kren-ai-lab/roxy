"""Tests for KmerDescriptor."""

from __future__ import annotations

import math

import pytest

from roxy.descriptors.composition.kmer import KmerDescriptor

SEQ = "ACDEFGHIKLMNPQRSTVWY"


def test_smoke():
    d = KmerDescriptor()
    df = d.compute([SEQ])
    assert df.shape[0] == 1
    assert "kmer_freq_AC" in df.columns


def test_default_k2_column_count():
    d = KmerDescriptor()
    out = d.compute_one(SEQ)
    # 3 base + 400 counts + 400 freqs + 1 freq_sum = 804
    assert len(out) == 804


def test_freq_sums_to_one():
    d = KmerDescriptor(include_counts=False)
    out = d.compute_one(SEQ * 5)
    assert math.isclose(out["frequency_sum"], 1.0, abs_tol=1e-9)


def test_empty_sequence():
    d = KmerDescriptor()
    out = d.compute_one("")
    assert out["length"] == 0
    assert out["total_kmers"] == 0
    assert math.isnan(out["freq_AC"])
    assert math.isnan(out["frequency_sum"])


def test_seq_shorter_than_k():
    d = KmerDescriptor(k=3)
    out = d.compute_one("AC")
    assert out["total_kmers"] == 0
    assert math.isnan(out["freq_ACA"])


def test_large_k_warns():
    with pytest.warns(UserWarning, match="k=4"):
        KmerDescriptor(k=4)


def test_registered():
    from roxy.descriptors import DESCRIPTOR_REGISTRY
    assert "kmer" in DESCRIPTOR_REGISTRY


def test_consistent_schema():
    d = KmerDescriptor()
    df = d.compute([SEQ, SEQ * 3, ""])
    assert df.shape[0] == 3
    assert len(df.columns) == len(d.compute_one(SEQ))
