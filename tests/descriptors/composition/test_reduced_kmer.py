"""Tests for ReducedKmerDescriptor."""

from __future__ import annotations

import math

import pytest

from roxy.descriptors.composition.reduced_kmer import ReducedKmerDescriptor

SEQ = "ACDEFGHIKLMNPQRSTVWY"


def test_smoke_default():
    d = ReducedKmerDescriptor()
    df = d.compute([SEQ])
    assert df.shape[0] == 1
    assert "reduced_kmer_freq_HH" in df.columns


def test_rd5_alphabet_size():
    d = ReducedKmerDescriptor(scheme="rd5")
    out = d.compute_one(SEQ)
    assert out["alphabet_size"] == 5
    # 5^2 = 25 freqs + 2 base cols = 27
    assert len(out) == 27


def test_rd3_alphabet_size():
    d = ReducedKmerDescriptor(scheme="rd3")
    out = d.compute_one(SEQ)
    assert out["alphabet_size"] == 3
    # 3^2 = 9 freqs + 2 base cols = 11
    assert len(out) == 11


def test_freq_sums_to_one():
    d = ReducedKmerDescriptor()
    out = d.compute_one(SEQ * 5)
    freqs = [v for k, v in out.items() if k.startswith("freq_")]
    assert math.isclose(sum(freqs), 1.0, abs_tol=1e-9)


def test_empty_sequence():
    d = ReducedKmerDescriptor()
    out = d.compute_one("")
    assert out["length"] == 0
    assert math.isnan(out["freq_HH"])


def test_custom_mapping():
    mapping = {"A": "X", "C": "Y", "D": "X", "E": "Y"}
    d = ReducedKmerDescriptor(mapping=mapping, k=1)
    out = d.compute_one("ACDE")
    assert out["alphabet_size"] == 2
    assert math.isclose(out["freq_X"], 0.5, abs_tol=1e-9)


def test_large_k_warns():
    with pytest.warns(UserWarning, match="k=4"):
        ReducedKmerDescriptor(k=4)


def test_registered():
    from roxy.descriptors import DESCRIPTOR_REGISTRY
    assert "reduced_kmer" in DESCRIPTOR_REGISTRY


def test_consistent_schema():
    d = ReducedKmerDescriptor()
    df = d.compute([SEQ, SEQ * 3, ""])
    assert df.shape[0] == 3
    assert len(df.columns) == len(d.compute_one(SEQ))
