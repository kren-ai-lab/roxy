"""Tests for KmerFullAlphabetDescriptor."""

from __future__ import annotations

import math

import pytest

from roxy.descriptors.composition.kmer_full_alphabet import KmerFullAlphabetDescriptor
from tests.descriptors._helpers import EMPTY, SEQ_ALL20, assert_keys_present

SEQ = SEQ_ALL20


def test_default_k2_features_present():
    d = KmerFullAlphabetDescriptor()
    out = d.compute_one(SEQ)
    assert_keys_present(
        out,
        ["length", "total_kmers", "unique_kmers", "count_AC", "freq_AC", "frequency_sum"],
    )


def test_freq_sums_to_one():
    d = KmerFullAlphabetDescriptor(include_counts=False)
    out = d.compute_one(SEQ * 5)
    assert math.isclose(out["frequency_sum"], 1.0, abs_tol=1e-9)


def test_empty_sequence():
    d = KmerFullAlphabetDescriptor()
    out = d.compute_one(EMPTY)
    assert out["length"] == 0
    assert out["total_kmers"] == 0
    assert math.isnan(out["freq_AC"])
    assert math.isnan(out["frequency_sum"])


def test_seq_shorter_than_k():
    d = KmerFullAlphabetDescriptor(k=3)
    out = d.compute_one("AC")
    assert out["total_kmers"] == 0
    assert math.isnan(out["freq_ACA"])


def test_large_k_warns():
    with pytest.warns(UserWarning, match="k=4"):
        KmerFullAlphabetDescriptor(k=4)


def test_consistent_schema():
    d = KmerFullAlphabetDescriptor()
    df = d.compute([SEQ, SEQ * 3, EMPTY])
    assert df.shape[0] == 3
    assert set(df.columns) == {f"kmer_full_alphabet_{key}" for key in d.compute_one(SEQ)}
