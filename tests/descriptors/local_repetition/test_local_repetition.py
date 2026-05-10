"""Tests for LocalRepetitionDescriptor."""

import math

import pytest

from roxy.descriptors.local_repetition.local_repetition import LocalRepetitionDescriptor

SEQ = "ACDEFGHIKLMNPQRSTVWY"
EMPTY = ""
POLY_A = "AAAAAAAAAA"  # high repetition


@pytest.fixture
def desc():
    return LocalRepetitionDescriptor()


def test_smoke(desc):
    df = desc.compute([SEQ, EMPTY])
    assert df.shape[0] == 2
    assert "local_repetition_repeated_fraction_k2" in df.columns
    assert "local_repetition_local_redundancy_mean_w8_k2" in df.columns


def test_empty_schema_consistent(desc):
    assert set(desc.compute_one(SEQ)) == set(desc.compute_one(EMPTY))


def test_empty_nan(desc):
    out = desc.compute_one(EMPTY)
    assert out["length"] == 0.0
    assert math.isnan(out["repeated_fraction_k2"])
    assert math.isnan(out["local_redundancy_mean_w8_k2"])


def test_column_count(desc):
    out = desc.compute_one(SEQ)
    # 2 base + 33 features = 35
    assert len(out) == 35


def test_poly_a_high_redundancy(desc):
    out = desc.compute_one(POLY_A)
    # all k2 words are "AA" → unique_fraction_k2 = 1/9 or similar, redundancy high
    assert out["redundancy_score_k2"] > 0.5
    assert out["repeated_fraction_k2"] > 0.5


def test_fractions_in_range(desc):
    out = desc.compute_one(SEQ)
    for key in ("repeated_fraction_k2", "unique_fraction_k3", "redundancy_score_k2"):
        v = out[key]
        if not math.isnan(v):
            assert 0.0 <= v <= 1.0


def test_registered():
    from roxy.descriptors import DESCRIPTOR_REGISTRY

    assert "local_repetition" in DESCRIPTOR_REGISTRY
