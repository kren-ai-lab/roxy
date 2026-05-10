"""Tests for MotifDescriptor."""

import math

import pytest

from roxy.descriptors.motif.motif import MotifDescriptor

SEQ = "ACDEFGHIKLMNPQRSTVWY"
EMPTY = ""
POLY_K = "KKKACDE"


@pytest.fixture
def desc():
    return MotifDescriptor()


def test_smoke(desc):
    df = desc.compute([SEQ, EMPTY])
    assert df.shape[0] == 2
    assert "motif_CxxC_present" in df.columns
    assert "motif_polyK_3plus_nterm_present_w10" in df.columns


def test_empty_schema_consistent(desc):
    assert set(desc.compute_one(SEQ)) == set(desc.compute_one(EMPTY))


def test_empty_zero_counts(desc):
    out = desc.compute_one(EMPTY)
    assert out["length"] == 0.0
    assert out["polyK_3plus_present"] == 0.0
    assert out["polyK_3plus_count"] == 0.0
    assert math.isnan(out["polyK_3plus_density"])


def test_column_count(desc):
    out = desc.compute_one(SEQ)
    # 2 base + 9 patterns x 5 stats = 47
    assert len(out) == 47


def test_polyk_detected():
    desc = MotifDescriptor()
    out = desc.compute_one(POLY_K)
    assert out["polyK_3plus_present"] == 1.0
    assert out["polyK_3plus_count"] == 1.0


def test_custom_patterns():
    desc = MotifDescriptor(patterns={"NxS": r"N.S"}, include_terminal=False)
    out = desc.compute_one(SEQ)
    assert "NxS_present" in out
    assert "NxS_nterm_present_w10" not in out
    # 2 base + 1 x 3 stats = 5
    assert len(out) == 5


def test_registered():
    from roxy.descriptors import DESCRIPTOR_REGISTRY

    assert "motif" in DESCRIPTOR_REGISTRY
