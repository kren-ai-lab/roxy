"""Tests for PatternDescriptor."""

import math

import pytest

from roxy.descriptors.motif.pattern import PatternDescriptor

SEQ = "ACDEFGHIKLMNPQRSTVWY"
EMPTY = ""
POLY_K = "KKKACDE"


@pytest.fixture
def desc():
    return PatternDescriptor()


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
    desc = PatternDescriptor()
    out = desc.compute_one(POLY_K)
    assert out["polyK_3plus_present"] == 1.0
    assert out["polyK_3plus_count"] == 1.0


def test_custom_patterns():
    desc = PatternDescriptor(patterns={"NxS": r"N.S"}, include_terminal=False)
    out = desc.compute_one(SEQ)
    assert "NxS_present" in out
    assert "NxS_nterm_present_w10" not in out
    # 2 base + 1 x 3 stats = 5
    assert len(out) == 5
