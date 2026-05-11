"""Tests for PatternDescriptor."""

import math

import pytest

from roxy.descriptors.motif.pattern import PatternDescriptor
from tests.descriptors._helpers import EMPTY, SEQ_ALL20, assert_keys_present

SEQ = SEQ_ALL20
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


def test_default_pattern_features_present(desc):
    out = desc.compute_one(SEQ)
    assert_keys_present(
        out,
        [
            "CxxC_present",
            "CxxC_count",
            "CxxC_density",
            "CxxC_nterm_present_w10",
            "polyK_3plus_present",
            "acidic_patch_3plus_count",
        ],
    )


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
    assert "NxS_count" in out
    assert "NxS_density" in out
