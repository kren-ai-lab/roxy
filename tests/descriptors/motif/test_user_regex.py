"""Tests for UserRegexDescriptor."""

import math

import pytest

from roxy.descriptors.motif.user_regex import UserRegexDescriptor

SEQ = "ACDEFGHIKLMNPQRSTVWY"
EMPTY = ""
KK_SEQ = "KKAACDE"  # has basic_pair


@pytest.fixture
def desc():
    return UserRegexDescriptor()


def test_empty_zero_and_nan(desc):
    out = desc.compute_one(EMPTY)
    assert out["length"] == 0.0
    assert out["basic_pair_present"] == 0.0
    assert math.isnan(out["basic_pair_density"])
    assert math.isnan(out["basic_pair_first_pos"])


def test_column_count(desc):
    out = desc.compute_one(SEQ)
    # 2 base + 8 patterns x 11 stats = 90
    assert len(out) == 90


def test_basic_pair_detected():
    desc = UserRegexDescriptor()
    out = desc.compute_one(KK_SEQ)
    assert out["basic_pair_present"] == 1.0
    assert out["basic_pair_count"] >= 1.0
    assert 0.0 < out["basic_pair_first_pos_norm"] <= 1.0


def test_custom_no_terminal():
    desc = UserRegexDescriptor(patterns={"any_K": r"K"}, include_terminal=False)
    out = desc.compute_one(SEQ)
    assert "any_K_present" in out
    assert "any_K_nterm_present_w10" not in out
    # 2 base + 1 x 9 = 11
    assert len(out) == 11
