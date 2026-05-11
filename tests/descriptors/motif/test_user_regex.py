"""Tests for UserRegexDescriptor."""

import math

import pytest

from roxy.descriptors.motif.user_regex import UserRegexDescriptor
from tests.descriptors._helpers import EMPTY, SEQ_ALL20, assert_keys_present

SEQ = SEQ_ALL20
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


def test_default_regex_features_present(desc):
    out = desc.compute_one(SEQ)
    assert_keys_present(
        out,
        [
            "basic_pair_present",
            "basic_pair_count",
            "basic_pair_density",
            "basic_pair_first_pos",
            "basic_pair_nterm_present_w10",
            "acidic_pair_count",
        ],
    )


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
    assert "any_K_count" in out
    assert "any_K_span_norm" in out
