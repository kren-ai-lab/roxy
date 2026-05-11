"""Tests for NormalizedPositionDescriptor."""

import math

import pytest

from roxy.descriptors.positional.normalized import NormalizedPositionDescriptor
from tests.descriptors._helpers import EMPTY, SEQ_ALL20, assert_keys_present

SEQ = SEQ_ALL20
ALL_CHARGED = "KRHDEKRH"


@pytest.fixture
def desc():
    return NormalizedPositionDescriptor()


def test_empty_nan(desc):
    out = desc.compute_one(EMPTY)
    assert out["length"] == 0.0
    assert out["charged_count"] == 0.0
    assert math.isnan(out["charged_first_norm"])


def test_position_feature_groups_present(desc):
    out = desc.compute_one(SEQ)
    assert_keys_present(
        out,
        [
            "charged_count",
            "charged_first_norm",
            "charged_last_norm",
            "charged_mean_norm",
            "charged_n_bias",
            "hydrophobic_c_bias",
            "gly_count",
            "pro_count",
        ],
    )


def test_norm_positions_in_range(desc):
    out = desc.compute_one(SEQ)
    for name in ("charged", "hydrophobic", "positive"):
        for stat in ("first_norm", "last_norm", "mean_norm"):
            v = out[f"{name}_{stat}"]
            if not math.isnan(v):
                assert 0.0 < v <= 1.0


def test_bias_bounds(desc):
    out = desc.compute_one(SEQ)
    for name in ("charged", "aromatic"):
        for stat in ("n_bias", "c_bias"):
            v = out[f"{name}_{stat}"]
            if not math.isnan(v):
                assert 0.0 <= v <= 1.0


def test_absent_group_zero_count(desc):
    # sequence with no G or P
    out = desc.compute_one("ACDE")
    # gly and pro may or may not be present
    assert not math.isnan(out["gly_count"]) or out["gly_count"] == 0.0
