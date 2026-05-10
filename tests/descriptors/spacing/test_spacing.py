"""Tests for SpacingDescriptor."""

import math

import pytest

from roxy.descriptors.spacing.spacing import SpacingDescriptor

SEQ = "ACDEFGHIKLMNPQRSTVWY"
EMPTY = ""
SINGLE = "K"  # only 1 charged — inter-event NaN


@pytest.fixture
def desc():
    return SpacingDescriptor()


def test_empty_nan(desc):
    out = desc.compute_one(EMPTY)
    assert out["length"] == 0.0
    assert out["charged_count"] == 0.0
    assert math.isnan(out["charged_mean"])
    assert math.isnan(out["positive_negative_cross_mean"])


def test_column_count(desc):
    out = desc.compute_one(SEQ)
    # 2 base + 8 groups x 16 stats + 4 cross = 134
    assert len(out) == 134


def test_density_in_range(desc):
    out = desc.compute_one(SEQ)
    for name in ("charged", "hydrophobic", "aromatic"):
        d = out[f"{name}_density"]
        assert 0.0 <= d <= 1.0


def test_single_residue_group_nan_dist(desc):
    # only 1 positive in seq → inter-event NaN, nn NaN
    out = desc.compute_one(SINGLE)
    assert math.isnan(out["positive_mean"])
    assert math.isnan(out["positive_nn_mean"])


def test_cross_group_nan_when_absent(desc):
    # no aromatic in "ACDE" → aromatic_polar_cross_mean = NaN
    out = desc.compute_one("ACDE")
    assert math.isnan(out["aromatic_polar_cross_mean"])
