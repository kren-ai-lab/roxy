"""Tests for SpacingDescriptor."""

import math

import pytest

from roxy.descriptors.motif.spacing import SpacingDescriptor
from tests.descriptors._helpers import EMPTY, SEQ_ALL20, assert_keys_present

SEQ = SEQ_ALL20
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


def test_spacing_feature_groups_present(desc):
    out = desc.compute_one(SEQ)
    assert_keys_present(
        out,
        [
            "charged_count",
            "charged_density",
            "charged_mean",
            "charged_nn_mean",
            "hydrophobic_density",
            "aromatic_polar_cross_mean",
            "positive_negative_cross_mean",
        ],
    )


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
