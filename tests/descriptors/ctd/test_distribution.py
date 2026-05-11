"""Tests for DistributionDescriptor."""

import math

import pytest

from roxy.descriptors.ctd.distribution import DistributionDescriptor
from tests.descriptors._helpers import EMPTY, SEQ_ALL20, assert_keys_present

SEQ = SEQ_ALL20


@pytest.fixture
def desc():
    return DistributionDescriptor()


def test_empty_nan(desc):
    out = desc.compute_one(EMPTY)
    assert out["length"] == 0.0
    assert math.isnan(out["charged_count"])
    assert math.isnan(out["hydrophobic_tercile_bin1_frac"])


def test_distribution_feature_groups_present(desc):
    out = desc.compute_one(SEQ)
    assert_keys_present(
        out,
        [
            "charged_count",
            "charged_tercile_bin1_frac",
            "charged_quartile_bin4_frac",
            "hydrophobic_tercile_spread",
            "aromatic_quartile_spread",
        ],
    )


def test_bin_fracs_sum_to_one_or_zero(desc):
    out = desc.compute_one(SEQ)
    for name in ("charged", "hydrophobic"):
        t = sum(out[f"{name}_tercile_bin{i}_frac"] for i in range(1, 4))
        assert math.isclose(t, 1.0, abs_tol=1e-9) or math.isclose(t, 0.0, abs_tol=1e-9)


def test_count_nonneg(desc):
    out = desc.compute_one(SEQ)
    for name in ("gly", "pro", "aromatic"):
        assert out[f"{name}_count"] >= 0


def test_spread_bounds(desc):
    out = desc.compute_one(SEQ)
    for name in ("charged", "positive"):
        s = out[f"{name}_tercile_spread"]
        if not math.isnan(s):
            assert 0.0 <= s <= 1.0
