"""Tests for DistributionDescriptor."""

import math

import pytest

from roxy.descriptors.distribution.distribution import DistributionDescriptor

SEQ = "ACDEFGHIKLMNPQRSTVWY"
EMPTY = ""


@pytest.fixture
def desc():
    return DistributionDescriptor()


def test_smoke(desc):
    df = desc.compute([SEQ, EMPTY])
    assert df.shape[0] == 2
    assert "distribution_charged_count" in df.columns
    assert "distribution_hydrophobic_tercile_bin1_frac" in df.columns


def test_empty_schema_consistent(desc):
    assert set(desc.compute_one(SEQ)) == set(desc.compute_one(EMPTY))


def test_empty_nan(desc):
    out = desc.compute_one(EMPTY)
    assert out["length"] == 0.0
    assert math.isnan(out["charged_count"])
    assert math.isnan(out["hydrophobic_tercile_bin1_frac"])


def test_column_count(desc):
    out = desc.compute_one(SEQ)
    # 2 + 8 groups x (1 count + 12 tercile + 15 quartile) = 2 + 8*28 = 226
    assert len(out) == 226


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


def test_registered():
    from roxy.descriptors import DESCRIPTOR_REGISTRY
    assert "distribution" in DESCRIPTOR_REGISTRY
