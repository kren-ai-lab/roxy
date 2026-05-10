"""Tests for PseAACDescriptor."""

import math

import pytest

from roxy.descriptors.pseudo.pseaac import PseAACDescriptor

SEQ = "ACDEFGHIKLMNPQRSTVWY"
EMPTY = ""
SHORT = "AC"  # shorter than default lam=5


@pytest.fixture
def desc():
    return PseAACDescriptor()


def test_smoke(desc):
    df = desc.compute([SEQ, EMPTY])
    assert df.shape[0] == 2
    assert "pseaac_length" in df.columns
    assert "pseaac_A" in df.columns
    assert "pseaac_theta_1" in df.columns


def test_empty_schema_consistent(desc):
    out_full = desc.compute_one(SEQ)
    out_empty = desc.compute_one(EMPTY)
    assert set(out_full.keys()) == set(out_empty.keys())


def test_empty_nan_values(desc):
    out = desc.compute_one(EMPTY)
    assert out["length"] == 0.0
    assert math.isnan(out["A"])
    assert math.isnan(out["theta_1"])
    assert math.isnan(out["feature_sum"])


def test_feature_sum_equals_one(desc):
    out = desc.compute_one(SEQ)
    assert math.isclose(out["feature_sum"], 1.0, abs_tol=1e-9)


def test_feature_sum_short_seq(desc):
    # Short seq: lags beyond len-1 fall back to 0 theta; sum still 1.0
    out = desc.compute_one(SHORT)
    assert math.isclose(out["feature_sum"], 1.0, abs_tol=1e-9)


def test_column_count(desc):
    out = desc.compute_one(SEQ)
    # 2 base + 20 AA + 5 theta + 1 feature_sum = 28
    assert len(out) == 28


def test_all_aa_features_present(desc):
    out = desc.compute_one(SEQ)
    for aa in "ACDEFGHIKLMNPQRSTVWY":
        assert aa in out
        assert not math.isnan(out[aa])


def test_custom_lam():
    desc = PseAACDescriptor(lam=3)
    out = desc.compute_one(SEQ)
    assert "theta_3" in out
    assert "theta_4" not in out
    assert len(out) == 2 + 20 + 3 + 1


def test_registered():
    from roxy.descriptors import DESCRIPTOR_REGISTRY
    assert "pseaac" in DESCRIPTOR_REGISTRY
