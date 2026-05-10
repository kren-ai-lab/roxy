"""Tests for QSODescriptor."""

import math

import pytest

from roxy.descriptors.qso.qso import QSODescriptor

SEQ = "ACDEFGHIKLMNPQRSTVWY"
EMPTY = ""


@pytest.fixture
def desc():
    return QSODescriptor()


def test_smoke(desc):
    df = desc.compute([SEQ, EMPTY])
    assert df.shape[0] == 2
    assert "qso_A" in df.columns
    assert "qso_tau_1" in df.columns
    assert "qso_feature_sum" in df.columns


def test_empty_schema_consistent(desc):
    assert set(desc.compute_one(SEQ)) == set(desc.compute_one(EMPTY))


def test_empty_nan(desc):
    out = desc.compute_one(EMPTY)
    assert out["length"] == 0.0
    assert math.isnan(out["A"])
    assert math.isnan(out["tau_1"])
    assert math.isnan(out["feature_sum"])


def test_column_count(desc):
    out = desc.compute_one(SEQ)
    # 2 base + 20 AA + 5 tau + 1 feature_sum = 28
    assert len(out) == 28


def test_feature_sum_approx_one(desc):
    out = desc.compute_one(SEQ)
    assert math.isclose(out["feature_sum"], 1.0, abs_tol=1e-9)


def test_custom_lam():
    d = QSODescriptor(lam=3)
    out = d.compute_one(SEQ)
    assert "tau_3" in out
    assert "tau_4" not in out
    # 2 + 20 + 3 + 1 = 26
    assert len(out) == 26


def test_registered():
    from roxy.descriptors import DESCRIPTOR_REGISTRY

    assert "qso" in DESCRIPTOR_REGISTRY
