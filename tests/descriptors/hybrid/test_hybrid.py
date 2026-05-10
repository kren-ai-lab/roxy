"""Tests for HybridDescriptor."""

import math

import pytest

from roxy.descriptors.hybrid.hybrid import HybridDescriptor

SEQ = "ACDEFGHIKLMNPQRSTVWY"
EMPTY = ""


@pytest.fixture
def desc():
    return HybridDescriptor()


def test_smoke(desc):
    df = desc.compute([SEQ, EMPTY])
    assert df.shape[0] == 2
    assert "hybrid_charge_family_mean" in df.columns
    assert "hybrid_global_family_amplitude" in df.columns


def test_empty_schema_consistent(desc):
    assert set(desc.compute_one(SEQ)) == set(desc.compute_one(EMPTY))


def test_empty_nan(desc):
    out = desc.compute_one(EMPTY)
    assert out["length"] == 0.0
    assert math.isnan(out["charge_family_mean"])
    assert math.isnan(out["global_family_std"])


def test_column_count(desc):
    out = desc.compute_one(SEQ)
    # 2 + 4 + 4 + 3 + 2 + 2 + 3 + 5 = 25
    assert len(out) == 25


def test_global_amplitude_consistent(desc):
    out = desc.compute_one(SEQ)
    expected = out["global_family_max"] - out["global_family_min"]
    assert math.isclose(out["global_family_amplitude"], expected, abs_tol=1e-9)


def test_registered():
    from roxy.descriptors import DESCRIPTOR_REGISTRY

    assert "hybrid" in DESCRIPTOR_REGISTRY
