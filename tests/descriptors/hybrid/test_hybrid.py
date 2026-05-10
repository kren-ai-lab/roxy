"""Tests for HybridDescriptor."""

import math

import pytest

from roxy.descriptors.hybrid.hybrid import HybridDescriptor

SEQ = "ACDEFGHIKLMNPQRSTVWY"
EMPTY = ""


@pytest.fixture
def desc():
    return HybridDescriptor()


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
