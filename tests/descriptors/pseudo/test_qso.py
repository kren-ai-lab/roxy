"""Tests for QSODescriptor."""

import math

import pytest

from roxy.descriptors.pseudo.qso import QSODescriptor
from tests.descriptors._helpers import EMPTY, SEQ_ALL20

SEQ = SEQ_ALL20


@pytest.fixture
def desc():
    return QSODescriptor()


def test_empty_nan(desc):
    out = desc.compute_one(EMPTY)
    assert out["length"] == 0.0
    assert math.isnan(out["A"])
    assert math.isnan(out["tau_1"])
    assert math.isnan(out["feature_sum"])


def test_feature_sum_approx_one(desc):
    out = desc.compute_one(SEQ)
    assert math.isclose(out["feature_sum"], 1.0, abs_tol=1e-9)


def test_custom_lam():
    d = QSODescriptor(lam=3)
    out = d.compute_one(SEQ)
    assert "tau_3" in out
    assert "tau_4" not in out
    assert "feature_sum" in out
