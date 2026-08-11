"""Tests for OrderDisorderDescriptor."""

import math

import pytest

from roxy.descriptors.physicochemical.order_disorder import OrderDisorderDescriptor
from tests.descriptors._helpers import EMPTY, SEQ_ALL20

SEQ = SEQ_ALL20
DISORDER_SEQ = "SAPREKGSAPREKG"
ORDER_SEQ = "CWYFILVCWYFILV"


@pytest.fixture
def desc():
    return OrderDisorderDescriptor()


def test_empty_nan_values(desc):
    out = desc.compute_one(EMPTY)
    assert out["length"] == 0.0
    assert math.isnan(out["disorder_fraction"])
    assert math.isnan(out["w5_disorder_mean"])
    assert math.isnan(out["terminal_disorder_asymmetry"])


def test_disorder_seq_high_fraction(desc):
    out = desc.compute_one(DISORDER_SEQ)
    assert out["disorder_fraction"] > out["order_fraction"]
    assert out["disorder_order_balance"] > 0


def test_order_seq_high_fraction(desc):
    out = desc.compute_one(ORDER_SEQ)
    assert out["order_fraction"] > out["disorder_fraction"]
    assert out["order_disorder_balance"] > 0


def test_window_columns_present(desc):
    out = desc.compute_one(SEQ)
    for ws in (5, 7):
        assert f"w{ws}_disorder_patch_fraction" in out
        assert f"w{ws}_order_patch_fraction" in out
        assert f"w{ws}_disorder_mean" in out
        assert f"w{ws}_order_mean" in out


def test_fraction_bounds(desc):
    out = desc.compute_one(SEQ)
    for key in ("disorder_fraction", "order_fraction", "flexibility_fraction"):
        assert 0.0 <= out[key] <= 1.0
