"""Tests for ChargeDescriptor."""

import math

import pytest

from roxy.descriptors.physicochemical.charge import ChargeDescriptor
from tests.descriptors._helpers import EMPTY, SEQ_ALL20

SEQ = SEQ_ALL20
BASIC_SEQ = "KRKRKR"
ACIDIC_SEQ = "DEDEDE"


@pytest.fixture
def desc():
    return ChargeDescriptor()


def test_empty_nan_values(desc):
    out = desc.compute_one(EMPTY)
    assert out["length"] == 0.0
    assert math.isnan(out["positive_fraction"])
    assert math.isnan(out["net_charge_ph7p0"])
    assert math.isnan(out["local_mean_ph7p0"])


def test_basic_seq_positive_charge(desc):
    out = desc.compute_one(BASIC_SEQ)
    assert out["net_charge_ph7p0"] > 0
    assert out["ncpr"] > 0


def test_acidic_seq_negative_charge(desc):
    out = desc.compute_one(ACIDIC_SEQ)
    assert out["net_charge_ph7p0"] < 0
    assert out["ncpr"] < 0


def test_ph_columns_present(desc):
    out = desc.compute_one(SEQ)
    for ph_tag in ("5p0", "7p0", "9p0"):
        assert f"net_charge_ph{ph_tag}" in out
        assert f"local_mean_ph{ph_tag}" in out
        assert f"terminal_asymmetry_ph{ph_tag}" in out


def test_custom_ph_values():
    desc = ChargeDescriptor(ph_values=(7.4,))
    out = desc.compute_one(SEQ)
    assert "net_charge_ph7p4" in out
    assert "net_charge_ph7p0" not in out


def test_fcr_bounds(desc):
    out = desc.compute_one(SEQ)
    assert 0.0 <= out["fcr"] <= 1.0
