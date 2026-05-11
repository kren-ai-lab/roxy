"""Tests for AutocorrelationDescriptor."""

import math

import pytest

from roxy.descriptors.autocorrelation.autocorrelation import AutocorrelationDescriptor
from tests.descriptors._helpers import EMPTY, SEQ_ALL20

SEQ = SEQ_ALL20
UNIFORM_SEQ = "AAAAAAAAAA"


@pytest.fixture
def desc():
    return AutocorrelationDescriptor()


def test_empty_nan_values(desc):
    out = desc.compute_one(EMPTY)
    assert out["length"] == 0.0
    assert math.isnan(out["mb_hydrophobicity_lag1"])
    assert math.isnan(out["moran_polarity_lag3"])
    assert math.isnan(out["geary_volume_lag5"])


def test_uniform_moran_nan(desc):
    # Uniform sequence → zero variance → Moran is NaN
    out = desc.compute_one(UNIFORM_SEQ)
    assert math.isnan(out["moran_hydrophobicity_lag1"])


def test_uniform_geary_nan(desc):
    out = desc.compute_one(UNIFORM_SEQ)
    assert math.isnan(out["geary_hydrophobicity_lag1"])


def test_all_scales_all_lags_present(desc):
    out = desc.compute_one(SEQ)
    scales = ("hydrophobicity", "polarity", "flexibility", "volume", "charge_proxy")
    for scale in scales:
        for lag in range(1, 6):
            assert f"mb_{scale}_lag{lag}" in out
            assert f"moran_{scale}_lag{lag}" in out
            assert f"geary_{scale}_lag{lag}" in out


def test_custom_scales_and_lags():
    from roxy.core.constants import KD
    desc = AutocorrelationDescriptor(scales={"kd": KD}, lags=(1, 2))
    out = desc.compute_one(SEQ)
    assert "mb_kd_lag1" in out
    assert "mb_kd_lag2" in out
    assert "mb_hydrophobicity_lag1" not in out
    assert "moran_kd_lag2" in out
    assert "geary_kd_lag2" in out
