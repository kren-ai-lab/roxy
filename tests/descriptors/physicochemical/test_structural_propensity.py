"""Tests for StructuralPropensityDescriptor."""

import math

import pytest

from roxy.descriptors.physicochemical.structural_propensity import StructuralPropensityDescriptor

SEQ = "ACDEFGHIKLMNPQRSTVWY"
EMPTY = ""
HELIX_SEQ = "AAEEEAAEEE"
SHEET_SEQ = "VIVIVIVIVI"


@pytest.fixture
def desc():
    return StructuralPropensityDescriptor()


def test_empty_nan_values(desc):
    out = desc.compute_one(EMPTY)
    assert out["length"] == 0.0
    assert math.isnan(out["helix_mean"])
    assert math.isnan(out["w5_helix_mean"])
    assert math.isnan(out["w5_helix_sheet_balance_mean"])


def test_helix_seq_high_propensity(desc):
    out = desc.compute_one(HELIX_SEQ)
    assert out["helix_mean"] > out["sheet_mean"]
    assert out["helix_sheet_balance"] > 0


def test_sheet_seq_high_propensity(desc):
    out = desc.compute_one(SHEET_SEQ)
    assert out["sheet_mean"] > out["helix_mean"]
    assert out["helix_sheet_balance"] < 0


def test_window_columns_present(desc):
    out = desc.compute_one(SEQ)
    for ws in (5, 7):
        for ss in ("helix", "sheet", "turn"):
            assert f"w{ws}_{ss}_mean" in out
            assert f"w{ws}_{ss}_high_fraction" in out
        assert f"w{ws}_helix_sheet_balance_mean" in out


def test_custom_window_sizes():
    desc = StructuralPropensityDescriptor(window_sizes=(3,))
    out = desc.compute_one(SEQ)
    assert "w3_helix_mean" in out
    assert "w5_helix_mean" not in out


def test_propensity_range(desc):
    out = desc.compute_one(SEQ)
    for ss in ("helix", "sheet", "turn"):
        assert 0.0 < out[f"{ss}_mean"] < 3.0
