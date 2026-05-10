"""Tests for CTDDescriptor."""

import math

import pytest

from roxy.descriptors.ctd.ctd import CTDDescriptor

SEQ = "ACDEFGHIKLMNPQRSTVWY"
EMPTY = ""
ALL_CLASS1_HYDRO = "RKEDQN"


@pytest.fixture
def desc():
    return CTDDescriptor()


def test_empty_nan_values(desc):
    out = desc.compute_one(EMPTY)
    assert out["length"] == 0.0
    assert math.isnan(out["hydrophobicity_comp_1"])
    assert math.isnan(out["polarity_trans_12"])
    assert math.isnan(out["charge_dist_1_050"])


def test_composition_sums_to_one(desc):
    out = desc.compute_one(SEQ)
    for prop in ("hydrophobicity", "polarity", "charge"):
        total = sum(out[f"{prop}_comp_{c}"] for c in ("1", "2", "3"))
        assert math.isclose(total, 1.0, abs_tol=1e-9)


def test_all_class1_hydro_composition():
    desc = CTDDescriptor()
    out = desc.compute_one(ALL_CLASS1_HYDRO)
    assert math.isclose(out["hydrophobicity_comp_1"], 1.0, abs_tol=1e-9)
    assert math.isclose(out["hydrophobicity_comp_2"], 0.0, abs_tol=1e-9)
    assert math.isclose(out["hydrophobicity_comp_3"], 0.0, abs_tol=1e-9)


def test_distribution_bounds(desc):
    out = desc.compute_one(SEQ)
    # Distribution values are normalized positions [0, 1]
    for prop in ("hydrophobicity", "polarity", "charge"):
        for c in ("1", "2", "3"):
            for dlabel in ("001", "025", "050", "075", "100"):
                val = out[f"{prop}_dist_{c}_{dlabel}"]
                if not math.isnan(val):
                    assert 0.0 < val <= 1.0


def test_column_count(desc):
    out = desc.compute_one(SEQ)
    # 2 base + 3 props x (3 comp + 3 trans + 15 dist) = 2 + 63 = 65
    assert len(out) == 65
