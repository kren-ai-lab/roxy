"""Tests for CTDClassicDescriptor."""

import math

import pytest

from roxy.descriptors.ctd.ctd_classic import CTDClassicDescriptor
from tests.descriptors._helpers import EMPTY, SEQ_ALL20, assert_keys_present

SEQ = SEQ_ALL20
ALL_CLASS1_HYDRO = "RKEDQN"


@pytest.fixture
def desc():
    return CTDClassicDescriptor()


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
    desc = CTDClassicDescriptor()
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


def test_ctd_feature_groups_present(desc):
    out = desc.compute_one(SEQ)
    assert_keys_present(
        out,
        [
            "hydrophobicity_comp_1",
            "hydrophobicity_trans_12",
            "hydrophobicity_dist_1_050",
            "polarity_comp_2",
            "polarity_trans_23",
            "polarity_dist_3_100",
            "charge_comp_3",
            "charge_trans_13",
            "charge_dist_2_025",
        ],
    )
