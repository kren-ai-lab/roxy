"""Tests for FamilySummaryDescriptor."""

import math

import pytest

from roxy.descriptors.hybrid.family_summary import FamilySummaryDescriptor
from tests.descriptors._helpers import EMPTY, SEQ_ALL20, assert_keys_present

SEQ = SEQ_ALL20


@pytest.fixture
def desc():
    return FamilySummaryDescriptor()


def test_empty_nan(desc):
    out = desc.compute_one(EMPTY)
    assert out["length"] == 0.0
    assert math.isnan(out["charge_family_mean"])
    assert math.isnan(out["global_family_std"])


def test_family_summary_features_present(desc):
    out = desc.compute_one(SEQ)
    assert_keys_present(
        out,
        [
            "charge_family_mean",
            "physchem_family_mean",
            "struct_family_mean",
            "orderdis_family_mean",
            "functional_family_mean",
            "complexity_family_mean",
            "global_family_amplitude",
        ],
    )


def test_global_amplitude_consistent(desc):
    out = desc.compute_one(SEQ)
    expected = out["global_family_max"] - out["global_family_min"]
    assert math.isclose(out["global_family_amplitude"], expected, abs_tol=1e-9)
