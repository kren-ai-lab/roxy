"""Tests for SequenceOrderDescriptor."""

import math

import pytest

from roxy.descriptors.pseudo.sequence_order import SequenceOrderDescriptor
from tests.descriptors._helpers import EMPTY, SEQ_ALL20, assert_keys_present

SEQ = SEQ_ALL20
SINGLE = "A"


@pytest.fixture
def desc():
    return SequenceOrderDescriptor()


def test_empty_nan(desc):
    out = desc.compute_one(EMPTY)
    assert out["length"] == 0.0
    assert math.isnan(out["charged_same_adj_frac"])
    assert math.isnan(out["charged_hydrophobic_transition_frac"])


def test_sequence_order_feature_groups_present(desc):
    out = desc.compute_one(SEQ)
    assert_keys_present(
        out,
        [
            "charged_same_adj_frac",
            "charged_cluster_frac_w3",
            "charged_lag1_coupling",
            "hydrophobic_mean_spacing_norm",
            "aromatic_lag2_coupling",
            "charged_hydrophobic_transition_frac",
        ],
    )


def test_fractions_in_range(desc):
    out = desc.compute_one(SEQ)
    for name in ("charged", "hydrophobic", "aromatic"):
        for suffix in ("same_adj_frac", "cluster_frac_w3", "lag1_coupling", "lag2_coupling"):
            v = out[f"{name}_{suffix}"]
            if not math.isnan(v):
                assert 0.0 <= v <= 1.0


def test_single_char_nan(desc):
    out = desc.compute_one(SINGLE)
    assert math.isnan(out["charged_same_adj_frac"])
    assert math.isnan(out["charged_hydrophobic_transition_frac"])
