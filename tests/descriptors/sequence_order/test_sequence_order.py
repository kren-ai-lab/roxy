"""Tests for SequenceOrderDescriptor."""

import math

import pytest

from roxy.descriptors.sequence_order.sequence_order import SequenceOrderDescriptor

SEQ = "ACDEFGHIKLMNPQRSTVWY"
EMPTY = ""
SINGLE = "A"


@pytest.fixture
def desc():
    return SequenceOrderDescriptor()


def test_smoke(desc):
    df = desc.compute([SEQ, EMPTY])
    assert df.shape[0] == 2
    assert "sequence_order_charged_same_adj_frac" in df.columns
    assert "sequence_order_disorder_order_transition_frac" in df.columns


def test_empty_schema_consistent(desc):
    assert set(desc.compute_one(SEQ)) == set(desc.compute_one(EMPTY))


def test_empty_nan(desc):
    out = desc.compute_one(EMPTY)
    assert out["length"] == 0.0
    assert math.isnan(out["charged_same_adj_frac"])
    assert math.isnan(out["charged_hydrophobic_transition_frac"])


def test_column_count(desc):
    out = desc.compute_one(SEQ)
    # 2 base + 6 groups x 6 stats + 4 cross = 42
    assert len(out) == 42


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


def test_registered():
    from roxy.descriptors import DESCRIPTOR_REGISTRY

    assert "sequence_order" in DESCRIPTOR_REGISTRY
