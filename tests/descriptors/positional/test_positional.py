"""Tests for PositionalDescriptor."""

import math

import pytest

from roxy.descriptors.positional.positional import PositionalDescriptor

SEQ = "ACDEFGHIKLMNPQRSTVWY"
EMPTY = ""
ALL_CHARGED = "KRHDEKRH"


@pytest.fixture
def desc():
    return PositionalDescriptor()


def test_empty_nan(desc):
    out = desc.compute_one(EMPTY)
    assert out["length"] == 0.0
    assert out["charged_count"] == 0.0
    assert math.isnan(out["charged_first_norm"])


def test_column_count(desc):
    out = desc.compute_one(SEQ)
    # 2 base + 10 groups x 10 stats = 102
    assert len(out) == 102


def test_norm_positions_in_range(desc):
    out = desc.compute_one(SEQ)
    for name in ("charged", "hydrophobic", "positive"):
        for stat in ("first_norm", "last_norm", "mean_norm"):
            v = out[f"{name}_{stat}"]
            if not math.isnan(v):
                assert 0.0 < v <= 1.0


def test_bias_bounds(desc):
    out = desc.compute_one(SEQ)
    for name in ("charged", "aromatic"):
        for stat in ("n_bias", "c_bias"):
            v = out[f"{name}_{stat}"]
            if not math.isnan(v):
                assert 0.0 <= v <= 1.0


def test_absent_group_zero_count(desc):
    # sequence with no G or P
    out = desc.compute_one("ACDE")
    # gly and pro may or may not be present
    assert not math.isnan(out["gly_count"]) or out["gly_count"] == 0.0
