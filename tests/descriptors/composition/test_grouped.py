"""Tests for GroupedCompositionDescriptor."""

from __future__ import annotations

import math

from roxy.descriptors.composition.grouped import GroupedCompositionDescriptor

SEQ = "ACDEFGHIKLMNPQRSTVWY"
SEQ_ALL_K = "KKKKKK"


def test_smoke():
    d = GroupedCompositionDescriptor()
    df = d.compute([SEQ])
    assert df.shape[0] == 1
    assert "grouped_composition_positive_count" in df.columns
    assert "grouped_composition_ratio_acidic_basic" in df.columns


def test_positive_count():
    d = GroupedCompositionDescriptor()
    out = d.compute_one(SEQ_ALL_K)
    assert out["positive_count"] == 6
    assert math.isclose(out["positive_frac"], 1.0, abs_tol=1e-9)


def test_empty_sequence():
    d = GroupedCompositionDescriptor()
    out = d.compute_one("")
    assert out["length"] == 0
    assert math.isnan(out["positive_frac"])
    assert math.isnan(out["ratio_acidic_basic"])


def test_ratio_zero_denominator():
    d = GroupedCompositionDescriptor()
    out = d.compute_one(SEQ_ALL_K)
    # All K → negative count = 0, ratio_acidic_basic = neg/pos = 0/6 = 0.0
    assert out["ratio_acidic_basic"] == 0.0
    # ratio_basic_acidic = pos/neg = 6/0 = NaN
    assert math.isnan(out["ratio_basic_acidic"])


def test_registered():
    from roxy.descriptors import DESCRIPTOR_REGISTRY
    assert "grouped_composition" in DESCRIPTOR_REGISTRY


def test_consistent_schema():
    d = GroupedCompositionDescriptor()
    df = d.compute([SEQ, SEQ_ALL_K, ""])
    assert df.shape[0] == 3
    assert len(df.columns) == len(d.compute_one(SEQ))
