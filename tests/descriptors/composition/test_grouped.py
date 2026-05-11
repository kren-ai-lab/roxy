"""Tests for GroupedCompositionDescriptor."""

from __future__ import annotations

import math

from roxy.descriptors.composition.grouped import GroupedCompositionDescriptor
from tests.descriptors._helpers import EMPTY, SEQ_ALL20

SEQ = SEQ_ALL20
SEQ_ALL_K = "KKKKKK"


def test_positive_count():
    d = GroupedCompositionDescriptor()
    out = d.compute_one(SEQ_ALL_K)
    assert out["positive_count"] == 6
    assert math.isclose(out["positive_frac"], 1.0, abs_tol=1e-9)


def test_empty_sequence():
    d = GroupedCompositionDescriptor()
    out = d.compute_one(EMPTY)
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


def test_consistent_schema():
    d = GroupedCompositionDescriptor()
    df = d.compute([SEQ, SEQ_ALL_K, EMPTY])
    assert df.shape[0] == 3
    assert set(df.columns) == {f"grouped_{key}" for key in d.compute_one(SEQ)}
