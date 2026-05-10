"""Tests for CompositionalBiasDescriptor."""

from __future__ import annotations

import math

from roxy.descriptors.composition.bias import CompositionalBiasDescriptor

SEQ = "ACDEFGHIKLMNPQRSTVWY"
SEQ_ALL_K = "KKKKKKKKKK"


def test_smoke():
    d = CompositionalBiasDescriptor()
    df = d.compute([SEQ])
    assert df.shape[0] == 1
    assert "compositional_bias_gini_like_inequality" in df.columns


def test_empty_sequence_full_schema():
    d = CompositionalBiasDescriptor()
    out_empty = d.compute_one("")
    out_seq = d.compute_one(SEQ)
    assert set(out_empty.keys()) == set(out_seq.keys())
    assert out_empty["length"] == 0
    assert math.isnan(out_empty["usage_mean"])
    assert math.isnan(out_empty["gini_like_inequality"])


def test_usage_mean_uniform():
    d = CompositionalBiasDescriptor()
    out = d.compute_one(SEQ)
    # Each of 20 AAs appears once → freq = 1/20
    assert math.isclose(out["usage_mean"], 1 / 20, abs_tol=1e-9)


def test_skew_all_positive():
    d = CompositionalBiasDescriptor()
    out = d.compute_one(SEQ_ALL_K)
    # All K → all positive, no negative → positive_negative_skew > 0
    assert out["positive_negative_skew"] > 0
    assert math.isnan(out["positive_negative_ratio"])


def test_kl_div_uniform_for_uniform_seq():
    d = CompositionalBiasDescriptor()
    out = d.compute_one(SEQ)
    # Perfect uniform → KL = 0
    assert math.isclose(out["kl_div_uniform"], 0.0, abs_tol=1e-9)


def test_registered():
    from roxy.descriptors import DESCRIPTOR_REGISTRY
    assert "compositional_bias" in DESCRIPTOR_REGISTRY


def test_consistent_schema():
    d = CompositionalBiasDescriptor()
    df = d.compute([SEQ, SEQ_ALL_K, ""])
    assert df.shape[0] == 3
    assert len(df.columns) == len(d.compute_one(SEQ))
