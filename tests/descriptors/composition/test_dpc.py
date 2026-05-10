"""Tests for DPCDescriptor."""

from __future__ import annotations

import math

from roxy.descriptors.composition.dpc import DPCDescriptor

SEQ = "ACDEFGHIKLMNPQRSTVWY"
SEQ_SHORT = "A"


def test_smoke():
    d = DPCDescriptor()
    df = d.compute([SEQ])
    assert df.shape[0] == 1
    assert "dpc_freq_AC" in df.columns


def test_freq_sums_to_one():
    d = DPCDescriptor()
    out = d.compute_one(SEQ * 5)
    s = out["frequency_sum"]
    assert math.isclose(s, 1.0, abs_tol=1e-9)


def test_total_dipeptides():
    d = DPCDescriptor()
    out = d.compute_one(SEQ)
    assert out["total_dipeptides"] == len(SEQ) - 1


def test_empty_sequence():
    d = DPCDescriptor()
    out = d.compute_one("")
    assert out["length"] == 0
    assert out["total_dipeptides"] == 0
    assert math.isnan(out["freq_AC"])
    assert math.isnan(out["frequency_sum"])


def test_short_sequence():
    d = DPCDescriptor()
    out = d.compute_one(SEQ_SHORT)
    assert out["total_dipeptides"] == 0
    assert math.isnan(out["freq_AA"])


def test_registered():
    from roxy.descriptors import DESCRIPTOR_REGISTRY
    assert "dpc" in DESCRIPTOR_REGISTRY


def test_consistent_schema():
    d = DPCDescriptor()
    df = d.compute([SEQ, SEQ * 5, ""])
    assert df.shape[0] == 3
    assert len(df.columns) == len(d.compute_one(SEQ))
