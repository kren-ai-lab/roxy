"""Tests for EntropyComplexityDescriptor."""

import math

import pytest

from roxy.descriptors.complexity.entropy_complexity import EntropyComplexityDescriptor
from tests.descriptors._helpers import EMPTY, SEQ_ALL20

SEQ = SEQ_ALL20
UNIFORM = "AAAAAAAAAA"


@pytest.fixture
def desc():
    return EntropyComplexityDescriptor()


def test_empty_values(desc):
    out = desc.compute_one(EMPTY)
    assert out["length"] == 0.0
    assert math.isnan(out["shannon_entropy"])
    assert out["longest_homopolymer_run"] == 0.0


def test_uniform_high_homopolymer(desc):
    out = desc.compute_one(UNIFORM)
    assert out["longest_homopolymer_run"] == float(len(UNIFORM))
    assert out["most_frequent_residue_fraction"] == 1.0


def test_uniform_zero_entropy(desc):
    out = desc.compute_one(UNIFORM)
    assert math.isclose(out["shannon_entropy"], 0.0, abs_tol=1e-9)


def test_diverse_high_entropy(desc):
    out = desc.compute_one(SEQ)
    assert out["shannon_entropy"] > 3.0  # ~4.32 for 20 equal AA


def test_linguistic_complexity_bounds(desc):
    out = desc.compute_one(SEQ)
    for k in (1, 2, 3):
        val = out[f"linguistic_complexity_k{k}"]
        assert 0.0 < val <= 1.0


def test_window_columns_present(desc):
    out = desc.compute_one(SEQ)
    assert "low_complexity_window_fraction_w5" in out
    assert "window_entropy_mean_w5" in out


def test_custom_window():
    desc = EntropyComplexityDescriptor(low_complexity_window=7)
    out = desc.compute_one(SEQ)
    assert "low_complexity_window_fraction_w7" in out
    assert "low_complexity_window_fraction_w5" not in out
