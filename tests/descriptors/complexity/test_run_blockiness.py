"""Tests for RunBlockinessDescriptor."""

import math

import pytest

from roxy.descriptors.complexity.run_blockiness import RunBlockinessDescriptor
from tests.descriptors._helpers import EMPTY, SEQ_ALL20, assert_keys_present

SEQ = SEQ_ALL20
POLY_K = "KKKKACDE"  # charged run of length 4


@pytest.fixture
def desc():
    return RunBlockinessDescriptor()


def test_empty_nan(desc):
    out = desc.compute_one(EMPTY)
    assert out["length"] == 0.0
    assert math.isnan(out["homopolymer_run_density"])
    assert math.isnan(out["charged_mean_run_length"])


def test_run_feature_groups_present(desc):
    out = desc.compute_one(SEQ)
    assert_keys_present(
        out,
        [
            "longest_homopolymer",
            "homopolymer_run_density",
            "charged_longest",
            "hydrophobic_longest_norm",
            "aromatic_mean_run_length",
        ],
    )


def test_poly_k_charged_run(desc):
    out = desc.compute_one(POLY_K)
    assert out["charged_longest"] == 4.0


def test_norm_in_range(desc):
    out = desc.compute_one(SEQ)
    for name in ("charged", "hydrophobic"):
        v = out[f"{name}_longest_norm"]
        if not math.isnan(v):
            assert 0.0 <= v <= 1.0
    v = out["switching_freq"] if "switching_freq" in out else out["charged_switching_freq"]
    if not math.isnan(v):
        assert 0.0 <= v <= 1.0
