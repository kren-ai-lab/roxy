"""Tests for GlobalBasicDescriptor."""

import math

import pytest

from roxy.descriptors.physicochemical.global_basic import GlobalBasicDescriptor

SEQ = "ACDEFGHIKLMNPQRSTVWY"
SEQ_REPEATED = "AAACCCDDDEEE"
EMPTY = ""


@pytest.fixture
def desc():
    return GlobalBasicDescriptor()


def test_smoke(desc):
    df = desc.compute([SEQ, SEQ_REPEATED, EMPTY])
    assert df.shape[0] == 3
    assert "global_basic_length" in df.columns
    assert "global_basic_molecular_weight" in df.columns


def test_length(desc):
    out = desc.compute_one(SEQ)
    assert out["length"] == len(SEQ)
    assert out["valid_residue_count"] == len(SEQ)


def test_empty_schema_consistent(desc):
    out_full = desc.compute_one(SEQ)
    out_empty = desc.compute_one(EMPTY)
    assert set(out_full.keys()) == set(out_empty.keys())


def test_empty_nan_values(desc):
    out = desc.compute_one(EMPTY)
    assert out["length"] == 0.0
    assert math.isnan(out["molecular_weight"])
    assert math.isnan(out["hydropathy_mean"])
    assert math.isnan(out["net_charge_ph7"])
    assert math.isnan(out["shannon_entropy"])
    assert out["longest_homopolymer_run"] == 0.0


def test_molecular_weight_positive(desc):
    out = desc.compute_one(SEQ)
    assert out["molecular_weight"] > 0


def test_group_fractions_sum_le_one(desc):
    out = desc.compute_one(SEQ)
    # positive + negative ≤ charged
    assert out["positive_fraction"] + out["negative_fraction"] <= out["charged_fraction"] + 1e-9


def test_all_aa_fraction_full_alphabet(desc):
    out = desc.compute_one(SEQ)
    # SEQ has all 20 AAs → unique = 20
    assert out["unique_residue_count"] == 20.0


def test_net_charge_range(desc):
    out = desc.compute_one(SEQ)
    # Reasonable bounds
    assert -20 < out["net_charge_ph7"] < 20


def test_linguistic_complexity_bounds(desc):
    out = desc.compute_one(SEQ)
    for k in (1, 2, 3):
        val = out[f"linguistic_complexity_k{k}"]
        assert 0.0 <= val <= 1.0


def test_shannon_entropy_positive(desc):
    out = desc.compute_one(SEQ)
    assert out["shannon_entropy"] > 0


def test_registered():
    from roxy.descriptors import DESCRIPTOR_REGISTRY
    assert "global_basic" in DESCRIPTOR_REGISTRY
