"""Tests for HydrophobicityDescriptor."""

import math

import pytest

from roxy.descriptors.physicochemical.hydrophobicity import HydrophobicityDescriptor

SEQ = "ACDEFGHIKLMNPQRSTVWY"
EMPTY = ""
HYDROPHOBIC_SEQ = "AVLIMFWCY"
HYDROPHILIC_SEQ = "RNDQEHKST"


@pytest.fixture
def desc():
    return HydrophobicityDescriptor()


def test_smoke(desc):
    df = desc.compute([SEQ, EMPTY])
    assert df.shape[0] == 2
    assert "hydrophobicity_length" in df.columns
    assert "hydrophobicity_hydropathy_mean" in df.columns


def test_empty_schema_consistent(desc):
    out_full = desc.compute_one(SEQ)
    out_empty = desc.compute_one(EMPTY)
    assert set(out_full.keys()) == set(out_empty.keys())


def test_empty_nan_values(desc):
    out = desc.compute_one(EMPTY)
    assert out["length"] == 0.0
    assert math.isnan(out["hydropathy_mean"])
    assert math.isnan(out["w5_hydropathy_mean"])
    assert math.isnan(out["w5_hydrophobic_patch_fraction"])


def test_hydrophobic_seq_high_balance(desc):
    out = desc.compute_one(HYDROPHOBIC_SEQ)
    assert out["hydrophobic_hydrophilic_balance"] > 0


def test_hydrophilic_seq_low_balance(desc):
    out = desc.compute_one(HYDROPHILIC_SEQ)
    assert out["hydrophobic_hydrophilic_balance"] < 0


def test_window_columns_present(desc):
    out = desc.compute_one(SEQ)
    for ws in (5, 7, 9):
        assert f"w{ws}_hydropathy_mean" in out
        assert f"w{ws}_amphipathicity_amplitude" in out
        assert f"w{ws}_hydrophobic_patch_fraction" in out


def test_custom_window_sizes():
    desc = HydrophobicityDescriptor(window_sizes=(3,), terminal_window=5)
    out = desc.compute_one(SEQ)
    assert "w3_hydropathy_mean" in out
    assert "w5_hydropathy_mean" not in out


def test_amphipathicity_proxy_nonneg(desc):
    out = desc.compute_one(SEQ)
    assert out["global_amphipathicity_proxy"] >= 0


def test_registered():
    from roxy.descriptors import DESCRIPTOR_REGISTRY
    assert "hydrophobicity" in DESCRIPTOR_REGISTRY
