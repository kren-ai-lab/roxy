"""Tests for FunctionalResidueDescriptor."""

import math

import pytest

from roxy.descriptors.functional_residue.functional_residue import FunctionalResidueDescriptor

SEQ = "ACDEFGHIKLMNPQRSTVWY"
EMPTY = ""


@pytest.fixture
def desc():
    return FunctionalResidueDescriptor()


def test_smoke(desc):
    df = desc.compute([SEQ, EMPTY])
    assert df.shape[0] == 2
    assert "functional_residue_catalytic_core_like_fraction" in df.columns
    assert "functional_residue_H_fraction" in df.columns
    assert "functional_residue_hbond_donors_per_residue" in df.columns


def test_empty_schema_consistent(desc):
    assert set(desc.compute_one(SEQ)) == set(desc.compute_one(EMPTY))


def test_empty_nan(desc):
    out = desc.compute_one(EMPTY)
    assert out["length"] == 0.0
    assert out["catalytic_core_like_count"] == 0.0
    assert math.isnan(out["catalytic_core_like_fraction"])
    assert math.isnan(out["basic_acidic_ratio"])
    assert math.isnan(out["hbond_donors_per_residue"])


def test_column_count(desc):
    out = desc.compute_one(SEQ)
    # 2 + 32 + 30 + 7 + 4 + 3 = 78
    assert len(out) == 78


def test_fractions_in_range(desc):
    out = desc.compute_one(SEQ)
    for name in ("catalytic_core_like", "aromatic_pi", "phosphorylation_prone_proxy"):
        v = out[f"{name}_fraction"]
        if not math.isnan(v):
            assert 0.0 <= v <= 1.0


def test_hbond_balance(desc):
    out = desc.compute_one(SEQ)
    expected = out["hbond_donors_per_residue"] - out["hbond_acceptors_per_residue"]
    assert math.isclose(out["hbond_balance"], expected, abs_tol=1e-9)


def test_registered():
    from roxy.descriptors import DESCRIPTOR_REGISTRY

    assert "functional_residue" in DESCRIPTOR_REGISTRY
