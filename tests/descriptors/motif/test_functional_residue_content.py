"""Tests for FunctionalResidueContentDescriptor."""

import math

import pytest

from roxy.descriptors.motif.functional_residue_content import FunctionalResidueContentDescriptor

SEQ = "ACDEFGHIKLMNPQRSTVWY"
EMPTY = ""


@pytest.fixture
def desc():
    return FunctionalResidueContentDescriptor()


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
