"""Tests for AAIndexDescriptor."""

import math

import pytest

from roxy.descriptors.aaindex.aaindex import AAIndexDescriptor, list_indices
from tests.descriptors._helpers import EMPTY, SEQ_ALL20, assert_keys_present

SEQ = SEQ_ALL20


@pytest.fixture
def desc():
    return AAIndexDescriptor()


def test_list_indices():
    codes = list_indices()
    assert len(codes) > 100
    assert "ANDN920101" in codes
    assert "KYTJ820101" in codes


def test_aaindex_bundled():
    """AAIndex data ships with the package and is available to the descriptor."""
    codes = list_indices()
    assert len(codes) == 566


def test_empty_nan(desc):
    out = desc.compute_one(EMPTY)
    assert out["length"] == 0.0
    assert math.isnan(out["ANDN920101_mean"])
    assert math.isnan(out["KYTJ820101_nterm_mean_w10"])


def test_default_features_present(desc):
    out = desc.compute_one(SEQ)
    assert_keys_present(
        out,
        [
            "length",
            "valid_residue_count",
            "ANDN920101_mean",
            "ANDN920101_std",
            "ANDN920101_nterm_mean_w10",
            "KYTJ820101_cterm_mean_w10",
        ],
    )


def test_custom_codes():
    d = AAIndexDescriptor(codes=["ANDN920101"], include_terminal=False)
    out = d.compute_one(SEQ)
    assert "ANDN920101_mean" in out
    assert "ANDN920101_nterm_mean_w10" not in out
    assert "ANDN920101_std" in out


def test_invalid_code():
    with pytest.raises(ValueError, match="Unknown AAIndex code"):
        AAIndexDescriptor(codes=["INVALID_CODE"])
