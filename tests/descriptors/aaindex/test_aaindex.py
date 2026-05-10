"""Tests for AAIndexDescriptor."""

import math

import pytest

from roxy.descriptors.aaindex.aaindex import AAIndexDescriptor, list_indices

SEQ = "ACDEFGHIKLMNPQRSTVWY"
EMPTY = ""


@pytest.fixture
def desc():
    return AAIndexDescriptor()


def test_list_indices():
    codes = list_indices()
    assert len(codes) > 100
    assert "ANDN920101" in codes
    assert "KYTJ820101" in codes


def test_empty_nan(desc):
    out = desc.compute_one(EMPTY)
    assert out["length"] == 0.0
    assert math.isnan(out["ANDN920101_mean"])
    assert math.isnan(out["KYTJ820101_nterm_mean_w10"])


def test_column_count(desc):
    out = desc.compute_one(SEQ)
    # 2 base + 4 codes x 7 stats = 30
    assert len(out) == 30


def test_custom_codes():
    d = AAIndexDescriptor(codes=["ANDN920101"], include_terminal=False)
    out = d.compute_one(SEQ)
    assert "ANDN920101_mean" in out
    assert "ANDN920101_nterm_mean_w10" not in out
    # 2 + 1 x 5 = 7
    assert len(out) == 7


def test_invalid_code():
    with pytest.raises(ValueError, match="Unknown AAIndex code"):
        AAIndexDescriptor(codes=["INVALID_CODE"])
