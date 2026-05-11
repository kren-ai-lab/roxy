"""Tests for TerminalDescriptor."""

import math

import pytest

from roxy.descriptors.positional.terminal import TerminalDescriptor
from tests.descriptors._helpers import EMPTY, SEQ_ALL20

SEQ = SEQ_ALL20
SHORT = "ACDE"  # shorter than window=5


@pytest.fixture
def desc():
    return TerminalDescriptor()


def test_empty_nan(desc):
    out = desc.compute_one(EMPTY)
    assert out["length"] == 0.0
    assert math.isnan(out["nterm5_hydropathy_mean"])
    assert math.isnan(out["cterm20_entropy"])


def test_short_seq_returns_full(desc):
    # seq shorter than window → terminal window = full seq
    out = desc.compute_one(SHORT)
    assert out["nterm5_length"] == float(len(SHORT))
    assert out["cterm5_length"] == float(len(SHORT))


def test_window_columns_present(desc):
    out = desc.compute_one(SEQ)
    for w in (5, 10, 20):
        for side in ("nterm", "cterm"):
            assert f"{side}{w}_length" in out
            assert f"{side}{w}_entropy" in out
            assert f"{side}{w}_hydrophobic_frac" in out
            assert f"{side}{w}_aac_A" in out


def test_aac_sums_to_one(desc):
    out = desc.compute_one(SEQ)
    for w in (5, 10, 20):
        for side in ("nterm", "cterm"):
            s = sum(out[f"{side}{w}_aac_{aa}"] for aa in "ACDEFGHIKLMNPQRSTVWY")
            assert math.isclose(s, 1.0, abs_tol=1e-9)


def test_custom_windows():
    desc = TerminalDescriptor(window_sizes=(3,))
    out = desc.compute_one(SEQ)
    assert "nterm3_length" in out
    assert "nterm5_length" not in out
