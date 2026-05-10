"""Tests for TerminalDescriptor."""

import math

import pytest

from roxy.descriptors.terminal.terminal import TerminalDescriptor

SEQ = "ACDEFGHIKLMNPQRSTVWY"
EMPTY = ""
SHORT = "ACDE"  # shorter than window=5


@pytest.fixture
def desc():
    return TerminalDescriptor()


def test_smoke(desc):
    df = desc.compute([SEQ, EMPTY])
    assert df.shape[0] == 2
    assert "terminal_nterm5_hydropathy_mean" in df.columns
    assert "terminal_cterm10_positive_frac" in df.columns


def test_empty_schema_consistent(desc):
    assert set(desc.compute_one(SEQ)) == set(desc.compute_one(EMPTY))


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


def test_registered():
    from roxy.descriptors import DESCRIPTOR_REGISTRY
    assert "terminal" in DESCRIPTOR_REGISTRY
