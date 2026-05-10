"""Tests for SlidingWindowDescriptor."""

import math

import pytest

from roxy.descriptors.sliding_window.sliding_window import SlidingWindowDescriptor

SEQ = "ACDEFGHIKLMNPQRSTVWY"
EMPTY = ""
SHORT = "ACK"  # shorter than window=5


@pytest.fixture
def desc():
    return SlidingWindowDescriptor()


def test_smoke(desc):
    df = desc.compute([SEQ, EMPTY])
    assert df.shape[0] == 2
    assert "sliding_window_win5_hydropathy_mean" in df.columns
    assert "sliding_window_win9_charged_frac_std" in df.columns


def test_empty_schema_consistent(desc):
    assert set(desc.compute_one(SEQ)) == set(desc.compute_one(EMPTY))


def test_empty_nan(desc):
    out = desc.compute_one(EMPTY)
    assert out["length"] == 0.0
    assert math.isnan(out["win5_hydropathy_mean"])
    assert math.isnan(out["win9_entropy_std"])


def test_short_seq_all_nan(desc):
    # seq shorter than min window → all window features NaN
    out = desc.compute_one(SHORT)
    assert math.isnan(out["win5_hydropathy_mean"])


def test_column_count(desc):
    out = desc.compute_one(SEQ)
    # 2 base + 3 windows x (5 profiles x 6 stats + 4 threshold) = 2 + 3*34 = 104
    assert len(out) == 104


def test_threshold_fractions_in_range(desc):
    out = desc.compute_one(SEQ)
    for w in (5, 7, 9):
        for key in (
            f"win{w}_hydropathy_high_fraction",
            f"win{w}_charged_high_fraction",
            f"win{w}_aromatic_high_fraction",
            f"win{w}_low_entropy_fraction",
        ):
            v = out[key]
            if not math.isnan(v):
                assert 0.0 <= v <= 1.0


def test_custom_windows():
    desc = SlidingWindowDescriptor(window_sizes=(3,))
    out = desc.compute_one(SEQ)
    assert "win3_hydropathy_mean" in out
    assert "win5_hydropathy_mean" not in out


def test_registered():
    from roxy.descriptors import DESCRIPTOR_REGISTRY

    assert "sliding_window" in DESCRIPTOR_REGISTRY
