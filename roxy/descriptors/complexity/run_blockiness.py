"""Run-length and blockiness descriptors for residue groups."""

from __future__ import annotations

import math
from itertools import groupby

import numpy as np

from roxy.core.constants import AA_GROUPS
from roxy.descriptors._utils import clean_sequence
from roxy.descriptors.base import BaseDescriptor
from roxy.descriptors.registry import register

_NAN = math.nan
_MIN_RUN_2 = 2
_MIN_RUN_3 = 3

_TRACKED_GROUPS = (
    "charged",
    "hydrophobic",
    "polar",
    "aromatic",
    "disorder_promoting",
    "order_promoting",
)


def _binary(seq: str, group: frozenset[str]) -> list[int]:
    return [1 if aa in group else 0 for aa in seq]


def _run_lengths(binary: list[int]) -> list[int]:
    return [len(list(g)) for v, g in groupby(binary) if v == 1]


def _longest_run(binary: list[int]) -> int:
    runs = _run_lengths(binary)
    return max(runs) if runs else 0


def _n_runs(binary: list[int]) -> int:
    return len(_run_lengths(binary))


def _mean_run(binary: list[int]) -> float:
    runs = _run_lengths(binary)
    return float(np.mean(runs)) if runs else _NAN


def _frac_in_long_runs(binary: list[int], seq_len: int, min_run: int) -> float:
    if seq_len == 0:
        return _NAN
    runs = _run_lengths(binary)
    residues = sum(r for r in runs if r >= min_run)
    return residues / seq_len


def _blockiness(binary: list[int]) -> float:
    total_hits = sum(binary)
    if total_hits == 0:
        return _NAN
    runs = _run_lengths(binary)
    if not runs:
        return _NAN
    return float(np.mean(runs) / total_hits)


def _switching_freq(binary: list[int]) -> float:
    if len(binary) < _MIN_RUN_2:
        return _NAN
    switches = sum(binary[i] != binary[i + 1] for i in range(len(binary) - 1))
    return switches / (len(binary) - 1)


@register("run_blockiness", family="complexity")
class RunBlockinessDescriptor(BaseDescriptor):
    """Run-length and blockiness statistics for residue groups.

    Computes homopolymer runs at the sequence level, then per-group:
    longest/count/mean/normalised run, density, fraction in long runs
    (min length 2 and 3), blockiness score, and switching frequency.

    Output columns (prefix ``run_blockiness_``):
        ``length``, ``valid_residue_count``,
        4 global homopolymer features,
        per group (6): 9 statistics each.
        Total: 2 + 4 + 6*9 = 60 columns.
    """

    def compute_one(self, sequence: str) -> dict[str, float]:
        """Compute run-blockiness features for a single sequence."""
        seq = clean_sequence(sequence)
        n = len(seq)

        # global homopolymer features
        homo_runs = [len(list(g)) for _, g in groupby(seq)] if seq else []
        longest_homo = max(homo_runs) if homo_runs else 0
        homo_density = len(homo_runs) / n if n > 0 else _NAN
        rbf2 = sum(r for r in homo_runs if r >= _MIN_RUN_2) / n if n > 0 else _NAN
        rbf3 = sum(r for r in homo_runs if r >= _MIN_RUN_3) / n if n > 0 else _NAN

        feats: dict[str, float] = {
            "length": float(n),
            "valid_residue_count": float(n),
            "longest_homopolymer": float(longest_homo),
            "homopolymer_run_density": homo_density,
            "repeated_block_fraction_len2": rbf2,
            "repeated_block_fraction_len3": rbf3,
        }

        for name in _TRACKED_GROUPS:
            group = frozenset(AA_GROUPS[name])
            binary = _binary(seq, group)
            longest = _longest_run(binary)

            feats[f"{name}_longest"] = float(longest)
            feats[f"{name}_n_runs"] = float(_n_runs(binary))
            feats[f"{name}_mean_run_length"] = _mean_run(binary)
            feats[f"{name}_longest_norm"] = longest / n if n > 0 else _NAN
            feats[f"{name}_run_density"] = _n_runs(binary) / n if n > 0 else _NAN
            feats[f"{name}_fraction_in_runs_len2"] = _frac_in_long_runs(binary, n, _MIN_RUN_2)
            feats[f"{name}_fraction_in_runs_len3"] = _frac_in_long_runs(binary, n, _MIN_RUN_3)
            feats[f"{name}_blockiness"] = _blockiness(binary)
            feats[f"{name}_switching_freq"] = _switching_freq(binary)

        return feats
