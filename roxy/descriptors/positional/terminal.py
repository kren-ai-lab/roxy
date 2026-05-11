"""Terminal region descriptors."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from roxy.core.constants import AA20, AA_GROUPS, KD, POLARITY
from roxy.descriptors.base import BaseDescriptor
from roxy.descriptors.composition._utils import clean_sequence
from roxy.descriptors.registry import register

_NAN = math.nan
_AA20_LIST: list[str] = sorted(AA20)


def _shannon_entropy(seq: str) -> float:
    if not seq:
        return _NAN
    counts = Counter(seq)
    total = len(seq)
    probs = np.array([c / total for c in counts.values()], dtype=float)
    return float(-(probs * np.log2(probs)).sum())


def _scale_mean(seq: str, scale: dict[str, float]) -> float:
    if not seq:
        return _NAN
    return float(np.mean([scale[aa] for aa in seq if aa in scale]))


def _feats_for_window(tsq: str, prefix: str) -> dict[str, float]:
    """Compute all features for one terminal window."""
    n = len(tsq)
    out: dict[str, float] = {f"{prefix}_length": float(n)}

    out[f"{prefix}_hydropathy_mean"] = _scale_mean(tsq, KD)
    out[f"{prefix}_polarity_mean"] = _scale_mean(tsq, POLARITY)
    out[f"{prefix}_entropy"] = _shannon_entropy(tsq)

    for name, group in AA_GROUPS.items():
        out[f"{prefix}_{name}_frac"] = (
            sum(aa in group for aa in tsq) / n if n else _NAN
        )

    pos_frac = out[f"{prefix}_positive_frac"]
    neg_frac = out[f"{prefix}_negative_frac"]
    hb_frac = out[f"{prefix}_hydrophobic_frac"]
    hl_frac = out[f"{prefix}_hydrophilic_frac"]

    out[f"{prefix}_positive_negative_balance"] = (
        pos_frac - neg_frac
        if not (math.isnan(pos_frac) or math.isnan(neg_frac))
        else _NAN
    )
    out[f"{prefix}_hydrophobic_hydrophilic_balance"] = (
        hb_frac - hl_frac
        if not (math.isnan(hb_frac) or math.isnan(hl_frac))
        else _NAN
    )

    if n:
        pos_n = sum(aa in AA_GROUPS["positive"] for aa in tsq)
        neg_n = sum(aa in AA_GROUPS["negative"] for aa in tsq)
        out[f"{prefix}_positive_negative_ratio"] = pos_n / neg_n if neg_n else _NAN
    else:
        out[f"{prefix}_positive_negative_ratio"] = _NAN

    counts = Counter(tsq)
    for aa in _AA20_LIST:
        out[f"{prefix}_aac_{aa}"] = (counts.get(aa, 0) / n) if n else _NAN

    return out


@register("terminal", family="positional")
class TerminalDescriptor(BaseDescriptor):
    """N- and C-terminal region features at multiple window sizes.

    For each window size and each terminus, computes scale means, entropy,
    amino acid composition, group fractions, and charge/hydrophobicity balances.

    Args:
        window_sizes: Tuple of residue counts defining terminal windows.

    Output columns (prefix ``terminal_``):
        ``length``, ``valid_residue_count``,
        per window x side: ``{side}term{N}_length``,
        ``{side}term{N}_hydropathy/polarity_mean``,
        ``{side}term{N}_entropy``,
        ``{side}term{N}_{group}_frac`` for 17 AA_GROUPS,
        ``{side}term{N}_positive_negative_balance/ratio``,
        ``{side}term{N}_hydrophobic_hydrophilic_balance``,
        ``{side}term{N}_aac_{AA}`` for 20 AAs.

    """

    def __init__(
        self,
        *,
        window_sizes: tuple[int, ...] = (5, 10, 20),
    ) -> None:
        """Initialize TerminalDescriptor."""
        self.window_sizes = tuple(window_sizes)

    def _nan_schema(self, feats: dict[str, float]) -> dict[str, float]:
        for w in self.window_sizes:
            for side in ("n", "c"):
                prefix = f"{side}term{w}"
                feats[f"{prefix}_length"] = 0.0
                feats[f"{prefix}_hydropathy_mean"] = _NAN
                feats[f"{prefix}_polarity_mean"] = _NAN
                feats[f"{prefix}_entropy"] = _NAN
                for name in AA_GROUPS:
                    feats[f"{prefix}_{name}_frac"] = _NAN
                feats[f"{prefix}_positive_negative_balance"] = _NAN
                feats[f"{prefix}_hydrophobic_hydrophilic_balance"] = _NAN
                feats[f"{prefix}_positive_negative_ratio"] = _NAN
                for aa in _AA20_LIST:
                    feats[f"{prefix}_aac_{aa}"] = _NAN
        return feats

    def compute_one(self, sequence: str) -> dict[str, float]:
        """Compute terminal region features for a single sequence."""
        seq = clean_sequence(sequence)
        n = len(seq)

        feats: dict[str, float] = {
            "length": float(n),
            "valid_residue_count": float(n),
        }

        if n == 0:
            return self._nan_schema(feats)

        for w in self.window_sizes:
            nterm = seq[:w]
            cterm = seq[-w:]
            feats.update(_feats_for_window(nterm, f"nterm{w}"))
            feats.update(_feats_for_window(cterm, f"cterm{w}"))

        return feats
