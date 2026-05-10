"""Motif and regex pattern descriptors."""

from __future__ import annotations

import math
import re

from roxy.descriptors.base import BaseDescriptor
from roxy.descriptors.composition._utils import clean_sequence
from roxy.descriptors.registry import register

_NAN = math.nan

PREDEFINED_PATTERNS: dict[str, str] = {
    "CxxC":             r"C.{2}C",
    "PxxP":             r"P.{2}P",
    "polyK_3plus":      r"K{3,}",
    "polyR_3plus":      r"R{3,}",
    "acidic_patch_3plus":  r"[DE]{3,}",
    "basic_patch_3plus":   r"[KRH]{3,}",
    "gly_rich_4plus":      r"(?:G.*){4,}",
    "proline_rich_4plus":  r"(?:P.*){4,}",
    "ser_thr_rich_4plus":  r"(?:[ST].*){4,}",
}


def _count(seq: str, pattern: str) -> int:
    return len(re.findall(pattern, seq))


def _has(seq: str, pattern: str) -> float:
    return float(_count(seq, pattern) > 0)


def _density(seq: str, pattern: str) -> float:
    n = len(seq)
    if n == 0:
        return _NAN
    return _count(seq, pattern) / n


def _terminal_has(seq: str, pattern: str, side: str, window: int) -> float:
    if not seq:
        return 0.0
    sub = seq[:window] if side == "N" else seq[-window:]
    return _has(sub, pattern)


@register("motif", family="motif")
class MotifDescriptor(BaseDescriptor):
    """Regex motif presence, count, density, and terminal occurrence descriptors.

    For each pattern, computes presence (0/1), count, density (count/length),
    and optionally N-/C-terminal presence within a fixed window.

    Args:
        patterns: Mapping of motif name → regex string. Defaults to 9 built-in
            patterns (CxxC, PxxP, polyK/R 3+, acidic/basic patches, Gly/Pro/Ser-Thr
            rich regions).
        include_terminal: If True, add N- and C-terminal window presence. Default True.
        terminal_window: Size of terminal window in residues. Default 10.

    Output columns (prefix ``motif_``):
        ``length``, ``valid_residue_count``,
        per pattern: ``{name}_present``, ``{name}_count``, ``{name}_density``,
        and if include_terminal: ``{name}_nterm_present_w{w}``,
        ``{name}_cterm_present_w{w}``.
        Total (defaults): 2 + 9*5 = 47 columns.

    """

    def __init__(
        self,
        *,
        patterns: dict[str, str] | None = None,
        include_terminal: bool = True,
        terminal_window: int = 10,
    ) -> None:
        """Initialize MotifDescriptor."""
        self.patterns = patterns if patterns is not None else dict(PREDEFINED_PATTERNS)
        self.include_terminal = include_terminal
        self.terminal_window = terminal_window

    def _nan_schema(self, feats: dict[str, float]) -> dict[str, float]:
        w = self.terminal_window
        for name in self.patterns:
            feats[f"{name}_present"] = 0.0
            feats[f"{name}_count"] = 0.0
            feats[f"{name}_density"] = _NAN
            if self.include_terminal:
                feats[f"{name}_nterm_present_w{w}"] = 0.0
                feats[f"{name}_cterm_present_w{w}"] = 0.0
        return feats

    def compute_one(self, sequence: str) -> dict[str, float]:
        """Compute motif features for a single sequence."""
        seq = clean_sequence(sequence)
        n = len(seq)

        feats: dict[str, float] = {
            "length": float(n),
            "valid_residue_count": float(n),
        }

        if n == 0:
            return self._nan_schema(feats)

        w = self.terminal_window
        for name, pattern in self.patterns.items():
            feats[f"{name}_present"] = _has(seq, pattern)
            feats[f"{name}_count"] = float(_count(seq, pattern))
            feats[f"{name}_density"] = _density(seq, pattern)
            if self.include_terminal:
                feats[f"{name}_nterm_present_w{w}"] = _terminal_has(seq, pattern, "N", w)
                feats[f"{name}_cterm_present_w{w}"] = _terminal_has(seq, pattern, "C", w)

        return feats
