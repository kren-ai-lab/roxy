"""User-defined regex motif descriptors with positional features."""

from __future__ import annotations

import math
import re

from roxy.descriptors.base import BaseDescriptor
from roxy.descriptors.registry import register

_NAN = math.nan

DEFAULT_PATTERNS: dict[str, str] = {
    "basic_pair": r"KR|RK|KK|RR",
    "acidic_pair": r"DE|ED|DD|EE",
    "gly_run_2plus": r"G{2,}",
    "pro_run_2plus": r"P{2,}",
    "ser_thr_patch": r"[ST]{3,}",
    "charged_triplet": r"[KRHDE]{3,}",
    "aromatic_pair": r"[FWYH]{2,}",
    "amide_pair": r"[NQ]{2,}",
}

_MIN_PAIR = 2


def _compile(pattern: str, *, overlapping: bool) -> re.Pattern[str]:
    if overlapping:
        return re.compile(f"(?=({pattern}))")
    return re.compile(pattern)


def _match_positions(seq: str, compiled: re.Pattern[str]) -> list[int]:
    return [m.start() + 1 for m in compiled.finditer(seq)]


def _norm_pos(pos: float, n: int) -> float:
    if n == 0 or math.isnan(pos):
        return _NAN
    return pos / n


@register("user_regex", family="motif")
class UserRegexDescriptor(BaseDescriptor):
    """User-defined regex motif descriptors with count, density, and positional features.

    For each pattern, computes presence, count, density, first/last position
    (raw and normalised), span, span_norm, and optionally N-/C-terminal presence.

    Args:
        patterns: Mapping of motif name → regex string. Defaults to 8 built-in
            patterns (charged pairs, poly runs, patches).
        overlapping: If True, use lookahead to count overlapping matches. Default False.
        include_terminal: If True, add N-/C-terminal window presence. Default True.
        terminal_window: Size of terminal window in residues. Default 10.

    Output columns (prefix ``user_regex_``):
        ``length``, ``valid_residue_count``,
        per pattern: 9 stats + (2 terminal if include_terminal=True).
        Total (defaults): 2 + 8*11 = 90 columns.

    """

    def __init__(
        self,
        *,
        patterns: dict[str, str] | None = None,
        overlapping: bool = False,
        include_terminal: bool = True,
        terminal_window: int = 10,
    ) -> None:
        """Initialize UserRegexDescriptor."""
        self.patterns = patterns if patterns is not None else dict(DEFAULT_PATTERNS)
        self.overlapping = overlapping
        self.include_terminal = include_terminal
        self.terminal_window = terminal_window
        self._compiled = {name: _compile(p, overlapping=overlapping) for name, p in self.patterns.items()}

    def _nan_schema(self, feats: dict[str, float]) -> dict[str, float]:
        w = self.terminal_window
        for name in self.patterns:
            feats[f"{name}_present"] = 0.0
            feats[f"{name}_count"] = 0.0
            feats[f"{name}_density"] = _NAN
            feats[f"{name}_first_pos"] = _NAN
            feats[f"{name}_last_pos"] = _NAN
            feats[f"{name}_first_pos_norm"] = _NAN
            feats[f"{name}_last_pos_norm"] = _NAN
            feats[f"{name}_span"] = _NAN
            feats[f"{name}_span_norm"] = _NAN
            if self.include_terminal:
                feats[f"{name}_nterm_present_w{w}"] = 0.0
                feats[f"{name}_cterm_present_w{w}"] = 0.0
        return feats

    def compute_one(self, sequence: str) -> dict[str, float]:
        """Compute regex motif features for a single sequence."""
        seq, n, feats = self._prepare_sequence(sequence)

        if n == 0:
            return self._nan_schema(feats)

        w = self.terminal_window
        for name, compiled in self._compiled.items():
            pos = _match_positions(seq, compiled)
            count = len(pos)
            first = float(pos[0]) if pos else _NAN
            last = float(pos[-1]) if pos else _NAN
            span = float(pos[-1] - pos[0]) if len(pos) >= _MIN_PAIR else _NAN

            feats[f"{name}_present"] = float(count > 0)
            feats[f"{name}_count"] = float(count)
            feats[f"{name}_density"] = count / n
            feats[f"{name}_first_pos"] = first
            feats[f"{name}_last_pos"] = last
            feats[f"{name}_first_pos_norm"] = _norm_pos(first, n)
            feats[f"{name}_last_pos_norm"] = _norm_pos(last, n)
            feats[f"{name}_span"] = span
            feats[f"{name}_span_norm"] = _norm_pos(span, n)

            if self.include_terminal:
                n_sub = seq[:w]
                c_sub = seq[-w:]
                feats[f"{name}_nterm_present_w{w}"] = float(bool(compiled.search(n_sub)))
                feats[f"{name}_cterm_present_w{w}"] = float(bool(compiled.search(c_sub)))

        return feats
