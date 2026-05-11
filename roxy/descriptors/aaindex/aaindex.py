"""AAIndex physicochemical scale descriptors."""

from __future__ import annotations

import contextlib
import csv
import math
import statistics
from importlib import resources

from roxy.descriptors._utils import clean_sequence
from roxy.descriptors.base import BaseDescriptor
from roxy.descriptors.registry import register

_NAN = math.nan

_DEFAULT_CODES = ("ANDN920101", "ARGP820101", "KYTJ820101", "FAUJ880104")


def _load_aaindex() -> dict[str, dict[str, float]]:
    """Load AAIndex data from the bundled CSV."""
    text = resources.files("roxy.data").joinpath("aaindex.csv").read_text(encoding="utf-8")

    reader = csv.DictReader(text.splitlines())
    result: dict[str, dict[str, float]] = {}
    for row in reader:
        code = row["index"]
        scale: dict[str, float] = {}
        for aa, val in row.items():
            if aa == "index":
                continue
            with contextlib.suppress(ValueError, TypeError):
                scale[aa] = float(val)
        result[code] = scale
    return result


_AAINDEX: dict[str, dict[str, float]] = _load_aaindex()


def list_indices() -> list[str]:
    """Return sorted list of all available AAIndex codes."""
    return sorted(_AAINDEX)


def _seq_values(seq: str, scale: dict[str, float]) -> list[float]:
    return [scale[aa] for aa in seq if aa in scale and not math.isnan(scale[aa])]


def _seq_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": _NAN, "std": _NAN, "min": _NAN, "max": _NAN, "median": _NAN}
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
        "median": float(statistics.median(values)),
    }


@register("aaindex", family="aaindex")
class AAIndexDescriptor(BaseDescriptor):
    """AAIndex physicochemical scale descriptors.

    For each selected AAIndex scale, computes mean, std, min, max, and median
    of residue-level values, plus optional N-/C-terminal means.

    Args:
        codes: Sequence of AAIndex codes to use. Defaults to 4 representative
            scales (ANDN920101, ARGP820101, KYTJ820101, FAUJ880104).
        include_terminal: If True, add N- and C-terminal mean columns. Default True.
        terminal_window: Terminal window size in residues. Default 10.

    Output columns (prefix ``aaindex_``):
        ``length``, ``valid_residue_count``,
        per code: ``{code}_mean``, ``{code}_std``, ``{code}_min``,
        ``{code}_max``, ``{code}_median``,
        and if include_terminal: ``{code}_nterm_mean_w{w}``,
        ``{code}_cterm_mean_w{w}``.
        Total (defaults): 2 + 4*7 = 30 columns.

    """

    def __init__(
        self,
        *,
        codes: tuple[str, ...] | list[str] | None = None,
        include_terminal: bool = True,
        terminal_window: int = 10,
    ) -> None:
        """Initialize AAIndexDescriptor."""
        self.codes = list(codes) if codes is not None else list(_DEFAULT_CODES)
        self.include_terminal = include_terminal
        self.terminal_window = terminal_window
        for code in self.codes:
            if code not in _AAINDEX:
                msg = f"Unknown AAIndex code: {code!r}. Use aaindex.list_indices() to see available codes."
                raise ValueError(msg)

    def _nan_schema(self, feats: dict[str, float]) -> dict[str, float]:
        w = self.terminal_window
        for code in self.codes:
            feats[f"{code}_mean"] = _NAN
            feats[f"{code}_std"] = _NAN
            feats[f"{code}_min"] = _NAN
            feats[f"{code}_max"] = _NAN
            feats[f"{code}_median"] = _NAN
            if self.include_terminal:
                feats[f"{code}_nterm_mean_w{w}"] = _NAN
                feats[f"{code}_cterm_mean_w{w}"] = _NAN
        return feats

    def compute_one(self, sequence: str) -> dict[str, float]:
        """Compute AAIndex features for a single sequence."""
        seq = clean_sequence(sequence)
        n = len(seq)

        feats: dict[str, float] = {
            "length": float(n),
            "valid_residue_count": float(n),
        }

        if n == 0:
            return self._nan_schema(feats)

        w = self.terminal_window
        for code in self.codes:
            scale = _AAINDEX[code]
            values = _seq_values(seq, scale)
            summary = _seq_summary(values)
            feats[f"{code}_mean"] = summary["mean"]
            feats[f"{code}_std"] = summary["std"]
            feats[f"{code}_min"] = summary["min"]
            feats[f"{code}_max"] = summary["max"]
            feats[f"{code}_median"] = summary["median"]

            if self.include_terminal:
                n_vals = _seq_values(seq[:w], scale)
                c_vals = _seq_values(seq[-w:], scale)
                feats[f"{code}_nterm_mean_w{w}"] = float(statistics.fmean(n_vals)) if n_vals else _NAN
                feats[f"{code}_cterm_mean_w{w}"] = float(statistics.fmean(c_vals)) if c_vals else _NAN

        return feats
