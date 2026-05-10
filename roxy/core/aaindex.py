"""AAIndex utilities for sequence-level descriptors.

The bundled CSV (``roxy/data/aaindex.csv``) contains 566 indices × 20 AAs
in wide format: rows = index codes, columns = one-letter AA codes.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from importlib.resources import files

import pandas as pd

from .constants import AA20
from .exceptions import AAIndexError

_AAINDEX_TABLE: pd.DataFrame | None = None


def load_aaindex() -> pd.DataFrame:
    """Load the bundled AAIndex table (566 indices × 20 AAs).

    Returns
    -------
    pandas.DataFrame
        Rows = AAIndex codes, columns = one-letter AA codes.
        Result is module-level cached after first call.

    Raises
    ------
    AAIndexError
        If the bundled CSV cannot be read.

    """
    global _AAINDEX_TABLE  # noqa: PLW0603,RUF100

    if _AAINDEX_TABLE is not None:
        return _AAINDEX_TABLE

    try:
        path = files("roxy").joinpath("data/aaindex.csv")
        df = pd.read_csv(path, index_col=0)
    except Exception as exc:
        msg = f"Could not load bundled AAIndex data: {exc}"
        raise AAIndexError(msg) from exc

    _AAINDEX_TABLE = df
    return df


def available_indices() -> list[str]:
    """Return the list of all AAIndex codes in the bundled table."""
    return list(load_aaindex().index)


def compute_aaindex_means(
    seq: str,
    index_codes: Iterable[str],
    table: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Compute mean AAIndex values over a sequence for given index codes.

    Parameters
    ----------
    seq:
        Amino-acid sequence (one-letter codes).
    index_codes:
        AAIndex identifiers to compute (e.g. ``["ANDN920101", "ARGP820101"]``).
    table:
        Optional pre-loaded AAIndex table. Loaded from bundle if ``None``.

    Returns
    -------
    dict
        ``{"aaindex_<CODE>_mean": float, ...}``

    """
    s = (seq or "").strip().upper().replace("*", "")
    if not s:
        return {f"aaindex_{code}_mean": float("nan") for code in index_codes}

    if table is None:
        table = load_aaindex()

    feats: dict[str, float] = {}
    for code in index_codes:
        if code not in table.index:
            warnings.warn(
                f"AAIndex code {code!r} not in bundled table; returning NaN.",
                RuntimeWarning,
                stacklevel=2,
            )
            feats[f"aaindex_{code}_mean"] = float("nan")
            continue

        scale = table.loc[code]
        vals = [scale[aa] for aa in s if aa in AA20 and not pd.isna(scale[aa])]
        feats[f"aaindex_{code}_mean"] = float(sum(vals) / len(vals)) if vals else float("nan")

    return feats


__all__ = ["available_indices", "compute_aaindex_means", "load_aaindex"]
