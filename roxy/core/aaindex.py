"""AAIndex handling utilities for sequence-level descriptors.

This module manages the AAIndex CSV file used to derive residue-level
properties for protein sequences. It provides functions to:

- determine the cache directory for Roxy,
- download or update the AAIndex CSV file,
- load the AAIndex table as a pandas DataFrame,
- compute per-sequence mean AAIndex values for one or more indices.

The AAIndex CSV is expected to be in *residue-wise* wide format, with one
row per amino acid and one column per index code, e.g.:

    residue, ANDN920101, ARGP820101, ARGP820102, ...
    A,      4.35,       0.61,      1.18,      ...
    L,      4.17,       1.53,      3.23,      ...
    ...

If your CSV uses a different schema, adapt the loader accordingly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import os
import warnings

import pandas as pd
import requests

from .constants import AA20, AAINDEX_URL, AAINDEX_FILENAME, ROXY_CACHE_SUBDIR


# ---------------------------------------------------------------------------
# Cache directory utilities
# ---------------------------------------------------------------------------

def get_cache_dir() -> Path:
    """Return the Roxy cache directory.

    The default location is ``~/.cache/roxy`` on Unix-like systems,
    and the analogous location on other platforms based on the
    ``XDG_CACHE_HOME`` or user home directory.

    Returns
    -------
    pathlib.Path
        Path to the cache directory. The directory is created if it
        does not already exist.
    """
    base = os.environ.get("XDG_CACHE_HOME", None)
    if base is None:
        base = os.path.join(Path.home(), ".cache")
    cache_dir = Path(base) / ROXY_CACHE_SUBDIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_aaindex_path() -> Path:
    """Return the full path to the AAIndex CSV file in the cache."""
    return get_cache_dir() / AAINDEX_FILENAME


# ---------------------------------------------------------------------------
# Download and update
# ---------------------------------------------------------------------------

def download_aaindex(url: Optional[str] = None, *, force: bool = False) -> Path:
    """Download or update the AAIndex CSV file in the cache directory.

    Parameters
    ----------
    url:
        URL to the AAIndex CSV file. If ``None``, :data:`AAINDEX_URL` is used.
        For robust operation, this should point directly to a CSV file.
    force:
        If ``True``, the file is downloaded even if it already exists.
        If ``False``, the download is skipped when the cached file exists.

    Returns
    -------
    pathlib.Path
        Path to the downloaded CSV file.
    """
    target = get_aaindex_path()
    if target.exists() and not force:
        return target

    use_url = url or AAINDEX_URL
    resp = requests.get(use_url, timeout=60)
    resp.raise_for_status()

    target.write_bytes(resp.content)
    return target


# ---------------------------------------------------------------------------
# Loading and lookup
# ---------------------------------------------------------------------------

_AAINDEX_TABLE: Optional[pd.DataFrame] = None


def load_aaindex(*, auto_download: bool = True) -> pd.DataFrame:
    """Load the AAIndex table from the cache.

    The returned table is indexed by ``residue`` (one-letter code)
    and has one column per AAIndex code.

    Parameters
    ----------
    auto_download:
        If ``True`` and the CSV file is missing, attempt to download it
        using :func:`download_aaindex`.

    Returns
    -------
    pandas.DataFrame
        AAIndex table in wide format (one row per residue, columns for
        each AAIndex code).

    Raises
    ------
    FileNotFoundError
        If the file does not exist and ``auto_download`` is ``False``.
    ValueError
        If the CSV does not have the expected columns.
    """
    global _AAINDEX_TABLE

    if _AAINDEX_TABLE is not None:
        return _AAINDEX_TABLE

    path = get_aaindex_path()
    if not path.exists():
        if not auto_download:
            raise FileNotFoundError(
                f"AAIndex CSV not found at {path!s}. "
                "Call `download_aaindex()` or set `auto_download=True`."
            )
        warnings.warn(
            "AAIndex CSV not found in cache; attempting download from AAINDEX_URL.",
            RuntimeWarning,
        )
        path = download_aaindex()

    df = pd.read_csv(path)
    # Normalise column names:
    df.columns = [c.strip() for c in df.columns]

    # Require a residue identifier column
    if "residue" not in df.columns:
        raise ValueError(
            "AAIndex CSV must contain a 'residue' column with the amino-acid codes."
        )

    # Normalise residue column
    df["residue"] = df["residue"].astype(str).str.strip().str.upper()
    df = df.set_index("residue")

    # Ensure that we have entries for all 20 amino acids
    missing_residues = [aa for aa in AA20 if aa not in df.index]
    if missing_residues:
        raise ValueError(
            "AAIndex CSV is missing rows for residues: "
            f"{', '.join(sorted(missing_residues))}."
        )

    # Optionally, you might want to restrict to AA20 rows only:
    df = df.loc[sorted(AA20)]

    _AAINDEX_TABLE = df
    return df


def compute_aaindex_means_for_sequence(
    seq: str,
    index_codes: Iterable[str],
    table: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """Compute mean AAIndex values for a sequence and a set of indices.

    For each AAIndex code, this function looks up the per-residue values
    in the AAIndex table and computes the mean over the sequence.

    Parameters
    ----------
    seq:
        Amino-acid sequence (one-letter codes).
    index_codes:
        Iterable of AAIndex identifiers (i.e. the column names in the
        AAIndex CSV, such as ``ANDN920101``).
    table:
        Optional AAIndex table. If ``None``, :func:`load_aaindex` is used
        to load the table from cache.

    Returns
    -------
    dict
        Mapping from feature name to mean value. Each key has the form
        ``aaindex_<CODE>_mean``.
    """
    s = (seq or "").strip().upper().replace("*", "")
    L = len(s)
    if L == 0:
        return {f"aaindex_{code}_mean": float("nan") for code in index_codes}

    if table is None:
        table = load_aaindex(auto_download=True)

    feats: Dict[str, float] = {}
    for code in index_codes:
        if code not in table.columns:
            warnings.warn(
                f"AAIndex code {code!r} not found in AAIndex table; "
                "returning NaN for this descriptor.",
                RuntimeWarning,
            )
            feats[f"aaindex_{code}_mean"] = float("nan")
            continue

        # This is a Series indexed by residue (A, C, D, ...)
        scale = table[code]

        vals: List[float] = []
        for aa in s:
            if aa in AA20:
                vals.append(scale[aa])

        if not vals:
            feats[f"aaindex_{code}_mean"] = float("nan")
        else:
            feats[f"aaindex_{code}_mean"] = float(sum(vals) / len(vals))

    return feats


def ensure_aaindex_available() -> None:
    """Ensure that the AAIndex CSV is present in the cache.

    This is a thin convenience wrapper around :func:`load_aaindex` with
    ``auto_download=True``. It can be called at library import time or
    from a post-install hook to guarantee that AAIndex is ready.

    Examples
    --------
    You can import and call this in :mod:`roxy.__init__`:

    >>> from roxy.core.aaindex import ensure_aaindex_available
    >>> ensure_aaindex_available()  # doctest: +SKIP

    or call it explicitly from your own installation scripts.
    """
    _ = load_aaindex(auto_download=True)
