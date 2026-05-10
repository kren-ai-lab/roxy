"""AAIndex handling utilities for sequence-level descriptors.

The AAIndex CSV is expected in residue-wise wide format:

    residue, ANDN920101, ARGP820101, ...
    A,      4.35,       0.61,      ...
    ...
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import requests

from roxy.logging import get_logger

from .config import get_cache_root
from .constants import AA20, AAINDEX_FILENAME, AAINDEX_URL
from .exceptions import AAIndexError

logger = get_logger(__name__)


def get_aaindex_path() -> Path:
    """Return the full path to the AAIndex CSV file in the cache."""
    return get_cache_root() / AAINDEX_FILENAME


# ---------------------------------------------------------------------------
# Download and update
# ---------------------------------------------------------------------------


def download_aaindex(url: str | None = None, *, force: bool = False) -> Path:
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

    Raises
    ------
    AAIndexError
        If the download fails or the remote server returns an error.

    """
    target = get_aaindex_path()
    if target.exists() and not force:
        logger.debug(
            "AAIndex CSV already present at %s; skipping download.", target
        )
        return target

    use_url = url or AAINDEX_URL
    logger.info("Downloading AAIndex CSV from %s to %s", use_url, target)

    try:
        resp = requests.get(use_url, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as exc:
        msg = f"Failed to download AAIndex CSV from {use_url}: {exc}"
        logger.error(msg)
        raise AAIndexError(msg) from exc

    target.write_bytes(resp.content)
    logger.info("AAIndex CSV saved to %s", target)
    return target


# ---------------------------------------------------------------------------
# Loading and lookup
# ---------------------------------------------------------------------------

_AAINDEX_TABLE: pd.DataFrame | None = None


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
        each index code).

    Raises
    ------
    AAIndexError
        If the CSV is missing and cannot be downloaded, cannot be read,
        or is structurally invalid.

    """
    global _AAINDEX_TABLE

    if _AAINDEX_TABLE is not None:
        return _AAINDEX_TABLE

    csv_path = get_aaindex_path()
    if not csv_path.exists():
        if not auto_download:
            msg = (
                f"AAIndex CSV not found at {csv_path}. Set auto_download=True "
                "or call `download_aaindex` explicitly."
            )
            logger.error(msg)
            raise AAIndexError(msg)

        logger.info(
            "AAIndex CSV not found at %s; attempting to download.", csv_path
        )
        download_aaindex()

    # At this point we expect the file to exist
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # pragma: no cover - I/O error path
        msg = f"Could not read AAIndex CSV at {csv_path}: {exc}"
        logger.error(msg)
        raise AAIndexError(msg) from exc

    # Normalise column names
    df.columns = [c.strip() for c in df.columns]

    # Require a residue identifier column
    if "residue" not in df.columns:
        msg = (
            "AAIndex CSV must contain a 'residue' column with the "
            "amino-acid codes."
        )
        logger.error(msg)
        raise AAIndexError(msg)

    # Normalise residue column
    df["residue"] = df["residue"].astype(str).str.strip().str.upper()
    df = df.set_index("residue")

    # Ensure that we have entries for all 20 amino acids
    missing_residues = [aa for aa in AA20 if aa not in df.index]
    if missing_residues:
        msg = (
            "AAIndex CSV is missing rows for residues: "
            f"{', '.join(sorted(missing_residues))}."
        )
        logger.error(msg)
        raise AAIndexError(msg)

    # Optionally, restrict to AA20 rows only
    df = df.loc[sorted(AA20)]

    _AAINDEX_TABLE = df
    logger.debug(
        "AAIndex table loaded with shape %s and %d indices.",
        df.shape,
        df.shape[1],
    )
    return df


def compute_aaindex_means_for_sequence(
    seq: str,
    index_codes: Iterable[str],
    table: pd.DataFrame | None = None,
) -> dict[str, float]:
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

    feats: dict[str, float] = {}
    for code in index_codes:
        if code not in table.columns:
            warnings.warn(
                f"AAIndex code {code!r} not found in AAIndex table; "
                "returning NaN for this descriptor.",
                RuntimeWarning,
                stacklevel=2,
            )
            feats[f"aaindex_{code}_mean"] = float("nan")
            continue

        # This is a Series indexed by residue (A, C, D, ...)
        scale = table[code]

        vals = [scale[aa] for aa in s if aa in AA20]

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
