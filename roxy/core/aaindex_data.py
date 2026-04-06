"""AAIndex handling utilities for sequence-level descriptors.

This module manages the AAIndex CSV file used to derive residue-level
properties for protein sequences. It provides functions to:

- determine the cache directory for Roxy,
- download or update the AAIndex CSV file,
- load the AAIndex table as a pandas DataFrame,
- compute per-sequence mean AAIndex values for one or more indices.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional

import os
import warnings

import requests

from roxy.core.constants import AA20, AAINDEX_FILENAME, AAINDEX_URL, ROXY_CACHE_SUBDIR
from roxy.core.exceptions import AAIndexError
from roxy.core.logging_utils import get_logger
from roxy.core.optional_deps import require_pandas

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    import pandas as pd

logger = get_logger(__name__)


def get_cache_dir() -> Path:
    """Return the Roxy cache directory."""
    base = os.environ.get("XDG_CACHE_HOME", None)
    if base is None:
        base = os.path.join(Path.home(), ".cache")
    cache_dir = Path(base) / ROXY_CACHE_SUBDIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_aaindex_path() -> Path:
    """Return the full path to the AAIndex CSV file in the cache."""
    return get_cache_dir() / AAINDEX_FILENAME


def download_aaindex(url: Optional[str] = None, *, force: bool = False) -> Path:
    """Download or update the AAIndex CSV file in the cache directory."""
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


_AAINDEX_TABLE: Optional[pd.DataFrame] = None


def load_aaindex(*, auto_download: bool = True) -> pd.DataFrame:
    """Load the AAIndex table from the cache."""
    global _AAINDEX_TABLE
    pd = require_pandas(purpose="AAIndex table loading and lookup")

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

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # pragma: no cover - I/O error path
        msg = f"Could not read AAIndex CSV at {csv_path}: {exc}"
        logger.error(msg)
        raise AAIndexError(msg) from exc

    df.columns = [c.strip() for c in df.columns]

    if "residue" not in df.columns:
        msg = (
            "AAIndex CSV must contain a 'residue' column with the "
            "amino-acid codes."
        )
        logger.error(msg)
        raise AAIndexError(msg)

    df["residue"] = df["residue"].astype(str).str.strip().str.upper()
    df = df.set_index("residue")

    missing_residues = [aa for aa in AA20 if aa not in df.index]
    if missing_residues:
        msg = (
            "AAIndex CSV is missing rows for residues: "
            f"{', '.join(sorted(missing_residues))}."
        )
        logger.error(msg)
        raise AAIndexError(msg)

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
    table: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """Compute mean AAIndex values for a sequence and a set of indices."""
    s = (seq or "").strip().upper().replace("*", "")
    if len(s) == 0:
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
    """Ensure that the AAIndex CSV is present in the cache."""
    _ = load_aaindex(auto_download=True)


__all__ = [
    "compute_aaindex_means_for_sequence",
    "download_aaindex",
    "ensure_aaindex_available",
    "get_aaindex_path",
    "get_cache_dir",
    "load_aaindex",
]
