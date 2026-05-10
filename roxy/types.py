"""Shared type aliases for Roxy."""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl

SequenceLike = str | Iterable[str]
FeatureFrame = pl.DataFrame

__all__ = ["FeatureFrame", "SequenceLike"]
