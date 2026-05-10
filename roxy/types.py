"""Shared type aliases for Roxy."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

SequenceLike = str | Iterable[str]
FeatureFrame = pd.DataFrame

__all__ = ["FeatureFrame", "SequenceLike"]
