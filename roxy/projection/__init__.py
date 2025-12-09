"""Roxy projection package.

This subpackage provides dimensionality reduction utilities for
projecting feature matrices into low-dimensional spaces suitable
for exploratory analysis and visualisation.

Currently supported methods are:

- PCA (:class:`PCAReducer`)
- t-SNE (:class:`TSNEReducer`)
- UMAP (:class:`UMAPReducer`, requires ``umap-learn``)

A convenience function :func:`project` is provided for quick use.
"""

from __future__ import annotations

from .reducers import (
    BaseReducer,
    PCAReducer,
    TSNEReducer,
    UMAPReducer,
    project,
)

__all__ = [
    "BaseReducer",
    "PCAReducer",
    "TSNEReducer",
    "UMAPReducer",
    "project",
]
