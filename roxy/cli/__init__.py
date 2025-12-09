"""CLI entry point for Roxy.

Usage (once installed):

    roxy describe-sequences ...
    roxy scale-data ...
    roxy select-features ...
    roxy full-pipeline ...
    roxy info --what descriptors
"""

from __future__ import annotations

from .main import app

__all__ = ["app"]
