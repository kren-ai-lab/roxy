"""Descriptor internals for Roxy.

Active sequence descriptor entrypoints live under ``roxy.sequence``.
This package only exposes shared descriptor-engine building blocks that
are still used inside the repository.
"""

from __future__ import annotations

from .base import BaseDescriptorEngine

__all__ = ["BaseDescriptorEngine"]
