"""Roxy descriptors subpackage."""

from __future__ import annotations

from .base import BaseDescriptor
from .registry import DESCRIPTOR_REGISTRY, register

__all__ = [
    "DESCRIPTOR_REGISTRY",
    "BaseDescriptor",
    "register",
]
