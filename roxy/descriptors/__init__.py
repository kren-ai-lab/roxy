"""Roxy descriptors subpackage."""

from __future__ import annotations

from . import composition as composition
from .base import BaseDescriptor
from .registry import DESCRIPTOR_REGISTRY, register

__all__ = [
    "DESCRIPTOR_REGISTRY",
    "BaseDescriptor",
    "register",
]
