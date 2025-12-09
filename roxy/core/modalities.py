from __future__ import annotations

"""Enumerations for supported data modalities in Roxy."""

from enum import Enum


class Modality(str, Enum):
    """Supported data modalities for Roxy."""

    SEQUENCE = "sequence"
    STRUCTURE = "structure"
    COMPOUND = "compound"
