"""Shared helpers for descriptor tests."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from roxy.core.constants import AA20_ORDERED

if TYPE_CHECKING:
    from collections.abc import Mapping

SEQ_ALL20 = "".join(AA20_ORDERED)
EMPTY = ""


def assert_keys_present(out: Mapping[str, float], keys: list[str]) -> None:
    missing = sorted(set(keys) - set(out))
    assert not missing


def assert_between_0_1_or_nan(value: float) -> None:
    assert math.isnan(value) or 0.0 <= value <= 1.0
