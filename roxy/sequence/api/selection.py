"""Descriptor-selection helpers for the sequence API."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from roxy.core.exceptions import SequenceInputError


def normalize_descriptor_selection(
    descriptors: Optional[Sequence[str]],
    *,
    descriptor_blocks: Optional[Sequence[str]] = None,
    aaindex_codes: Optional[Iterable[str]] = None,
) -> List[str]:
    """Return the ordered descriptor-block selection for the API."""
    if descriptors is not None and descriptor_blocks is not None:
        raise SequenceInputError(
            "Use either `descriptors` or `descriptor_blocks`, not both."
        )

    selected_blocks = (
        descriptor_blocks if descriptor_blocks is not None else descriptors
    )
    if selected_blocks is not None:
        return list(selected_blocks)

    selected = ["aac", "global_basic"]
    if aaindex_codes is not None:
        codes = [str(code).strip() for code in aaindex_codes if str(code).strip()]
        if codes:
            selected.append("aaindex_mean")
    return selected


# TODO:
# - Revisit whether default descriptor selection should live closer to
#   the sequence block registry once selection policy becomes stable.

__all__ = ["normalize_descriptor_selection"]
