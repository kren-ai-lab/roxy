"""AAIndex-backed sequence descriptor helpers."""

from __future__ import annotations

from typing import Dict, Iterable, List

from roxy.core.aaindex_data import (
    compute_aaindex_means_for_sequence as _compute_mean_descriptors,
    load_aaindex as _load_aaindex,
)
from roxy.sequence.descriptors._base import (
    DescriptorRecord,
    FunctionDescriptorBlock,
    prepare_descriptor_sequence,
)


def aaindex_backend_available() -> bool:
    """Return whether the packaged AAIndex backend is available."""
    return True


def normalize_aaindex_codes(index_codes: Iterable[str]) -> List[str]:
    """Normalize AAIndex codes into a materialized, stripped list."""
    return [str(code).strip() for code in index_codes if str(code).strip()]


def empty_aaindex_mean_descriptors(index_codes: Iterable[str]) -> Dict[str, float]:
    """Return NaN-filled AAIndex mean descriptors for the requested codes."""
    return {
        f"aaindex_{code}_mean": float("nan")
        for code in normalize_aaindex_codes(index_codes)
    }


def validate_aaindex_codes(
    index_codes: Iterable[str],
    *,
    auto_download: bool = True,
) -> Dict[str, bool]:
    """Check whether requested AAIndex codes exist in the core AAIndex table."""
    codes = normalize_aaindex_codes(index_codes)
    if not codes:
        return {}

    table = _load_aaindex(auto_download=auto_download)
    return {code: code in table.columns for code in codes}


def compute_sequence_aaindex_means(
    sequence: str,
    *,
    index_codes: Iterable[str],
) -> Dict[str, float]:
    """Compute per-sequence AAIndex mean descriptors."""
    codes = normalize_aaindex_codes(index_codes)
    if not codes:
        return {}

    return _compute_mean_descriptors(
        prepare_descriptor_sequence(sequence),
        index_codes=codes,
    )


class AAIndexDescriptors(FunctionDescriptorBlock):
    """Descriptor block for per-index AAIndex mean sequence descriptors."""

    def __init__(self, *, index_codes: Iterable[str]) -> None:
        self.index_codes = normalize_aaindex_codes(index_codes)
        super().__init__(
            "aaindex_mean",
            "aaindex",
            aaindex_mean_descriptors,
            preserve_feature_names=True,
            known_feature_names=tuple(
                f"aaindex_{code}_mean" for code in self.index_codes
            ),
            index_codes=self.index_codes,
        )

    def transform_sequence(self, sequence: str) -> DescriptorRecord:
        """Transform one sequence into AAIndex mean descriptors."""
        return super().transform_sequence(sequence)


def aaindex_mean_descriptors(
    sequence: str,
    *,
    index_codes: Iterable[str],
) -> Dict[str, float]:
    """Return the current AAIndex descriptor block for a sequence."""
    return compute_sequence_aaindex_means(
        sequence,
        index_codes=index_codes,
    )


__all__ = [
    "AAIndexDescriptors",
    "aaindex_backend_available",
    "aaindex_mean_descriptors",
    "compute_sequence_aaindex_means",
    "empty_aaindex_mean_descriptors",
    "normalize_aaindex_codes",
    "validate_aaindex_codes",
]
