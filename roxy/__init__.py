"""Roxy

Roxy is a Python package for modular extraction of protein sequence
descriptors.

The top-level namespace stays intentionally small. Dataset
orchestration, EDA, projection, reporting, and non-sequence descriptor
APIs are not re-exported here.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence, Union

from roxy.sequence.cleaning import InvalidResiduePolicy

from . import sequence

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    import pandas as pd


try:
    __version__ = version("roxy")
except PackageNotFoundError:  # pragma: no cover - local source tree fallback
    __version__ = "0.1.0"


def describe_sequences(
    data: Union[Iterable[str], "pd.DataFrame"],
    *,
    sequence_column: str = "sequence",
    descriptors: Optional[Sequence[str]] = None,
    aaindex_codes: Optional[Iterable[str]] = None,
    pH: float = 7.0,
    include_histidine_in_charge: bool = False,
    include_aac_counts: bool = False,
    invalid_policy: InvalidResiduePolicy = "strict",
    allow_empty: bool = False,
    min_length: Optional[int] = None,
    remove_internal_whitespace: bool = True,
    remove_terminal_stop: bool = True,
) -> "pd.DataFrame":
    """Describe protein sequences using the sequence-focused public API."""
    from roxy.sequence.api import describe_sequences as _describe_sequences

    return _describe_sequences(
        data,
        sequence_column=sequence_column,
        descriptors=descriptors,
        aaindex_codes=aaindex_codes,
        pH=pH,
        include_histidine_in_charge=include_histidine_in_charge,
        include_aac_counts=include_aac_counts,
        invalid_policy=invalid_policy,
        allow_empty=allow_empty,
        min_length=min_length,
        remove_internal_whitespace=remove_internal_whitespace,
        remove_terminal_stop=remove_terminal_stop,
    )


def validate_sequences(
    data: Union[Iterable[str], "pd.DataFrame"],
    *,
    sequence_column: str = "sequence",
    allow_empty: bool = False,
    invalid_policy: InvalidResiduePolicy = "strict",
    min_length: Optional[int] = None,
    remove_internal_whitespace: bool = True,
    remove_terminal_stop: bool = True,
    check_duplicates: bool = False,
    require_unique: bool = False,
) -> "pd.DataFrame":
    """Validate canonical protein sequences using the public API."""
    from roxy.sequence.api import validate_sequences as _validate_sequences

    return _validate_sequences(
        data,
        sequence_column=sequence_column,
        allow_empty=allow_empty,
        invalid_policy=invalid_policy,
        min_length=min_length,
        remove_internal_whitespace=remove_internal_whitespace,
        remove_terminal_stop=remove_terminal_stop,
        check_duplicates=check_duplicates,
        require_unique=require_unique,
    )


def describe_fasta(
    fasta_path: str,
    *,
    pH: float = 7.0,
    include_histidine_in_charge: bool = False,
    aaindex_codes: Optional[Iterable[str]] = None,
    invalid_policy: InvalidResiduePolicy = "strict",
    allow_empty: bool = False,
    min_length: Optional[int] = None,
    remove_internal_whitespace: bool = True,
    remove_terminal_stop: bool = True,
) -> "pd.DataFrame":
    """Describe protein sequences from a FASTA file via the public API."""
    from roxy.sequence.api import describe_fasta as _describe_fasta

    return _describe_fasta(
        fasta_path,
        pH=pH,
        include_histidine_in_charge=include_histidine_in_charge,
        aaindex_codes=aaindex_codes,
        invalid_policy=invalid_policy,
        allow_empty=allow_empty,
        min_length=min_length,
        remove_internal_whitespace=remove_internal_whitespace,
        remove_terminal_stop=remove_terminal_stop,
    )


def list_available_descriptors() -> List[str]:
    """List the modular sequence descriptor blocks exposed by the API."""
    from roxy.sequence.api import (
        list_available_descriptors as _list_available_descriptors,
    )

    return _list_available_descriptors()


# TODO:
# - Re-export additional sequence-facing APIs only after they are stable.
# - Introduce `SequenceDescriptorPipeline` once the sequence API supports a
#   first-class pipeline abstraction.

__all__ = [
    "__version__",
    "describe_fasta",
    "describe_sequences",
    "list_available_descriptors",
    "sequence",
    "validate_sequences",
]
