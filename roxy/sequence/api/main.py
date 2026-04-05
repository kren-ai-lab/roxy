"""Public API entrypoints for sequence descriptor workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence

from roxy.core.exceptions import SequenceError
from roxy.sequence.base.registry import describe_with_block
from roxy.sequence.base.registry import (
    list_available_descriptors as _list_available_descriptors,
)
from roxy.sequence.preprocessing.cleaning import (
    InvalidResiduePolicy,
    SequenceCleaningConfig,
    clean_sequence,
)
from roxy.sequence.preprocessing.validation import (
    assert_valid_sequence,
    validate_sequences as _validate_sequences,
)

from .io import coerce_sequence_series, read_fasta_records, require_pandas
from .selection import normalize_descriptor_selection
from .types import SequenceInput

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    import pandas as pd


def build_cleaning_config(
    *,
    invalid_policy: InvalidResiduePolicy = "strict",
    remove_internal_whitespace: bool = True,
    remove_terminal_stop: bool = True,
) -> SequenceCleaningConfig:
    """Build the shared cleaning policy used by the sequence API."""
    return SequenceCleaningConfig(
        remove_internal_whitespace=remove_internal_whitespace,
        remove_terminal_stop=remove_terminal_stop,
        invalid_policy=invalid_policy,
    )


def describe_sequences(
    data: SequenceInput,
    *,
    sequence_column: str = "sequence",
    descriptors: Optional[Sequence[str]] = None,
    descriptor_blocks: Optional[Sequence[str]] = None,
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
    """Describe protein sequences using modular sequence descriptor blocks.

    Parameters
    ----------
    data:
        Iterable of sequences or a DataFrame containing a sequence column.
    sequence_column:
        Name of the sequence column when ``data`` is a DataFrame.
    descriptors:
        Optional ordered list of descriptor-block names to apply. If not
        provided, the API currently uses ``["aac", "global_basic"]`` and
        adds ``"aaindex_mean"`` when AAIndex codes are supplied.
    descriptor_blocks:
        Explicit alias for ``descriptors``. This is the preferred name
        for new code because it matches the modular sequence-family
        architecture.
    aaindex_codes:
        Optional iterable of AAIndex codes used by the ``aaindex_mean`` block.
    pH:
        pH used by charge-related global/basic summaries.
    include_histidine_in_charge:
        Whether histidine contributes to the net-charge estimate.
    include_aac_counts:
        Whether the AAC block should also emit absolute counts.
    invalid_policy:
        How non-canonical residues are handled before descriptor computation.
    allow_empty:
        Whether empty sequences are accepted after cleaning.
    min_length:
        Optional minimum cleaned sequence length required before
        descriptor computation.
    remove_internal_whitespace:
        Whether internal whitespace is removed during preprocessing.
    remove_terminal_stop:
        Whether a single terminal ``*`` stop marker is removed.

    Returns
    -------
    pandas.DataFrame
        Sequence descriptor table indexed like the input when possible.
    """
    pd = require_pandas()
    sequence_series, original_index = coerce_sequence_series(
        data,
        sequence_column=sequence_column,
    )
    selected = normalize_descriptor_selection(
        descriptors,
        descriptor_blocks=descriptor_blocks,
        aaindex_codes=aaindex_codes,
    )
    cleaning_config = build_cleaning_config(
        invalid_policy=invalid_policy,
        remove_internal_whitespace=remove_internal_whitespace,
        remove_terminal_stop=remove_terminal_stop,
    )

    rows = []
    for row_label, sequence in sequence_series.items():
        try:
            cleaned_sequence = assert_valid_sequence(
                sequence,
                cleaning_config=cleaning_config,
                allow_empty=allow_empty,
                min_length=min_length,
                invalid_policy=invalid_policy,
            )
        except SequenceError as exc:
            raise type(exc)(
                f"Sequence preprocessing failed at row {row_label!r}: {exc}"
            ) from exc

        feats = {}
        for block_name in selected:
            if block_name == "global_basic":
                block_feats = describe_with_block(
                    block_name,
                    cleaned_sequence,
                    pH=pH,
                    include_histidine_in_charge=include_histidine_in_charge,
                )
            elif block_name == "aac":
                block_feats = describe_with_block(
                    block_name,
                    cleaned_sequence,
                    include_counts=include_aac_counts,
                )
            elif block_name == "aaindex_mean":
                block_feats = describe_with_block(
                    block_name,
                    cleaned_sequence,
                    index_codes=aaindex_codes or [],
                )
            else:
                block_feats = describe_with_block(block_name, cleaned_sequence)
            feats.update(dict(block_feats))
        rows.append(feats)

    df = pd.DataFrame(rows)
    if original_index is not None:
        df.index = original_index
    return df


def validate_sequences(
    data: SequenceInput,
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
    """Validate protein sequences through the shared validation layer."""
    cleaning_config = build_cleaning_config(
        invalid_policy=invalid_policy,
        remove_internal_whitespace=remove_internal_whitespace,
        remove_terminal_stop=remove_terminal_stop,
    )
    return _validate_sequences(
        data,
        sequence_column=sequence_column,
        cleaning_config=cleaning_config,
        allow_empty=allow_empty,
        min_length=min_length,
        check_duplicates=check_duplicates,
        require_unique=require_unique,
    )


def describe_fasta(
    fasta_path: str,
    *,
    descriptors: Optional[Sequence[str]] = None,
    descriptor_blocks: Optional[Sequence[str]] = None,
    pH: float = 7.0,
    include_histidine_in_charge: bool = False,
    aaindex_codes: Optional[Iterable[str]] = None,
    invalid_policy: InvalidResiduePolicy = "strict",
    allow_empty: bool = False,
    min_length: Optional[int] = None,
    remove_internal_whitespace: bool = True,
    remove_terminal_stop: bool = True,
) -> "pd.DataFrame":
    """Describe sequences from a FASTA file via the sequence API."""
    pd = require_pandas()
    records = read_fasta_records(fasta_path)
    sequence_values = [sequence for _, sequence in records]
    cleaning_config = build_cleaning_config(
        invalid_policy=invalid_policy,
        remove_internal_whitespace=remove_internal_whitespace,
        remove_terminal_stop=remove_terminal_stop,
    )

    descriptors = describe_sequences(
        sequence_values,
        pH=pH,
        include_histidine_in_charge=include_histidine_in_charge,
        descriptors=descriptors,
        descriptor_blocks=descriptor_blocks,
        aaindex_codes=aaindex_codes,
        invalid_policy=invalid_policy,
        allow_empty=allow_empty,
        min_length=min_length,
        remove_internal_whitespace=remove_internal_whitespace,
        remove_terminal_stop=remove_terminal_stop,
    )
    descriptors.insert(
        0,
        "sequence",
        [
            clean_sequence(sequence, config=cleaning_config) or ""
            for sequence in sequence_values
        ],
    )
    descriptors.insert(0, "id", [record_id for record_id, _ in records])

    df = pd.DataFrame(descriptors)
    col_order = ["id", "sequence"] + [
        column for column in df.columns if column not in {"id", "sequence"}
    ]
    return df[col_order]


def list_available_descriptors() -> List[str]:
    """Return the names of registered modular sequence descriptor blocks."""
    return _list_available_descriptors()


__all__ = [
    "build_cleaning_config",
    "describe_fasta",
    "describe_sequences",
    "list_available_descriptors",
    "validate_sequences",
]
