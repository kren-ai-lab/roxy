"""Public API entrypoints for sequence descriptor workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Sequence, Tuple

from roxy.core.exceptions import SequenceCollectionError, SequenceError
from roxy.sequence.cleaning import (
    InvalidResiduePolicy,
    SequenceCleaningConfig,
    clean_sequence,
)
from roxy.sequence.registry import describe_with_block
from roxy.sequence.registry import list_available_descriptors as _list_available_descriptors
from roxy.sequence.validation import (
    assert_valid_sequence,
    validate_sequences as _validate_sequences,
)

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    import pandas as pd

SequenceInput = Any


def _require_pandas() -> "type[pd]":
    """Import pandas lazily for DataFrame-based API operations."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "pandas is required for the sequence API DataFrame outputs."
        ) from exc
    return pd


def _normalize_descriptor_selection(
    descriptors: Optional[Sequence[str]],
    *,
    aaindex_codes: Optional[Iterable[str]] = None,
) -> List[str]:
    """Return the ordered descriptor-block selection for the API."""
    if descriptors is not None:
        return list(descriptors)

    selected = ["aac", "global_basic"]
    if aaindex_codes is not None:
        codes = [str(code).strip() for code in aaindex_codes if str(code).strip()]
        if codes:
            selected.append("aaindex_mean")
    return selected


def _coerce_sequence_series(
    data: SequenceInput,
    *,
    sequence_column: str = "sequence",
) -> Tuple["pd.Series", Optional["pd.Index"]]:
    """Normalize API input into a sequence Series plus an optional index."""
    pd = _require_pandas()

    if isinstance(data, str):
        raise SequenceCollectionError(
            "Expected an iterable of sequences or a DataFrame, not a single "
            "string. Use sequence-level validation helpers for one sequence."
        )

    if isinstance(data, pd.DataFrame):
        if sequence_column not in data.columns:
            raise SequenceCollectionError(
                f"Sequence column {sequence_column!r} not found in input DataFrame."
            )
        series = data[sequence_column]
        return series, data.index

    try:
        values = list(data)
    except TypeError as exc:
        raise SequenceCollectionError(
            "Expected an iterable of sequence values or a pandas DataFrame."
        ) from exc

    return pd.Series(values, dtype="object"), None


def _build_cleaning_config(
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


def _read_fasta_records(fasta_path: str) -> List[Tuple[str, str]]:
    """Read FASTA records as ``(id, sequence)`` pairs."""
    records: List[Tuple[str, str]] = []
    current_id: Optional[str] = None
    current_seq: List[str] = []

    with open(fasta_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records.append((current_id, "".join(current_seq)))
                current_id = line[1:].strip()
                current_seq = []
                continue
            current_seq.append(line)

    if current_id is not None:
        records.append((current_id, "".join(current_seq)))

    return records


def describe_sequences(
    data: SequenceInput,
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
    pd = _require_pandas()
    sequence_series, original_index = _coerce_sequence_series(
        data,
        sequence_column=sequence_column,
    )
    selected = _normalize_descriptor_selection(
        descriptors,
        aaindex_codes=aaindex_codes,
    )
    cleaning_config = _build_cleaning_config(
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
    cleaning_config = _build_cleaning_config(
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
    pd = _require_pandas()
    records = _read_fasta_records(fasta_path)
    sequence_values = [sequence for _, sequence in records]
    cleaning_config = _build_cleaning_config(
        invalid_policy=invalid_policy,
        remove_internal_whitespace=remove_internal_whitespace,
        remove_terminal_stop=remove_terminal_stop,
    )

    descriptors = describe_sequences(
        sequence_values,
        pH=pH,
        include_histidine_in_charge=include_histidine_in_charge,
        descriptors=None,
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
    "describe_fasta",
    "describe_sequences",
    "list_available_descriptors",
    "validate_sequences",
]
