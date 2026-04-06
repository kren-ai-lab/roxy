"""Explicit public API for sequence descriptor workflows."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Iterable, Optional, Sequence

from roxy.core.optional_deps import require_pandas
from roxy.sequence.cleaning import InvalidResiduePolicy, clean_sequence
from roxy.sequence.io import coerce_sequence_series, read_fasta_records
from roxy.sequence.pipeline import (
    SequenceDescriptorPipeline,
    build_cleaning_config,
    build_descriptor_blocks,
    normalize_descriptor_selection,
)
from roxy.sequence.registry import list_available_descriptors as list_registered_descriptors
from roxy.sequence.validation import inspect_sequences

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    import pandas as pd


def describe_sequences(
    data: object,
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
    """Describe protein sequences through the explicit pipeline."""
    selected = normalize_descriptor_selection(
        descriptors,
        descriptor_blocks=descriptor_blocks,
        aaindex_codes=aaindex_codes,
    )
    blocks = build_descriptor_blocks(
        selected,
        aaindex_codes=aaindex_codes,
        pH=pH,
        include_histidine_in_charge=include_histidine_in_charge,
        include_aac_counts=include_aac_counts,
    )
    pipeline = SequenceDescriptorPipeline(
        blocks,
        cleaning_config=build_cleaning_config(
            invalid_policy=invalid_policy,
            remove_internal_whitespace=remove_internal_whitespace,
            remove_terminal_stop=remove_terminal_stop,
        ),
        allow_empty=allow_empty,
        min_length=min_length,
        on_validation_error="raise",
        include_sequence=False,
        include_cleaned_sequence=False,
        include_validation_metadata=False,
    )
    return pipeline.transform(
        data,
        sequence_column=sequence_column,
    )


def validate_sequences(
    data: object,
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
    """Return a DataFrame validation report without running descriptor blocks."""
    pd = require_pandas(purpose="sequence API DataFrame validation reports")
    sequence_series, original_index = coerce_sequence_series(
        data,
        sequence_column=sequence_column,
    )
    cleaning_config = build_cleaning_config(
        invalid_policy=invalid_policy,
        remove_internal_whitespace=remove_internal_whitespace,
        remove_terminal_stop=remove_terminal_stop,
    )
    results = inspect_sequences(
        sequence_series.tolist(),
        cleaning_config=cleaning_config,
        allow_empty=allow_empty,
        min_length=min_length,
    )

    rows = []
    for result in results:
        row = asdict(result)
        row["invalid_count"] = len(result.invalid_residues)
        row["error_count"] = len(result.errors)
        rows.append(row)

    df = pd.DataFrame(rows)
    if original_index is not None:
        df.index = original_index

    if check_duplicates or require_unique:
        cleaned_series = df["cleaned"].fillna("<missing>")
        duplicate_counts = cleaned_series.map(cleaned_series.value_counts())
        df["duplicate_count"] = duplicate_counts
        df["is_duplicate"] = duplicate_counts > 1

        if require_unique:
            duplicate_mask = df["is_duplicate"] & df["cleaned"].notna()
            df.loc[duplicate_mask, "errors"] = df.loc[
                duplicate_mask, "errors"
            ].apply(lambda errors: list(errors) + ["Sequence is duplicated."])
            df.loc[duplicate_mask, "error_count"] = df.loc[
                duplicate_mask, "errors"
            ].apply(len)
            df.loc[duplicate_mask, "is_valid"] = False

    return df


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
    """Describe sequences from a FASTA file through the explicit pipeline."""
    pd = require_pandas(purpose="sequence API DataFrame outputs")
    records = read_fasta_records(fasta_path)
    input_frame = pd.DataFrame(
        {
            "sequence_id": [record_id for record_id, _ in records],
            "sequence": [sequence for _, sequence in records],
        }
    )
    descriptors_df = describe_sequences(
        input_frame,
        sequence_column="sequence",
        descriptors=descriptors,
        descriptor_blocks=descriptor_blocks,
        aaindex_codes=aaindex_codes,
        pH=pH,
        include_histidine_in_charge=include_histidine_in_charge,
        invalid_policy=invalid_policy,
        allow_empty=allow_empty,
        min_length=min_length,
        remove_internal_whitespace=remove_internal_whitespace,
        remove_terminal_stop=remove_terminal_stop,
    )

    cleaning_config = build_cleaning_config(
        invalid_policy=invalid_policy,
        remove_internal_whitespace=remove_internal_whitespace,
        remove_terminal_stop=remove_terminal_stop,
    )
    descriptors_df.insert(
        0,
        "sequence",
        [clean_sequence(sequence, config=cleaning_config) for _, sequence in records],
    )
    descriptors_df.insert(0, "id", [record_id for record_id, _ in records])
    if "sequence_id" in descriptors_df.columns:
        descriptors_df = descriptors_df.drop(columns=["sequence_id"])
    return descriptors_df


def list_available_descriptors() -> list[str]:
    """Return the stable names of descriptor blocks supported by the pipeline."""
    return list_registered_descriptors()


__all__ = [
    "describe_fasta",
    "describe_sequences",
    "list_available_descriptors",
    "validate_sequences",
]
