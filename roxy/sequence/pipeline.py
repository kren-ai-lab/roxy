"""Sequence descriptor pipeline primitives."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Literal, Optional, Sequence

from roxy.core.exceptions import (
    EmptySequenceError,
    InvalidSequenceError,
    MissingSequenceError,
    SequenceCollectionError,
    SequenceInputError,
    SequenceTooShortError,
)
from roxy.core.optional_deps import require_pandas
from roxy.sequence.cleaning import InvalidResiduePolicy, SequenceCleaningConfig
from roxy.sequence.descriptors._base import SequenceDescriptorBlock
from roxy.sequence.descriptors.aaindex.aaindex import AAIndexDescriptors
from roxy.sequence.descriptors.basic.composition import AminoAcidCompositionDescriptors
from roxy.sequence.descriptors.basic.global_basic import BasicGlobalDescriptors
from roxy.sequence.descriptors.basic.grouped import GroupedCompositionDescriptors
from roxy.sequence.descriptors.kmers.dpc import DipeptideDescriptors
from roxy.sequence.io import coerce_sequence_series
from roxy.sequence.validation import (
    SequenceValidationError,
    SequenceValidationResult,
    inspect_sequences,
)

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    import pandas as pd

SequenceInput = Any
ValidationMode = Literal["raise", "report"]

SUPPORTED_DESCRIPTOR_BLOCKS: tuple[str, ...] = (
    "aac",
    "aaindex_mean",
    "dpc",
    "global_basic",
    "grouped",
)


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


def normalize_descriptor_selection(
    descriptors: Optional[Sequence[str]],
    *,
    descriptor_blocks: Optional[Sequence[str]] = None,
    aaindex_codes: Optional[Iterable[str]] = None,
) -> list[str]:
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
def build_descriptor_blocks(
    descriptor_names: Sequence[str],
    *,
    aaindex_codes: Optional[Iterable[str]] = None,
    pH: float = 7.0,
    include_histidine_in_charge: bool = False,
    include_aac_counts: bool = False,
) -> list[SequenceDescriptorBlock]:
    """Build explicit descriptor-block instances from stable names."""
    blocks: list[SequenceDescriptorBlock] = []
    normalized_aaindex_codes = [
        str(code).strip() for code in (aaindex_codes or []) if str(code).strip()
    ]

    for descriptor_name in descriptor_names:
        if descriptor_name == "aac":
            blocks.append(
                AminoAcidCompositionDescriptors(include_counts=include_aac_counts)
            )
            continue

        if descriptor_name == "grouped":
            blocks.append(GroupedCompositionDescriptors())
            continue

        if descriptor_name == "global_basic":
            blocks.append(
                BasicGlobalDescriptors(
                    include_histidine_in_charge=include_histidine_in_charge,
                )
            )
            continue

        if descriptor_name == "aaindex_mean":
            blocks.append(AAIndexDescriptors(index_codes=normalized_aaindex_codes))
            continue

        if descriptor_name == "dpc":
            blocks.append(DipeptideDescriptors())
            continue

        raise SequenceInputError(
            f"Unsupported descriptor block: {descriptor_name!r}."
        )

    return blocks


class SequenceDescriptorPipeline:
    """Explicit pipeline that cleans, validates, and transforms sequences."""

    def __init__(
        self,
        blocks: Sequence[SequenceDescriptorBlock],
        *,
        cleaning_config: Optional[SequenceCleaningConfig] = None,
        allow_empty: bool = False,
        min_length: Optional[int] = None,
        on_validation_error: ValidationMode = "raise",
        include_sequence: bool = True,
        include_cleaned_sequence: bool = True,
        include_validation_metadata: bool = True,
    ) -> None:
        if not blocks:
            raise SequenceInputError("`blocks` must contain at least one descriptor block.")
        if on_validation_error not in {"raise", "report"}:
            raise SequenceInputError(
                "`on_validation_error` must be either 'raise' or 'report'."
            )
        if min_length is not None and min_length < 0:
            raise SequenceInputError("`min_length` must be non-negative.")

        self.blocks = self._validate_blocks(blocks)
        self.cleaning_config = (
            cleaning_config if cleaning_config is not None else SequenceCleaningConfig()
        )
        self.allow_empty = allow_empty
        self.min_length = min_length
        self.on_validation_error = on_validation_error
        self.include_sequence = include_sequence
        self.include_cleaned_sequence = include_cleaned_sequence
        self.include_validation_metadata = include_validation_metadata

    def transform(
        self,
        data: SequenceInput,
        *,
        sequence_column: str = "sequence",
        sequence_id_column: Optional[str] = None,
        metadata_columns: Optional[Sequence[str]] = None,
    ) -> "pd.DataFrame":
        """Run the full pipeline and return one descriptor row per sequence."""
        pd = require_pandas(purpose="sequence descriptor pipeline outputs")
        input_rows, original_index = self._coerce_input_rows(
            data,
            sequence_column=sequence_column,
            sequence_id_column=sequence_id_column,
            metadata_columns=metadata_columns,
        )

        ordered_feature_columns = self._initial_feature_columns()
        seen_feature_columns = set(ordered_feature_columns)
        output_rows: list[dict[str, object]] = []

        for input_row in input_rows:
            result = self._inspect_input_row(input_row["sequence"])
            output_row = self._build_metadata_row(input_row, result)

            if not result.is_valid:
                if self.on_validation_error == "raise":
                    self._raise_validation_error(result, input_row)
                output_rows.append(output_row)
                continue

            cleaned_sequence = result.cleaned or ""
            for block in self.blocks:
                try:
                    block_features = block.transform_sequence(cleaned_sequence)
                except Exception as exc:
                    raise type(exc)(
                        self._descriptor_error_message(block, input_row, exc)
                    ) from exc

                for feature_name, feature_value in block_features.items():
                    output_row[feature_name] = feature_value
                    if feature_name not in seen_feature_columns:
                        ordered_feature_columns.append(feature_name)
                        seen_feature_columns.add(feature_name)

            output_rows.append(output_row)

        df = pd.DataFrame(output_rows)
        if original_index is not None:
            df.index = original_index

        metadata_order = self._metadata_columns(
            output_rows,
            feature_columns=ordered_feature_columns,
        )
        ordered_columns = metadata_order + [
            column for column in ordered_feature_columns if column in df.columns
        ]
        remaining_columns = [
            column for column in df.columns if column not in set(ordered_columns)
        ]
        return df[ordered_columns + remaining_columns]

    def _validate_blocks(
        self,
        blocks: Sequence[SequenceDescriptorBlock],
    ) -> list[SequenceDescriptorBlock]:
        """Validate the provided descriptor blocks."""
        normalized_blocks: list[SequenceDescriptorBlock] = []
        for block in blocks:
            if not isinstance(block, SequenceDescriptorBlock):
                raise SequenceInputError(
                    "Each block must expose `family_name`, `column_prefix`, "
                    "and `transform_sequence(sequence)`."
                )
            normalized_blocks.append(block)
        return normalized_blocks

    def _coerce_input_rows(
        self,
        data: SequenceInput,
        *,
        sequence_column: str,
        sequence_id_column: Optional[str],
        metadata_columns: Optional[Sequence[str]],
    ) -> tuple[list[dict[str, object]], Optional["pd.Index"]]:
        """Normalize supported inputs into row dictionaries plus optional index."""
        pd = require_pandas(purpose="sequence descriptor pipeline inputs")

        if isinstance(data, str):
            raise SequenceCollectionError(
                "Expected an iterable of sequences or a DataFrame, not a single string."
            )

        if isinstance(data, pd.DataFrame):
            if sequence_column not in data.columns:
                raise SequenceCollectionError(
                    f"Sequence column {sequence_column!r} not found in input DataFrame."
                )

            resolved_sequence_id_column = sequence_id_column
            if resolved_sequence_id_column is None:
                if "sequence_id" in data.columns and "sequence_id" != sequence_column:
                    resolved_sequence_id_column = "sequence_id"
                elif "id" in data.columns and "id" != sequence_column:
                    resolved_sequence_id_column = "id"

            resolved_metadata_columns = list(metadata_columns or [])
            if (
                resolved_sequence_id_column is not None
                and resolved_sequence_id_column not in resolved_metadata_columns
                and resolved_sequence_id_column in data.columns
            ):
                resolved_metadata_columns.insert(0, resolved_sequence_id_column)

            rows: list[dict[str, object]] = []
            for row_label, row in data.iterrows():
                row_data: dict[str, object] = {
                    "row_label": row_label,
                    "sequence": row[sequence_column],
                }
                for column_name in resolved_metadata_columns:
                    if column_name == sequence_column:
                        continue
                    if column_name in data.columns:
                        row_data[column_name] = row[column_name]
                rows.append(row_data)

            return rows, data.index

        try:
            values = list(data)
        except TypeError as exc:
            raise SequenceCollectionError(
                "Expected an iterable of sequence values or a pandas DataFrame."
            ) from exc

        rows = [
            {
                "row_label": index,
                "sequence": value,
            }
            for index, value in enumerate(values)
        ]
        return rows, None

    def _inspect_input_row(self, sequence: object) -> SequenceValidationResult:
        """Inspect one input sequence with the configured policy."""
        results = inspect_sequences(
            [sequence],
            cleaning_config=self.cleaning_config,
            allow_empty=self.allow_empty,
            min_length=self.min_length,
        )
        return results[0]

    def _initial_feature_columns(self) -> list[str]:
        """Collect predeclared block feature names in block order."""
        ordered_feature_columns: list[str] = []
        seen_feature_columns: set[str] = set()

        for block in self.blocks:
            feature_names_method = getattr(block, "feature_names", None)
            if not callable(feature_names_method):
                continue
            declared_feature_names = feature_names_method()
            if not declared_feature_names:
                continue
            for feature_name in declared_feature_names:
                if feature_name not in seen_feature_columns:
                    ordered_feature_columns.append(feature_name)
                    seen_feature_columns.add(feature_name)

        return ordered_feature_columns

    def _build_metadata_row(
        self,
        input_row: dict[str, object],
        result: SequenceValidationResult,
    ) -> dict[str, object]:
        """Build the non-descriptor columns for one output row."""
        output_row: dict[str, object] = {}

        for key, value in input_row.items():
            if key in {"row_label", "sequence"}:
                continue
            output_row[key] = value

        if self.include_sequence:
            output_row["sequence"] = input_row["sequence"]
        if self.include_cleaned_sequence:
            output_row["cleaned_sequence"] = result.cleaned

        if self.include_validation_metadata:
            output_row["is_valid"] = result.is_valid
            output_row["validation_errors"] = list(result.errors)
        return output_row

    def _metadata_columns(
        self,
        rows: Sequence[dict[str, object]],
        *,
        feature_columns: Sequence[str],
    ) -> list[str]:
        """Return the stable metadata-column order for the final DataFrame."""
        preferred_columns = [
            "sequence_id",
            "id",
            "sequence",
            "cleaned_sequence",
            "is_valid",
            "validation_errors",
        ]
        available = {key for row in rows for key in row}
        feature_column_set = set(feature_columns)

        ordered = [
            column
            for column in preferred_columns
            if column in available and column not in feature_column_set
        ]
        extras = [
            column
            for column in available
            if column not in set(ordered) and column not in feature_column_set
        ]
        return ordered + sorted(extras)

    def _raise_validation_error(
        self,
        result: SequenceValidationResult,
        input_row: dict[str, object],
    ) -> None:
        """Raise a specific validation exception with row context."""
        message = self._validation_error_message(result, input_row)

        if result.is_missing:
            raise MissingSequenceError(message)
        if result.normalized is None and not result.is_missing:
            raise SequenceInputError(message)
        if result.is_empty:
            raise EmptySequenceError(message)
        if result.min_length is not None and result.length < result.min_length:
            raise SequenceTooShortError(message)
        if result.invalid_residues:
            raise InvalidSequenceError(message)
        raise SequenceValidationError(message)

    def _validation_error_message(
        self,
        result: SequenceValidationResult,
        input_row: dict[str, object],
    ) -> str:
        """Build a row-aware validation error message."""
        location = f"row {input_row['row_label']!r}"
        sequence_id = input_row.get("sequence_id", input_row.get("id"))
        if sequence_id is not None:
            location += f" (sequence_id={sequence_id!r})"

        if result.errors:
            return f"Sequence validation failed at {location}: {' '.join(result.errors)}"
        return f"Sequence validation failed at {location}."

    def _descriptor_error_message(
        self,
        block: SequenceDescriptorBlock,
        input_row: dict[str, object],
        exc: BaseException,
    ) -> str:
        """Build a row-aware block execution error message."""
        location = f"row {input_row['row_label']!r}"
        sequence_id = input_row.get("sequence_id", input_row.get("id"))
        if sequence_id is not None:
            location += f" (sequence_id={sequence_id!r})"
        return (
            f"Descriptor block {block.family_name!r} failed at {location}: {exc}"
        )


__all__ = [
    "SUPPORTED_DESCRIPTOR_BLOCKS",
    "SequenceDescriptorPipeline",
    "SequenceInput",
    "ValidationMode",
    "build_cleaning_config",
    "build_descriptor_blocks",
    "normalize_descriptor_selection",
]
