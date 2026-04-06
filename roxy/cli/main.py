"""Sequence-focused command-line interface for Roxy.

This module intentionally exposes only sequence-descriptor-oriented
commands.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional

import typer

from roxy.core.optional_deps import require_pandas as _require_pandas
from roxy.sequence.api import (
    describe_fasta as describe_fasta_api,
    describe_sequences as describe_sequences_api,
    list_available_descriptors as list_available_descriptors_api,
    validate_sequences as validate_sequences_api,
)

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    import pandas as pd

app = typer.Typer(
    help=(
        "Roxy command-line interface for modular protein sequence "
        "descriptor extraction."
    ),
    no_args_is_help=True,
)


# TODO:
# - Add descriptor-family selection flags once the active API contract is
#   stable.
# - Add richer preprocessing/validation reporting only when the sequence
#   CLI surface is stable.


def _load_table(path: Path) -> "pd.DataFrame":
    """Load a CSV or Parquet table into a DataFrame."""
    pd = _require_pandas(purpose="CLI commands that read or write tables")

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    raise ValueError(
        f"Unsupported file extension {suffix!r} for {path}. "
        "Supported: .csv, .parquet."
    )


def _save_table(df: "pd.DataFrame", path: Path) -> None:
    """Save a DataFrame to CSV or Parquet based on the file extension."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=True)
        return
    if suffix in {".parquet", ".pq"}:
        df.to_parquet(path)
        return

    raise ValueError(
        f"Unsupported output extension {suffix!r} for {path}. "
        "Use .csv or .parquet."
    )


def _normalize_aaindex_codes(
    aaindex_codes: Optional[Iterable[str]],
) -> Optional[List[str]]:
    """Normalize AAIndex codes provided through the CLI."""
    if aaindex_codes is None:
        return None
    codes = [code.strip() for code in aaindex_codes if code and code.strip()]
    return codes or None


@app.command("describe-sequences")
def describe_sequences(
    input_path: Path = typer.Argument(
        ...,
        help="Input table with at least one protein sequence column.",
    ),
    output_path: Path = typer.Option(
        Path("roxy_sequence_descriptors.parquet"),
        "--output-path",
        "-o",
        help="Path where the descriptor table will be written.",
    ),
    sequence_column: str = typer.Option(
        "sequence",
        "--sequence-column",
        "-s",
        help="Name of the column containing amino-acid sequences.",
    ),
    aaindex_codes: Optional[List[str]] = typer.Option(
        None,
        "--aaindex-code",
        help=(
            "Optional AAIndex code to include. Repeat the option to pass "
            "multiple codes."
        ),
    ),
    pH: float = typer.Option(
        7.0,
        "--ph",
        help="pH value used for charge-related descriptors.",
    ),
    include_histidine_in_charge: bool = typer.Option(
        False,
        "--include-histidine-in-charge",
        help="Whether to include histidine in the net-charge estimate.",
    ),
) -> None:
    """Compute sequence descriptors from a tabular input file."""
    df = _load_table(input_path)
    if sequence_column not in df.columns:
        raise typer.BadParameter(
            f"Sequence column {sequence_column!r} was not found in {input_path}."
        )

    codes = _normalize_aaindex_codes(aaindex_codes)
    descriptors = describe_sequences_api(
        df,
        sequence_column=sequence_column,
        aaindex_codes=codes,
        pH=pH,
        include_histidine_in_charge=include_histidine_in_charge,
    )

    _save_table(descriptors, output_path)
    typer.echo(f"Saved sequence descriptors to: {output_path}")


@app.command("describe-fasta")
def describe_fasta(
    fasta_path: Path = typer.Argument(
        ...,
        help="Input FASTA file with protein sequences.",
    ),
    output_path: Path = typer.Option(
        Path("roxy_fasta_descriptors.parquet"),
        "--output-path",
        "-o",
        help="Path where the descriptor table will be written.",
    ),
    aaindex_codes: Optional[List[str]] = typer.Option(
        None,
        "--aaindex-code",
        help=(
            "Optional AAIndex code to include. Repeat the option to pass "
            "multiple codes."
        ),
    ),
    pH: float = typer.Option(
        7.0,
        "--ph",
        help="pH value used for charge-related descriptors.",
    ),
    include_histidine_in_charge: bool = typer.Option(
        False,
        "--include-histidine-in-charge",
        help="Whether to include histidine in the net-charge estimate.",
    ),
) -> None:
    """Compute sequence descriptors directly from a FASTA file."""
    if not fasta_path.exists():
        raise typer.BadParameter(f"FASTA file not found: {fasta_path}")

    codes = _normalize_aaindex_codes(aaindex_codes)
    descriptors = describe_fasta_api(
        str(fasta_path),
        pH=pH,
        include_histidine_in_charge=include_histidine_in_charge,
        aaindex_codes=codes,
    )
    _save_table(descriptors, output_path)
    typer.echo(f"Saved FASTA sequence descriptors to: {output_path}")


@app.command("list-descriptors")
def list_descriptors() -> None:
    """List active descriptor blocks exposed by the sequence API."""
    typer.echo("Active sequence descriptor blocks:")
    for name in list_available_descriptors_api():
        typer.echo(f"  - {name}")


@app.command("validate-sequences")
def validate_sequences(
    input_path: Path = typer.Argument(
        ...,
        help="Input table with at least one protein sequence column.",
    ),
    sequence_column: str = typer.Option(
        "sequence",
        "--sequence-column",
        "-s",
        help="Name of the column containing amino-acid sequences.",
    ),
    max_examples: int = typer.Option(
        10,
        "--max-examples",
        help="Maximum number of invalid-sequence examples to print.",
    ),
) -> None:
    """Validate whether table rows contain canonical protein sequences."""
    df = _load_table(input_path)
    if sequence_column not in df.columns:
        raise typer.BadParameter(
            f"Sequence column {sequence_column!r} was not found in {input_path}."
        )

    report = validate_sequences_api(
        df,
        sequence_column=sequence_column,
    )
    invalid_report = report.loc[~report["is_valid"]]

    if invalid_report.empty:
        typer.echo("All sequences passed canonical protein-sequence validation.")
        return

    typer.echo(
        f"Found {len(invalid_report)} sequence(s) that failed sequence validation."
    )
    for idx, row in invalid_report.head(max_examples).iterrows():
        invalid = row["invalid_residues"]
        typer.echo(
            f"  - row={idx!r} invalid={','.join(invalid) or '<empty>'} "
            f"sequence={row['cleaned']!r}"
        )

    raise typer.Exit(code=1)


def main() -> None:
    """Entry point for `python -m roxy.cli`."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
