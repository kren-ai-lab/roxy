"""``roxy compute`` — run descriptors on a sequence file."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import typer

from roxy import __version__
from roxy.cli._provenance import utc_now, write_provenance
from roxy.cli._utils import _resolve_names
from roxy.cli.init_config import _default_params
from roxy.cli.validate import load_config
from roxy.core.exceptions import RoxyIdError
from roxy.core.io import read_sequences, write_table
from roxy.descriptors import DESCRIPTOR_REGISTRY

if TYPE_CHECKING:
    from roxy.descriptors.base import BaseDescriptor

_BASE_COLS = {"length", "valid_residue_count"}


def _build_instances(
    config: dict[str, dict[str, object]] | None,
    names: list[str],
) -> list[tuple[str, BaseDescriptor]]:
    if config is not None:
        return [(n, DESCRIPTOR_REGISTRY[n](**p)) for n, p in config.items()]
    return [(n, DESCRIPTOR_REGISTRY[n]()) for n in names]


def _run_descriptors(
    instances: list[tuple[str, BaseDescriptor]],
    seqs: list[str],
    ids: list[str],
    *,
    no_progress: bool,
) -> tuple[list[pl.DataFrame], list[str]]:
    frames: list[pl.DataFrame] = []
    desc_names: list[str] = []

    if no_progress:
        for name, inst in instances:
            frames.append(inst.compute(seqs, ids=ids))
            desc_names.append(name)
        return frames, desc_names

    from rich.progress import Progress  # noqa: PLC0415

    with Progress() as progress:
        task = progress.add_task("Computing descriptors...", total=len(instances))
        for name, inst in instances:
            progress.update(task, description=f"[cyan]{name}[/cyan]")
            frames.append(inst.compute(seqs, ids=ids))
            desc_names.append(name)
            progress.advance(task)
    return frames, desc_names


def _concat_frames(frames: list[pl.DataFrame], desc_names: list[str]) -> pl.DataFrame:
    """Horizontally concat descriptor frames, keeping id and base cols only from first."""
    _drop_from_rest = _BASE_COLS | {"id"}
    parts: list[pl.DataFrame] = []
    for i, (df, name) in enumerate(zip(frames, desc_names, strict=True)):
        if i == 0:
            rename = {f"{name}_{col}": col for col in _BASE_COLS if f"{name}_{col}" in df.columns}
            parts.append(df.rename(rename))
        else:
            drop = [c for col in _drop_from_rest for c in (f"{name}_{col}", col) if c in df.columns]
            parts.append(df.drop(drop))
    return functools.reduce(pl.DataFrame.hstack, parts)


def compute(  # noqa: PLR0917
    input_path: Path = typer.Argument(..., metavar="INPUT", help="FASTA, CSV, or Parquet file."),
    config: Path | None = typer.Option(
        None, "--config", "-c", help="YAML config file with descriptor params."
    ),
    all_: bool = typer.Option(False, "--all", help="Run all registered descriptors."),
    descriptors: list[str] = typer.Option([], "--descriptor", "-d", help="Descriptor name (repeatable)."),
    family: list[str] = typer.Option(
        [], "--family", "-f", help="Family name — runs all descriptors in it (repeatable)."
    ),
    output: Path = typer.Option(..., "--output", "-o", help="Output file (.csv or .parquet)."),
    seq_col: str = typer.Option("sequence", "--seq-col", help="Sequence column name (CSV/Parquet)."),
    id_col: str | None = typer.Option(None, "--id-col", help="ID column to preserve (CSV/Parquet)."),
    no_progress: bool = typer.Option(False, "--no-progress", help="Disable progress bar."),
) -> None:
    """Compute descriptors for sequences in INPUT and write to OUTPUT."""
    started = utc_now()
    if config is not None and (all_ or descriptors or family):
        msg = "--config is mutually exclusive with --all / -d / -f."
        raise typer.BadParameter(msg)

    if config is not None:
        cfg = load_config(config)
        effective = {n: {**_default_params(DESCRIPTOR_REGISTRY[n]), **p} for n, p in cfg.items()}
        instances = _build_instances(cfg, [])
    else:
        names = _resolve_names(all_, descriptors, family)
        if not names:
            msg = "Specify --config, --all, -d <name>, or -f <family>."
            raise typer.BadParameter(msg)
        unknown = [n for n in names if n not in DESCRIPTOR_REGISTRY]
        if unknown:
            msg = f"Unknown descriptors: {', '.join(unknown)}"
            raise typer.BadParameter(msg)
        effective = {n: _default_params(DESCRIPTOR_REGISTRY[n]) for n in names}
        instances = _build_instances(None, names)

    try:
        records = read_sequences(input_path, seq_col=seq_col, id_col=id_col)
    except RoxyIdError as exc:
        raise typer.BadParameter(str(exc)) from exc

    ids = [r[0] for r in records]
    seqs = [r[1] for r in records]

    frames, desc_names = _run_descriptors(instances, seqs, ids, no_progress=no_progress)
    result = _concat_frames(frames, desc_names)
    write_table(result, output)

    write_provenance(
        output,
        tool="roxy",
        version=__version__,
        command="compute",
        created_at=started,
        parameters={"descriptors": effective, "seq_col": seq_col, "id_col": id_col},
        inputs=[input_path],
        packages=["roxylib", "numpy", "polars", "scipy", "scikit-learn"],
        output_info={"n_rows": result.shape[0], "n_cols": result.shape[1] - 1},
    )
    typer.echo(f"Wrote {result.shape[0]} sequences x {result.shape[1]} features → {output}")
